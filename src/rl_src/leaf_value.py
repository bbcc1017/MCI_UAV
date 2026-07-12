"""P1 NCRP — 신선한 리프 가치망 (계획 §4.1 표 #1).

절단 롤아웃 플래너(planner_policy.TruncatedRolloutPlanner)가 h-결정 지평에서 롤아웃을 끊고
남은 가치를 부트스트랩할 때 쓰는 **무할인 리프 가치망**을 수집·학습·평가한다.

왜 신선한 망인가(계획 부록 A-2): 챔피언 MaskablePPO 크리틱 V 는 VecNormalize 의
norm_reward 단위 + γ0.99 할인으로 학습돼 있어, 플래너가 누적하는 **무할인 raw r_woG**
suffix 와 단위가 맞지 않는다(합산 불가). 그래서 별도로 (정규화 obs355, suffix-to-go) 회귀망을
학습한다.

★ 라벨 단위 결정 = pdrwog(=Σr_woG/preventable_woG) suffix-to-go (계획 §4.1 표 권장):
  라벨 y_t = (그 결정스텝 t 부터 에피소드 끝까지의 무할인 r_woG 합) / preventable_woG.
  preventable_woG 는 사고규모에 비례하므로, r_woG 합을 그대로 쓰면 지역/규모마다 스케일이
  달라 학습이 어렵다. preventable 로 나눈 pdrwog 단위는 사고규모 불변(0~1 권)이라 전국
  단일 회귀망이 성립한다. planner_policy 는 이 예측값을 다시 ×preventable_woG 로 환산해
  r_woG 단위 suffix 누적에 더한다(단위 정합은 planner_policy 주석 참조).

수집 관례:
  - 대상 = sigungu_osrm_manifest(250 학습풀) 전 지역 × --eps_per_region(기본 20).
  - 시드 = 20000 + region_idx*1000 + ep (판정 CRN 11000 오염 절대 금지 — 계획 §4.1·§8).
  - obs = 챔피언 vecnorm **동결 정규화본**(make_feature_env(norm) 의 _NormObs 출력) 그대로 —
    플래너가 롤아웃 중 마주치는 obs 와 동일 정규화(단위 일관).
  - 챔피언 greedy 로 에피소드 진행, 매 스텝 (obs_t, r_woG_t) 저장 → 종료 후 역방향 누적으로
    suffix-to-go 라벨 확정(단조 비증가는 아님 — 후반 결정도 미래 입원 보상을 일부 남김).

CLI: collect / train / eval 서브커맨드. load_leaf(path) 헬퍼가 플래너에 콜백 제공.

예(미니 수집): PYTHONIOENCODING=utf-8 python src/rl_src/leaf_value.py collect \
    --regions_limit 5 --eps_per_region 3 --workers 4 --out /tmp/leaf_smoke.npz
예(학습):     python src/rl_src/leaf_value.py train --dataset /tmp/leaf_smoke.npz \
    --epochs 3 --out /tmp/leaf_smoke.pt
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MCI_REWARD_MODE", "woG")  # info['r_woG'] 직접 읽음(보상모드 무관)

import numpy as np
import warnings as _warnings
_warnings.filterwarnings("ignore", message=r".*action_masks.*")   # NormObs 경유 접근 경고
_warnings.filterwarnings("ignore", category=UserWarning, module=r".*gymnasium.*")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
LEAF_SEED0 = 20000     # 리프 수집 전용 시드 오프셋(CRN 11000 과 분리)


def _log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


# ================================================================= 리프 가치망
def make_leaf_net(in_dim=355, hidden=(256, 256)):
    """MLP[256,256] ReLU, in_dim→1 회귀 헤드."""
    import torch.nn as nn
    layers, d = [], in_dim
    for hdim in hidden:
        layers += [nn.Linear(d, hdim), nn.ReLU()]
        d = hdim
    layers += [nn.Linear(d, 1)]
    return nn.Sequential(*layers)


def load_leaf(path, device="cpu"):
    """저장된 리프 가치망 → callable(obs_batch (B,355) np.ndarray) -> (B,) np.ndarray.
    플래너(planner_policy)가 이 콜백을 leaf_fn 으로 받아 부트스트랩에 사용한다.
    반환값 단위 = pdrwog suffix(=Σr_woG/preventable) 예측 → 플래너가 ×preventable 로 환산."""
    import torch as th
    ckpt = th.load(path, map_location=device)
    net = make_leaf_net(ckpt["in_dim"], tuple(ckpt["hidden"]))
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()

    def fn(obs_batch):
        x = np.asarray(obs_batch, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        with th.no_grad():
            t = th.as_tensor(x, device=device)
            return net(t).squeeze(-1).cpu().numpy().reshape(-1)
    return fn


# ================================================================= collect
def _collect_worker(job):
    """지역 1곳의 챔피언 greedy 에피소드들에서 (정규화 obs, pdrwog suffix-to-go) 수집."""
    region_idx, region, cfg, model_dir, eps = job
    from rollout_oracle import _set_env_vars
    _set_env_vars()                                  # essential+load · occ
    import torch as th
    th.set_num_threads(1)
    from sb3_contrib import MaskablePPO
    from hospital_set_extractor import HospitalSetExtractor  # noqa: F401 (역직렬화)
    from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401
    from viper_distill import make_feature_env, load_vecnorm, _suppress_stdout
    try:
        vn = os.path.join(model_dir, "vecnormalize.pkl")
        norm = load_vecnorm(vn) if os.path.exists(vn) else None
        model = MaskablePPO.load(os.path.join(model_dir, "final_model.zip"), device="cpu")
        obs_rows, y_rows = [], []
        with _suppress_stdout():
            fac = make_feature_env(cfg, norm)
            for ep in eps:
                seed = LEAF_SEED0 + region_idx * 1000 + ep
                env = fac(seed=seed)
                obs, _ = env.reset(seed=seed)
                done = False
                ep_obs, ep_r = [], []
                while not done:
                    mask = env.action_masks()
                    a, _ = model.predict(obs, action_masks=mask, deterministic=True)
                    ep_obs.append(np.asarray(obs, dtype=np.float32).copy())  # 결정 전 obs_t
                    obs, _r, term, trunc, info = env.step(int(a))
                    ep_r.append(float(info.get("r_woG", 0.0)))               # a_t 의 r_woG
                    done = term or trunc
                prev = float(env.unwrapped.preventable_woG)
                if prev <= 0 or not ep_obs:
                    continue
                # suffix-to-go / preventable : 역방향 누적 → 각 결정스텝 라벨
                suffix = 0.0
                ys = [0.0] * len(ep_r)
                for t in range(len(ep_r) - 1, -1, -1):
                    suffix += ep_r[t]
                    ys[t] = suffix / prev
                obs_rows.extend(ep_obs)
                y_rows.extend(ys)
        obs_arr = (np.stack(obs_rows).astype(np.float32) if obs_rows
                   else np.zeros((0, 0), np.float32))
        y_arr = np.asarray(y_rows, dtype=np.float32)
        return {"ok": True, "region": region, "region_idx": region_idx,
                "obs": obs_arr, "y": y_arr, "n_ep": len(eps)}
    except Exception as e:
        import traceback
        return {"ok": False, "region": region, "err": (str(e) + traceback.format_exc())[:500]}


def cmd_collect(A):
    manifest = json.load(open(A.manifest, encoding="utf-8"))
    all_keys = list(manifest.keys())
    keys = all_keys[:A.regions_limit] if A.regions_limit > 0 else all_keys
    jobs = [(i, k, manifest[k], A.model_dir, list(range(A.eps_per_region)))
            for i, k in enumerate(keys)]
    _log(f"[leaf.collect] regions={len(keys)} eps/region={A.eps_per_region} "
         f"workers={A.workers} seed0={LEAF_SEED0} out={A.out}")

    obs_all, y_all, rid_all = [], [], []
    t0, n_fail = time.time(), 0
    with Pool(min(A.workers, len(jobs)), maxtasksperchild=1) as pool:
        for j, r in enumerate(pool.imap_unordered(_collect_worker, jobs), 1):
            if r["ok"]:
                if r["obs"].shape[0]:
                    obs_all.append(r["obs"])
                    y_all.append(r["y"])
                    rid_all.append(np.full(r["obs"].shape[0], r["region_idx"], dtype=np.int32))
                _log(f"  [{j}/{len(jobs)}] {r['region']} +{r['obs'].shape[0]}샘플 "
                     f"({r['n_ep']}ep, {time.time()-t0:.0f}s)")
            else:
                n_fail += 1
                _log(f"  [{j}/{len(jobs)}] FAIL {r['region']}: {r['err'][:200]}")
    if not obs_all:
        raise SystemExit("수집 샘플 0 — 실패 로그 확인")
    obs = np.concatenate(obs_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    rid = np.concatenate(rid_all, axis=0)
    os.makedirs(os.path.dirname(os.path.abspath(A.out)) or ".", exist_ok=True)
    np.savez(A.out, obs=obs, y=y, region_ids=rid, region_keys=np.array(keys))
    print(f"[leaf.collect] 저장 {A.out}  N={obs.shape[0]} obs_dim={obs.shape[1]} "
          f"regions={len(np.unique(rid))} fail={n_fail}", flush=True)
    print(f"  y(pdrwog suffix): min={y.min():.4f} max={y.max():.4f} mean={y.mean():.4f} "
          f"std={y.std():.4f}  wall={time.time()-t0:.0f}s", flush=True)


# ================================================================= train / eval
def _region_split(region_ids, val_frac=0.1, seed=0):
    """지역 단위 train/val 분할 → val_mask(bool). 같은 지역이 양쪽에 안 가게(리키지 방지)."""
    uniq = np.unique(region_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_regions = set(perm[:n_val].tolist())
    return np.isin(region_ids, list(val_regions)), sorted(val_regions)


def cmd_train(A):
    import torch as th
    import torch.nn as nn
    th.set_num_threads(min(8, os.cpu_count() or 8))   # 단일프로세스 학습이라 소폭 병렬 허용
    d = np.load(A.dataset)
    obs, y, rid = d["obs"].astype(np.float32), d["y"].astype(np.float32), d["region_ids"]
    val_mask, val_regions = _region_split(rid, A.val_frac, seed=0)
    Xtr, ytr = obs[~val_mask], y[~val_mask]
    Xva, yva = obs[val_mask], y[val_mask]
    _log(f"[leaf.train] N={obs.shape[0]} obs_dim={obs.shape[1]}  "
         f"train={Xtr.shape[0]} val={Xva.shape[0]} (val지역 {len(val_regions)}/{len(np.unique(rid))})")

    dev = th.device(A.device)
    net = make_leaf_net(obs.shape[1], (256, 256)).to(dev)
    opt = th.optim.Adam(net.parameters(), lr=A.lr)
    lossf = nn.MSELoss()
    Xtr_t = th.as_tensor(Xtr, device=dev)
    ytr_t = th.as_tensor(ytr, device=dev)
    Xva_t = th.as_tensor(Xva, device=dev)
    yva_t = th.as_tensor(yva, device=dev)
    n = Xtr_t.shape[0]

    best = {"mae": float("inf"), "state": None, "mse": float("inf"), "epoch": -1}
    for ep in range(A.epochs):
        net.train()
        perm = th.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, A.batch):
            idx = perm[i:i + A.batch]
            opt.zero_grad()
            pred = net(Xtr_t[idx]).squeeze(-1)
            loss = lossf(pred, ytr_t[idx])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * idx.shape[0]  # detach: grad 텐서 스칼라화 경고 방지
        net.eval()
        with th.no_grad():
            pv = net(Xva_t).squeeze(-1)
            vmse = float(((pv - yva_t) ** 2).mean())
            vmae = float((pv - yva_t).abs().mean())
        _log(f"  [ep {ep}] train_mse={tot/n:.5f}  val_mse={vmse:.5f} val_mae={vmae:.5f}")
        if vmae < best["mae"]:
            best = {"mae": vmae, "mse": vmse, "epoch": ep,
                    "state": {k: v.cpu().clone() for k, v in net.state_dict().items()}}

    os.makedirs(os.path.dirname(os.path.abspath(A.out)) or ".", exist_ok=True)
    th.save({"state_dict": best["state"], "in_dim": int(obs.shape[1]), "hidden": [256, 256],
             "val_mse": best["mse"], "val_mae": best["mae"], "best_epoch": best["epoch"],
             "n_train": int(Xtr.shape[0]), "n_val": int(Xva.shape[0]),
             "dataset": os.path.abspath(A.dataset)}, A.out)
    print(f"[leaf.train] 저장 {A.out}  best(ep{best['epoch']}) val_mse={best['mse']:.5f} "
          f"val_mae={best['mae']:.5f}", flush=True)


def cmd_eval(A):
    import torch as th
    net_fn = load_leaf(A.model, device=A.device)
    d = np.load(A.dataset)
    obs, y, rid = d["obs"].astype(np.float32), d["y"].astype(np.float32), d["region_ids"]
    val_mask, val_regions = _region_split(rid, A.val_frac, seed=0)
    Xva, yva = obs[val_mask], y[val_mask]
    pred = net_fn(Xva)
    mae = float(np.abs(pred - yva).mean())
    mse = float(((pred - yva) ** 2).mean())
    corr = float(np.corrcoef(pred, yva)[0, 1]) if Xva.shape[0] > 1 else float("nan")
    print(f"[leaf.eval] val N={Xva.shape[0]} (지역 {len(val_regions)})", flush=True)
    print(f"  corr={corr:.4f}  MAE={mae:.5f}  MSE={mse:.5f}", flush=True)
    print(f"  target: min={yva.min():.4f} max={yva.max():.4f} mean={yva.mean():.4f}", flush=True)
    print(f"  pred  : min={pred.min():.4f} max={pred.max():.4f} mean={pred.mean():.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="P1 NCRP 리프 가치망 (collect/train/eval)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("collect", help="챔피언 greedy 에피소드 → (obs, pdrwog suffix) npz")
    pc.add_argument("--model_dir", default=os.path.join(REPO, "results/rl/redesign/v4_plr2_s0"))
    pc.add_argument("--manifest", default=os.path.join(REPO, "scenarios/manifests/sigungu_osrm_manifest.json"))
    pc.add_argument("--eps_per_region", type=int, default=20)
    pc.add_argument("--regions_limit", type=int, default=0, help=">0 이면 앞 N 지역만(부분 수집)")
    pc.add_argument("--workers", type=int, default=8)
    pc.add_argument("--out", default=os.path.join(REPO, "results/rl/redesign/leaf_dataset.npz"))

    pt = sub.add_parser("train", help="npz → MLP[256,256] 회귀 학습")
    pt.add_argument("--dataset", default=os.path.join(REPO, "results/rl/redesign/leaf_dataset.npz"))
    pt.add_argument("--epochs", type=int, default=20)
    pt.add_argument("--batch", type=int, default=4096)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--val_frac", type=float, default=0.1)
    pt.add_argument("--device", default="cpu")
    pt.add_argument("--out", default=os.path.join(REPO, "results/rl/redesign/leaf_value.pt"))

    pe = sub.add_parser("eval", help="val 산점 수치 요약(상관·MAE)")
    pe.add_argument("--dataset", default=os.path.join(REPO, "results/rl/redesign/leaf_dataset.npz"))
    pe.add_argument("--model", default=os.path.join(REPO, "results/rl/redesign/leaf_value.pt"))
    pe.add_argument("--val_frac", type=float, default=0.1)
    pe.add_argument("--device", default="cpu")

    A = ap.parse_args()
    if A.cmd == "collect":
        cmd_collect(A)
    elif A.cmd == "train":
        cmd_train(A)
    elif A.cmd == "eval":
        cmd_eval(A)


if __name__ == "__main__":
    main()
