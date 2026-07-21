"""v6 B1 — NCRP(비천리안) 라벨 수집기 — champion 방문상태(DAgger)에 플래너 라벨.

exit_labels.py(성능트랙 S4)의 골격을 그대로 승계하되, 라벨러를 **천리안 오라클
(rollout_oracle.q_rollout)**에서 **배포 가능한 NCRP 플래너(planner_policy.
TruncatedRolloutPlanner)**로 교체한다. 이유는 v3 ExIt 의 실패 교훈(docs/성능극대화_사다리
§S4): 천리안 라벨은 obs 로 예측 불가(BC acc 0.19)였다 — rng 비트복제로 미래를 내다본
라벨이라 상태(obs)에 예측 정보가 남지 않는다. NCRP 라벨은 **미래-무지 몬테카를로**로
만들어지므로(현재 obs 가 담은 정보만으로 결정) obs 예측가능성이 열려 있을 수 있다.
P4(ExIt-online) 착수 전 관문 = "NCRP 라벨이 obs 로 예측 가능한가"(특히 switched 결정
한정) 를 측정하는 것 → 이 스크립트가 그 데이터를 만들고, ncrp_label_probe.py 가 잰다.

DAgger 계약(exit_labels 동일): 에피소드 **진행은 항상 champion greedy**(상태분포 =
champion 방문분포), 각 결정 시점의 플래너 결정을 **라벨로만** 저장한다. 플래너는 원본 env
를 deepcopy 로만 건드리므로(planner_policy docstring: act() 전후 원본 obs/mask/ev_manager
불변) "라벨만 뽑고 진행은 greedy" 패턴에 그대로 맞는다.

라벨러 차이(vs exit_labels):
  - exit_labels: q_rollout(clairvoyant deepcopy) argmax = 천리안 오라클.
  - 여기: TruncatedRolloutPlanner(K,h,m, leaf_fn=None, clairvoyant=False, switch_margin=0)
    .act(env, ep_seed, obs) — 복제 후 재시드(미래무지)·h 절단·후보간 CRN·m 회 평균 후
    엄격개선 스위치. last_info 로 switched/n_cand/ms 회수 → 부가 배열로 저장.

재사용 의존(중복 구현 금지):
  planner_policy: TruncatedRolloutPlanner(라벨러 본체 — deepcopy 롤아웃 전부 내부 처리).
  rollout_oracle: _set_env_vars(essential+load·occ — env 빌드 전 설정).
  viper_distill: make_feature_env·load_vecnorm·_masked_probs·_suppress_stdout.
  score_cma: select_tune_regions(--tune_pool40 시군구 균등 40 튜닝풀).
  pointer_policy/hospital_set_extractor: MaskablePPO.load 전 import(역직렬화 필수).

설계 결정:
  - obs = champion vecnorm 동결 정규화본(make_feature_env(norm) 출력) 그대로 저장 —
    BC(bc_pretrain)·이후 PPO resume 과 정규화 일관(exit_labels 동일).
  - 라벨 대상 = 유효액션 ≥2 인 결정 전부(유효 1개는 masked NLL gradient 0 → 제외).
    dedup 후 후보 1개(lookahead 미수행)면 라벨=greedy·switched=False·n_cand=0 로 저장(진행에
    영향 없는 '쉬운' 샘플 — probe 가 switched 로 필터). greedy 는 planner 내부 greedy 와
    비트 동일하게 _masked_probs argmax 로 산출 → switched=last_info["switched"] 정합.
  - seed0 기본 21000: 평가 시드(11000~)·플래너 재시드(777000~)와 분리된 라벨 전용 대역.
  - 재개 = 청크 방식(exit_labels 동일): 잡 결과를 <out>.chunks/*.pkl 로 즉시 저장, 재실행
    시 기존 청크의 (region, ep) 스킵, 마지막에 전 청크 병합 → --out.
  - 비용 감각: 결정당 롤아웃 = K×m×(≤h) step. B0 CSV ms_per_dec≈2878(m16) → 결정당 ~3s
    (m16) → 10만 결정 규모는 워커 다수 필수. ⚠️ 본 수집은 절대 여기서 실행하지 말 것
    (스모크만) — B0 승자(K/h/m) 확정 후 메인이 실행.

예(스모크): PYTHONIOENCODING=utf-8 python src/rl_src/ncrp_labels.py \
    --regions 종로구_11110,중구_11140 --n_eps 2 --K 8 --h 10 --m 2 --workers 2 \
    --out /home/ryu/.claude/jobs/14788e01/tmp/ncrp_smoke.pkl
예(본수집): ... --tune_pool40 --n_eps 30 --K 8 --h 10 --m 8 --workers 32 \
    --out results/rl/redesign/ncrp_labels_v6.pkl   # (K/h/m 은 B0 승자로 교체)
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import pickle
import subprocess
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MCI_REWARD_MODE", "woG")  # 플래너는 info['r_woG'] 직접 읽음(모드 무관)

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SEED0_DEFAULT = 21000  # 평가 시드(11000~)·재시드(777000~)와 분리된 라벨 전용 시드 풀


def _log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def label_episode(fac, model, seed, K, h, m, reseed_base, clairvoyant, switch_margin):
    """champion greedy 로 1 에피소드 진행(DAgger 상태분포), 각 결정(유효액션≥2)마다 NCRP
    플래너 라벨 산출. 진행 액션 = 항상 champion greedy(라벨은 저장만) — DAgger 핵심."""
    from planner_policy import TruncatedRolloutPlanner
    from viper_distill import _masked_probs
    env = fac(seed=seed)
    # 플래너는 에피소드 스코프(내부 _n_dec/_dest_tab/_cloner 상태 보유) → ep 마다 새로 생성
    planner = TruncatedRolloutPlanner(model, K=K, h=h, m=m, leaf_fn=None,
                                      clairvoyant=clairvoyant, reseed_base=reseed_base,
                                      switch_margin=switch_margin)
    obs, _ = env.reset(seed=seed)
    done = False
    S = {"obs": [], "actions": [], "masks": [], "steps": [],
         "greedy": [], "switched": [], "n_cand": [], "plan_ms": [],
         # (v7) 후보 롤아웃 가치(pdrwog 단위, planner last_info). lookahead 미수행 시 nan.
         "q_greedy": [], "q_best": [], "q_exec": [], "dpdr": []}
    n_dec = n_switch = 0
    step = 0
    while not done:
        mask = np.asarray(env.action_masks(), dtype=bool)
        valid = np.flatnonzero(mask)
        if valid.size <= 1:
            # 유효액션 ≤1 — masked NLL gradient 0 이라 샘플 제외, 진행만
            g = int(valid[0]) if valid.size else 0
        else:
            # greedy 를 planner 내부와 동일한 _masked_probs argmax 로 선산출(1 forward pass —
            # 롤아웃 K×m×h 대비 무시가능) → 진행에 사용 + switched=last_info 와 정합 보장
            probs = _masked_probs(model, obs, mask)
            g = int(np.argmax(probs))
            a_label = planner.act(env, ep_seed=seed, obs=obs)  # ★라벨(원본 env 무접촉)
            li = planner.last_info
            switched = bool(li["switched"])
            if li["lookahead"]:        # dedup 후 후보≥2 였던 실제 lookahead 결정
                n_dec += 1
            if switched:
                n_switch += 1
            S["obs"].append(np.asarray(obs, dtype=np.float32).copy())
            S["actions"].append(int(a_label))
            S["masks"].append(mask.copy())
            S["steps"].append(step)
            S["greedy"].append(g)
            S["switched"].append(switched)
            S["n_cand"].append(int(li["n_cand"]))
            S["plan_ms"].append(float(li["ms"]))
            _qn = lambda k: (float(li[k]) if li.get(k) is not None else float("nan"))
            S["q_greedy"].append(_qn("q_greedy")); S["q_best"].append(_qn("q_best"))
            S["q_exec"].append(_qn("q_exec")); S["dpdr"].append(_qn("dpdr"))
        obs, _r, term, trunc, _info = env.step(int(g))  # ★진행 = champion greedy(라벨 아님)
        done = term or trunc
        step += 1
    S["n_dec"], S["n_switch"] = n_dec, n_switch
    return S


# ---------------------------------------------------------------- 병렬 워커
def worker(job):
    """(region, cfg, model_dir, seed0, ep 리스트, K, h, m, reseed_base, clairvoyant,
    switch_margin) → per-ep 라벨 dict 목록."""
    (region, cfg, model_dir, seed0, eps, K, h, m,
     reseed_base, clairvoyant, switch_margin) = job
    from rollout_oracle import _set_env_vars
    _set_env_vars()                                  # essential+load · occ (env 빌드 전)
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
        data = []
        with _suppress_stdout():
            fac = make_feature_env(cfg, norm)
            for ep in eps:
                t0 = time.time()
                S = label_episode(fac, model, seed0 + ep, K, h, m,
                                  reseed_base, clairvoyant, switch_margin)
                S["ep"] = ep
                S["sec"] = round(time.time() - t0, 2)
                data.append(S)
        return {"ok": True, "region": region, "data": data}
    except Exception as e:
        import traceback
        return {"ok": False, "region": region, "err": (str(e) + traceback.format_exc())[:500]}


# ---------------------------------------------------------------- 청크 병합
def merge_chunks(chunk_dir, region_order, out, meta):
    """<out>.chunks/*.pkl 전부 → bc_dataset 규약 pickle 1개. (region, ep) 중복은 첫 것 유지."""
    merged = {}
    for fn in sorted(os.listdir(chunk_dir)):
        if not fn.endswith(".pkl"):
            continue
        try:
            with open(os.path.join(chunk_dir, fn), "rb") as f:
                c = pickle.load(f)
        except Exception:
            _log(f"[ncrp-labels] ⚠️ 손상 청크 스킵(병합): {fn}")
            continue
        for e in c["data"]:
            merged.setdefault((c["region"], e["ep"]), e)
    if not merged:
        raise SystemExit("병합할 청크가 없음 — 라벨 0건")

    def _key(t):  # region_order(매니페스트/튜닝풀 순) → ep 순(결정적 순서)
        return (region_order.index(t[0]) if t[0] in region_order else len(region_order), t[0], t[1])

    obs_l, act_l, mask_l, reg_l, ep_l, st_l = [], [], [], [], [], []
    gr_l, sw_l, nc_l, ms_l = [], [], [], []
    qg_l, qb_l, qe_l, dp_l = [], [], [], []   # (v7) 후보 가치 배열
    n_dec = n_switch = 0
    per_region = {}
    for (region, ep) in sorted(merged, key=_key):
        e = merged[(region, ep)]
        n = len(e["actions"])
        n_dec += e["n_dec"]
        n_switch += e["n_switch"]
        pr = per_region.setdefault(region, [0, 0, 0])  # [samples, n_dec, n_switch]
        pr[0] += n; pr[1] += e["n_dec"]; pr[2] += e["n_switch"]
        if n == 0:
            continue
        obs_l.extend(e["obs"]); act_l.extend(e["actions"]); mask_l.extend(e["masks"])
        reg_l.extend([region] * n)
        ep_l.extend([ep] * n); st_l.extend(e["steps"])
        gr_l.extend(e["greedy"]); sw_l.extend(e["switched"])
        nc_l.extend(e["n_cand"]); ms_l.extend(e["plan_ms"])
        _nan = [float("nan")] * n   # 구 청크(q값 없음) 호환
        qg_l.extend(e.get("q_greedy", _nan)); qb_l.extend(e.get("q_best", _nan))
        qe_l.extend(e.get("q_exec", _nan)); dp_l.extend(e.get("dpdr", _nan))

    if not act_l:
        raise SystemExit("수집된 라벨 샘플 0건(전 결정 유효액션 ≤1?) — 파라미터 확인")
    obs_arr = np.stack(obs_l).astype(np.float32)
    mask_arr = np.stack(mask_l).astype(np.bool_)
    payload = {
        # ---- bc_dataset.py 규약(필수 키 — bc_pretrain 이 읽는 부분) ----
        "obs": obs_arr,
        "actions": np.asarray(act_l, dtype=np.int64),
        "masks": mask_arr,
        "regions": reg_l,
        "obs_dim": int(obs_arr.shape[1]),
        "n_actions": int(mask_arr.shape[1]),
        # ---- 분석용 부가 배열(별도 키 — bc_pretrain 은 무시, probe 가 소비) ----
        "greedy_actions": np.asarray(gr_l, dtype=np.int64),
        "switched": np.asarray(sw_l, dtype=np.bool_),
        "n_cand": np.asarray(nc_l, dtype=np.int32),
        "plan_ms": np.asarray(ms_l, dtype=np.float32),
        "eps": np.asarray(ep_l, dtype=np.int32),
        "steps": np.asarray(st_l, dtype=np.int32),
        # ---- (v7) 후보 롤아웃 가치(pdrwog 단위) — 가치게이트·value-target 학습용 ----
        "q_greedy": np.asarray(qg_l, dtype=np.float32),
        "q_best": np.asarray(qb_l, dtype=np.float32),
        "q_exec": np.asarray(qe_l, dtype=np.float32),
        "dpdr": np.asarray(dp_l, dtype=np.float32),
        # ---- 출처 메타(dict) + 요약 ----
        "meta": meta,
        "n_dec_total": int(n_dec),
        "n_switch_total": int(n_switch),
        "switch_rate": float(n_switch / n_dec) if n_dec else 0.0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    _log(f"[ncrp-labels] 병합 저장: {out}")
    _log(f"  샘플 N={len(act_l)} obs_dim={payload['obs_dim']} n_actions={payload['n_actions']} "
         f"(에피소드 {len(merged)})")
    _log(f"  스위치율(라벨≠greedy / lookahead 결정) = {n_switch}/{n_dec} "
         f"= {payload['switch_rate']:.3f}")
    for r in sorted(per_region, key=lambda x: (region_order.index(x) if x in region_order
                                               else len(region_order), x)):
        s, d, w = per_region[r]
        _log(f"    {r}: 샘플 {s}, 스위치 {w}/{d} ({w / d:.3f})" if d else f"    {r}: 샘플 {s}, 결정 0")
    return payload


def _resolve_pairs(A):
    """--tune_pool40 / --regions / 전체 → [(region, cfg), ...] (planner_eval 관례 승계)."""
    if A.tune_pool40:
        from score_cma import select_tune_regions
        sig = os.path.join(REPO, "scenarios/manifests/sigungu_osrm_manifest.json")
        pairs = select_tune_regions(sig, 40)
        if A.regions:
            want = set(A.regions.split(","))
            pairs = [(k, c) for k, c in pairs if k in want]
        return pairs
    manifest = json.load(open(A.manifest, encoding="utf-8"))
    keys = [k for k in A.regions.split(",") if k in manifest] if A.regions else list(manifest.keys())
    return [(k, manifest[k]) for k in keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.path.join(REPO, "results/rl/redesign/v4_plr2_s0"),
                    help="champion 디렉터리(final_model.zip + vecnormalize.pkl)")
    ap.add_argument("--manifest", default=os.path.join(REPO, "scenarios/manifests/sigungu_osrm_manifest.json"))
    ap.add_argument("--regions", default="", help="쉼표구분 매니페스트 키 서브셋(생략시 전체/튜닝풀)")
    ap.add_argument("--tune_pool40", action="store_true",
                    help="시군구 sigcd 균등 40지역 튜닝풀 사용(score_cma.select_tune_regions)")
    ap.add_argument("--n_eps", type=int, default=30)
    ap.add_argument("--K", type=int, default=8, help="masked-prob 상위 후보 수(플래너)")
    ap.add_argument("--h", type=int, default=10, help="롤아웃 결정 지평(h<0=무한=종단까지)")
    ap.add_argument("--m", type=int, default=8, help="비천리안 몬테카를로 롤아웃 수")
    ap.add_argument("--reseed_base", type=int, default=777000,
                    help="비천리안 재시드 베이스(평가 11000·라벨 21000 과 분리된 대역)")
    ap.add_argument("--clairvoyant", action="store_true",
                    help="재시드 생략(=천리안 오라클 라벨 — v3 대조/디버그용, 기본 비천리안)")
    ap.add_argument("--switch_margin", type=float, default=0.0,
                    help="스위치 마진 ε(pdrwog 단위) — 0=엄격개선(스펙 기본)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed0", type=int, default=SEED0_DEFAULT)
    ap.add_argument("--chunk", type=int, default=5, help="잡당 에피소드 수(부하 균형)")
    ap.add_argument("--out", required=True, help="출력 pickle (bc_dataset 규약)")
    A = ap.parse_args()

    pairs = _resolve_pairs(A)
    if not pairs:
        raise SystemExit("대상 지역 0개 — --regions/--tune_pool40 확인")
    region_order = [k for k, _ in pairs]

    # ---- 재개: 기존 청크의 (region, ep) 스킵 ----
    chunk_dir = A.out + ".chunks"
    os.makedirs(chunk_dir, exist_ok=True)
    done = set()
    for fn in os.listdir(chunk_dir):
        if not fn.endswith(".pkl"):
            continue
        try:
            with open(os.path.join(chunk_dir, fn), "rb") as f:
                c = pickle.load(f)
            done.update((c["region"], e["ep"]) for e in c["data"])
        except Exception:
            _log(f"[ncrp-labels] ⚠️ 손상 청크 무시(재개 스캔): {fn}")
    if done:
        _log(f"[ncrp-labels] 재개 — 기존 청크 (region,ep) {len(done)}건 스킵")

    jobs = []
    for k, cfg in pairs:
        todo = [ep for ep in range(A.n_eps) if (k, ep) not in done]
        for i in range(0, len(todo), A.chunk):
            jobs.append((k, cfg, A.model_dir, A.seed0, todo[i:i + A.chunk], A.K, A.h, A.m,
                         A.reseed_base, A.clairvoyant, A.switch_margin))
    _log(f"[ncrp-labels] regions={len(pairs)} n_eps={A.n_eps} K={A.K} h={A.h} m={A.m} "
         f"clairvoyant={A.clairvoyant} margin={A.switch_margin} reseed_base={A.reseed_base} "
         f"seed0={A.seed0} jobs={len(jobs)} workers={A.workers} out={A.out}")

    if jobs:
        t0, n_ep_done, n_fail = time.time(), 0, 0
        with Pool(min(A.workers, len(jobs)), maxtasksperchild=1) as pool:
            for j, r in enumerate(pool.imap_unordered(worker, jobs), 1):
                if r["ok"]:
                    eps = [e["ep"] for e in r["data"]]
                    tag = f"{r['region']}__e{min(eps)}-{max(eps)}.pkl"
                    with open(os.path.join(chunk_dir, tag), "wb") as f:
                        pickle.dump({"region": r["region"], "data": r["data"]}, f,
                                    protocol=pickle.HIGHEST_PROTOCOL)
                    n_ep_done += len(eps)
                    nd = sum(e["n_dec"] for e in r["data"])
                    ns = sum(e["n_switch"] for e in r["data"])
                    sec = np.mean([e["sec"] for e in r["data"]])
                    _log(f"  [{j}/{len(jobs)}] {r['region']} +{len(eps)}ep 스위치 {ns}/{nd} "
                         f"{sec:.0f}s/ep (누적 {n_ep_done}ep, {time.time() - t0:.0f}s)")
                else:
                    n_fail += 1
                    _log(f"  [{j}/{len(jobs)}] FAIL {r['region']}: {r['err'][:200]}")
        _log(f"[ncrp-labels] 수집 완료 ep={n_ep_done} fail_jobs={n_fail} wall={time.time() - t0:.0f}s")
    else:
        _log("[ncrp-labels] 할 일 없음(전부 완료) — 병합만 수행")

    meta = {"K": A.K, "h": A.h, "m": A.m, "model_dir": A.model_dir, "git_sha": _git_sha(),
            "seed0": A.seed0, "manifest": A.manifest, "clairvoyant": A.clairvoyant,
            "reseed_base": A.reseed_base, "switch_margin": A.switch_margin}
    payload = merge_chunks(chunk_dir, region_order, A.out, meta)
    # 핵심 요약은 stdout 에도(detached 재확인용)
    print(f"[ncrp-labels] N={len(payload['actions'])} switch_rate={payload['switch_rate']:.3f} "
          f"→ {A.out}", flush=True)


if __name__ == "__main__":
    main()
