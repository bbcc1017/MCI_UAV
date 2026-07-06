"""ExIt-lite 라벨 수집 (성능트랙 S4 — 1단계) — champion 방문상태(DAgger)에 롤아웃 오라클 라벨.

S0(rollout_oracle.py)에서 검증된 1-step lookahead(top-K deepcopy 롤아웃, headroom Δ=+0.023)를
"측정"에서 "라벨 생성"으로 전환한다: champion 정책이 greedy 로 에피소드를 진행하고(상태분포
= champion 방문분포 — DAgger 핵심: **에피소드 진행은 라벨과 무관하게 champion 액션**), 각
결정 시점의 오라클 argmax(후보별 suffix r_woG 롤아웃 최대)를 라벨로만 저장한다. 출력 pickle
은 bc_dataset.py 규약(obs/actions/masks/obs_dim/n_actions[+regions])을 그대로 따르므로
train_ppo_bc.bc_pretrain(→ exit_distill.py)에 바로 먹일 수 있다.

재사용 의존(중복 구현 금지):
  rollout_oracle: _set_env_vars(essential+load·occ)·_dest_table(stay dedup 테이블)·
    Cloner(deepcopy/replay 상태복제)·q_rollout(후보 Q 추정) — lookahead 로직 전부 재사용.
    lookahead_episode 와의 유일한 차이 = "실행 액션이 항상 champion greedy"(라벨은 저장만).
  viper_distill: make_feature_env·load_vecnorm·_masked_probs·_suppress_stdout.
  pointer_policy/hospital_set_extractor: MaskablePPO.load 전 import(역직렬화 필수).

설계 결정:
  - obs = champion vecnorm **동결 정규화본**(make_feature_env(norm)의 _NormObs 출력) 그대로
    저장 — exit_distill(BC)·이후 PPO resume(vecnormalize.pkl 복사)과 정규화 일관.
  - 라벨 대상 = 유효액션 ≥2 인 결정 상태 전부. lookahead 는 dedup 후 후보 ≥2 일 때만 실행
    (아니면 라벨=greedy, q_*=NaN) — 유효액션 1개 상태는 masked NLL gradient 0 이라 제외.
  - 동률(q 같음)이면 greedy 유지 = rollout_oracle 과 동일(보수적 스위치).
  - seed0 기본 21000: paired 평가 시드(11000~)와 에피소드 풀 분리(평가 오염 방지).
  - 재개 = 청크 방식: 잡(지역×ep 묶음) 결과를 <out>.chunks/*.pkl 로 즉시 저장, 재실행 시
    기존 청크의 (region, ep) 스킵, 마지막에 전 청크 병합 → --out. 병합만 다시 하려면 그냥
    재실행(할 일 0개여도 병합은 수행).
  - 메타 부가배열(regions/eps/steps/greedy_actions/q_greedy/q_oracle): 분석용 별도 키 —
    bc_pretrain/load_bc_dataset 은 필수 키만 읽으므로 무시된다.
  - 비용 감각: S0 실측 ~19s/ep(K=8, 워커36). 본실행 ~80지역×30ep 규모.

예(스모크): PYTHONIOENCODING=utf-8 python src/rl_src/exit_labels.py \
    --regions 서울,강원 --n_eps 2 --topk 4 --workers 8 --out /tmp/exit_smoke.pkl
예(본실행): ... --regions_csv results/rl/redesign/oracle_headroom_sido17.csv --regions_topn 80 \
    --n_eps 30 --topk 8 --workers 36 --out results/rl/redesign/exit_labels_v1.pkl
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import json
import pickle
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MCI_REWARD_MODE", "woG")  # q_rollout 은 info['r_woG'] 직접 읽음(모드 무관)

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SEED0_DEFAULT = 21000  # 평가 시드(11000~)와 분리된 라벨 수집 전용 시드 풀


def _log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def label_episode(fac, model, seed, topk, cloner):
    """champion greedy 로 1 에피소드 진행(DAgger 상태분포), 각 결정(유효액션≥2)마다 오라클
    라벨 산출. 후보 구성·롤아웃·동률규칙은 rollout_oracle.lookahead_episode 와 동일 —
    차이는 '실행 액션 = 항상 champion greedy'(오라클 라벨은 저장만) 하나뿐."""
    from rollout_oracle import _dest_table, q_rollout
    from viper_distill import _masked_probs
    env = fac(seed=seed)
    obs, _ = env.reset(seed=seed)
    H = env.unwrapped.H
    done = False
    prefix = []          # replay 폴백용 실행 액션 기록(항상 champion 액션)
    dest_tab = None
    S = {"obs": [], "actions": [], "masks": [], "steps": [],
         "greedy": [], "q_greedy": [], "q_oracle": []}
    n_dec = n_switch = 0
    step = 0
    while not done:
        mask = np.asarray(env.action_masks(), dtype=bool)
        if dest_tab is None:
            dest_tab = _dest_table(len(mask), H)
        valid = np.flatnonzero(mask)
        if valid.size <= 1:
            # 유효액션 ≤1 — masked NLL gradient 0 이라 샘플 제외, 진행만
            g = int(valid[0]) if valid.size else 0
        else:
            probs = _masked_probs(model, obs, mask)
            g = int(np.argmax(probs))  # deterministic greedy = masked argmax
            # ---- 후보 구성(rollout_oracle 과 동일: top-K + stay dedup + greedy 보장) ----
            order = np.argsort(-probs)
            cand, seen_stay = [], False
            for x in order[:topk]:
                x = int(x)
                if not mask[x] or probs[x] <= 0:
                    continue
                if dest_tab[x] == 0:   # stay 는 (c,m) 무관 동일 no-op → 1개만
                    if seen_stay:
                        continue
                    seen_stay = True
                cand.append(x)
            if g not in cand:          # 안전 가드(order[0]=g 라 항상 포함이긴 함)
                cand.append(g)
            label, q_g, q_o = g, float("nan"), float("nan")
            if len(cand) > 1:
                n_dec += 1
                qs = [q_rollout(cloner.clone(env, seed, prefix), model, a) for a in cand]
                gi, bi = cand.index(g), int(np.argmax(qs))
                q_g, q_o = float(qs[gi]), float(qs[bi])
                if qs[bi] > qs[gi]:    # 엄격 개선일 때만 스위치(동률=greedy 유지)
                    label = cand[bi]
                    if label != g:
                        n_switch += 1
            S["obs"].append(np.asarray(obs, dtype=np.float32).copy())
            S["actions"].append(int(label))
            S["masks"].append(mask.copy())
            S["steps"].append(step)
            S["greedy"].append(g)
            S["q_greedy"].append(q_g)
            S["q_oracle"].append(q_o)
        obs, _r, term, trunc, _info = env.step(int(g))  # ★진행은 champion 액션(라벨 아님)
        prefix.append(int(g))
        done = term or trunc
        step += 1
    S["n_dec"], S["n_switch"] = n_dec, n_switch
    return S


# ---------------------------------------------------------------- 병렬 워커
def worker(job):
    """(region, cfg, model_dir, seed0, ep 리스트, topk, clone_mode) → per-ep 라벨 dict 목록."""
    region, cfg, model_dir, seed0, eps, topk, clone_mode = job
    from rollout_oracle import _set_env_vars, Cloner
    _set_env_vars()
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
            cloner = Cloner(clone_mode, cfg, norm)
            for ep in eps:
                t0 = time.time()
                S = label_episode(fac, model, seed0 + ep, topk, cloner)
                S["ep"] = ep
                S["sec"] = round(time.time() - t0, 2)
                data.append(S)
        return {"ok": True, "region": region, "data": data}
    except Exception as e:
        import traceback
        return {"ok": False, "region": region, "err": (str(e) + traceback.format_exc())[:500]}


# ---------------------------------------------------------------- 지역 선정
def regions_from_csv(path, topn, manifest_keys):
    """CSV 상위 N 지역 선정. 헤드룸 CSV(pdr_base/pdr_oracle 컬럼)면 지역평균
    Δ=pdr_base−pdr_oracle 내림차순, 아니면 weight 컬럼 내림차순. region 값은 매니페스트
    키와 동일해야 한다(시군구 CSV 의 sigcd 매칭 이슈는 호출측에서 키를 맞춰 줄 것)."""
    with open(path, encoding="utf-8-sig") as f:  # 시군구 CSV 관례상 BOM 대응
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"빈 CSV: {path}")
    cols = set(rows[0].keys())
    if {"pdr_base", "pdr_oracle"} <= cols:
        acc = {}
        for r in rows:
            acc.setdefault(r["region"], []).append(float(r["pdr_base"]) - float(r["pdr_oracle"]))
        score = {k: float(np.mean(v)) for k, v in acc.items()}
    elif "weight" in cols:
        score = {r["region"]: float(r["weight"]) for r in rows}
    else:
        raise SystemExit(f"CSV 형식 인식 불가(pdr_base/pdr_oracle 또는 weight 필요): {sorted(cols)}")
    ranked = [k for k in sorted(score, key=lambda k: -score[k]) if k in manifest_keys]
    unknown = [k for k in score if k not in manifest_keys]
    if unknown:
        _log(f"[labels] ⚠️ CSV 지역 {len(unknown)}개가 매니페스트에 없음(예: {unknown[:3]}) — 제외")
    return ranked[:topn]


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
            _log(f"[labels] ⚠️ 손상 청크 스킵(병합): {fn}")
            continue
        for e in c["data"]:
            merged.setdefault((c["region"], e["ep"]), e)
    if not merged:
        raise SystemExit("병합할 청크가 없음 — 라벨 0건")

    def _key(t):  # 매니페스트 순 region → ep 순(결정적 순서)
        return (region_order.index(t[0]) if t[0] in region_order else len(region_order), t[0], t[1])

    obs_l, act_l, mask_l, reg_l, ep_l, st_l, gr_l, qg_l, qo_l = [], [], [], [], [], [], [], [], []
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
        ep_l.extend([ep] * n); st_l.extend(e["steps"]); gr_l.extend(e["greedy"])
        qg_l.extend(e["q_greedy"]); qo_l.extend(e["q_oracle"])

    obs_arr = np.stack(obs_l).astype(np.float32)
    mask_arr = np.stack(mask_l).astype(np.bool_)
    dims = ({obs_arr.shape[1]}, {mask_arr.shape[1]})
    payload = {
        # ---- bc_dataset.py 규약(필수 키 — bc_pretrain 이 읽는 부분) ----
        "obs": obs_arr,
        "actions": np.asarray(act_l, dtype=np.int64),
        "masks": mask_arr,
        "regions": reg_l,
        "obs_dim": int(obs_arr.shape[1]),
        "n_actions": int(mask_arr.shape[1]),
        # ---- 분석용 부가 배열(별도 키 — bc_pretrain 은 무시) ----
        "eps": np.asarray(ep_l, dtype=np.int32),
        "steps": np.asarray(st_l, dtype=np.int32),
        "greedy_actions": np.asarray(gr_l, dtype=np.int64),
        "q_greedy": np.asarray(qg_l, dtype=np.float32),
        "q_oracle": np.asarray(qo_l, dtype=np.float32),
        # ---- 출처 메타 ----
        **meta,
        "n_dec_total": int(n_dec),
        "n_switch_total": int(n_switch),
        "switch_rate": float(n_switch / n_dec) if n_dec else 0.0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    _log(f"[labels] 병합 저장: {out}")
    _log(f"  샘플 N={len(act_l)} obs_dim={payload['obs_dim']} n_actions={payload['n_actions']} "
         f"(에피소드 {len(merged)}) dims_check={dims}")
    _log(f"  스위치율(라벨≠greedy / lookahead 결정) = {n_switch}/{n_dec} "
         f"= {payload['switch_rate']:.3f}")
    for r in sorted(per_region, key=lambda x: (region_order.index(x) if x in region_order
                                               else len(region_order), x)):
        s, d, w = per_region[r]
        _log(f"    {r}: 샘플 {s}, 스위치 {w}/{d} ({w / d:.3f})" if d else f"    {r}: 샘플 {s}, 결정 0")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.path.join(REPO, "results/rl/redesign/L3_pointer_s0"),
                    help="champion 모델 디렉터리(final_model.zip + vecnormalize.pkl)")
    ap.add_argument("--manifest", default=os.path.join(REPO, "scenarios/manifests/sido_osrm_manifest.json"))
    ap.add_argument("--regions", default="", help="쉼표구분 매니페스트 키 서브셋(생략시 전체/CSV)")
    ap.add_argument("--regions_csv", default="",
                    help="region_weights(region,weight) 또는 headroom(pdr_base/pdr_oracle) CSV — "
                         "상위 --regions_topn 지역 선정(--regions 미지정 시)")
    ap.add_argument("--regions_topn", type=int, default=80)
    ap.add_argument("--n_eps", type=int, default=30)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--workers", type=int, default=36)
    ap.add_argument("--seed0", type=int, default=SEED0_DEFAULT)
    ap.add_argument("--chunk", type=int, default=5, help="잡당 에피소드 수(부하 균형)")
    ap.add_argument("--clone", choices=["deepcopy", "replay"], default="deepcopy")
    ap.add_argument("--out", required=True, help="출력 pickle (bc_dataset 규약)")
    A = ap.parse_args()

    manifest = json.load(open(A.manifest, encoding="utf-8"))
    if A.regions:
        keys = [k for k in A.regions.split(",") if k in manifest]
    elif A.regions_csv:
        keys = regions_from_csv(A.regions_csv, A.regions_topn, set(manifest))
    else:
        keys = list(manifest.keys())
    if not keys:
        raise SystemExit("대상 지역 0개 — --regions/--regions_csv 확인")

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
            _log(f"[labels] ⚠️ 손상 청크 무시(재개 스캔): {fn}")
    if done:
        _log(f"[labels] 재개 — 기존 청크 (region,ep) {len(done)}건 스킵")

    jobs = []
    for k in keys:
        todo = [ep for ep in range(A.n_eps) if (k, ep) not in done]
        for i in range(0, len(todo), A.chunk):
            jobs.append((k, manifest[k], A.model_dir, A.seed0, todo[i:i + A.chunk], A.topk, A.clone))
    _log(f"[labels] regions={len(keys)} n_eps={A.n_eps} topk={A.topk} clone={A.clone} "
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
        _log(f"[labels] 수집 완료 ep={n_ep_done} fail_jobs={n_fail} wall={time.time() - t0:.0f}s")
    else:
        _log("[labels] 할 일 없음(전부 완료) — 병합만 수행")

    meta = {"manifest": A.manifest, "model_dir": A.model_dir, "topk": A.topk,
            "seed0": A.seed0, "clone": A.clone}
    payload = merge_chunks(chunk_dir, keys, A.out, meta)
    # 핵심 요약은 stdout 에도(detached 재확인용)
    print(f"[labels] N={len(payload['actions'])} switch_rate={payload['switch_rate']:.3f} "
          f"→ {A.out}", flush=True)


if __name__ == "__main__":
    main()
