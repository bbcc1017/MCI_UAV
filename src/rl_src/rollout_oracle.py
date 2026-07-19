"""롤아웃 오라클 headroom 측정 — L3 정책의 1-step 정책개선(Bertsekas lookahead) 상한 추정 (플랜 S0).

목적: L3(results/rl/redesign/L3_pointer_s0)가 도달가능 상한에 얼마나 가까운지 측정.
매 결정 시점에 masked-prob 상위 K 후보(+greedy 포함)를 각각 "그 액션 실행 후 L3 greedy 로
에피소드 종단까지" 롤아웃(env 상태 deepcopy)해 suffix r_woG 누적이 최선인 액션을 실행 —
같은 시드의 순수 L3 greedy 에피소드와 paired 비교. prefix 보상은 후보 간 공통이므로
suffix 비교 = 에피소드 총합 비교와 동치. Δ = mean(pdr_base − pdr_oracle) = headroom.

재사용 의존: viper_distill(make_feature_env·load_vecnorm·_suppress_stdout·_masked_probs),
pointer_policy/hospital_set_extractor(모델 역직렬화 — MaskablePPO.load 전 import 필수),
loadbalance_heuristic._codec_from_mask(액션 코덱 — mask 길이로 uav0=96/기본 192 자동),
paired_eval_ladder 의 워커 패턴(Pool maxtasksperchild=1·스레드 핀·env var 워커 설정).

핵심 설계:
  - 상태 복제 = copy.deepcopy(env). rng(np.random.Generator)까지 비트단위 복제 →
    greedy 후보의 롤아웃 = 순수 greedy 연속과 완전 동일(1-step 개선 보장 성립).
    --validate 로 결정론 사전 검증. 실패 시 --clone replay 폴백(별도 캐시 env 를
    reset(seed) 후 실행 액션 prefix 재생 — 2026-07-03 reset(seed) 재현성 계약 이용).
  - dest=0(stay)은 (c,m) 무관 동일 no-op(EventManager.proceed_action:87) → stay 후보는
    최고확률 1개로 dedup. 유효액션 ≤1 또는 dedup 후 후보 1개면 lookahead 생략(n_dec 미집계).
  - 동률(q 같음)이면 greedy 유지(switch 아님) — 보수적 headroom.
  - 재개 가능: --out CSV 의 기존 (region, ep) 행은 스킵.

예(검증):  PYTHONIOENCODING=utf-8 python src/rl_src/rollout_oracle.py --validate --regions 서울,강원 --n_eps 2
예(본측정): ... --n_eps 100 --topk 8 --workers 40 --out results/rl/redesign/oracle_headroom_sido17.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import copy
import csv
import json
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MCI_REWARD_MODE", "woG")  # 평가는 info['r_woG'] 직접 읽음(모드 무관)

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SEED0_DEFAULT = 11000
H_DEFAULT = 47


def _set_env_vars(respect_existing=False):
    """L3 학습과 동일 조건(essential+load·occ) — env 빌드/step 전에 설정.

    respect_existing(v6): True 면 이미 설정된 env var 를 존중(setdefault) —
    planner_eval --obs_variant/--h_pad 로 v6 모델(essential+load+valid, 402)을
    평가할 때 워커의 하드 강제가 main 설정을 덮어쓰는 것을 방지. 기본 False
    (오라클/리프 등 기존 경로는 하드 강제 유지 — 잔류 env var 오염 방어)."""
    if respect_existing:
        os.environ.setdefault("MCI_OBS_VARIANT", "essential+load")
        os.environ.setdefault("MCI_CAP_GATE", "occ")
    else:
        os.environ["MCI_OBS_VARIANT"] = "essential+load"
        os.environ["MCI_CAP_GATE"] = "occ"


def _dest_table(mask_len, H):
    """flat action → dest 룩업 테이블. 인코딩은 loadbalance_heuristic._codec_from_mask
    (mask 길이 기반 — uav0 auto-pin 96/기본 192 자동 정합)를 그대로 사용해 역산."""
    from loadbalance_heuristic import _codec_from_mask
    enc = _codec_from_mask(mask_len, H)
    n_mode = 2 if mask_len == 2 * (H + 1) * 2 else 1
    tab = np.zeros(mask_len, dtype=np.int32)
    for c in range(2):
        for d in range(H + 1):
            for m in range(n_mode):
                tab[enc(c, d, m)] = d
    return tab


class Cloner:
    """env 상태 복제기. deepcopy(기본) 또는 replay 폴백(fresh env reset(seed)+prefix 재생)."""

    def __init__(self, mode, cfg, norm):
        self.mode = mode
        self._cfg, self._norm = cfg, norm
        self._fac2 = None  # replay 용 별도 캐시 env (deepcopy 모드에선 미생성)

    def clone(self, env, seed, prefix):
        if self.mode == "deepcopy":
            return copy.deepcopy(env)
        # replay 폴백: reset(seed) 가 ev_manager rng 를 재시드(2026-07-03 수정)하므로
        # 같은 시드 + 같은 액션열 재생 = 동일 상태 (단 prefix 길이에 비례해 느림).
        if self._fac2 is None:
            from viper_distill import make_feature_env
            self._fac2 = make_feature_env(self._cfg, self._norm)
        e = self._fac2(seed=seed)
        e.reset(seed=seed)
        for a in prefix:
            e.step(a)
        return e


def q_rollout(env_copy, model, action):
    """복제 env 에 action 적용 후 L3 greedy(마스크 적용 deterministic)로 종단까지 —
    이 시점 이후의 r_woG 누적(=Q 추정) 반환."""
    obs, _r, term, trunc, info = env_copy.step(int(action))
    w = info.get("r_woG", 0.0)
    done = term or trunc
    while not done:
        mask = env_copy.action_masks()
        a, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, _r, term, trunc, info = env_copy.step(int(a))
        w += info.get("r_woG", 0.0)
        done = term or trunc
    return w


def lookahead_episode(fac, model, seed, topk, cloner):
    """한 에피소드: (1) 순수 L3 greedy 베이스라인 → (2) 같은 시드에서 오라클(1-step
    lookahead) 에피소드. 반환 (pdr_base, pdr_oracle, n_dec, n_switch)."""
    from viper_distill import _masked_probs
    env = fac(seed=seed)

    # ---- (1) baseline: 순수 greedy (paired 기준) ----
    obs, _ = env.reset(seed=seed)
    done, w_base = False, 0.0
    while not done:
        mask = env.action_masks()
        a, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, _r, term, trunc, info = env.step(int(a))
        w_base += info.get("r_woG", 0.0)
        done = term or trunc
    prev = env.unwrapped.preventable_woG
    pdr_base = 1.0 - w_base / prev if prev > 0 else 0.0

    # ---- (2) oracle: 매 결정 top-K 후보 롤아웃 argmax ----
    obs, _ = env.reset(seed=seed)
    prev_o = env.unwrapped.preventable_woG
    H = env.unwrapped.H
    done, w = False, 0.0
    n_dec = n_switch = 0
    prefix = []          # replay 폴백용 실행 액션 기록
    dest_tab = None
    while not done:
        mask = np.asarray(env.action_masks(), dtype=bool)
        if dest_tab is None:
            dest_tab = _dest_table(len(mask), H)
        valid = np.flatnonzero(mask)
        if valid.size <= 1:
            a_exec = int(valid[0]) if valid.size else 0
        else:
            probs = _masked_probs(model, obs, mask)
            g = int(np.argmax(probs))  # deterministic greedy = masked argmax
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
            if len(cand) <= 1:
                a_exec = g
            else:
                n_dec += 1
                qs = [q_rollout(cloner.clone(env, seed, prefix), model, a) for a in cand]
                gi, bi = cand.index(g), int(np.argmax(qs))
                if qs[bi] > qs[gi]:    # 엄격 개선일 때만 스위치(동률=greedy 유지)
                    a_exec = cand[bi]
                    if a_exec != g:
                        n_switch += 1
                else:
                    a_exec = g
        obs, _r, term, trunc, info = env.step(a_exec)
        prefix.append(a_exec)
        w += info.get("r_woG", 0.0)
        done = term or trunc
    pdr_oracle = 1.0 - w / prev_o if prev_o > 0 else 0.0
    return pdr_base, pdr_oracle, n_dec, n_switch


# ---------------------------------------------------------------- 복제 결정론 검증
def validate_clone(cfg_path, seed, n_steps, model_dir):
    """env 를 n_steps//2 스텝 진행 → copy.deepcopy → 원본·복제본에 같은 액션열 주입,
    mask/obs/reward/time/r_woG 완전일치 검증. 전체 전제(deepcopy 결정론) 확인용."""
    _set_env_vars()
    import torch as th
    th.set_num_threads(1)
    from sb3_contrib import MaskablePPO
    from hospital_set_extractor import HospitalSetExtractor  # noqa: F401 (역직렬화)
    from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401
    from viper_distill import make_feature_env, load_vecnorm, _suppress_stdout

    # 에피소드가 greedy 기준 ~45스텝으로 짧음 → 사전구간을 짧게 잡아 비교 구간 확보
    n_pre = max(1, min(n_steps // 2, 15))
    res = {"cfg": cfg_path, "seed": seed, "ok": True, "n_cmp": 0, "note": ""}
    with _suppress_stdout():
        vn = os.path.join(model_dir, "vecnormalize.pkl")
        norm = load_vecnorm(vn) if os.path.exists(vn) else None
        model = MaskablePPO.load(os.path.join(model_dir, "final_model.zip"), device="cpu")
        fac = make_feature_env(cfg_path, norm)
        env = fac(seed=seed)
        obs, _ = env.reset(seed=seed)
        done, steps = False, 0
        while not done and steps < n_pre:  # 전반부: greedy 로 중간 상태까지
            mask = env.action_masks()
            a, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, _r, term, trunc, _i = env.step(int(a))
            done = term or trunc
            steps += 1
        if done:
            res["note"] = f"에피소드가 사전구간 {steps}스텝에 종료 — 비교 0스텝(재시도 요망)"
            return res
        t0 = time.time()
        clone = copy.deepcopy(env)
        res["deepcopy_sec"] = time.time() - t0
        for _t in range(n_steps - n_pre):  # 후반부: 같은 액션열로 병행 진행·비교
            m_o = np.asarray(env.action_masks())
            m_c = np.asarray(clone.action_masks())
            if not np.array_equal(m_o, m_c):
                res.update(ok=False, note=f"mask 불일치 @cmp{res['n_cmp']}")
                return res
            a, _ = model.predict(obs, action_masks=m_o, deterministic=True)
            oo, ro, to, tro, io = env.step(int(a))
            oc, rc, tc, trc, ic = clone.step(int(a))
            same = (np.array_equal(np.asarray(oo), np.asarray(oc)) and ro == rc
                    and to == tc and tro == trc
                    and io.get("time") == ic.get("time")
                    and io.get("r_woG") == ic.get("r_woG"))
            if not same:
                res.update(ok=False, note=f"step 결과 불일치 @cmp{res['n_cmp']}")
                return res
            obs = oo
            res["n_cmp"] += 1
            if to or tro:
                res["note"] = f"종단까지 {res['n_cmp']}스텝 전부 일치"
                break
    return res


# ---------------------------------------------------------------- 병렬 워커
def worker(job):
    """(region, cfg, model_dir, seed0, ep 리스트, topk, clone_mode) → per-ep 행 목록."""
    region, cfg, model_dir, seed0, eps, topk, clone_mode = job
    _set_env_vars()
    import torch as th
    th.set_num_threads(1)
    from sb3_contrib import MaskablePPO
    from hospital_set_extractor import HospitalSetExtractor  # noqa: F401
    from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401
    from viper_distill import make_feature_env, load_vecnorm, _suppress_stdout
    try:
        vn = os.path.join(model_dir, "vecnormalize.pkl")
        norm = load_vecnorm(vn) if os.path.exists(vn) else None
        model = MaskablePPO.load(os.path.join(model_dir, "final_model.zip"), device="cpu")
        rows = []
        with _suppress_stdout():
            fac = make_feature_env(cfg, norm)
            cloner = Cloner(clone_mode, cfg, norm)
            for ep in eps:
                t0 = time.time()
                pdr_b, pdr_o, nd, ns = lookahead_episode(fac, model, seed0 + ep, topk, cloner)
                rows.append({"region": region, "ep": ep,
                             "pdr_base": pdr_b, "pdr_oracle": pdr_o,
                             "n_dec": nd, "n_switch": ns,
                             "sec": round(time.time() - t0, 2)})
        return {"ok": True, "region": region, "rows": rows}
    except Exception as e:
        import traceback
        return {"ok": False, "region": region, "err": (str(e) + traceback.format_exc())[:500]}


def _log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


COLS = ["region", "ep", "pdr_base", "pdr_oracle", "n_dec", "n_switch", "sec"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.path.join(REPO, "results/rl/redesign/L3_pointer_s0"))
    ap.add_argument("--manifest", default=os.path.join(REPO, "scenarios/manifests/sido_osrm_manifest.json"))
    ap.add_argument("--regions", default="", help="쉼표구분 매니페스트 키 서브셋(생략시 전체)")
    ap.add_argument("--key_filter", default="", help="매니페스트 키 부분문자열 필터")
    ap.add_argument("--n_eps", type=int, default=100)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=SEED0_DEFAULT)
    ap.add_argument("--chunk", type=int, default=5, help="잡당 에피소드 수(부하 균형)")
    ap.add_argument("--clone", choices=["deepcopy", "replay"], default="deepcopy")
    ap.add_argument("--validate", action="store_true", help="deepcopy 결정론 검증만 하고 종료")
    ap.add_argument("--validate_steps", type=int, default=120)
    ap.add_argument("--out", default=os.path.join(REPO, "results/rl/redesign/oracle_headroom_sido17.csv"))
    A = ap.parse_args()

    manifest = json.load(open(A.manifest, encoding="utf-8"))
    keys = [k for k in A.regions.split(",") if k in manifest] if A.regions else list(manifest.keys())
    if A.key_filter:
        keys = [k for k in keys if A.key_filter in k]
    if not keys:
        raise SystemExit("대상 지역 0개 — --regions/--key_filter 확인")

    # ---- 검증 모드 ----
    if A.validate:
        all_ok = True
        for k in keys:
            for ep in range(A.n_eps):
                r = validate_clone(manifest[k], A.seed0 + ep, A.validate_steps, A.model_dir)
                print(f"[validate] {k} seed={A.seed0+ep}: ok={r['ok']} n_cmp={r['n_cmp']} "
                      f"deepcopy={r.get('deepcopy_sec', float('nan')):.3f}s {r['note']}", flush=True)
                all_ok &= r["ok"]
        print(f"[validate] 종합: {'PASS' if all_ok else 'FAIL — replay 폴백 필요(--clone replay)'}", flush=True)
        return

    # ---- 재개: 기존 CSV 의 (region, ep) 스킵 ----
    done = set()
    if os.path.exists(A.out):
        with open(A.out, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["region"], int(r["ep"])))
        _log(f"[oracle] 재개 — 기존 {len(done)}행 스킵")

    jobs = []
    for k in keys:
        todo = [ep for ep in range(A.n_eps) if (k, ep) not in done]
        for i in range(0, len(todo), A.chunk):
            jobs.append((k, manifest[k], A.model_dir, A.seed0, todo[i:i + A.chunk], A.topk, A.clone))
    _log(f"[oracle] regions={len(keys)} n_eps={A.n_eps} topk={A.topk} clone={A.clone} "
         f"jobs={len(jobs)} workers={A.workers} out={A.out}")
    if not jobs:
        _log("[oracle] 할 일 없음(전부 완료)")
        return

    new_file = not os.path.exists(A.out)
    fout = open(A.out, "a", newline="", encoding="utf-8")
    wcsv = csv.DictWriter(fout, fieldnames=COLS)
    if new_file:
        wcsv.writeheader()
        fout.flush()

    t0, n_rows, n_fail = time.time(), 0, 0
    with Pool(min(A.workers, len(jobs)), maxtasksperchild=1) as pool:
        for j, r in enumerate(pool.imap_unordered(worker, jobs), 1):
            if r["ok"]:
                for row in r["rows"]:
                    wcsv.writerow(row)
                    n_rows += 1
                fout.flush()
                d = np.mean([row["pdr_base"] - row["pdr_oracle"] for row in r["rows"]])
                s = np.mean([row["sec"] for row in r["rows"]])
                _log(f"  [{j}/{len(jobs)}] {r['region']} +{len(r['rows'])}ep Δ={d:+.4f} "
                     f"{s:.0f}s/ep (누적 {n_rows}행, {time.time()-t0:.0f}s)")
            else:
                n_fail += 1
                _log(f"  [{j}/{len(jobs)}] FAIL {r['region']}: {r['err'][:200]}")
    fout.close()
    _log(f"[oracle] 완료 rows={n_rows} fail_jobs={n_fail} wall={time.time()-t0:.0f}s → {A.out}")

    # 요약(Δ = pdr_base − pdr_oracle, 양수 = 오라클이 PDR 낮춤 = headroom)
    per = {}
    with open(A.out, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            per.setdefault(r["region"], []).append(float(r["pdr_base"]) - float(r["pdr_oracle"]))
    ds = [d for v in per.values() for d in v]
    _log(f"[oracle] Δ 전체 mean={np.mean(ds):+.4f} (지역 {len(per)}, 에피소드 {len(ds)})")
    for k in sorted(per, key=lambda x: -np.mean(per[x])):
        _log(f"    {k}: Δ={np.mean(per[k]):+.4f} (n={len(per[k])})")


if __name__ == "__main__":
    main()
