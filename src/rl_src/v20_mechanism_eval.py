# -*- coding: utf-8 -*-
"""v20 기전 계측 — PDR 을 "누가 수술실에 못 들어갔나" 와 "언제 들어갔나" 로 정확히 분해한다.

## 왜 필요한가

보상 정의(`MCIEnvironment_gymnasium.py:178-207`)를 그대로 쓰면 손실이 **항등식으로** 쪼개진다.

    preventable_woG = sum_{R,Y 구조} p(t_rescue)      ← 구조 즉시 처치했다면 얻는 최대
    reward_woG      = sum_{R,Y 처치개시} p(t_admit)   ← p_admit 은 **수술실 배정 순간**
    ------------------------------------------------------------------
    preventable - reward =  sum_{진입}    [p(t_rescue) - p(t_admit)]   ← 지연 손실
                          + sum_{미진입}  p(t_rescue)                  ← 미진입 손실

    PDR_woG = 지연손실/preventable + 미진입손실/preventable

이 분해가 필요한 이유는 부하항의 기전을 잘못 읊었기 때문이다. 처음에는 "부하항이 대기시간을
줄인다" 고 설명했는데, 처치를 시작한 환자만 보면 97.8% 가 대기 0 이다(인계 5분 정확히).
그건 **생존자 편향**이었다 — 줄에서 못 빠져나온 환자는 `care_start` 가 안 찍혀 표본에 없다.
실제로 부하항이 바꾸는 것은 **수술실에 들어간 환자 수**다. 그것을 지표로 못 박아 둔다.

## 게이트가 두 개다 (혼동 주의)

    전원(diversion) 게이트 : 병원 도착 시 n_occupied < 수술실수 + 병상수 (중앙값 약 14) — 거의 안 걸린다
    수술실 게이트          : 처치 시작 시 n_idle > 0, 즉 수술실수 (중앙값 2)  — 이쪽이 병목

## 계측 방식

시뮬 코어를 수정하지 않는다. `EventManager.enable_trace` 를 켜면 rescue / hospital_arrival /
care_start / care_complete / diversion 이 시각·환자id·병원id·등급과 함께 기록되므로 전부 사후 재구성된다.
⚠️ enable_trace 는 리스트에 append 만 하고 난수를 쓰지 않으므로 결과가 바뀌지 않는다.
그 사실을 `--gate` 로 실측 검증한다(같은 좌표·시드에서 trace on/off 의 PDR 비교).

정책 스펙·시드·CSV 규약은 `v17_rule_eval.py` 와 동일하다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(__file__))

REPO = Path(__file__).resolve().parents[2]
COLS = [
    "region", "policy", "episode", "seed",
    "pdr_woG", "reward_woG", "preventable_woG",
    # ★ 분해 (두 항의 합 = pdr_woG, 항등식으로 검증한다)
    "share_delay", "share_missing",
    # 수술실 진입
    "n_resc_RY", "n_arr_RY", "n_or_RY", "or_rate_RY",
    "n_resc_R", "n_or_R", "or_rate_R", "n_resc_Y", "n_or_Y", "or_rate_Y",
    # 시각·대기
    "t_admit_med_R", "t_admit_med_Y", "wait_med_RY", "wait_p90_RY", "frac_wait_pos",
    # 즉시 진입 / 병상 대기 후 진입 (from_queue 플래그로 구분)
    "n_or_now", "n_or_queued", "frac_queued", "wait_med_queued",
    # 분산·전원
    "n_hosp_used", "gini_hosp", "n_diversion", "sim_time", "n_decisions",
]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _gini(counts) -> float:
    x = np.sort(np.asarray([c for c in counts if c > 0], float))
    if x.size == 0:
        return 0.0
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def rollout(factory, policy, seed: int):
    """한 에피소드 실행 후 기전 지표를 반환. trace 를 켠 상태로 돌린다."""
    env = factory(seed=seed)
    u = env.unwrapped
    # ⚠️ 계측을 reset 前에 켜야 한다. EventManager.start() 가 사고 발생과 첫 decision epoch 까지를
    #    처리하면서 그 구간의 rescue 를 이미 소비하므로, reset 後에 켜면 초기 구조환자가 표본에서
    #    누락되고 분해 항등식이 깨진다(첫 시도에서 잔차 0.72 로 FAIL 났던 원인).
    #    start() 가 trace_log 를 비우므로 여기서 켜 두면 초기 이벤트부터 온전히 기록된다.
    u.ev_manager.enable_trace = True
    obs, _ = env.reset(seed=seed)
    done, reward, n_dec, info = False, 0.0, 0, {}
    while not done:
        action = policy(obs, env.action_masks(), u)
        n_dec += 1
        obs, _, term, trunc, info = env.step(action)
        reward += info.get("r_woG", 0.0)
        done = term or trunc
    tr = u.ev_manager.get_trace()
    prev = float(u.preventable_woG)

    resc, arr, care = {}, {}, {}
    div = 0
    hos = {}
    for r in tr:
        e = r["event"]
        if e == "rescue" and r["severity"] in (0, 1):
            resc[r["patient_id"]] = (r["time"], r["severity"])
        elif e == "hospital_arrival" and r.get("admitted") and r["severity"] in (0, 1):
            arr[r["patient_id"]] = (r["time"], r["hospital_id"])
        elif e == "care_start" and r["severity"] in (0, 1):
            care[r["patient_id"]] = (r["time"], r["hospital_id"], r["severity"],
                                     bool(r.get("from_queue", False)))
            hos[r["hospital_id"]] = hos.get(r["hospital_id"], 0) + 1
        elif e == "diversion":
            div += 1

    sp = u.getSurvProb
    loss_delay = loss_missing = 0.0
    ta = {0: [], 1: []}
    n_or = {0: 0, 1: 0}
    n_rc = {0: 0, 1: 0}
    waits, wq = [], []
    n_now = n_q = 0
    for pid, (t_r, sev) in resc.items():
        n_rc[sev] += 1
        p_r = sp(t_r, sev)
        if pid in care:
            t_a = care[pid][0]
            loss_delay += p_r - sp(t_a, sev)
            n_or[sev] += 1
            ta[sev].append(t_a)
            if care[pid][3]:
                n_q += 1
                if pid in arr:
                    wq.append(t_a - arr[pid][0])
            else:
                n_now += 1
            if pid in arr:
                waits.append(t_a - arr[pid][0])
        else:
            loss_missing += p_r
    pdr = 1.0 - reward / prev if prev > 0 else 0.0
    med = lambda v: float(np.median(v)) if v else float("nan")
    nR, nY = n_rc[0], n_rc[1]
    tot = nR + nY
    w = np.asarray(waits, float)
    return {
        "pdr_woG": pdr, "reward_woG": reward, "preventable_woG": prev,
        "share_delay": loss_delay / prev if prev > 0 else 0.0,
        "share_missing": loss_missing / prev if prev > 0 else 0.0,
        "n_resc_RY": tot, "n_arr_RY": len(arr), "n_or_RY": n_or[0] + n_or[1],
        "or_rate_RY": (n_or[0] + n_or[1]) / tot if tot else float("nan"),
        "n_resc_R": nR, "n_or_R": n_or[0], "or_rate_R": n_or[0] / nR if nR else float("nan"),
        "n_resc_Y": nY, "n_or_Y": n_or[1], "or_rate_Y": n_or[1] / nY if nY else float("nan"),
        "t_admit_med_R": med(ta[0]), "t_admit_med_Y": med(ta[1]),
        "wait_med_RY": med(list(w)), "wait_p90_RY": float(np.percentile(w, 90)) if w.size else float("nan"),
        "frac_wait_pos": float((w > 1e-9).mean()) if w.size else float("nan"),
        "n_or_now": n_now, "n_or_queued": n_q,
        "frac_queued": n_q / (n_now + n_q) if (n_now + n_q) else float("nan"),
        "wait_med_queued": med(wq),
        "n_hosp_used": len(hos), "gini_hosp": _gini(hos.values()), "n_diversion": div,
        "sim_time": float(info.get("time", np.nan)), "n_decisions": n_dec,
    }


def worker(job):
    region, cfg, specs, n_eps, seed0 = job
    try:
        os.environ.update(MCI_CAP_GATE="occ", MCI_OBS_VARIANT="essential+load+valid",
                          MCI_H_PAD="47", MCI_REWARD_MODE="woG")
        from viper_distill import _suppress_stdout, make_feature_env
        from v17_rule_eval import build_rule_policies

        rows = []
        with _suppress_stdout():
            policies = build_rule_policies(specs, region=region)
            factory = make_feature_env(cfg, None)
            for ep in range(n_eps):
                seed = seed0 + ep
                for name, pol in policies:
                    m = rollout(factory, pol, seed)
                    m.update(region=region, policy=name, episode=ep, seed=seed)
                    rows.append({k: m[k] for k in COLS})
        return {"ok": True, "region": region, "rows": rows}
    except Exception as exc:
        import traceback
        return {"ok": False, "region": region, "err": (str(exc) + traceback.format_exc())[:1500]}


def gate(args) -> None:
    """trace on/off 가 결과를 바꾸지 않는지 실측 확인."""
    os.environ.update(MCI_CAP_GATE="occ", MCI_OBS_VARIANT="essential+load+valid",
                      MCI_H_PAD="47", MCI_REWARD_MODE="woG")
    from viper_distill import _suppress_stdout, make_feature_env
    from v17_rule_eval import build_rule_policies, rollout as plain_rollout

    man = json.load(open(args.manifest, encoding="utf-8"))
    keys = [k for k in args.regions.split(",") if k in man][:4] or list(man)[:4]
    specs = [x for x in args.policies.split(";") if x]
    d = []
    with _suppress_stdout():
        for k in keys:
            fac = make_feature_env(man[k], None)
            pols = build_rule_policies(specs, region=k)
            for ep in range(args.n_eps):
                for name, pol in pols:
                    a = plain_rollout(fac, pol, args.seed0 + ep)[1]      # trace off
                    b = rollout(fac, pol, args.seed0 + ep)["pdr_woG"]    # trace on
                    d.append(abs(a - b))
    d = np.asarray(d)
    print(f"[gate] trace on/off PDR 비교 n={d.size}  maxΔ={d.max():.3e}  "
          f"{'PASS (계측이 결과를 바꾸지 않는다)' if d.max() == 0 else 'FAIL'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(REPO / "scenarios/manifests/v19/tradeoff250_manifest.json"))
    p.add_argument("--policies", required=True, help="세미콜론(;) 구분 — v17_rule_eval 과 동일 문법")
    p.add_argument("--regions", default="")
    p.add_argument("--n_eps", type=int, default=10)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--out", default="")
    p.add_argument("--gate", action="store_true", help="trace on/off 비트동일 검증만 수행")
    args = p.parse_args()
    sys.path.insert(0, str(REPO / "src/sim_src"))
    if args.gate:
        gate(args)
        return
    if not args.out:
        p.error("--out 필요")

    mp = Path(args.manifest).resolve()
    man = json.load(open(mp, encoding="utf-8"))
    keys = [k for k in args.regions.split(",") if k in man] if args.regions else list(man)
    specs = [x for x in args.policies.split(";") if x]
    from v17_rule_eval import build_rule_policies
    cases = [n for n, _ in build_rule_policies(specs)]

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        seen = {}
        with open(out, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.setdefault(row["region"], set()).add((row["policy"], int(row["episode"])))
        exp = len(cases) * args.n_eps
        done = {k for k, v in seen.items() if len(v) == exp}
        bad = set(seen) - done
        if bad:
            raise RuntimeError(f"부분 기록 지역(수동 정리 필요): {sorted(bad)[:3]}")
    jobs = [(k, man[k], specs, args.n_eps, args.seed0) for k in keys if k not in done]
    print(f"[mech] regions={len(keys)} remaining={len(jobs)} cases={len(cases)} "
          f"n_eps={args.n_eps} seed={args.seed0}..{args.seed0+args.n_eps-1}", flush=True)

    new = not out.exists()
    fo = open(out, "a", newline="", encoding="utf-8")
    wr = csv.DictWriter(fo, fieldnames=COLS)
    if new:
        wr.writeheader(); fo.flush()
    t0, n = time.time(), 0
    if jobs:
        with Pool(min(args.workers, len(jobs)), maxtasksperchild=1) as pool:
            for i, r in enumerate(pool.imap_unordered(worker, jobs), 1):
                if not r["ok"]:
                    fo.close(); raise RuntimeError(f"{r['region']} 실패: {r['err']}")
                wr.writerows(r["rows"]); fo.flush(); n += len(r["rows"])
                if i % 25 == 0 or i == len(jobs):
                    print(f"  [{i}/{len(jobs)}] {r['region']} total={n} wall={time.time()-t0:.0f}s", flush=True)
    fo.close()

    # 분해 항등식 검증 — share_delay + share_missing 이 pdr_woG 와 같아야 한다
    import pandas as pd
    d = pd.read_csv(out)
    resid = (d.share_delay + d.share_missing - d.pdr_woG).abs()
    print(f"[mech] 분해 항등식 최대 잔차 {resid.max():.3e} "
          f"{'PASS' if resid.max() < 1e-9 else 'FAIL — 분해가 안 닫힌다'}")
    meta = {
        "schema_version": 1, "manifest": str(mp), "manifest_sha256": sha256_file(mp),
        "policy_specs": specs, "cases": cases, "n_regions": len(keys),
        "n_eps_per_region": args.n_eps, "seed_start": args.seed0,
        "environment": {"MCI_CAP_GATE": "occ", "MCI_OBS_VARIANT": "essential+load+valid", "MCI_H_PAD": "47"},
        "scenario_knobs": {k: v for k, v in sorted(os.environ.items())
                           if k.startswith("MCI_") and k not in
                           ("MCI_CAP_GATE", "MCI_OBS_VARIANT", "MCI_H_PAD", "MCI_REWARD_MODE")},
        "decomposition_max_residual": float(resid.max()),
        "n_rows": len(d), "output": str(out), "output_sha256": sha256_file(out),
        "note": "p_admit = 수술실 배정 순간. share_delay + share_missing = pdr_woG (항등식).",
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    print(f"[mech] 완료 rows={len(d)} wall={(time.time()-t0)/60:.1f}분 → {out}", flush=True)


if __name__ == "__main__":
    main()
