# -*- coding: utf-8 -*-
"""v22 — 현장 파이프라인의 코어 스케일링 실측.

지금까지의 예산 산술("24코어 3분이면 N500 에서 12,600 에피소드")은 **병렬효율 1.0 가정**이었다.
이 측정이 그 가정을 실측으로 바꾼다. 교수님 피드백 4항("실행 시간 체크")의 마지막 미결 항목이다.

왜 `v22_field_assist.py` 를 쓰는가
  현장 사례는 **좌표 1개**다. `v17_rule_eval.py` 의 Pool 은 **지역 단위 샤딩**이라 좌표가 하나면
  워커가 하나 = 병렬도 0 이고, 그걸로 재면 스케일링이 아니라 직렬 시간을 재게 된다.
  `v22_field_assist.py` 는 (후보 × 시드) 격자를 샤딩하므로 현장 병렬도를 실제로 반영한다.

방법론 (src/sim_src_upgrade/bench/bench_core.py:1-17 승계)
  · 공유 노드라 벽시계가 요동한다 → **min-of-N** 이 경합이 가장 적었던 회차이므로 가장 강건한 추정량.
  · **인터리브**: 바깥 루프가 반복(rep), 안쪽이 워커 수. loadavg 드리프트가 특정 워커 설정에
    편중되지 않게 한다(같은 설정을 연속으로 N회 돌리면 그 구간의 부하가 통째로 그 설정에 실린다).
  · 벽시계와 CPU 시간을 둘 다 기록하고, 측정 전후 loadavg 를 남긴다.
  · BLAS/OpenMP 4종을 1로 고정한다 — 스레드 수가 부동소수 결과를 바꾼 선례가 있다.

병렬효율 = T(1) / (k · T(k)). 1.0 이 이상적, 낮을수록 고정비(cold)와 경합이 먹는다.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = "/home/ryu/anaconda3/envs/UAV/bin/python"
ASSIST = REPO / "src/rl_src/v22_field_assist.py"
BLAS = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}


def one_run(workers: int, patients: int, k: int, lat: float, lon: float,
            lam, yhold, tag: str, out_dir: Path) -> dict:
    """파이프라인 1회. 반환 = 그 실행의 timing 블록 + 우리 쪽 벽시계."""
    env = dict(os.environ); env.update(BLAS)
    cmd = [PY, str(ASSIST), "--lat", str(lat), "--lon", str(lon),
           "--patients", str(patients), "--families", "Q",
           "--lam", *[str(x) for x in lam], "--yhold", *[str(y) for y in yhold],
           "--search", "grid", "--no_outer_all",
           "--inner_seeds", str(k), "--outer_seeds", "2",
           "--workers", str(workers), "--tag", tag,
           "--out_dir", str(out_dir), "--no_save_rows", "--quiet"]
    la0 = os.getloadavg()
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall = time.perf_counter() - t0
    la1 = os.getloadavg()
    if p.returncode != 0:
        return {"ok": False, "rc": p.returncode, "err": (p.stderr or p.stdout)[-1200:]}
    jf = out_dir / f"{tag}.json"
    if not jf.exists():
        return {"ok": False, "rc": 0, "err": f"산출 JSON 없음: {jf}"}
    d = json.load(open(jf, encoding="utf-8"))
    t = d.get("timing", {})
    st = t.get("stages", {})
    return {"ok": True, "outer_wall_s": wall,
            "inner_wall_s": t.get("total_wall_s"),
            "T5_wall_s": (st.get("T5_rollout") or {}).get("wall_s"),
            "T3_wall_s": (st.get("T3_env") or {}).get("wall_s"),
            "env_cold_mean_wall_s": (t.get("env_cold") or {}).get("mean_wall_s"),
            "import_s": t.get("import_s"),
            "proc_age_s": t.get("process_age_s_at_main"),
            "cpu_children_s": t.get("total_cpu_children_s"),
            "rollout_cpu_ms_sum": t.get("rollout_cpu_ms_sum"),
            "episodes": (d.get("search") or {}).get("episodes_total"),
            "n_candidates": (d.get("search") or {}).get("n_candidates"),
            "loadavg": [la0[0], la1[0]]}


def cmd_run(args) -> None:
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lam = [float(x) for x in args.lam.split(",")]
    yh = [float(x) for x in args.yhold.split(",")]
    workers = [int(x) for x in args.workers.split(",")]
    sizes = [int(x) for x in args.patients.split(",")]
    ks = {int(a.split(":")[0]): int(a.split(":")[1]) for a in args.seeds_by_n.split(",")}

    rec = {}
    print(f"[설정] 후보 {len(lam)}×{len(yh)}={len(lam)*len(yh)} · 워커 {workers} · N {sizes} "
          f"· 시드 {ks} · 반복 {args.reps} (인터리브 · min-of-N)")
    for rep in range(args.reps):                      # ★바깥이 반복, 안쪽이 워커 = 인터리브
        for n in sizes:
            for w in workers:
                tag = f"cs_n{n}_w{w}_r{rep}"
                r = one_run(w, n, ks[n], args.lat, args.lon, lam, yh, tag, out_dir)
                key = (n, w)
                rec.setdefault(key, []).append(r)
                ok = "OK" if r.get("ok") else "FAIL"
                extra = (f"wall={r['outer_wall_s']:7.2f}s T5={r['T5_wall_s']:6.2f}s "
                         f"ep={r['episodes']}" if r.get("ok") else r.get("err", "")[:70])
                print(f"  rep{rep} N={n:<4} w={w:<3} {ok} {extra}", flush=True)

    print()
    print("=" * 104)
    print("코어 스케일링 — min-of-N. 효율 = T(1)/(k·T(k)).  T5 = 롤아웃 단계만(cold 제외)")
    print("=" * 104)
    table = {}
    for n in sizes:
        base_outer = base_t5 = None
        print(f"\n[N={n}]  후보 {len(lam)*len(yh)} × 내부시드 {ks[n]} = {len(lam)*len(yh)*ks[n]} 에피소드")
        print(f"{'워커':>5} {'전체wall min':>13} {'T5 min':>10} {'효율(전체)':>11} {'효율(T5)':>10} "
              f"{'env cold':>9} {'자식CPU':>10}")
        for w in workers:
            rs = [x for x in rec.get((n, w), []) if x.get("ok")]
            if not rs:
                print(f"{w:>5}  실패"); continue
            ow = min(x["outer_wall_s"] for x in rs)
            t5 = min(x["T5_wall_s"] for x in rs if x["T5_wall_s"] is not None)
            cold = min(x["env_cold_mean_wall_s"] for x in rs if x["env_cold_mean_wall_s"] is not None)
            cpu = min(x["cpu_children_s"] for x in rs if x["cpu_children_s"] is not None)
            if base_outer is None:
                base_outer, base_t5 = ow, t5
            eo = base_outer / (w * ow); e5 = base_t5 / (w * t5)
            table[f"N{n}_w{w}"] = {"outer_wall_min_s": ow, "T5_wall_min_s": t5,
                                   "eff_outer": eo, "eff_T5": e5,
                                   "env_cold_s": cold, "cpu_children_s": cpu,
                                   "n_ok": len(rs)}
            print(f"{w:>5} {ow:13.2f} {t5:10.2f} {eo:11.3f} {e5:10.3f} {cold:9.2f} {cpu:10.1f}")

    # 실측 기반 예산 판정
    print()
    print("=" * 104)
    print("실측 기반 예산 판정 — 3분/5분에 들어가는 에피소드 수 (T5 처리량 × 예산, cold 차감)")
    print("=" * 104)
    verdict = {}
    for n in sizes:
        eps = len(lam) * len(yh) * ks[n]
        print(f"\n[N={n}]")
        print(f"{'워커':>5} {'T5 처리량(ep/s)':>15} {'cold(s)':>8} {'3분 가용ep':>12} {'5분 가용ep':>12}")
        for w in workers:
            key = f"N{n}_w{w}"
            if key not in table: continue
            t5 = table[key]["T5_wall_min_s"]; cold = table[key]["env_cold_s"]
            thr = eps / t5
            # cold 는 워커당 1회지만 병렬이므로 한 번의 배리어로 계산
            e180 = max(0.0, 180 - cold - 3.0) * thr
            e300 = max(0.0, 300 - cold - 3.0) * thr
            verdict[key] = {"throughput_ep_s": thr, "ep_180s": e180, "ep_300s": e300}
            print(f"{w:>5} {thr:15.1f} {cold:8.2f} {e180:12.0f} {e300:12.0f}")
    out = {"schema_version": 1,
           "purpose": "현장 파이프라인 코어 스케일링 — 병렬효율 1.0 가정의 실측 대체",
           "method": {"estimator": "min-of-N (벽시계) · 인터리브(바깥=반복, 안=워커)",
                      "reference": "src/sim_src_upgrade/bench/bench_core.py:1-17",
                      "driver": "src/rl_src/v22_field_assist.py ((후보×시드) 샤딩)",
                      "why_not_rule_eval": "v17_rule_eval.py 는 지역 샤딩이라 좌표 1개면 병렬도 0",
                      "reps": args.reps, "blas_threads": BLAS,
                      "candidates": len(lam) * len(yh), "seeds_by_n": ks},
           "coord": {"lat": args.lat, "lon": args.lon},
           "scaling": table, "budget_verdict": verdict,
           "raw": {f"N{k[0]}_w{k[1]}": v for k, v in rec.items()}}
    op = REPO / args.out_json
    op.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[기록] {op}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, default=37.648903)
    ap.add_argument("--lon", type=float, default=127.505374)
    ap.add_argument("--workers", default="1,2,4,8,16,24,32")
    ap.add_argument("--patients", default="100,500")
    ap.add_argument("--seeds_by_n", default="100:16,500:8",
                    help="N:내부시드 수. N=500 은 에피소드가 5배 비싸 시드를 줄인다")
    ap.add_argument("--lam", default="8,12,18,26,36,50,70,95")
    ap.add_argument("--yhold", default="0,2,4,8")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out_dir", default="results/field/timing/core_runs")
    ap.add_argument("--out_json", default="results/field/timing/core_scaling.json")
    ap.set_defaults(func=cmd_run)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
