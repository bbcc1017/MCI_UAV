#!/usr/bin/env python
"""자율주행 폐루프 평가 CSV 비교 — DS/RC/IP 와 **종방향 승차감**을 한 표로 낸다.

왜 필요한가: 구 하네스는 횡방향 지표만 재서 "가감속을 수시로 반복"이 원리적으로 안 보였다.
이 리포터는 두 축을 항상 같이 찍어 한쪽만 좋아진 걸 개선으로 오독하지 않게 한다.

사용: road_eval_report.py <csv> [csv ...]
"""
import csv, io, math, os, sys

def load(path):
    with io.open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def num(r, k, d=float("nan")):
    v = r.get(k)
    try: return float(v)
    except (TypeError, ValueError): return d

def summarize(path):
    rows = load(path)
    n = len(rows)
    if n == 0: return None
    g = lambda k: [num(r, k) for r in rows]
    finite = lambda xs: [x for x in xs if x == x]
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    # 승차감은 **움직인 에피소드만** — 스폰 직후 죽은 에피소드의 -1 은 평균을 왜곡한다.
    comfort = [r for r in rows if num(r, "jerk_rms", -1) >= 0]
    reasons = {}
    for r in rows: reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    return dict(
        label=os.path.basename(path).replace("road_eval_", "").replace(".csv", ""),
        n=n, ds=mean(finite(g("ds"))), rc=mean(finite(g("rc"))), ip=mean(finite(g("ip"))),
        success=sum(1 for r in rows if r["success"] == "1"),
        red=int(sum(finite(g("red")))), ped=sum(1 for r in rows if r["collision_ped"] == "1"),
        veh=sum(1 for r in rows if r["collision_veh"] == "1"),
        stat=sum(1 for r in rows if r["collision_static"] == "1"),
        off=sum(1 for r in rows if r["off_lane"] == "1"),
        stall=reasons.get("stall", 0),
        cn=len(comfort),
        jerk=mean([num(r, "jerk_rms") for r in comfort]),
        jraw=mean([num(r, "jerk_rms_raw") for r in comfort]),
        jexc=mean([num(r, "jerk_exceed_rate") for r in comfort]),
        amax=mean([num(r, "abs_accel_max") for r in comfort]),
        latmax=mean([num(r, "abs_lat_accel_max") for r in comfort]),
        pedal=mean([num(r, "pedal_reversals") for r in comfort]),
        reasons=reasons,
    )

def main(paths):
    ss = [s for s in (summarize(p) for p in paths) if s]
    if not ss: print("데이터 없음"); return 1
    hdr = ("label", "n", "DS", "RC", "IP", "도달", "적신호", "보행", "차량", "정적", "이탈", "정체",
           "저크RMS", "저크초과", "저크생", "|a|max", "|alat|max", "페달반전")
    w = [16, 4, 6, 6, 6, 5, 6, 5, 5, 5, 5, 5, 8, 8, 7, 7, 9, 9]
    print(" ".join(h.rjust(x) for h, x in zip(hdr, w)))
    print(" ".join("-" * x for x in w))
    for s in ss:
        cells = (s["label"], s["n"], f"{s['ds']:.3f}", f"{s['rc']:.3f}", f"{s['ip']:.3f}",
                 s["success"], s["red"], s["ped"], s["veh"], s["stat"], s["off"], s["stall"],
                 f"{s['jerk']:.3f}", f"{s['jexc']:.4f}", ("-" if s['jraw']!=s['jraw'] else f"{s['jraw']:.1f}"), f"{s['amax']:.2f}",
                 f"{s['latmax']:.2f}", f"{s['pedal']:.1f}")
        print(" ".join(str(c).rjust(x) for c, x in zip(cells, w)))
    if len(ss) >= 2:
        a, b = ss[0], ss[-1]
        print(f"\n{b['label']} − {a['label']}:  DS {b['ds']-a['ds']:+.3f} · "
              f"저크RMS {b['jerk']-a['jerk']:+.3f} m/s³ · 페달반전 {b['pedal']-a['pedal']:+.1f}/에피")
    print("\n종료 사유:")
    for s in ss:
        top = sorted(s["reasons"].items(), key=lambda kv: -kv[1])
        print(f"  {s['label']}: " + " ".join(f"{k}={v}" for k, v in top))
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) if len(sys.argv) > 1 else (print(__doc__) or 1))
