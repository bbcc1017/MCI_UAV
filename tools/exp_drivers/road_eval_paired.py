#!/usr/bin/env python
"""자율주행 폐루프 평가 **페어드** 비교 — 같은 시드 에피소드끼리 짝지어 차이를 낸다.

왜 페어드인가: 60에피 요약평균은 시드 분산에 묻힌다(실측 v6 판정에서 저크 42% 악화가
페어드로는 노이즈 안이었다). 에피소드 짝의 차이 분포로 95%CI 와 승/무/패를 같이 낸다.

사용: road_eval_paired.py <기준.csv> <비교.csv> [비교2.csv ...]
"""
import csv, io, math, os, sys

# (열, 좋은방향)  ↑=높을수록 좋다 / ↓=낮을수록 좋다
METRICS = [("ds", "up"), ("rc", "up"), ("ip", "up"), ("progress_m", "up"),
           ("jerk_rms", "dn"), ("jerk_exceed_rate", "dn"), ("pedal_reversals", "dn"),
           ("steer_cmd_reversals", "dn"), ("lat_rms_straight", "dn"),
           ("abs_accel_max", "dn"), ("abs_lat_accel_max", "dn"), ("decisions", "up")]

def load(path):
    with io.open(path, encoding="utf-8-sig") as f:
        return {int(r["episode"]): r for r in csv.DictReader(f)}

def num(row, key):
    try: return float(row[key])
    except (KeyError, TypeError, ValueError): return float("nan")

def name(path):
    return os.path.basename(path).replace("road_eval_", "").replace(".csv", "")

def counts(rows):
    """에피소드 단위 사건 수 — 평균이 아니라 개수로 봐야 저빈도 사건이 안 묻힌다."""
    n = len(rows)
    ok = lambda k: sum(1 for r in rows.values() if r.get(k) == "1")
    reasons = {}
    for r in rows.values(): reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    return dict(n=n, success=ok("success"), veh=ok("collision_veh"), ped=ok("collision_ped"),
                stat=ok("collision_static"), off=ok("off_lane"),
                red=int(sum(num(r, "red") for r in rows.values())),
                stall=reasons.get("stall", 0), reasons=reasons)

def main(paths):
    base = load(paths[0])
    print("기준선 = %s" % name(paths[0]))
    print()
    print("%-14s %5s %5s %5s %5s %5s %5s %5s" % ("팔", "n", "도달", "차량", "보행", "정적", "이탈", "정체"))
    for p in paths:
        c = counts(load(p))
        print("%-14s %5d %5d %5d %5d %5d %5d %5d" % (
            name(p), c["n"], c["success"], c["veh"], c["ped"], c["stat"], c["off"], c["stall"]))
    for p in paths[1:]:
        cmp_ = load(p)
        common = sorted(set(base) & set(cmp_))
        print()
        print("── %s vs %s (공통 에피 %d) ──" % (name(p), name(paths[0]), len(common)))
        print("%-20s %10s %10s %+11s %9s  %-12s %s" % (
            "지표", "기준", "비교", "차", "95%CI", "승/무/패", "판정"))
        for key, dirn in METRICS:
            pairs = [(num(base[e], key), num(cmp_[e], key)) for e in common]
            # 승차감 열은 안 움직인 에피에서 -1 sentinel 이다 — 양쪽 다 유효한 짝만 쓴다.
            pairs = [(a, b) for a, b in pairs if a == a and b == b and not (a < 0 and b < 0)]
            pairs = [(a, b) for a, b in pairs if a >= 0 and b >= 0]
            if len(pairs) < 2: continue
            d = [b - a for a, b in pairs]
            m = sum(d) / len(d)
            var = sum((x - m) ** 2 for x in d) / len(d)
            ci = 1.96 * math.sqrt(var) / math.sqrt(len(d))
            w = sum(1 for x in d if x > 1e-9); l = sum(1 for x in d if x < -1e-9)
            good = (m > 0) if dirn == "up" else (m < 0)
            verdict = "무의미" if abs(m) <= ci else ("우세" if good else "열세")
            print("%-20s %10.4f %10.4f %+11.4f %9.4f  %-12s %s (n=%d)" % (
                key, sum(a for a, _ in pairs) / len(pairs), sum(b for _, b in pairs) / len(pairs),
                m, ci, "%d/%d/%d" % (w, len(d) - w - l, l), verdict, len(d)))

if __name__ == "__main__":
    if len(sys.argv) < 3: sys.exit(__doc__)
    main(sys.argv[1:])
