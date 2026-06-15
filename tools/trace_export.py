# -*- coding: utf-8 -*-
"""시뮬 trace JSON(딕셔너리) → Unity JsonUtility 친화 평탄 형식.

입력 `results/{EXP}/trace_{COORD}.json` 구조:
  { "<rule>__iter<N>": [ {"time":.., "event":"..", "patient_id":.., ...}, ... ], ... }
출력:
  { "traces": [ { "key":"<rule>__iter<N>",
                  "events":[ {"time","evt","patient_id","vehicle","vehicle_id","hospital_id","severity","n_patients"} ] } ] }
  - "event" → "evt" (C# 키워드 회피), 누락 필드는 -1/"" 로 정규화.

사용:
  python trace_export.py --trace "results/exp_.../trace_(lat,lon).json" [--out trace_flat.json]
"""
from __future__ import annotations
import argparse, json, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FIELDS_INT = ["patient_id", "vehicle_id", "hospital_id", "severity", "n_patients"]


def norm_event(e):
    out = {
        "time": float(e.get("time", 0.0)),
        "evt": str(e.get("event", "")),
        "vehicle": str(e.get("vehicle", "")),
    }
    for k in FIELDS_INT:
        v = e.get(k, None)
        try:
            out[k] = int(v) if v is not None else -1
        except (ValueError, TypeError):
            out[k] = -1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="입력 trace JSON (딕셔너리)")
    ap.add_argument("--out", default=None, help="출력(기본=<입력>_flat.json)")
    args = ap.parse_args()

    data = json.load(open(args.trace, encoding="utf-8"))
    if isinstance(data, list):  # 단일 trace 리스트로 온 경우
        data = {"trace__iter0": data}
    if not isinstance(data, dict):
        print("[ERROR] trace 최상위가 dict/list 아님"); sys.exit(2)

    traces = []
    for key in sorted(data.keys()):
        evs = data[key] or []
        events = [norm_event(e) for e in evs if isinstance(e, dict)]
        events.sort(key=lambda r: r["time"])
        traces.append({"key": key, "events": events, "n_events": len(events),
                       "t_end": events[-1]["time"] if events else 0.0})

    out = args.out or (os.path.splitext(args.trace)[0] + "_flat.json")
    json.dump({"traces": traces}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[OK] {out}")
    print(f"  시나리오(키) {len(traces)}개")
    for t in traces[:8]:
        print(f"   - {t['key']}: {t['n_events']} events, t_end={t['t_end']:.1f}min")
    if len(traces) > 8:
        print(f"   ... (+{len(traces)-8})")


if __name__ == "__main__":
    main()
