# -*- coding: utf-8 -*-
"""지역특화 교사 250개의 **대각 평가** — 각 시군구 모델을 자기 지역 좌표에서만 돌린다 (v18 E5).

``v17_ppo_eval.py`` 는 1 모델 = 1 호출 전제라 250개를 돌리려면 프로세스 기동·모델 로드가
지배한다. 여기서는 잡 하나가 (지역, 그 지역 모델) 쌍이고 워커 안에서 그 지역 모델만
로드하므로 전국 1회 순회로 끝난다. rollout·seed·CSV 규약과 워커 자체를 ``v17_ppo_eval``
에서 그대로 import 해 배관 차이를 원천 차단한다.

전국모델과의 비교는 같은 좌표·같은 seed 의 기존 CSV(`ppo_eval250_seed0_29.csv` 등)와
지역 단위로 짝지어 계산한다 — 이 스크립트는 대각 행만 만든다.

사용
----
    python src/rl_src/sigungu_diag_eval.py --model_root results/rl/sigungu250/wave1 \
        --policy_name PPO_SIGUNGU_W1 --out results/scoreboard/v18/sigungu_diag_wave1.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from v17_ppo_eval import COLS, sha256_file, worker  # noqa: E402  동일 배관 재사용

REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "scenarios/manifests/sigungu250/_index.json"
EVAL_MANIFEST = REPO / "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index", default=str(INDEX))
    p.add_argument("--manifest", default=str(EVAL_MANIFEST),
                   help="평가 좌표. 기본은 대표점250(지역키에 p 접미 없음)")
    p.add_argument("--model_root", required=True, help="<root>/<지역>/final_model.zip")
    p.add_argument("--policy_name", default="PPO_SIGUNGU_LOCAL")
    p.add_argument("--obs_variant", default="essential+load+valid")
    p.add_argument("--regions", default="")
    p.add_argument("--wave", type=int, default=0)
    p.add_argument("--n_eps", type=int, default=30)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    idx = json.load(open(a.index, encoding="utf-8"))["regions"]
    manifest = json.load(open(a.manifest, encoding="utf-8"))
    root = Path(a.model_root).resolve()

    if a.regions:
        keys = [r for r in a.regions.split(",")]
    elif a.wave:
        keys = [r for r, v in idx.items() if v["wave"] == a.wave]
    else:
        keys = list(idx)
    keys = sorted(k for k in keys if k in manifest)

    missing_model = [k for k in keys if not (root / k / "final_model.zip").exists()]
    if missing_model:
        print(f"[diag] 모델 없는 지역 {len(missing_model)}개 제외 (예: {missing_model[:3]})")
        keys = [k for k in keys if k not in missing_model]
    if not keys:
        raise SystemExit("평가할 (지역, 모델) 쌍이 없다")

    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            seen: dict[str, set] = {}
            for row in csv.DictReader(f):
                if row["policy"] != a.policy_name:
                    continue
                seen.setdefault(row["region"], set()).add(int(row["seed"]))
        done = {k for k, v in seen.items() if len(v) == a.n_eps}
        bad = set(seen) - done
        if bad:
            raise RuntimeError(f"부분 기록 지역: {sorted(bad)[:3]}")

    jobs = [(k, manifest[k], str(root / k), a.n_eps, a.seed0, a.policy_name,
             a.obs_variant)
            for k in keys if k not in done]
    print(f"[diag] 지역 {len(keys)} · 남은 {len(jobs)} · n_eps={a.n_eps} "
          f"seed={a.seed0}..{a.seed0+a.n_eps-1} workers={a.workers}")
    if not jobs:
        print("[diag] 전부 완료 상태")
        return

    t0 = time.time()
    write_header = not out.exists()
    n_ok = 0
    with open(out, "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=COLS)
        if write_header:
            wr.writeheader()
        with Pool(min(a.workers, len(jobs)), maxtasksperchild=1) as pool:
            for res in pool.imap_unordered(worker, jobs):
                if not res["ok"]:
                    print(f"  ✗ {res['region']}: {res['err'][:200]}")
                    continue
                wr.writerows(res["rows"]); f.flush()
                n_ok += 1
                avg = sum(r["pdr_woG"] for r in res["rows"]) / len(res["rows"])
                print(f"  [{n_ok}/{len(jobs)}] {res['region']} pdr={avg:.4f} "
                      f"wall={time.time()-t0:.0f}s", flush=True)

    meta = {
        "script": "sigungu_diag_eval.py", "policy_name": a.policy_name,
        "eval_manifest": str(Path(a.manifest).resolve()),
        "model_root": str(root), "n_regions": len(keys),
        "n_eps": a.n_eps, "seed0": a.seed0,
        "model_sha256": {k: sha256_file(root / k / "final_model.zip") for k in keys[:5]},
        "model_sha256_note": "앞 5개만 기록(전수는 용량 과다)",
        "wall_min": round((time.time() - t0) / 60, 2),
    }
    Path(str(out) + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[diag] 완료 {n_ok}지역 · {meta['wall_min']}분 → {out}")


if __name__ == "__main__":
    main()
