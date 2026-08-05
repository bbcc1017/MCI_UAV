# -*- coding: utf-8 -*-
"""후보랭킹 증류 NPZ를 state_key suffix 기준으로 무손실 부분집합화."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--suffixes", required=True, help="쉼표구분, 예: _p0,_p1,_p2")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    src, dst = Path(args.input).resolve(), Path(args.output).resolve()
    if dst.exists() or Path(str(dst) + ".meta.json").exists():
        raise FileExistsError(f"기존 산출물 보호: {dst}")
    z = np.load(src, allow_pickle=True)
    suffixes = tuple(x for x in args.suffixes.split(",") if x)
    state_key = z["state_key"].astype(str)
    keep = np.asarray([x.endswith(suffixes) for x in state_key], dtype=bool)
    state_ids = np.flatnonzero(keep)
    ncand = np.asarray(z["ncand"], dtype=int)
    offsets = np.asarray(z["offsets"], dtype=np.int64)
    row_ids = np.concatenate([
        np.arange(offsets[i], offsets[i + 1], dtype=np.int64) for i in state_ids
    ])
    row_keys = {"X", "target", "weight", "chosen", "cand_action", "ppo_prob"}
    state_keys = {
        "ncand", "teacher_action", "behavior_action", "ppo_action", "state_key",
        "state_seed", "decision_index", "teacher_switched", "teacher_in_milp",
        "planner_lookahead", "planner_dpdr", "planner_q_greedy", "planner_q_exec",
        "planner_n_cand", "planner_n_extra", "milp_action0", "milp_action1",
    }
    out = {}
    for key in z.files:
        if key == "offsets":
            continue
        arr = z[key]
        if key in row_keys or (arr.ndim >= 1 and len(arr) == offsets[-1]):
            out[key] = arr[row_ids]
        elif key in state_keys or (arr.ndim >= 1 and len(arr) == len(ncand)):
            out[key] = arr[state_ids]
        elif key.startswith("episode_"):
            # 원본 수집은 좌표당 1 episode. episode_key로 독립 선택한다.
            if "episode_key" not in z.files:
                raise ValueError("episode_key 없이 episode 배열 존재")
            ep_keep = np.asarray([str(x).endswith(suffixes) for x in z["episode_key"]], dtype=bool)
            out[key] = arr[ep_keep]
        else:
            out[key] = arr
    out["offsets"] = np.concatenate([[0], np.cumsum(out["ncand"], dtype=np.int64)])
    if out["offsets"][-1] != len(out["X"]):
        raise RuntimeError("부분집합 offsets 불일치")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out)
    meta = {
        "schema_version": 1, "source": str(src), "suffixes": suffixes,
        "n_source_states": int(len(ncand)), "n_states": int(len(state_ids)),
        "n_candidate_rows": int(len(row_ids)), "output": str(dst),
    }
    Path(str(dst) + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
