"""한 시드의 멀티시드 재현 파이프라인을 한 번에 실행한다 (피드백 #2·#4 재현성).

train(500k) → collect_decisions → analyze_policy → distill_policy → 지표 JSON.

이 스크립트 자체를 백그라운드 태스크로 띄우면 내부 subprocess 는 tool 타임아웃과
무관하게 끝까지 돈다. 산출물(repro_seed<S>.json)은 에이전트 생사와 무관하게 디스크에
남으므로, 모니터링이 끊겨도 결과는 보존된다.

sim_src 무수정 — 기존 스크립트들만 호출. 항상 MCI_REDUCED_OBS=1, CUDA_VISIBLE_DEVICES="".

사용:
  python src/rl_src/run_seed_repro.py --seed 1
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

PY = "/home/RYU/anaconda3/envs/UAV/bin/python"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def run(cmd, log_path):
    env = dict(os.environ, MCI_REDUCED_OBS="1", CUDA_VISIBLE_DEVICES="")
    with open(log_path, "w") as f:
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {' '.join(cmd)} — 로그 {log_path}")


def compute_metrics(tag, out_csv, adir="results/analysis"):
    """발견 재현 여부를 수치로 추출."""
    obs = np.load(os.path.join(ROOT, adir, f"decisions_{tag}.npz"))["obs"]
    meta = pd.read_csv(os.path.join(ROOT, adir, f"decisions_{tag}_meta.csv"))
    with open(os.path.join(ROOT, adir, f"decisions_{tag}_labels.json"), encoding="utf-8") as f:
        info = json.load(f)
    labels = info["labels"]
    hp = info.get("hospital_props", {})
    li = {c: i for i, c in enumerate(labels)}
    H = sum(1 for c in labels if c.startswith("h") and c.endswith("_occ"))

    # 발견 1 — 이송수단: 'n_amb>0 → AMB, else UAV' 규칙의 실제 일치율 (free_mode 부분집합)
    namb = obs[:, li["n_amb_at_site"]]
    rule_mode = np.where(namb > 0.5, 0, 1)
    fm = meta["free_mode"] == 1
    mode_rule_acc = float((rule_mode[fm.values] == meta.loc[fm, "rl_mode"].values).mean())

    # 발견 2 — 우선순위: R·Y 동시 대기서 RL↔START 일치, RL Red 비율
    both = (obs[:, li["atsite_Red"]] > 0) & (obs[:, li["atsite_Yellow"]] > 0)
    selA = both & meta["rl_class"].isin([0, 1]).values
    prio_start_agree = float(meta.loc[selA, "agree_class"].mean()) if selA.sum() else None
    prio_red_ratio = float((meta.loc[selA, "rl_class"] == 0).mean()) if selA.sum() else None

    # 발견 3 — 라우팅: 선택병원 점유 RL vs 휴리스틱, Tier3 사용률
    occ = obs[:, [li[f"h{i}_occ"] for i in range(H)]]
    sent = meta[(meta["rl_dest"] > 0) & (meta["heur_dest"] > 0)]
    rows = sent.index.values
    rl_d = sent["rl_dest"].values - 1
    he_d = sent["heur_dest"].values - 1
    rl_occ = float(occ[rows, rl_d].mean())
    he_occ = float(occ[rows, he_d].mean())
    rl_t3 = float(np.mean([d in set(hp.get(meta.at[r, "region"], {}).get("tier3_idx", []))
                           for r, d in zip(rows, rl_d)]))
    he_t3 = float(np.mean([d in set(hp.get(meta.at[r, "region"], {}).get("tier3_idx", []))
                           for r, d in zip(rows, he_d)]))

    # 증류 성능
    df = pd.read_csv(os.path.join(ROOT, out_csv))
    ppo_margin = float(df["PPO_vs_heur"].mean())
    dist_margin = float(df["distill_vs_heur"].mean())

    return {
        "n_decisions": int(len(meta)),
        "agree_global": {"class": float(meta["agree_class"].mean()),
                         "mode": float(meta["agree_mode"].mean()),
                         "full": float(meta["agree_full"].mean())},
        "mode_rule_acc": mode_rule_acc,
        "priority_start_agree": prio_start_agree,
        "priority_red_ratio": prio_red_ratio,
        "route_occ_rl": rl_occ, "route_occ_heur": he_occ, "route_occ_gap": rl_occ - he_occ,
        "route_tier3_rl": rl_t3, "route_tier3_heur": he_t3,
        "ppo_vs_heur": ppo_margin, "distill_vs_heur": dist_margin,
        "distill_regions_won": int((df["distill_vs_heur"] > 0).sum()), "n_regions": int(len(df)),
        "distill_retention_pct": float(dist_margin / ppo_margin * 100) if ppo_margin else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--manifest", default="scenarios/manifests/plan1nat_manifest.json")
    ap.add_argument("--heur_csv", default="results/plan1nat_f3_eval.csv")
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--collect_ep", type=int, default=50)
    ap.add_argument("--distill_ep", type=int, default=100)
    ap.add_argument("--skip_train", action="store_true", help="이미 학습된 모델 재사용")
    a = ap.parse_args()
    S = a.seed
    tag = f"plan1nat_f3_seed{S}"
    log_dir = f"results/rl/{tag}/national/ppo"
    model = f"{log_dir}/final_model.zip"
    os.makedirs(os.path.join(ROOT, "results/analysis/repro_logs"), exist_ok=True)
    logp = lambda step: f"results/analysis/repro_logs/seed{S}_{step}.log"

    if not a.skip_train:
        sys.stderr.write(f"[seed{S}] 1/4 학습 500k ...\n"); sys.stderr.flush()
        run([PY, "src/rl_src/train_ppo.py", "--config_path", a.manifest,
             "--total_timesteps", str(a.timesteps), "--n_envs", "4", "--vec", "subproc",
             "--seed", str(S), "--log_dir", log_dir], logp("train"))
    sys.stderr.write(f"[seed{S}] 2/4 의사결정 로깅 ...\n"); sys.stderr.flush()
    run([PY, "src/rl_src/collect_decisions.py", "--manifest", a.manifest, "--model", model,
         "--heur_csv", a.heur_csv, "--n_episodes", str(a.collect_ep), "--tag", tag], logp("collect"))
    sys.stderr.write(f"[seed{S}] 3/4 서로게이트 분석 ...\n"); sys.stderr.flush()
    run([PY, "src/rl_src/analyze_policy.py", "--tag", tag], logp("analyze"))
    sys.stderr.write(f"[seed{S}] 4/4 증류 평가 ...\n"); sys.stderr.flush()
    out_csv = f"results/{tag}_distill_eval.csv"
    run([PY, "src/rl_src/distill_policy.py", "--manifest", a.manifest, "--model", model,
         "--heur_csv", a.heur_csv, "--n_episodes", str(a.distill_ep), "--tag", tag,
         "--out_csv", out_csv], logp("distill"))

    metrics = compute_metrics(tag, out_csv)
    metrics["seed"] = S
    out_json = os.path.join(ROOT, "results/analysis", f"repro_seed{S}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"DONE seed{S} → {out_json}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
