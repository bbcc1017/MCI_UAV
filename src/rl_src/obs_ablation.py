"""obs_reduced(221차원) 피처 중요도 — permutation ablation (역방향 분석).

챔피언 MaskablePPO 모델에 실제 결정점 obs(decisions_*.npz)를 넣고, 피처(또는 피처
그룹)를 셔플(permute)했을 때 정책의 예측 액션이 baseline 대비 얼마나 바뀌는지로
기여도를 정량화한다. 액션은 평탄 Discrete → (class, dest, mode) 3축으로 디코딩해
축별 민감도를 따로 본다.

목적: 221차원 중 정책이 실제로 의존하는 피처를 찾아 (1) obs 를 더 줄이거나
재배치할 근거, (2) 라우팅 의존 구조(병원별 점유 vs 자원 ETA)를 정량화.

주의:
- decisions_*.npz 엔 action mask 가 저장돼 있지 않다. baseline·ablated 모두 mask
  없이(deterministic argmax over 전체 액션) 예측하므로, 둘 사이의 *상대* 변화(중요도)
  는 유효하나 절대 액션은 실제 마스킹 롤아웃과 다를 수 있다. meta 의 rl_class/dest/mode
  (실제 마스킹 액션)와 baseline 일치율을 함께 출력해 sanity check.
- sim_src 무수정. 기존 산출물만 읽는다.

사용:
  CUDA_VISIBLE_DEVICES="" python src/rl_src/obs_ablation.py --tag plan1nat_f3 \
    --model results/rl/plan1nat_f3/national/ppo/final_model.zip
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ADIR = "results/analysis"


def group_of(label):
    """라벨 → 해석용 그룹."""
    if label.startswith("h") and label[1:2].isdigit():
        if label.endswith("_idle"):  return "hosp_idle"
        if label.endswith("_queue"): return "hosp_queue"
        if label.endswith("_occ"):   return "hosp_occ"
    if label.startswith("psent_"):  return "psent"
    if label.startswith("atsite_"): return "atsite"
    if label.startswith("pa_"):     return "patient_state"
    if label.startswith("ve_amb_"): return "ve_amb_eta" if "ETA" in label else "ve_amb_other"
    if label.startswith("ve_uav_"): return "ve_uav_eta" if "ETA" in label else "ve_uav_other"
    if label in ("n_amb_at_site", "n_uav_at_site"): return "counts_at_site"
    if label == "time": return "time"
    return "other"


def decode(flat, n_dest, n_mode=2):
    """평탄 Discrete → (class, dest, mode). env_wrapper 인코딩 c*(n_dest*n_mode)+d*n_mode+m."""
    m = flat % n_mode
    d = (flat // n_mode) % n_dest
    c = flat // (n_dest * n_mode)
    return c, d, m


def predict_actions(model, obs, masks=None, batch=8192):
    out = np.empty(len(obs), dtype=np.int64)
    for i in range(0, len(obs), batch):
        mb = None if masks is None else masks[i:i + batch]
        a, _ = model.predict(obs[i:i + batch], action_masks=mb, deterministic=True)
        out[i:i + batch] = np.asarray(a).reshape(-1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="plan1nat_f3")
    ap.add_argument("--model", default="results/rl/plan1nat_f3/national/ppo/final_model.zip")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out_csv", default=None)
    a = ap.parse_args()
    out_csv = a.out_csv or os.path.join(ADIR, f"obs_ablation_{a.tag}.csv")
    rng = np.random.default_rng(a.seed)

    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(a.model)

    npz = np.load(os.path.join(ADIR, f"decisions_{a.tag}.npz"))
    obs = npz["obs"].astype(np.float32)
    masks = npz["mask"] if "mask" in npz.files else None
    with open(os.path.join(ADIR, f"decisions_{a.tag}_labels.json"), encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    meta = pd.read_csv(os.path.join(ADIR, f"decisions_{a.tag}_meta.csv"))
    N, D = obs.shape
    assert D == len(labels), f"obs dim {D} != labels {len(labels)}"
    H = sum(1 for c in labels if c.startswith("h") and c.endswith("_occ"))
    n_dest = H + 1
    mask_state = "마스크 적용" if masks is not None else "마스크 없음(노마스크 — 잠정)"
    print(f"[obs_ablation] {a.tag}: N={N} obs decisions, D={D}, H={H} hospitals, "
          f"n_dest={n_dest} | {mask_state}")
    if masks is None:
        print("  ⚠ decisions npz 에 mask 가 없다 → 마스크 재수집 후 재실행 권장 "
              "(노마스크 baseline 은 실제 정책과 괴리).")

    # baseline
    base = predict_actions(model, obs, masks)
    bc, bd, bm = decode(base, n_dest)

    # sanity: meta 의 실제 마스킹 액션과 baseline 일치율 (마스크 적용 시 ≈1.0 이어야 정상)
    sane = {
        "class": float((bc == meta["rl_class"].values).mean()),
        "dest":  float((bd == meta["rl_dest"].values).mean()),
        "mode":  float((bm == meta["rl_mode"].values).mean()),
    }
    print(f"[sanity] baseline vs 실제 수집 액션 일치율: "
          f"class={sane['class']:.3f} dest={sane['dest']:.3f} mode={sane['mode']:.3f}")

    # 개별 피처 permutation importance
    rows = []
    for j in range(D):
        col = obs[:, j].copy()
        obs[:, j] = rng.permutation(col)
        a_p = predict_actions(model, obs, masks)
        obs[:, j] = col  # 복원
        pc, pd_, pm = decode(a_p, n_dest)
        rows.append({
            "feature": labels[j], "group": group_of(labels[j]),
            "imp_class": float((pc != bc).mean()),
            "imp_dest":  float((pd_ != bd).mean()),
            "imp_mode":  float((pm != bm).mean()),
            "imp_any":   float((a_p != base).mean()),
        })
        if (j + 1) % 40 == 0:
            print(f"  ... {j+1}/{D} features")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # 그룹 집계 (합 = 그룹 전체가 정책에 미치는 누적 영향)
    g = df.groupby("group")[["imp_class", "imp_dest", "imp_mode", "imp_any"]].agg(["sum", "mean", "max"])
    g_sum = df.groupby("group")[["imp_class", "imp_dest", "imp_mode", "imp_any"]].sum().sort_values("imp_any", ascending=False)

    print("\n=== 그룹별 누적 중요도 (sum over features in group) ===")
    print(g_sum.round(3).to_string())
    print("\n=== 개별 피처 top-20 (imp_any) ===")
    top = df.sort_values("imp_any", ascending=False).head(20)
    print(top[["feature", "group", "imp_class", "imp_dest", "imp_mode", "imp_any"]].round(3).to_string(index=False))
    print("\n=== imp_any 가 ~0 인 (정책이 거의 안 보는) 피처 수 ===")
    for thr in (0.001, 0.005, 0.01):
        print(f"  imp_any < {thr}: {(df['imp_any'] < thr).sum()}/{D}")

    # 그림: 그룹별 누적 + top 개별
    try:
        from plot_variant_eval import _set_korean_font
        _set_korean_font()
    except Exception:
        pass
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))
    gp = g_sum.reset_index()
    y = np.arange(len(gp))
    ax[0].barh(y, gp["imp_dest"], 0.4, label="dest(목적지)", color="#d62728")
    ax[0].barh(y, gp["imp_class"], 0.4, left=gp["imp_dest"], label="class(우선순위)", color="#1f77b4")
    ax[0].barh(y, gp["imp_mode"], 0.4, left=gp["imp_dest"] + gp["imp_class"], label="mode(이송)", color="#2ca02c")
    ax[0].set_yticks(y); ax[0].set_yticklabels(gp["group"]); ax[0].invert_yaxis()
    ax[0].set_xlabel("Σ permutation importance (축별 누적)"); ax[0].legend()
    ax[0].set_title("(a) obs 그룹별 누적 정책 기여도")
    ax[0].grid(axis="x", alpha=0.3)

    t = df.sort_values("imp_any", ascending=False).head(20).iloc[::-1]
    yy = np.arange(len(t))
    ax[1].barh(yy, t["imp_any"], color="#9467bd")
    ax[1].set_yticks(yy); ax[1].set_yticklabels(t["feature"], fontsize=8)
    ax[1].set_xlabel("imp_any (액션 변화율)")
    ax[1].set_title("(b) 개별 피처 top-20")
    ax[1].grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(ADIR, f"fig_obs_ablation_{a.tag}.png")
    plt.savefig(out_png, dpi=150); plt.close()
    print(f"\n[저장] {out_csv}, {out_png}")


if __name__ == "__main__":
    main()
