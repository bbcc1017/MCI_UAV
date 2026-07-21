# -*- coding: utf-8 -*-
"""v7 가치 예측 게이트 — NCRP 개선분(dpdr)이 obs 로 예측 가능한가.

배경: ExIt(행동 BC)는 3중 기각(v3·v4·v6) — 플래너가 greedy 를 뒤집는 이유가 미래 난수
실현(obs 부재 정보)이라 **행동**은 예측 불가. 가설: **크리틱(가치)은 기댓값을 배우므로
난수가 평균으로 씻겨** 예측 가능할 수 있다 → value-target 개입의 학습가능성 관문.

측정(ncrp_labels 가 q_greedy/q_best/q_exec/dpdr 를 pdrwog 단위로 저장):
  - obs → q_greedy   : 기준선(크리틱이 원래 배우는 상태가치 — 잘 예측될 것)
  - obs → q_best     : NCRP 개선가치(value-target 이 배울 목표)
  - obs → dpdr(=q_best−q_greedy) : **개선분 — 핵심.** 이게 예측되면 "어디서 개선 여지가
    큰지"를 크리틱이 알 수 있어 advantage 재조정이 유효. 노이즈면(R²≈0) value 도 익사.
분할: 지역 단위(ncrp_label_probe 관례 — 일반화·누수 차단). 회귀기: Ridge(선형)+HGB(비선형).
go/no-go 임계는 출력 안 함(수치만 — 판정은 main). B1(행동 BC switched acc 0.125)의 가치판.

예: python src/rl_src/value_gate_probe.py --labels results/rl/redesign/ncrp_value_labels.pkl \
      --out_json results/rl/redesign/value_gate_probe.json
"""
import argparse, json, pickle, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="ncrp_labels 병합 pkl(q_* 포함)")
    ap.add_argument("--val_frac", type=float, default=0.2, help="held-out 지역 비율")
    ap.add_argument("--out_json", default=None)
    A = ap.parse_args()

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    d = pickle.load(open(A.labels, "rb"))
    obs = np.asarray(d["obs"], dtype=np.float32)
    reg = np.asarray(d["regions"])
    qg, qb, dp = (np.asarray(d[k], dtype=np.float64) for k in ("q_greedy", "q_best", "dpdr"))
    sw = np.asarray(d["switched"], dtype=bool)
    # lookahead 수행 샘플만(dpdr 유한). 미수행(nan)은 dpdr=0 이라 게이트 무의미.
    keep = np.isfinite(dp) & np.isfinite(qg) & np.isfinite(qb)
    obs, reg, qg, qb, dp, sw = obs[keep], reg[keep], qg[keep], qb[keep], dp[keep], sw[keep]
    n = len(dp)
    print(f"[vgate] 라벨={A.labels}")
    print(f"[vgate] lookahead 샘플 N={n} (전체 {len(keep)}), 지역={len(set(reg))}")
    print(f"[vgate] dpdr: mean={dp.mean():.4f} std={dp.std():.4f} p90={np.percentile(dp,90):.4f} "
          f"| switched 비율={sw.mean():.3f} | switched dpdr mean={dp[sw].mean() if sw.any() else 0:.4f}")

    regions = sorted(set(reg.tolist()))
    rng = np.random.default_rng(0)
    val_set = set(rng.choice(regions, max(1, int(len(regions)*A.val_frac)), replace=False).tolist())
    va = np.isin(reg, list(val_set)); tr = ~va
    print(f"[vgate] 분할(지역): train {tr.sum()}({len(regions)-len(val_set)}지역) / "
          f"val {va.sum()}({len(val_set)}지역)")

    sc = StandardScaler().fit(obs[tr])
    Xtr, Xva = sc.transform(obs[tr]), sc.transform(obs[va])

    def fit(y, tag):
        out = {}
        for mname, mdl in [("ridge", Ridge(alpha=1.0)),
                           ("hgb", HistGradientBoostingRegressor(max_iter=300, max_depth=6,
                                                                 learning_rate=0.05, random_state=0))]:
            X_tr = Xtr if mname == "ridge" else obs[tr]   # HGB 는 원 스케일
            X_va = Xva if mname == "ridge" else obs[va]
            mdl.fit(X_tr, y[tr])
            p = mdl.predict(X_va)
            r2 = r2_score(y[va], p); mae = mean_absolute_error(y[va], p)
            out[mname] = {"r2": float(r2), "mae": float(mae)}
            print(f"    {tag:9s} [{mname:5s}] R²={r2:+.3f}  MAE={mae:.4f}")
        return out

    print("[vgate] held-out 회귀 (R² 높을수록 예측가능):")
    rep = {"n": int(n), "n_regions": len(regions), "n_val_regions": len(val_set),
           "dpdr_mean": float(dp.mean()), "dpdr_std": float(dp.std()),
           "switch_rate": float(sw.mean()),
           "q_greedy": fit(qg, "q_greedy"), "q_best": fit(qb, "q_best"), "dpdr": fit(dp, "dpdr")}
    # switched 한정 dpdr 예측(B1 대응 — 뒤집은 결정의 개선분)
    if sw.sum() > 50 and (va & sw).sum() > 10:
        ytr, yva = dp[tr & sw], dp[va & sw]
        mdl = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.05, random_state=0)
        mdl.fit(obs[tr & sw], ytr); p = mdl.predict(obs[va & sw])
        r2 = r2_score(yva, p)
        rep["dpdr_switched"] = {"hgb_r2": float(r2), "n_val": int((va & sw).sum())}
        print(f"    dpdr(switched만) [hgb ] R²={r2:+.3f} (n_val={int((va&sw).sum())})")

    print(f"\n[vgate] 요약: dpdr HGB R²={rep['dpdr']['hgb']['r2']:+.3f} vs "
          f"기준선 q_greedy HGB R²={rep['q_greedy']['hgb']['r2']:+.3f}")
    if A.out_json:
        json.dump(rep, open(A.out_json, "w"), ensure_ascii=False, indent=1)
        print(f"[vgate] JSON 저장: {A.out_json}")


if __name__ == "__main__":
    main()
