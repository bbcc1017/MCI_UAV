"""지역 학습 가중치 산출 (플랜 v2 S3 부품).

train_ppo_feature 의 `--region_weights` 는 매니페스트 지역을 균등이 아니라 **가중 샘플링**
하도록 CSV(컬럼 region,weight)를 읽는다. 이 모듈은 그 CSV 를 두 신호로 만든다:

  * regret  : paired CSV(region, PDR_<model>, PDR_lb_T4)에서 모델이 LB-T4 에 **뒤지는**
              지역(regret = max(PDR_model − PDR_lb_T4, 0), PDR 은 낮을수록 좋음)에 가중.
              "약한 지역 더 학습" 신호.
  * headroom: oracle_headroom CSV(region, ep, pdr_base, pdr_oracle, …)에서 지역별
              Δ = mean(pdr_base − pdr_oracle)(오라클이 열어주는 개선 여지)에 가중.

공통 공식(둘 다 균등 하한 + softmax 혼합, 합=1):
    w_r = floor·(1/N) + (1−floor)·softmax(signal_r / τ)
  τ 기본 = 신호의 표준편차(문서화; 0 이면 1 로 대체 → 균등). floor 기본 0.5.

출력 CSV 는 **BOM 없이** region,weight (train_ppo_feature 가 utf-8-sig 로 읽으므로 BOM
있어도 무해하나, 관례상 순수 utf-8 로 저장).

CLI 예(headroom):  PYTHONIOENCODING=utf-8 python src/rl_src/region_weights.py \
  --csv results/rl/redesign/oracle_headroom_sido17.csv --mode headroom \
  --out results/rl/redesign/region_weights_headroom.csv
CLI 예(regret):    python src/rl_src/region_weights.py \
  --csv results/rl/redesign/program_eval.csv --mode regret --model_col PDR_rl \
  --out results/rl/redesign/region_weights_regret.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

# regret 모드에서 '모델'이 아니라 '기준선'으로 취급할 PDR 컬럼(모델 자동선택서 제외).
_BASELINE_COLS = {"PDR_lb_T4", "PDR_lb_adaptT", "PDR_heur", "PDR_adaptT", "PDR_lb"}


def _softmax_mix(signal, floor, tau):
    """w = floor·균등 + (1−floor)·softmax(signal/τ). τ None 이면 std(신호)(0→1)."""
    signal = np.asarray(signal, dtype=float)
    N = len(signal)
    if tau is None:
        tau = float(signal.std())
    if not np.isfinite(tau) or tau <= 0:
        tau = 1.0
    z = signal / tau
    z -= z.max()
    sm = np.exp(z)
    sm /= sm.sum()
    w = floor / N + (1.0 - floor) * sm
    return w, tau


def _read_concat(csv_paths):
    """단일 경로 또는 리스트 → 모든 행 concat DataFrame."""
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    frames = [pd.read_csv(p, encoding="utf-8-sig") for p in csv_paths]
    return pd.concat(frames, ignore_index=True)


def _pick_model_col(df, model_col, baseline_col):
    """regret 모델 컬럼 자동선택 — PDR_rl 우선, 없으면 baseline/기준선 아닌 첫 PDR_ 컬럼."""
    if model_col:
        return model_col
    pdr_cols = [c for c in df.columns if c.startswith("PDR_")]
    if "PDR_rl" in pdr_cols:
        return "PDR_rl"
    for c in pdr_cols:
        if c != baseline_col and c not in _BASELINE_COLS:
            return c
    raise ValueError(f"모델 PDR 컬럼을 못 찾음 (컬럼: {list(df.columns)})")


def compute(csv_paths, mode="regret", floor=0.5, tau=None,
            model_col=None, baseline_col="PDR_lb_T4"):
    """지역 가중 DataFrame(region, weight, signal) 반환.

    regret : paired CSV(region, PDR_model, PDR_lb_T4) 지역당 1행 가정.
    headroom: oracle_headroom CSV(region, ep, pdr_base, pdr_oracle) 지역당 다행 → 평균 Δ.
    """
    df = _read_concat(csv_paths)
    if mode == "regret":
        mc = _pick_model_col(df, model_col, baseline_col)
        if baseline_col not in df.columns:
            raise ValueError(f"baseline 컬럼 {baseline_col} 없음 (컬럼: {list(df.columns)})")
        g = df.groupby("region", sort=False)
        regions = list(g.groups.keys())
        sig = np.array([max(float(df.loc[idx, mc].mean() - df.loc[idx, baseline_col].mean()), 0.0)
                        for idx in [g.groups[r] for r in regions]])
        meta = f"regret(model={mc} − {baseline_col}, clip≥0)"
    elif mode == "headroom":
        for c in ("pdr_base", "pdr_oracle"):
            if c not in df.columns:
                raise ValueError(f"headroom 컬럼 {c} 없음 (컬럼: {list(df.columns)})")
        df = df.assign(_delta=df["pdr_base"] - df["pdr_oracle"])
        agg = df.groupby("region", sort=False)["_delta"].mean()
        regions = list(agg.index)
        sig = agg.values.astype(float)
        meta = "headroom(mean(pdr_base − pdr_oracle))"
    else:
        raise ValueError(f"mode 는 regret|headroom (got {mode})")

    w, tau_used = _softmax_mix(sig, floor, tau)
    out = pd.DataFrame({"region": regions, "weight": w, "signal": sig})
    out.attrs["meta"] = meta
    out.attrs["tau"] = tau_used
    out.attrs["floor"] = floor
    return out


def _summary(df):
    o = df.sort_values("weight", ascending=False).reset_index(drop=True)
    print(f"  신호정의: {df.attrs.get('meta')}  τ={df.attrs.get('tau'):.4g}  "
          f"floor={df.attrs.get('floor')}  N={len(df)}", flush=True)
    print(f"  weight 합={df['weight'].sum():.4f}(=1) min={df['weight'].min():.4g} "
          f"max={df['weight'].max():.4g} (균등={1.0/len(df):.4g})", flush=True)
    print("\n  === 상위 10 (가중 큼 = 더 자주 학습) ===", flush=True)
    print(f"  {'region':>16} {'weight':>9} {'signal':>9}", flush=True)
    for _, r in o.head(10).iterrows():
        print(f"  {str(r['region']):>16} {r['weight']:>9.5f} {r['signal']:>9.5f}", flush=True)
    print("\n  === 하위 10 ===", flush=True)
    for _, r in o.tail(10).iterrows():
        print(f"  {str(r['region']):>16} {r['weight']:>9.5f} {r['signal']:>9.5f}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="쉼표 구분 경로(여러 개면 행 concat)")
    ap.add_argument("--mode", choices=["regret", "headroom"], default="regret")
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=None, help="미지정 시 std(신호)")
    ap.add_argument("--model_col", default=None, help="regret 모델 PDR 컬럼(미지정 자동)")
    ap.add_argument("--baseline_col", default="PDR_lb_T4")
    ap.add_argument("--out", default=None)
    A = ap.parse_args()

    paths = [p for p in A.csv.split(",") if p]
    df = compute(paths, mode=A.mode, floor=A.floor, tau=A.tau,
                 model_col=A.model_col, baseline_col=A.baseline_col)
    print(f"=== region_weights (mode={A.mode}, csv={paths}) ===", flush=True)
    _summary(df)
    if A.out:
        os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
        # BOM 없이 region,weight (train_ppo_feature --region_weights 형식)
        df[["region", "weight"]].to_csv(A.out, index=False, encoding="utf-8")
        print(f"\n저장 {A.out} (region,weight, BOM 없음)", flush=True)


if __name__ == "__main__":
    main()
