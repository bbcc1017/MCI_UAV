"""스코어 정책 정태 피팅 (플랜 v2 추출 트랙 B2).

collect_score_dataset 가 저장한 long-format npz(후보 φ·복원 S·chosen·offsets)에서
선형 스코어 `score = w·φ` 의 w 를 두 방법으로 적합한다:

  1) fit_condlogit — 조건부 로짓(McFadden) MLE. 각 이송 결정의 **적격 후보집합**에서
     P(선택) = softmax(w·φ) 로 보고, 볼록 음의 로그가능도(+L2)를 scipy L-BFGS 로 최소화.
     타깃은 RL 이 실제로 고른 후보(chosen) — 행동복제형 적합이라 S 복원 오프셋 문제 무관.
     criticality(loggap) 표본가중 옵션(불확실↔확신 결정 가중).
  2) fit_ridge_logits — 복원 스코어 S 를 타깃으로 한 ridge 회귀(부호 진단용). S 는
     (모드별 stay 기준) 상대 스코어라 **(결정,모드) 그룹 내 평균 중심화** 후 회귀해야
     그룹 오프셋(f_class+g_mode+S[0,m])이 소거된다(모드 상수열 is_uav 는 그룹내 불변→0).

두 방법의 w **부호가 일치**하고 상식적(eta<0, p_sent<0, occ_ratio<0, is_tier3>0 등)인지가
1차 위생 점검. CLI 는 부호 비교표·지표(pseudo_r2·top1_acc·kendall_tau)를 JSON 으로 저장.

--selfcheck: 알려진 w* 로 softmax 샘플한 합성 데이터에서 조건부로짓이 w* 를 복원하는지
  (상관·최대오차) 확인 — 최적화·그래디언트 구현의 단위검증(의미론 아님).

예 피팅:  PYTHONIOENCODING=utf-8 python src/rl_src/fit_score.py \
  --npz results/rl/redesign/score_dataset.npz --weight loggap \
  --out results/rl/redesign/score_fit.json
예 단위검증: python src/rl_src/fit_score.py --selfcheck
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from score_features import PHI_NAMES, K_PHI  # noqa: E402


# ------------------------------------------------------------------ 조건부 로짓 코어
def _condlogit_core(phi, offsets, chosen_idx, weights, l2, x0=None):
    """조건부 로짓 MLE 코어 (npz/합성 공용).

    phi (N_cand,K) · offsets (n_dec+1,) CSR · chosen_idx (n_dec,) 결정별 선택 후보의 로컬 idx
    · weights (n_dec,) 표본가중(평균 1 정규화 권장) · l2 릿지계수.
    반환 dict(w, se, ll, ll_null, pseudo_r2, top1_acc, converged, n_dec).
    """
    from scipy.optimize import minimize

    phi = np.asarray(phi, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.int64)
    chosen_idx = np.asarray(chosen_idx, dtype=np.int64)
    n_dec = len(offsets) - 1
    K = phi.shape[1]
    if weights is None:
        weights = np.ones(n_dec, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # 결정별 슬라이스(≥2 후보만 학습에 기여 — 후보 1개면 선택 확정, 정보 0).
    slices = [(int(offsets[d]), int(offsets[d + 1])) for d in range(n_dec)]

    def nll_and_grad(w):
        nll = 0.0
        grad = np.zeros(K, dtype=np.float64)
        for d, (s, e) in enumerate(slices):
            if e - s < 2:
                continue
            wt = weights[d]
            u = phi[s:e] @ w                        # (n_cand,)
            u -= u.max()                            # logsumexp 안정화
            ex = np.exp(u)
            Z = ex.sum()
            p = ex / Z                              # softmax 확률
            ci = chosen_idx[d]
            nll -= wt * (u[ci] - np.log(Z))
            m = p @ phi[s:e]                         # E[φ]
            grad -= wt * (phi[s + ci] - m)
        nll += l2 * float(w @ w)
        grad += 2.0 * l2 * w
        return nll, grad

    x0 = np.zeros(K) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    res = minimize(nll_and_grad, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-8})
    w = res.x

    # ---- 지표(비가중·데이터 로그가능도) + 표준오차(가중 Fisher 정보) ----
    ll = 0.0
    ll_null = 0.0
    top1 = 0
    n_used = 0
    Hess = np.zeros((K, K), dtype=np.float64)
    for d, (s, e) in enumerate(slices):
        if e - s < 2:
            continue
        n_used += 1
        u = phi[s:e] @ w
        u -= u.max()
        ex = np.exp(u)
        Z = ex.sum()
        p = ex / Z
        ci = chosen_idx[d]
        ll += (u[ci] - np.log(Z))
        ll_null += -np.log(e - s)                    # 균등선택(w=0) 기준
        if int(np.argmax(u)) == ci:
            top1 += 1
        m = p @ phi[s:e]
        # 가중 Fisher: Σ wt (E[φφ'] − m m')
        Ephi2 = (phi[s:e] * p[:, None]).T @ phi[s:e]
        Hess += weights[d] * (Ephi2 - np.outer(m, m))
    Hess += 2.0 * l2 * np.eye(K)
    try:
        cov = np.linalg.inv(Hess)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        se = np.full(K, np.nan)
    pseudo_r2 = float(1.0 - ll / ll_null) if ll_null != 0 else float("nan")
    return dict(w=w, se=se, ll=float(ll), ll_null=float(ll_null),
                pseudo_r2=pseudo_r2, top1_acc=float(top1 / max(n_used, 1)),
                converged=bool(res.success), n_dec=int(n_used), niter=int(res.nit))


def _chosen_idx_from_bool(offsets, chosen):
    """결정별 chosen(bool)에서 로컬 선택 idx 배열 — 선택 0개 결정은 −1(학습서 스킵)."""
    n_dec = len(offsets) - 1
    idx = np.full(n_dec, -1, dtype=np.int64)
    for d in range(n_dec):
        s, e = int(offsets[d]), int(offsets[d + 1])
        loc = np.flatnonzero(chosen[s:e])
        if loc.size:
            idx[d] = int(loc[0])
    return idx


def _kendall_tau_vs_S(w, phi, S, offsets, cand_m, mode_chosen):
    """(선택 모드 내) 후보에서 w·φ 랭킹과 복원 S 랭킹의 Kendall τ 평균.

    S 는 모드별 상대 스코어라 모드 내에서만 랭킹 비교가 정합(모드간 오프셋 소거됨)."""
    from scipy.stats import kendalltau
    n_dec = len(offsets) - 1
    taus = []
    for d in range(n_dec):
        s, e = int(offsets[d]), int(offsets[d + 1])
        if e - s < 2:
            continue
        mc = mode_chosen[d]
        sel = np.flatnonzero(cand_m[s:e] == mc)
        if sel.size < 2:
            continue
        score = phi[s:e][sel] @ w
        if np.ptp(score) == 0 or np.ptp(S[s:e][sel]) == 0:
            continue
        t, _ = kendalltau(score, S[s:e][sel])
        if np.isfinite(t):
            taus.append(t)
    return float(np.mean(taus)) if taus else float("nan")


# ------------------------------------------------------------------ 공개 API
def fit_condlogit(npz, weight="loggap", l2=1e-4):
    """npz 경로(또는 로드된 dict)에서 조건부 로짓 적합 → dict(w, se, ll, pseudo_r2,
    top1_acc, kendall_tau, ...). weight="loggap" 면 결정별 criticality 표본가중(평균1 정규화)."""
    d = np.load(npz, allow_pickle=True) if isinstance(npz, str) else npz
    phi = np.asarray(d["phi"], dtype=np.float64)
    offsets = np.asarray(d["offsets"], dtype=np.int64)
    chosen = np.asarray(d["chosen"], dtype=bool)
    chosen_idx = _chosen_idx_from_bool(offsets, chosen)

    n_dec = len(offsets) - 1
    if weight == "loggap":
        lg = np.asarray(d["loggap"], dtype=np.float64)
        lg = np.clip(lg, 0.0, None)
        w_samp = lg / lg.mean() if lg.mean() > 0 else np.ones(n_dec)
    else:
        w_samp = np.ones(n_dec, dtype=np.float64)
    # 선택 0개(chosen 없음) 결정은 가중 0 → 학습서 배제(코어의 slice<2 스킵과 별개 방어).
    w_samp = np.where(chosen_idx >= 0, w_samp, 0.0)
    chosen_idx = np.where(chosen_idx >= 0, chosen_idx, 0)

    out = _condlogit_core(phi, offsets, chosen_idx, w_samp, l2)
    # 복원 S 랭킹과의 정합(있으면)
    if "S" in d and "cand_m" in d and "mode_chosen" in d:
        out["kendall_tau"] = _kendall_tau_vs_S(
            out["w"], phi, np.asarray(d["S"], dtype=np.float64),
            offsets, np.asarray(d["cand_m"]), np.asarray(d["mode_chosen"]))
    else:
        out["kendall_tau"] = float("nan")
    return out


def fit_ridge_logits(npz, alpha=1.0):
    """복원 S 를 타깃으로 한 ridge 회귀(부호 진단) → w_ridge (K,).

    (결정,모드) 그룹 내 평균 중심화 후 회귀 — 그룹 오프셋(f_class+g_mode+S[0,m]) 소거.
    모드 상수열(is_uav)은 그룹내 불변→0 으로 소거되어 계수 미식별(≈0, 정상)."""
    from sklearn.linear_model import Ridge
    d = np.load(npz, allow_pickle=True) if isinstance(npz, str) else npz
    phi = np.asarray(d["phi"], dtype=np.float64)
    S = np.asarray(d["S"], dtype=np.float64)
    offsets = np.asarray(d["offsets"], dtype=np.int64)
    cand_m = np.asarray(d["cand_m"])
    n_dec = len(offsets) - 1

    Xc, yc = [], []
    for dd in range(n_dec):
        s, e = int(offsets[dd]), int(offsets[dd + 1])
        for mv in np.unique(cand_m[s:e]):
            g = s + np.flatnonzero(cand_m[s:e] == mv)
            if g.size < 2:
                continue                              # 단일 후보 그룹은 중심화 후 0
            Xc.append(phi[g] - phi[g].mean(axis=0))
            yc.append(S[g] - S[g].mean())
    if not Xc:
        return np.zeros(phi.shape[1])
    Xc = np.vstack(Xc)
    yc = np.concatenate(yc)
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(Xc, yc)
    return reg.coef_.astype(np.float64)


# ------------------------------------------------------------------ selfcheck
def _selfcheck(n_dec=8000, seed=0):
    """알려진 w* 로 softmax 샘플 → 조건부로짓 복원 오차 검증(최적화·그래디언트 단위검증)."""
    rng = np.random.default_rng(seed)
    K = K_PHI
    # 상식적 부호를 흉내낸 임의 w* (스케일 식별 확인용 — 절대값 다양)
    w_true = np.array([-1.6, -0.5, 0.9, -0.7, -0.3, -0.4, -0.8, 1.1, -0.6, -0.5, 0.4, -0.9])[:K]
    phi_rows, off, chosen_idx = [], [0], []
    for _ in range(n_dec):
        nc = int(rng.integers(3, K + 1))
        phi = rng.standard_normal((nc, K))
        u = phi @ w_true
        p = np.exp(u - u.max())
        p /= p.sum()
        ci = int(rng.choice(nc, p=p))
        phi_rows.append(phi.astype(np.float64))
        off.append(off[-1] + nc)
        chosen_idx.append(ci)
    phi = np.vstack(phi_rows)
    offsets = np.asarray(off, dtype=np.int64)
    chosen_idx = np.asarray(chosen_idx, dtype=np.int64)

    out = _condlogit_core(phi, offsets, chosen_idx, None, l2=1e-6)
    w_hat = out["w"]
    err = np.abs(w_hat - w_true)
    corr = float(np.corrcoef(w_hat, w_true)[0, 1])
    print("=== fit_condlogit selfcheck (합성 softmax) ===", flush=True)
    print(f"  결정 {n_dec}  수렴={out['converged']}  niter={out['niter']}  "
          f"pseudo_r2={out['pseudo_r2']:.3f}  top1={out['top1_acc']:.3f}", flush=True)
    print(f"  {'idx':>3} {'name':>12} {'w_true':>8} {'w_hat':>8} {'se':>7} {'|err|':>7}", flush=True)
    for i, nm in enumerate(PHI_NAMES):
        print(f"  {i:>3} {nm:>12} {w_true[i]:>8.3f} {w_hat[i]:>8.3f} "
              f"{out['se'][i]:>7.3f} {err[i]:>7.3f}", flush=True)
    ok_sign = bool((np.sign(w_hat) == np.sign(w_true)).all())
    ok = bool(corr > 0.98 and err.max() < 0.25 and ok_sign)
    print(f"\n  corr(w_hat,w_true)={corr:.4f}  max|err|={err.max():.4f}  부호일치={ok_sign}", flush=True)
    print("  ✅ 복원 성공" if ok else "  ❌ 복원 실패 — 최적화/그래디언트 점검", flush=True)
    return 0 if ok else 1


# ------------------------------------------------------------------ CLI
def _sign_table(w_cl, w_rg, se):
    """조건부로짓 vs ridge 부호 비교표(문자열 리스트)."""
    lines = [f"  {'idx':>3} {'name':>12} {'w_cl':>9} {'se':>7} {'w_ridge':>9} {'부호일치':>8}"]
    agree = 0
    n = 0
    for i, nm in enumerate(PHI_NAMES):
        sc, sr = np.sign(w_cl[i]), np.sign(w_rg[i])
        # ridge 는 모드상수열(is_uav)을 0 으로 소거 → 부호비교서 제외
        skip = (nm == "is_uav" and abs(w_rg[i]) < 1e-6)
        same = "—" if skip else ("O" if sc == sr else "X")
        if not skip:
            n += 1
            agree += int(sc == sr)
        lines.append(f"  {i:>3} {nm:>12} {w_cl[i]:>9.4f} {se[i]:>7.4f} {w_rg[i]:>9.4f} {same:>8}")
    lines.append(f"  부호일치 {agree}/{n}")
    return lines, agree, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--weight", choices=["loggap", "none"], default="loggap")
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--ridge_alpha", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()

    if A.selfcheck:
        sys.exit(_selfcheck())
    if not A.npz:
        ap.error("--npz 필요(또는 --selfcheck).")

    weight = None if A.weight == "none" else A.weight
    cl = fit_condlogit(A.npz, weight=weight, l2=A.l2)
    w_rg = fit_ridge_logits(A.npz, alpha=A.ridge_alpha)
    w_cl = cl["w"]

    print(f"=== fit_condlogit (npz={A.npz}, weight={A.weight}, l2={A.l2}) ===", flush=True)
    print(f"  결정 {cl['n_dec']}  수렴={cl['converged']}  ll={cl['ll']:.1f}  "
          f"pseudo_r2={cl['pseudo_r2']:.4f}  top1_acc={cl['top1_acc']:.4f}  "
          f"kendall_tau={cl['kendall_tau']:.4f}", flush=True)
    lines, agree, ntot = _sign_table(w_cl, w_rg, cl["se"])
    print("\n=== 부호 비교표 (조건부로짓 vs ridge[복원S]) ===", flush=True)
    for ln in lines:
        print(ln, flush=True)

    if A.out:
        os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
        payload = {
            "phi_names": PHI_NAMES,
            "w": {nm: float(w_cl[i]) for i, nm in enumerate(PHI_NAMES)},
            "w_vec": [float(x) for x in w_cl],
            "se": {nm: float(cl["se"][i]) for i, nm in enumerate(PHI_NAMES)},
            "w_ridge": {nm: float(w_rg[i]) for i, nm in enumerate(PHI_NAMES)},
            "metrics": {"ll": cl["ll"], "ll_null": cl["ll_null"],
                        "pseudo_r2": cl["pseudo_r2"], "top1_acc": cl["top1_acc"],
                        "kendall_tau": cl["kendall_tau"], "n_dec": cl["n_dec"],
                        "converged": cl["converged"]},
            "sign_agreement": {"agree": agree, "total": ntot},
            "config": {"npz": A.npz, "weight": A.weight, "l2": A.l2,
                       "ridge_alpha": A.ridge_alpha},
        }
        with open(A.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n저장 {A.out}", flush=True)


if __name__ == "__main__":
    main()
