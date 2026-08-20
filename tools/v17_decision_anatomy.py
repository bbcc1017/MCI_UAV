"""PPO 교사 의사결정 심층 해부 — 마스크 제약을 걷어낸 뒤 무엇이 남는가.

입력(재수집 0, 기존 자산 재사용):
  * results/scoreboard/v10/distill/data/ppo_train1000_seed5000.npz
      후보 1,835,939행 × 43특징 + `target`(=PPO 확률, 상태별 합 1) + cand_action + offsets
  * results/scoreboard/v17/fieldrules/static_train1000.npz
      좌표 1,000 × 병원 47 정적 물리량(도로/직선 km, tier, 헬기장)

출력: PNG/SVG 3장 + 수치 JSON.
  fig1  제약 해부 + 자유도(실질 선택지 수·확신도)
  fig2  선호 곡면 — 거리 × 부하 × PPO 확률 3차원 + 등가선 + 드러난 교환율
  fig3  축별 결정 — 수단(가용성 vs 진짜 선택) · 등급 · 2항 점수 일치도

설계 원칙: 물리량은 obs 평탄화가 아니라 정적표에서 읽는다(v17 규율). 거리는 수단별
정의를 따른다 — AMB=도로 km, UAV=직선 km.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

rcParams["font.family"] = "NanumGothic"
rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parents[1]
NPZ = REPO / "results/scoreboard/v10/distill/data/ppo_train1000_seed5000.npz"
STATIC = REPO / "results/scoreboard/v17/fieldrules/static_train1000.npz"
OUT = REPO / "results/scoreboard/v17/anatomy"
H_PAD = 47

C_R, C_Y = "#c0392b", "#e08a1e"
C_AMB, C_UAV = "#2c6fbb", "#1f9d76"
C_GRAY = "#8a8a8a"


def decode(a: int, h_pad: int = H_PAD):
    n_dest = h_pad + 1
    return a // (n_dest * 2), (a % (n_dest * 2)) // 2, a % 2


# ------------------------------------------------------------------ 자료 조립
def build():
    z = np.load(NPZ, allow_pickle=True)
    st = np.load(STATIC, allow_pickle=False)
    fn = [str(x) for x in z["feature_names"]]
    col = {n: i for i, n in enumerate(fn)}
    X, off, cand, teach = z["X"], z["offsets"], z["cand_action"], z["teacher_action"]
    prob = z["target"].astype(np.float64)
    keys = np.asarray([str(x) for x in z["state_key"]])
    skeys = {str(k): i for i, k in enumerate(st["keys"])}

    n_state = len(off) - 1
    dec = {int(a): decode(int(a)) for a in np.unique(cand)}

    # 후보 단위 파생열
    N = X.shape[0]
    cls = np.empty(N, np.int8)
    mode = np.empty(N, np.int8)
    dest = np.empty(N, np.int16)
    sidx = np.empty(N, np.int32)
    for s in range(n_state):
        a, b = int(off[s]), int(off[s + 1])
        sidx[a:b] = s
        for j in range(a, b):
            c, d, m = dec[int(cand[j])]
            cls[j], dest[j], mode[j] = c, d, m

    is_stay = dest == 0
    load = X[:, col["cand_occ"]] + X[:, col["cand_in_flight"]]   # 명 (지금 안고 있는 환자)
    p_sent = X[:, col["cand_p_sent"]]
    occ_ratio = X[:, col["cand_occ_ratio"]]

    # 정적 물리량 조인
    si = np.asarray([skeys[k] for k in keys], np.int32)
    d_road_t, d_euc_t = st["d_road"], st["d_euc"]
    tier_t, heli_t, Hs = st["tier"], st["heli"], st["H"]
    h = np.clip(dest.astype(np.int32) - 1, 0, H_PAD - 1)
    ss = si[sidx]
    km = np.where(mode == 1, d_euc_t[ss, h], d_road_t[ss, h]).astype(np.float64)
    km[is_stay] = np.nan
    tier3 = (tier_t[ss, h] == 3).astype(np.float64)
    heli = (heli_t[ss, h] > 0.5).astype(np.float64)

    d = dict(
        X=X, col=col, off=off, prob=prob, cls=cls, mode=mode, dest=dest, sidx=sidx,
        is_stay=is_stay, load=load, p_sent=p_sent, occ_ratio=occ_ratio, km=km,
        tier3=tier3, heli=heli, teach=teach, dec=dec, si=si, keys=keys,
        d_road_t=d_road_t, d_euc_t=d_euc_t, tier_t=tier_t, heli_t=heli_t, Hs=Hs,
        n_state=n_state,
    )
    return d


# ------------------------------------------------------- A. 제약 해부 · 자유도
def anatomy(d):
    off, n_state = d["off"], d["n_state"]
    cls, mode, is_stay, prob = d["cls"], d["mode"], d["is_stay"], d["prob"]
    si, tier_t, heli_t, Hs = d["si"], d["tier_t"], d["heli_t"], d["Hs"]
    teach, dec = d["teach"], d["dec"]

    # 축별 적격 후보 수 (마스크가 이미 반영된 실제 후보집합)
    elig = np.zeros((n_state, 2, 2), np.int32)
    n_cand = np.zeros(n_state, np.int32)
    stay_p = np.zeros(n_state)
    perplex = np.zeros(n_state)
    gap12 = np.zeros(n_state)
    t_cls = np.zeros(n_state, np.int8)
    t_mode = np.zeros(n_state, np.int8)
    t_dest = np.zeros(n_state, np.int16)
    for s in range(n_state):
        a, b = int(off[s]), int(off[s + 1])
        sl = slice(a, b)
        real = ~is_stay[sl]
        cc, mm = cls[sl], mode[sl]
        for c in (0, 1):
            for m in (0, 1):
                elig[s, c, m] = int(np.sum(real & (cc == c) & (mm == m)))
        n_cand[s] = int(real.sum())
        p = prob[sl]
        stay_p[s] = float(p[~real].sum())
        q = np.clip(p, 1e-12, 1.0)
        perplex[s] = float(np.exp(-(q * np.log(q)).sum()))
        srt = np.sort(p)[::-1]
        gap12[s] = float(srt[0] - (srt[1] if srt.size > 1 else 0.0))
        t_cls[s], t_dest[s], t_mode[s] = dec[int(teach[s])]

    # 병원 구성(정적) — 깔때기 계산용
    H = Hs[si]
    n_t3 = np.asarray([int((tier_t[i, :Hs[i]] == 3).sum()) for i in si])
    n_hl = np.asarray([int((heli_t[i, :Hs[i]] > 0.5).sum()) for i in si])
    n_t3hl = np.asarray([int(((tier_t[i, :Hs[i]] == 3) & (heli_t[i, :Hs[i]] > 0.5)).sum())
                         for i in si])

    # 자격 제약만의 효과 = 그 축이 "열려 있었을 때"의 병원 수 (가용성 0 제외)
    cond_med = {}
    for cn, c in (("R", 0), ("Y", 1)):
        for mn, m in (("amb", 0), ("uav", 1)):
            v = elig[:, c, m]
            cond_med[f"{cn}_{mn}"] = float(np.median(v[v > 0])) if (v > 0).any() else 0.0

    free_mode = (elig[:, :, 0].sum(1) > 0) & (elig[:, :, 1].sum(1) > 0)
    # 등급별로 정확히: 선택된 등급에서 두 수단 모두 열려 있었나
    fm = np.asarray([elig[s, t_cls[s], 0] > 0 and elig[s, t_cls[s], 1] > 0
                     for s in range(n_state)])
    fc = np.asarray([elig[s, 0].sum() > 0 and elig[s, 1].sum() > 0
                     for s in range(n_state)])
    n_own = np.asarray([elig[s, t_cls[s], t_mode[s]] for s in range(n_state)])

    A = dict(
        elig=elig, n_cand=n_cand, stay_p=stay_p, perplex=perplex, gap12=gap12,
        t_cls=t_cls, t_mode=t_mode, t_dest=t_dest, H=H, n_t3=n_t3, n_hl=n_hl,
        n_t3hl=n_t3hl, free_mode=fm, free_class=fc, n_own=n_own,
        free_mode_any=free_mode, cond_med=cond_med,
    )
    return A


# --------------------------------------------- C. 선호 곡면 · 드러난 교환율
def surface(d, A):
    """선택된 (등급, 수단) 안의 후보만 남겨 거리·부하·PPO확률 3원 관계를 본다."""
    off = d["off"]
    keep = np.zeros(len(d["km"]), bool)
    for s in range(d["n_state"]):
        a, b = int(off[s]), int(off[s + 1])
        sl = slice(a, b)
        keep[a:b] = ((~d["is_stay"][sl]) & (d["cls"][sl] == A["t_cls"][s])
                     & (d["mode"][sl] == A["t_mode"][s]))
    km, load, prob = d["km"][keep], d["load"][keep], d["prob"][keep]
    sid = d["sidx"][keep]
    # 병원 고정효과용 그룹 키: 좌표 × 병원 × 수단 × 등급
    gkey = (d["si"][sid].astype(np.int64) * 1000
            + d["dest"][keep].astype(np.int64)) * 4 \
        + d["mode"][keep].astype(np.int64) * 2 + d["cls"][keep].astype(np.int64)
    # 상태 내 확률 재정규화 → "같은 등급·수단 안에서의 목적지 선호"
    ssum = np.bincount(sid, weights=prob, minlength=d["n_state"])
    pn = prob / np.maximum(ssum[sid], 1e-12)

    # 드러난 교환율: 선택 vs 최근접(같은 축) — Δkm / Δ부하
    ch_km = np.full(d["n_state"], np.nan)
    ch_ld = np.full(d["n_state"], np.nan)
    nr_km = np.full(d["n_state"], np.nan)
    nr_ld = np.full(d["n_state"], np.nan)
    n_in = np.full(d["n_state"], 0, np.int32)
    order = np.argsort(sid, kind="stable")
    sid_s, km_s, ld_s, pn_s = sid[order], km[order], load[order], pn[order]
    bounds = np.searchsorted(sid_s, np.arange(d["n_state"] + 1))
    for s in range(d["n_state"]):
        a, b = bounds[s], bounds[s + 1]
        if b - a < 1:
            continue
        k, l, p = km_s[a:b], ld_s[a:b], pn_s[a:b]
        n_in[s] = b - a
        j = int(np.argmax(p))
        i0 = int(np.argmin(k))
        ch_km[s], ch_ld[s] = k[j], l[j]
        nr_km[s], nr_ld[s] = k[i0], l[i0]
    dkm, dld = ch_km - nr_km, nr_ld - ch_ld      # 더 간 거리 / 덜어낸 부하
    # 상태별 argmax = 실제 선택 (선택률 표면용)
    is_pick = np.zeros(len(km), bool)
    top = np.full(d["n_state"], -1, np.int64)
    for s in range(d["n_state"]):
        a, b = bounds[s], bounds[s + 1]
        if b - a >= 1:
            top[s] = a + int(np.argmax(pn_s[a:b]))
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    good = top[top >= 0]
    is_pick[order[good]] = True
    # 후보 수 교란 제거: lift = 확률 / (1/후보수) = 균등 배분 대비 선호 배수
    lift = pn * np.maximum(n_in[sid], 1)
    # 병원 고정효과 제거 = 같은 좌표·같은 병원이 "비었을 때 vs 찼을 때"만 비교
    y = np.log2(np.clip(lift, 2.0 ** -8, None))
    uq, inv, cnt = np.unique(gkey, return_inverse=True, return_counts=True)
    gsum = np.bincount(inv, weights=y, minlength=len(uq))
    gmean = gsum / cnt
    y_within = y - gmean[inv]
    big = cnt[inv] >= 20
    S = dict(km=km, load=load, pn=pn, sid=sid, dkm=dkm, dld=dld, is_pick=is_pick,
             lift=lift, y=y, y_within=y_within, big=big, gkey=gkey,
             ch_km=ch_km, ch_ld=ch_ld, nr_km=nr_km, nr_ld=nr_ld, n_in=n_in)
    return S


def cells(d, A, n_boot: int = 2000):
    """셀(등급×수단)별 재량 폭·최근접률·교환율(km/명, 분/명 + 시군구 클러스터 CI)."""
    off, n = d["off"], d["n_state"]
    st = np.load(STATIC, allow_pickle=False)
    t_amb, t_uav = st["t_amb"], st["t_uav"]
    si = d["si"]
    sig = np.asarray([k.rsplit("_", 2)[-2] for k in d["keys"]])
    rows = []
    for cn, c in (("Red", 0), ("Yellow", 1)):
        for mn, mm in (("AMB", 0), ("UAV", 1)):
            sel = np.flatnonzero((A["t_cls"] == c) & (A["t_mode"] == mm))
            near = tot = agr = 0
            lk, lt, clus = [], [], []
            for s_ in sel:
                a, b = int(off[s_]), int(off[s_ + 1])
                msk = ((~d["is_stay"][a:b]) & (d["cls"][a:b] == c)
                       & (d["mode"][a:b] == mm))
                if msk.sum() < 2:
                    continue
                hh = (d["dest"][a:b][msk] - 1).astype(int)
                k = d["km"][a:b][msk]
                l = d["load"][a:b][msk]
                p = d["prob"][a:b][msk]
                tt = (t_uav if mm == 1 else t_amb)[si[s_], hh]
                j, i0 = int(np.argmax(p)), int(np.argmin(k))
                tot += 1
                near += int(j == i0)
                agr += int(int(np.argmin(k + 6.37 * l)) == j)
                dl = l[i0] - l[j]
                if dl > 0:
                    lk.append((k[j] - k[i0]) / dl)
                    lt.append((tt[j] - tt[i0]) / dl)
                    clus.append(sig[s_])
            lk, lt, clus = np.asarray(lk), np.asarray(lt), np.asarray(clus)
            ci = [float("nan")] * 2
            if lt.size > 20:
                uc = np.unique(clus)
                idx = {u: np.flatnonzero(clus == u) for u in uc}
                rng = np.random.default_rng(7)
                bs = [np.median(lt[np.concatenate(
                    [idx[u] for u in rng.choice(uc, len(uc), replace=True)])])
                    for _ in range(n_boot)]
                ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            rows.append(dict(
                cell=f"{cn}+{mn}", n=int(sel.size), share=100.0 * sel.size / n,
                med_cand=float(np.median(A["n_own"][sel])),
                nearest_pct=100.0 * near / max(tot, 1),
                agree637_pct=100.0 * agr / max(tot, 1),
                lam_km=float(np.median(lk)) if lk.size else float("nan"),
                lam_min=float(np.median(lt)) if lt.size else float("nan"),
                lam_min_ci=ci,
                perplex=float(np.median(A["perplex"][sel])),
                indiff_pct=100.0 * float((A["gap12"][sel] < 0.1).mean()),
                n_trade=int(lk.size)))
    return rows


def unit_contest(d, A, km_grid=(0, 2, 4, 6.37, 9, 12, 16, 20, 25),
                 min_grid=(0, 1, 2, 3, 4, 5, 6, 8, 10)):
    """단일 가중을 km에 걸 때와 분에 걸 때, PPO 목적지 argmax 일치율 대결."""
    off, n = d["off"], d["n_state"]
    st = np.load(STATIC, allow_pickle=False)
    t_amb, t_uav = st["t_amb"], st["t_uav"]
    si = d["si"]
    pre = []
    for s_ in range(n):
        c, mm = int(A["t_cls"][s_]), int(A["t_mode"][s_])
        a, b = int(off[s_]), int(off[s_ + 1])
        msk = ((~d["is_stay"][a:b]) & (d["cls"][a:b] == c) & (d["mode"][a:b] == mm))
        if msk.sum() < 2:
            continue
        hh = (d["dest"][a:b][msk] - 1).astype(int)
        pre.append((d["km"][a:b][msk], d["load"][a:b][msk],
                    (t_uav if mm == 1 else t_amb)[si[s_], hh],
                    int(np.argmax(d["prob"][a:b][msk]))))
    out = {}
    for unit, grid in (("km", km_grid), ("min", min_grid)):
        sc = {}
        for lam in grid:
            ok = sum(int(np.argmin((k if unit == "km" else t) + lam * l) == j)
                     for k, l, t, j in pre)
            sc[float(lam)] = ok / len(pre)
        best = max(sc, key=sc.get)
        out[unit] = dict(grid=sc, best_lam=best, best_acc=sc[best])
    out["n"] = len(pre)
    return out


def score_agreement(d, A, lams=(0.0, 6.37, 12.0)):
    """PPO의 목적지 argmax가 `km + λ·부하` argmin과 일치하는 비율."""
    off = d["off"]
    out = {f"lam_{l:g}": 0 for l in lams}
    tot = 0
    for s in range(d["n_state"]):
        a, b = int(off[s]), int(off[s + 1])
        sl = slice(a, b)
        m = ((~d["is_stay"][sl]) & (d["cls"][sl] == A["t_cls"][s])
             & (d["mode"][sl] == A["t_mode"][s]))
        if m.sum() < 2:
            continue
        k = d["km"][a:b][m]
        l = d["load"][a:b][m]
        p = d["prob"][a:b][m]
        pick = int(np.argmax(p))
        tot += 1
        for lam in lams:
            if int(np.argmin(k + lam * l)) == pick:
                out[f"lam_{lam:g}"] += 1
    return {k: v / max(tot, 1) for k, v in out.items()} | {"n": tot}


# ------------------------------------------------------------------ 그림 1
def fig1(A, res):
    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)

    # (a) 제약 깔때기
    ax = fig.add_subplot(gs[0, 0])
    cm = A["cond_med"]
    lab = ["전체 병원", "Yellow+AMB", "Yellow+UAV\n(헬기장)", "Red+AMB\n(Tier3)",
           "Red+UAV\n(Tier3∩헬기장)"]
    val = [np.median(A["H"]), cm["Y_amb"], cm["Y_uav"], cm["R_amb"], cm["R_uav"]]
    cols = [C_GRAY, C_AMB, C_UAV, C_R, "#7b1f6e"]
    b = ax.barh(range(5)[::-1], val, color=cols, height=0.62)
    ax.set_yticks(range(5)[::-1]); ax.set_yticklabels(lab, fontsize=9)
    for r, v in zip(b, val):
        ax.text(v + 0.7, r.get_y() + r.get_height() / 2, f"{v:.0f}", va="center", fontsize=10)
    ax.set_xlabel("적격 병원 수 (중위)")
    ax.set_title("(1) 자격 제약 깔때기 (축이 열렸을 때의 중위)", fontsize=11, weight="bold")
    ax.set_xlim(0, max(val) * 1.18)

    # (b) 축별 적격 수 분포
    ax = fig.add_subplot(gs[0, 1])
    data = [A["elig"][:, 1, 0], A["elig"][:, 1, 1], A["elig"][:, 0, 0], A["elig"][:, 0, 1]]
    names = ["Y+AMB", "Y+UAV", "R+AMB", "R+UAV"]
    vp = ax.violinplot(data, showmedians=True, widths=0.82)
    for pc, c in zip(vp["bodies"], [C_AMB, C_UAV, C_R, "#7b1f6e"]):
        pc.set_facecolor(c); pc.set_alpha(0.55)
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(names)
    ax.set_ylabel("적격 병원 수 (0 = 가용성 없음)")
    ax.set_title("(2) 축마다 선택지 폭이 다르다", fontsize=11, weight="bold")
    ax.axhline(1, color="k", ls=":", lw=1)
    z0 = [float((x == 0).mean()) * 100 for x in data]
    ax.text(0.02, 0.96, "적격 0개 비율\n" + " · ".join(f"{n} {v:.0f}%" for n, v in zip(names, z0)),
            transform=ax.transAxes, va="top", fontsize=8.5, color="#444")

    # (c) 실질 선택지 수
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(A["n_cand"], bins=40, color=C_GRAY, alpha=0.55, label="법적 후보 수")
    ax.hist(A["perplex"], bins=40, color="#b5179e", alpha=0.75, label="실질 선택지 수")
    ax.set_xlabel("개수"); ax.set_ylabel("결정 수")
    ax.set_title("(3) 후보는 많고 고민은 적다", fontsize=11, weight="bold")
    ax.legend(fontsize=9)
    ax.text(0.97, 0.55, f"중위 {np.median(A['n_cand']):.0f}개 →"
                        f" {np.median(A['perplex']):.1f}개",
            transform=ax.transAxes, ha="right", fontsize=10, color="#b5179e", weight="bold")

    # (d) 확신도
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(A["gap12"], bins=50, color="#2b6cb0", alpha=0.8)
    ax.set_xlabel("1순위와 2순위의 확률차"); ax.set_ylabel("결정 수")
    ax.set_title("(4) 얼마나 확신하나", fontsize=11, weight="bold")
    ax.axvline(np.median(A["gap12"]), color="k", ls="--", lw=1)
    ax.text(0.97, 0.9, f"중위 {np.median(A['gap12']):.3f}\n"
                       f"0.1 미만 = 사실상 무차별 {float((A['gap12'] < 0.1).mean())*100:.0f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5)

    # (e) 자유도 스펙트럼
    ax = fig.add_subplot(gs[1, 1])
    seg = [("수단 강제\n(한쪽만 대기)", 1 - A["free_mode"].mean()),
           ("수단 자유", A["free_mode"].mean()),
           ("등급 강제", 1 - A["free_class"].mean()),
           ("등급 자유", A["free_class"].mean()),
           ("목적지 1개", float((A["n_own"] <= 1).mean())),
           ("목적지 2개 이상", float((A["n_own"] > 1).mean()))]
    y = [0, 0, 1, 1, 2, 2]
    left = 0.0
    for i, (nm, v) in enumerate(seg):
        if i % 2 == 0:
            left = 0.0
        c = "#d9d9d9" if i % 2 == 0 else [C_UAV, C_R, "#2b6cb0"][i // 2]
        ax.barh(y[i], v, left=left, color=c, height=0.6)
        if v > 0.08:
            ax.text(left + v / 2, y[i], f"{v*100:.1f}%", ha="center", va="center",
                    fontsize=9.5, color="k" if i % 2 == 0 else "w", weight="bold")
        left += v
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["수단 축", "등급 축", "목적지 축"])
    ax.set_xlim(0, 1); ax.set_xlabel("결정 비중 (회색 = 제약·가용성이 이미 정해준 몫)")
    ax.set_title("(5) 어느 축에 진짜 재량이 있나", fontsize=11, weight="bold")

    # (f) 대기(보류) 확률
    ax = fig.add_subplot(gs[1, 2])
    sp = A["stay_p"]
    ax.hist(np.log10(np.clip(sp, 1e-8, 1)), bins=50, color="#7b7b7b", alpha=0.85)
    ax.set_xlabel("대기(현장 잔류)에 준 확률, log10"); ax.set_ylabel("결정 수")
    ax.set_title("(6) 보류는 고려조차 안 된다", fontsize=11, weight="bold")
    ax.text(0.03, 0.95, f"중위 {np.median(sp):.2e}\n평균 {sp.mean():.2e}\n"
                        f"1% 넘는 결정 {float((sp > 0.01).mean())*100:.2f}%\n"
                        f"argmax가 대기 {res['stay_argmax_pct']:.2f}%",
            transform=ax.transAxes, va="top", fontsize=9.5)

    fig.suptitle("PPO 교사 의사결정 해부 (1) — 제약이 먼저 자르고, 남은 재량은 좁다",
                 fontsize=13.5, weight="bold")
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"anatomy_constraints.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 그림 2
def _pick_surface(S):
    """(거리, 부하) 격자별 선호 배수(lift) 표면. lift 1 = 균등 배분과 같음."""
    kb = np.linspace(0, 60, 25)
    lb = np.arange(0, 7)
    Z = np.full((len(lb) - 1, len(kb) - 1), np.nan)
    N = np.zeros_like(Z)
    ki = np.digitize(S["km"], kb) - 1
    li = np.digitize(S["load"], lb) - 1
    ok = (ki >= 0) & (ki < len(kb) - 1) & (li >= 0) & (li < len(lb) - 1)
    for a in range(len(lb) - 1):
        for b in range(len(kb) - 1):
            sel = ok & (li == a) & (ki == b)
            c = int(sel.sum())
            if c >= 60:
                Z[a, b] = float(np.mean(S["lift"][sel]))
                N[a, b] = c
    return kb, lb, Z, N


def fig2(S, res, S_keys):
    fig = plt.figure(figsize=(16.4, 5.6))
    kb, lb, Z, _ = _pick_surface(S)
    kc = (kb[:-1] + kb[1:]) / 2
    lc = (lb[:-1] + lb[1:]) / 2

    # (a) 3차원 선택률 표면
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    KK, LL = np.meshgrid(kc, lc)
    Zp = np.where(np.isfinite(Z), Z, np.nan)
    ax.plot_surface(KK, LL, Zp, cmap="viridis", edgecolor="#333", linewidth=0.3,
                    rstride=1, cstride=1, alpha=0.96, antialiased=True)
    ax.contour(KK, LL, Zp, levels=[1.0], colors="#ff2d55", linewidths=2.5,
               offset=0.0)
    ax.set_xlabel("거리 (km)", labelpad=8)
    ax.set_ylabel("병원 부하 (명)", labelpad=6)
    ax.set_zlabel("선호 배수 (균등 = 1)", labelpad=8)
    ax.view_init(elev=24, azim=-121)
    ax.set_title("(1) 원자료 선호 곡면 (거리 축만 신뢰)", fontsize=11, weight="bold")

    # (b) 부하의 진짜 효과 — 병원 고정효과 전 / 후
    ax = fig.add_subplot(1, 3, 2)
    lv = np.arange(0, 6)
    raw, wit, rlo, rhi, wlo, whi = [], [], [], [], [], []
    rng = np.random.default_rng(11)
    sig = np.asarray([k.rsplit("_", 2)[-2] for k in S_keys])[S["sid"]]
    for v in lv:
        m0 = S["load"] == v
        m1 = m0 & S["big"]
        raw.append(float(np.mean(S["y"][m0])) if m0.sum() > 30 else np.nan)
        wit.append(float(np.mean(S["y_within"][m1])) if m1.sum() > 30 else np.nan)
        for arr, msk, lo, hi in ((S["y"], m0, rlo, rhi), (S["y_within"], m1, wlo, whi)):
            if msk.sum() > 30:
                cl = sig[msk]
                uc = np.unique(cl)
                idx = {u: np.flatnonzero(cl == u) for u in uc}
                a_ = arr[msk]
                bs = [np.mean(a_[np.concatenate(
                    [idx[u] for u in rng.choice(uc, len(uc), replace=True)])])
                    for _ in range(400)]
                lo.append(float(np.percentile(bs, 2.5)))
                hi.append(float(np.percentile(bs, 97.5)))
            else:
                lo.append(np.nan); hi.append(np.nan)
    raw, wit = np.asarray(raw), np.asarray(wit)
    ax.errorbar(lv, raw, yerr=[raw - np.asarray(rlo), np.asarray(rhi) - raw],
                fmt="s--", color="#b0b0b0", lw=2, ms=7, capsize=4,
                label="원자료 (교란 있음)")
    ax.errorbar(lv, wit, yerr=[wit - np.asarray(wlo), np.asarray(whi) - wit],
                fmt="o-", color="#c0392b", lw=2.4, ms=7, capsize=4,
                label="같은 병원 안에서 비교")
    ax.axhline(0, color="k", lw=1, ls=":")
    ax.set_xlabel("그 병원이 지금 안고 있는 환자 (명)")
    ax.set_ylabel("선호도 log2 (0 = 그 병원의 평균)")
    ax.set_title("(2) 부하 효과는 교란을 벗겨야 보인다", fontsize=11, weight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.text(0.97, 0.95, "부하는 과거 선택의 결과다\n(좋은 병원이 먼저 찬다)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#555")

    # (c) 드러난 교환율
    ax = fig.add_subplot(1, 3, 3)
    rng = np.random.default_rng(0)
    dk, dl = S["dkm"], S["dld"]
    m = np.isfinite(dk) & np.isfinite(dl) & (dl > 0)
    ax.scatter(dl[m] + rng.normal(0, 0.08, int(m.sum())), dk[m], s=6, alpha=0.16,
               color="#2b6cb0")
    bins = np.arange(1, 8)
    med = [np.median(dk[m & (dl == v)]) if (m & (dl == v)).sum() >= 20 else np.nan
           for v in bins]
    ax.plot(bins, med, "o-", color="#c0392b", lw=2, ms=6, label="구간 중위")
    ax.plot(bins, res["revealed_lambda"] * bins, ls="--", color="k", lw=1.6,
            label=f"기울기 {res['revealed_lambda']:.1f} km/명")
    ax.set_xlabel("덜어낸 부하 (명)"); ax.set_ylabel("더 간 거리 (km)")
    ax.set_ylim(-2, 60)
    ax.set_title("(3) 부하 1명을 피하려 얼마를 더 가나", fontsize=11, weight="bold")
    ax.legend(fontsize=9)
    ax.text(0.97, 0.05, f"최근접을 버린 결정 {res['not_nearest_pct']:.1f}%\n"
                        f"그중 부하가 더 낮은 곳 {res['trade_ok_pct']:.1f}%",
            transform=ax.transAxes, ha="right", fontsize=9)

    fig.suptitle("PPO 교사 의사결정 해부 (2) — 목적지 선택은 거리와 부하의 교환이다",
                 fontsize=13.5, weight="bold")
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"anatomy_surface.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 그림 3
def fig3(d, A, res):
    fig = plt.figure(figsize=(15.5, 4.8))

    # (a) 수단 축 — 자유선택 구간에서 UAV 확률 vs 최근접 tier3 도로거리
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(res["uav_curve_x"], res["uav_curve_y"], "o-", color=C_UAV, lw=2, ms=5,
            label="UAV 선택률 (수단 자유)")
    ax.plot(res["uav_all_x"], res["uav_all_y"], "s--", color=C_GRAY, lw=1.5, ms=4,
            label="UAV 선택률 (전체 결정)")
    ax.axvline(12, color=C_R, ls=":", lw=2)
    ax.text(12.4, 0.06, "12 km", color=C_R, fontsize=10, weight="bold")
    ax.set_xlabel("최근접 Tier3 도로거리 (km)"); ax.set_ylabel("UAV 선택 비율")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(1) 수단 — 임계값은 자유선택에서만 보인다", fontsize=11, weight="bold")
    ax.legend(fontsize=8.5, loc="lower right")

    # (b) 등급 축
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(res["red_curve_x"], res["red_curve_y"], "o-", color=C_R, lw=2, ms=5)
    ax.axvline(14, color="#333", ls=":", lw=2)
    ax.text(14.5, 0.8, "14명", fontsize=10, weight="bold")
    ax.set_xlabel("현장 Yellow 대기 인원 (명)"); ax.set_ylabel("Red 선택 비율")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(2) 등급 — Yellow가 밀리면 Red를 접는다", fontsize=11, weight="bold")

    # (c) 2항 점수 일치도
    ax = fig.add_subplot(1, 3, 3)
    ag = res["agreement"]
    names = ["무작위\n(1/후보수)", "최근접만\n(가중 0)", "가중 6.37\n(모방 추정)", "가중 12\n(성능 최적)"]
    vals = [res["random_baseline"], ag["lam_0"], ag["lam_6.37"], ag["lam_12"]]
    cols = ["#cfcfcf", C_GRAY, "#2b6cb0", "#1f9d76"]
    b = ax.bar(names, vals, color=cols, width=0.62)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v*100:.1f}%",
                ha="center", fontsize=10.5, weight="bold")
    ax.set_ylabel("PPO 목적지와 일치")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("(3) 두 항이면 목적지를 얼마나 설명하나", fontsize=11, weight="bold")

    fig.suptitle("PPO 교사 의사결정 해부 (3) — 축마다 다른 변수를 쓴다",
                 fontsize=13.5, weight="bold")
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"anatomy_axes.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 그림 4
def fig4(CL, UC):
    fig = plt.figure(figsize=(15.5, 4.9))
    names = [r["cell"] for r in CL]
    cols = [C_R, "#7b1f6e", C_AMB, C_UAV]

    ax = fig.add_subplot(1, 3, 1)
    v = [r["lam_km"] for r in CL]
    b = ax.bar(names, v, color=cols, width=0.6)
    for r, x in zip(b, v):
        ax.text(r.get_x() + r.get_width() / 2, x + 0.5, f"{x:.1f}", ha="center",
                fontsize=10.5, weight="bold")
    ax.set_ylabel("교환율 (km / 환자 1명)")
    ax.set_title(f"(1) 거리 단위로 보면 {max(v)/min(v):.1f}배 흩어진다",
                 fontsize=11, weight="bold")
    ax.set_ylim(0, max(v) * 1.22)

    ax = fig.add_subplot(1, 3, 2)
    v = [r["lam_min"] for r in CL]
    lo = [r["lam_min"] - r["lam_min_ci"][0] for r in CL]
    hi = [r["lam_min_ci"][1] - r["lam_min"] for r in CL]
    ax.bar(names, v, color=cols, width=0.6, yerr=[lo, hi], capsize=5)
    for i, x in enumerate(v):
        ax.text(i, x + 0.55, f"{x:.1f}", ha="center", fontsize=10.5, weight="bold")
    ax.set_ylabel("교환율 (분 / 환자 1명)")
    ax.set_title(f"(2) 시간 단위로 바꾸면 {max(v)/min(v):.1f}배로 좁아진다",
                 fontsize=11, weight="bold")
    ax.set_ylim(0, max(v) * 1.35)
    ax.axhspan(min(v[1:]), max(v[1:]), color="#ffe08a", alpha=0.35, zorder=0)

    ax = fig.add_subplot(1, 3, 3)
    for unit, c, lab in (("km", "#2b6cb0", "거리(km)에 가중"),
                         ("min", "#1f9d76", "시간(분)에 가중")):
        g = UC[unit]["grid"]
        xs = sorted(g)
        ax.plot([x / max(xs) for x in xs], [g[x] * 100 for x in xs], "o-",
                color=c, lw=2, ms=5, label=lab)
    ax.axhline(UC["km"]["grid"][0.0] * 100, color="k", ls=":", lw=1.4)
    ax.text(0.02, UC["km"]["grid"][0.0] * 100 + 0.12, "최근접만 (가중 0)", fontsize=9)
    ax.set_xlabel("가중치 (각 단위 격자에서의 상대 위치)")
    ax.set_ylabel("PPO 목적지와 일치 (%)")
    ax.set_title("(3) 어느 단위로 써도 일치율 천장은 25%", fontsize=11, weight="bold")
    ax.legend(fontsize=9, loc="lower right")

    fig.suptitle("PPO 교사 의사결정 해부 (4) — 교환율의 진짜 단위는 거리가 아니라 시간이다",
                 fontsize=13.5, weight="bold")
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"anatomy_exchange.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 곡선 계산
def curves(d, A):
    off = d["off"]
    n = d["n_state"]
    si, tier_t, d_road_t, Hs = d["si"], d["tier_t"], d["d_road_t"], d["Hs"]
    near_t3 = np.asarray([
        float(d_road_t[i, :Hs[i]][tier_t[i, :Hs[i]] == 3].min())
        if (tier_t[i, :Hs[i]] == 3).any() else np.nan for i in si])
    ys = np.asarray([d["X"][int(off[s]), d["col"]["yellow_at_site"]] for s in range(n)])

    def rate(x, y, edges):
        cx, cy = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (x >= a) & (x < b) & np.isfinite(x)
            if m.sum() >= 30:
                cx.append((a + b) / 2); cy.append(float(y[m].mean()))
        return cx, cy

    ed = np.array([0, 5, 10, 15, 20, 30, 40, 60, 200.0])
    is_uav = (A["t_mode"] == 1).astype(float)
    fx, fy = rate(near_t3[A["free_mode"]], is_uav[A["free_mode"]], ed)
    ax_, ay_ = rate(near_t3, is_uav, ed)

    edy = np.array([0, 2, 5, 8, 11, 14, 17, 21, 40.0])
    is_red = (A["t_cls"] == 0).astype(float)
    fc = A["free_class"]
    rx, ry = rate(ys[fc], is_red[fc], edy)
    return dict(uav_curve_x=fx, uav_curve_y=fy, uav_all_x=ax_, uav_all_y=ay_,
                red_curve_x=rx, red_curve_y=ry, near_t3=near_t3, yellow_at_site=ys)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/5] 자료 조립", flush=True)
    d = build()
    print("[2/5] 제약 해부", flush=True)
    A = anatomy(d)
    print("[3/5] 선호 곡면", flush=True)
    S = surface(d, A)
    print("[4/5] 일치도 · 곡선", flush=True)
    ag = score_agreement(d, A)
    CL = cells(d, A)
    UC = unit_contest(d, A)
    cv = curves(d, A)

    dk, dl = S["dkm"], S["dld"]
    m = np.isfinite(dk) & np.isfinite(dl) & (dl > 0)
    lam = float(np.median(dk[m] / dl[m]))
    not_near = float(np.mean(np.isfinite(dk) & (dk > 1e-9))) * 100
    trade_ok = float(np.mean(dl[np.isfinite(dk) & (dk > 1e-9)] > 0)) * 100
    stay_argmax = float(np.mean(A["t_dest"] == 0)) * 100
    rnd = float(np.mean(1.0 / np.maximum(A["n_own"], 1)))

    res = dict(
        n_state=int(d["n_state"]), n_cand=int(len(d["km"])),
        median_H=float(np.median(A["H"])),
        cond_median_elig=A["cond_med"], median_elig=dict(Y_amb=float(np.median(A["elig"][:, 1, 0])),
                         Y_uav=float(np.median(A["elig"][:, 1, 1])),
                         R_amb=float(np.median(A["elig"][:, 0, 0])),
                         R_uav=float(np.median(A["elig"][:, 0, 1]))),
        zero_elig_pct=dict(Y_amb=float((A["elig"][:, 1, 0] == 0).mean()) * 100,
                           Y_uav=float((A["elig"][:, 1, 1] == 0).mean()) * 100,
                           R_amb=float((A["elig"][:, 0, 0] == 0).mean()) * 100,
                           R_uav=float((A["elig"][:, 0, 1] == 0).mean()) * 100),
        median_n_cand=float(np.median(A["n_cand"])),
        median_perplexity=float(np.median(A["perplex"])),
        mean_perplexity=float(A["perplex"].mean()),
        median_gap12=float(np.median(A["gap12"])),
        indifferent_pct=float((A["gap12"] < 0.1).mean()) * 100,
        free_mode_pct=float(A["free_mode"].mean()) * 100,
        free_class_pct=float(A["free_class"].mean()) * 100,
        single_dest_pct=float((A["n_own"] <= 1).mean()) * 100,
        median_own_dest=float(np.median(A["n_own"])),
        stay_prob_median=float(np.median(A["stay_p"])),
        stay_prob_mean=float(A["stay_p"].mean()),
        stay_prob_gt1pct=float((A["stay_p"] > 0.01).mean()) * 100,
        stay_argmax_pct=stay_argmax,
        revealed_lambda=lam,
        not_nearest_pct=not_near, trade_ok_pct=trade_ok,
        agreement=ag, random_baseline=rnd, cells=CL, unit_contest=UC,
        **{k: v for k, v in cv.items() if k.endswith(("_x", "_y"))},
    )

    print("[5/5] 그림", flush=True)
    fig1(A, res)
    fig2(S, res, d["keys"])
    fig3(d, A, res)
    fig4(CL, UC)

    def _j(v):
        if isinstance(v, np.ndarray):
            return [float(x) for x in v]
        if isinstance(v, (list, tuple)):
            return [_j(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _j(x) for k, x in v.items()}
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        return v

    ser = {k: _j(v) for k, v in res.items()}
    (OUT / "anatomy.json").write_text(json.dumps(ser, ensure_ascii=False, indent=2))
    for k in ("median_H", "median_elig", "zero_elig_pct", "median_n_cand",
              "median_perplexity", "median_gap12", "indifferent_pct",
              "free_mode_pct", "free_class_pct", "single_dest_pct", "median_own_dest",
              "stay_prob_median", "stay_prob_mean", "stay_prob_gt1pct",
              "stay_argmax_pct", "revealed_lambda", "not_nearest_pct",
              "trade_ok_pct", "agreement", "random_baseline"):
        print(f"  {k} = {res[k]}", flush=True)
    print("  --- 셀별 ---", flush=True)
    for r in CL:
        print(f"  {r['cell']:11s} n={r['n']:6d} ({r['share']:4.1f}%) 후보중위 {r['med_cand']:3.0f} "
              f"최근접 {r['nearest_pct']:5.1f}% 무차별 {r['indiff_pct']:5.1f}% "
              f"교환율 {r['lam_km']:6.2f} km/명 = {r['lam_min']:5.2f} 분/명 "
              f"CI[{r['lam_min_ci'][0]:.2f},{r['lam_min_ci'][1]:.2f}]", flush=True)
    print(f"  단위대결 km 최적 {UC['km']['best_lam']} → {UC['km']['best_acc']*100:.1f}% / "
          f"분 최적 {UC['min']['best_lam']} → {UC['min']['best_acc']*100:.1f}%", flush=True)
    print(f"→ {OUT}", flush=True)


if __name__ == "__main__":
    main()
