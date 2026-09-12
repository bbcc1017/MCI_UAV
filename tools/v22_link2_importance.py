# -*- coding: utf-8 -*-
"""v22 링크 2 재적합 — 현행 교사(v19 field PPO) 결정로그의 변수 중요도.

**링크 2 의 역할은 "카드에 넣을 변수 집합을 고르는 것"이고, 숫자는 폐루프가 정한다.**
이 도구는 그 문장을 데이터로 뒷받침하기 위한 것이며, 교사 행동 재현율을 폐루프 성능의
근거로 쓰지 않는다. 그 오류의 실측 반례가 이 저장소에 있다 — v20 에서 교사 top-1 재현율
순서(K 0.3146 > 최근접 0.2372 > T 0.2033 > H 0.1959 > Q 0.1792)가 폐루프 PDR 순서와
정확히 반대였고, 채택 카드(Q)가 재현율 최하위였다. `모방 ≠ 성능` 은 6회 재확인된 결론이다.

측정하는 것
  1. 변수 중요도 — (a) McFadden 조건부 로짓 표준화 계수, (b) 후보랭킹 GBDT gain 중요도.
  2. 중요도 ≠ 정보이득 — 같은 표에 **추가 설명력**(leave-one-out ΔpseudoR²·Δ재현율,
     전진선택 증분, GBDT 순열 Δ)을 나란히 낸다. v17 에서 신규 특징이 중요도 질량 51.7%를
     먹고도 재현율은 +0.004 였다.
  3. 부분상관 — 원거리성(ETA)을 통제하기 전/후. 통제 없이는 인프라 특징이 전부 "중요"해 보인다.
  4. 정보수준 절단 — 병원 census(통신 필요)를 뺀 팔과 자기 발송기록을 뺀 팔을 **둘 다** 잰다.
     v21 정본의 비대칭(통신 끊기 +0.00076 vs 내 발송 안 세기 +0.0929)이 결정로그에도 보이는가.
  5. 17 시도 로그로 중요도 순위의 지역 간 안정성(Kendall τ).

통계 커널은 재구현하지 않는다.
  * `src/rl_src/v17_field_rules.py` : `_cond_logit`(조건부 로짓 MLE) · `_lambda_fit`(λ 계수비)
    · `_topk_acc`(top-1 재현율) · `cluster_boot_index`(시군구 클러스터 부트스트랩) · `decode`
  * `tools/v20_threshold_report.py` : `ci`(1.96·sd/√n) · `paired`/`wtl`(지역별 CI 승무패)
    — 정보수준 팔 비교는 `(지역 × 상태)` 직사각 NLL 행렬에 이 커널을 그대로 적용한다.
    ⚠️ 판정선 0.00053 은 PDR 단위의 CRN paired 실측치다. 여기 수치는 **로그우도·재현율 단위**라
    그 선을 적용할 수 없다 — 팔 비교는 전부 자체 paired 95%CI 로만 판정한다.

시뮬 재실행 0회. 입력은 이미 완주된 결정로그뿐이다.

사용:
    python tools/v22_link2_importance.py sample --m 160
    python tools/v22_link2_importance.py fit
    python tools/v22_link2_importance.py sido --m 160
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import numpy.lib.format as _nfmt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "rl_src"))
sys.path.insert(0, str(REPO / "tools"))

from v17_field_rules import (  # noqa: E402  통계 커널 재사용 (수정 금지 파일)
    H_PAD,
    LOGIT_FEATURES_CARD,
    _cond_logit,
    _lambda_fit,
    _topk_acc,
    cluster_boot_index,
    decode,
)
from v20_threshold_report import ci, paired, wtl  # noqa: E402

CARDS = REPO / "results/scoreboard/v19/cards"
DEC_NATIONAL = CARDS / "dec_national.npz"
STATIC = REPO / "results/scoreboard/v18/static_sigungu30.npz"
TRAIN_MANIFEST = REPO / "scenarios/manifests/sigungu30_train6000_manifest.json"
OUTDIR = REPO / "results/field/link2"
CACHE = Path(os.environ.get(
    "V22_LINK2_CACHE",
    "/tmp/claude-1002/-home-ryu-MCI-UAV/50486f3e-5bdb-4d88-ae63-316418000be8/scratchpad/link2_cache"))

SEED = 20260911

# ------------------------------------------------------------------ 특징 정의
# 파생 특징: 카드의 함수형과 정보수준 사다리를 결정로그 위에서 재현하기 위한 것.
#   max_capa(수술실수) 는 43 열 스키마에 없다 — X 의 max_send 는 `수술실수 + 병상수`
#   (ScenarioManager: max_send_coeff=[1,1]) 라서 카드의 서버수와 다르다. 따라서 좌표별
#   hospital_info.csv 의 '수술실수' 를 병원 인덱스 순서로 읽어 붙인다(정합성은 sample 에서 검사).
DERIVED = [
    "max_capa",          # 수술실수 = 카드 hinge 의 서버수. 병원 명부(무통신).
    "load",              # occ + in_flight = 카드 채택 부하항.
    "hinge1_capa",       # max(load + 1 − 수술실수, 0) = 부하 초과분.
    "hingerate_capa",    # 초과분 / 수술실수 = v22 Q족 `hingerate`.
    "hinge1_send",       # max(load + 1 − 발송상한, 0) = 발송 게이트 기준 초과분.
    "psent_hinge1",      # max(내가 보낸 누적 + 1 − 수술실수, 0) = 무통신 부하 대리값(I1).
    "red_x_ysite",       # is_red × 현장 Yellow 수 = yhold 규칙의 상호작용.
    "red_x_rsite",       # is_red × 현장 Red 수.
    "uav_x_ambavail",    # is_uav × 대기 AMB 수 = 수단 게이트.
    "uav_x_uavavail",    # is_uav × 대기 UAV 수.
    "uav_x_fleetcrit",   # is_uav × fleet_critical.
    "eta_x_rho",         # 도달시간(분) × rho = 혼잡 시 거리 가중 변화.
    "load_x_rho",        # 부하 × rho.
]

# 사전등록 — 카드가 실제로 쓰는 변수 집합(도달시간·부하 초과분·수술실수·등급·수단 가용).
CARD_SET = ["eta_raw_min", "hingerate_capa", "max_capa", "is_red", "is_uav",
            "uav_x_ambavail", "red_x_ysite"]

# 정보수준 블록. v21 정본의 비대칭을 재게 하려면 OWN 과 CENSUS 를 반드시 분리해야 한다.
BLK_CENSUS = ["cand_cap_remain", "cand_occ", "cand_occ_ratio", "rho",
              "red_at_hospital", "yellow_at_hospital", "red_done", "yellow_done",
              "total_cap_remain"]                       # 병원 전산 → 통신 필요
BLK_OWN = ["cand_p_sent", "cand_p_sent_rel", "cand_in_flight",
           "total_p_sent", "total_in_flight", "psent_hinge1"]   # 현장 지휘소 자기 기록
BLK_MIXED = ["load", "hinge1_capa", "hingerate_capa", "hinge1_send",
             "load_x_rho", "eta_x_rho"]                 # census 와 자기기록을 함께 쓰는 항
BLK_TELEMETRY = ["amb_busy", "amb_min_return", "amb_mean_return",
                 "uav_busy", "uav_min_return", "uav_mean_return",
                 "fleet_critical", "cand_arrive_min", "uav_x_fleetcrit"]
BLK_UNRESCUED = ["red_unrescued", "yellow_unrescued", "red_in_transport",
                 "yellow_in_transport"]

ETA_CONTROLS = ["eta_raw_min", "eta_norm", "eta_rank"]


# --------------------------------------------------------------- npz 스트리밍
def _stream(npz_path: Path, member: str, keep: np.ndarray | None, chunk: int = 400_000):
    """압축 npz 멤버를 순차 스트리밍하며 keep(불리언 행 마스크)만 모은다.

    전량 dense 적재(X 는 1.9GB)를 피하려고 쓴다. 한 번 통과가 약 4초다.
    """
    out = []
    with zipfile.ZipFile(npz_path) as z, z.open(member) as f:
        ver = _nfmt.read_magic(f)
        sh, forder, dt = _nfmt._read_array_header(f, ver)
        assert not forder, f"{member}: fortran order 미지원"
        ncol = int(np.prod(sh[1:])) if len(sh) > 1 else 1
        rb = ncol * dt.itemsize
        n = 0
        while n < sh[0]:
            k = min(chunk, sh[0] - n)
            buf = f.read(k * rb)
            if len(buf) != k * rb:
                raise IOError(f"{member}: 불완전 읽기 {len(buf)} != {k*rb}")
            a = np.frombuffer(buf, dtype=dt)
            a = a.reshape((k,) + sh[1:]) if len(sh) > 1 else a
            out.append(a if keep is None else a[keep[n:n + k]])
            n += k
    return np.concatenate(out) if out else np.zeros((0,) + sh[1:], dt)


def _lazy(npz_path: Path, *members):
    z = np.load(npz_path, allow_pickle=False)
    return [z[m] for m in members]


# ------------------------------------------------------------------ 표본 추출
def _max_capa_table(keys: list[str], manifest: dict, workers: int = 24):
    """좌표 key → 수술실수 배열. hospital_info.csv 행 순서 = 병원 인덱스."""
    from multiprocessing import Pool
    jobs = [(k, manifest[k]) for k in keys]
    with Pool(min(workers, max(1, len(jobs)))) as pool:
        res = pool.map(_capa_worker, jobs, chunksize=16)
    return {k: v for k, v in res}


def _capa_worker(job):
    import pandas as pd
    k, cfg = job
    d = Path(cfg).parent
    hi = pd.read_csv(d / "hospital_info.csv")
    return k, (np.asarray(hi["수술실수"], np.float32),
               np.asarray(hi["병상수"], np.float32))


def cmd_sample(args) -> None:
    t0 = time.time()
    dec = Path(args.decisions)
    meta = json.load(open(str(dec) + ".meta.json", encoding="utf-8"))
    sk, off, nc, teach, chosen_all_n = _lazy(dec, "state_key", "offsets", "ncand",
                                             "teacher_action", "ncand")
    del chosen_all_n
    feat_names = [str(x) for x in _lazy(dec, "feature_names")[0]]
    sk = np.asarray([str(x) for x in sk])
    n_states = len(sk)
    sig = np.asarray([k.rsplit("_", 2)[-2] for k in sk])

    # 균형 표본: 시군구별로 m 개 상태를 비복원 무작위 추출한다.
    # 무작위 순열 순서로 훑어 '보류(dest=0) 아님 & 배차후보 ≥ 2' 인 상태부터 m 개를 채워
    # (지역 × 상태) 직사각 구조를 유지한다 — paired/wtl 커널이 직사각 행렬을 요구한다.
    dec_td = {int(a): decode(int(a))[1] for a in np.unique(teach)}
    td = np.asarray([dec_td[int(a)] for a in teach])
    rng = np.random.default_rng(args.seed)
    codes = np.unique(sig)
    picked, n_stay = [], 0
    for c in codes:
        idx = np.flatnonzero(sig == c)
        rng.shuffle(idx)
        got = []
        for s in idx:
            if td[s] == 0:
                n_stay += 1
                continue
            got.append(int(s))
            if len(got) >= args.m:
                break
        if len(got) < args.m:
            raise RuntimeError(f"{c}: 유효 상태 {len(got)} < m={args.m}")
        picked.append(np.sort(np.asarray(got, int)))
    states = np.concatenate(picked)
    dist_of = np.repeat(np.arange(len(codes)), args.m)
    order = np.argsort(states, kind="stable")
    states_sorted = states[order]
    dist_sorted = dist_of[order]

    # 선택 상태의 후보 행만 스트리밍으로 회수
    keep = np.zeros(int(off[-1]), bool)
    for s in states_sorted:
        keep[int(off[s]):int(off[s + 1])] = True
    X = _stream(dec, "X.npy", keep).astype(np.float64)
    target = _stream(dec, "target.npy", keep).astype(np.float64)
    chosen = _stream(dec, "chosen.npy", keep)
    cact = _stream(dec, "cand_action.npy", keep)
    print(f"[sample] 상태 {len(states)} · 회수행 {len(X)} ({time.time()-t0:.0f}s)", flush=True)

    # 상태별 블록 경계 (스트리밍은 원본 행 순서 = 상태 오름차순)
    sizes = np.asarray([int(off[s + 1]) - int(off[s]) for s in states_sorted], int)
    bnd = np.concatenate([[0], np.cumsum(sizes)])
    row_state = np.repeat(np.arange(len(states_sorted)), sizes)

    # 배차 후보(dest>0)만 남긴다 — 보류행은 43 열이 전부 0 인 퇴화행이라
    # 넣으면 is_stay 와 '낮은 ETA' 사이에 인공 상관이 생긴다. 보류 채택은 v17 실측 2.6e-5.
    dmap = {int(a): decode(int(a)) for a in np.unique(cact)}
    cls = np.asarray([dmap[int(a)][0] for a in cact], np.int8)
    dst = np.asarray([dmap[int(a)][1] for a in cact], np.int16)
    mod = np.asarray([dmap[int(a)][2] for a in cact], np.int8)
    disp = dst > 0
    n_stay_rows = int((~disp).sum())

    # 정적표(거리)·수술실수 결합
    st = np.load(STATIC, allow_pickle=False)
    st_idx = {str(k): i for i, k in enumerate(st["keys"])}
    manifest = json.load(open(TRAIN_MANIFEST, encoding="utf-8"))
    used_keys = sorted(set(sk[states_sorted]))
    capa = _max_capa_table(used_keys, manifest, args.workers)
    print(f"[sample] 좌표 {len(used_keys)} 의 수술실수 로드 ({time.time()-t0:.0f}s)", flush=True)

    nrow = len(X)
    d_km = np.zeros(nrow)          # AMB=도로거리, UAV=직선거리 (카드 규약)
    mcapa = np.zeros(nrow)
    bad_send = 0
    for si, s in enumerate(states_sorted):
        a, b = int(bnd[si]), int(bnd[si + 1])
        key = str(sk[s])
        ii = st_idx[key]
        Hn = int(st["H"][ii])
        dr, de = st["d_road"][ii, :Hn], st["d_euc"][ii, :Hn]
        cap, que = capa[key]
        h = dst[a:b] - 1
        ok = h >= 0
        hh = h[ok]
        d_km[a:b][ok] = np.where(mod[a:b][ok] == 1, de[hh], dr[hh])
        mcapa[a:b][ok] = cap[hh]
        # 정합성: X 의 max_send == 수술실수 + 병상수 (스키마 검증)
        ms = X[a:b, feat_names.index("max_send")][ok]
        bad_send += int(np.sum(np.abs(ms - (cap[hh] + que[hh])) > 1e-3))
    if bad_send:
        raise RuntimeError(f"max_send != 수술실수+병상수 인 행 {bad_send}개 — 스키마 불일치")

    # 파생 특징
    def col(n):
        return X[:, feat_names.index(n)]

    load = col("cand_occ") + col("cand_in_flight")
    capa_pos = np.maximum(mcapa, 1.0)
    send_pos = np.maximum(col("max_send"), 1.0)
    hin_capa = np.maximum(load + 1.0 - capa_pos, 0.0)
    D = {
        "max_capa": mcapa,
        "load": load,
        "hinge1_capa": hin_capa,
        "hingerate_capa": hin_capa / capa_pos,
        "hinge1_send": np.maximum(load + 1.0 - send_pos, 0.0),
        "psent_hinge1": np.maximum(col("cand_p_sent") + 1.0 - capa_pos, 0.0),
        "red_x_ysite": col("is_red") * col("yellow_at_site"),
        "red_x_rsite": col("is_red") * col("red_at_site"),
        "uav_x_ambavail": col("is_uav") * col("amb_available"),
        "uav_x_uavavail": col("is_uav") * col("uav_available"),
        "uav_x_fleetcrit": col("is_uav") * col("fleet_critical"),
        "eta_x_rho": col("eta_raw_min") * col("rho"),
        "load_x_rho": load * col("rho"),
    }
    names = feat_names + DERIVED
    Xa = np.column_stack([X] + [D[k] for k in DERIVED])

    # 배차 후보만 남기고 그룹 재구성 + 교사 soft 라벨 재정규화
    Xa, target, chosen = Xa[disp], target[disp], chosen[disp]
    d_km, cls, mod, row_state = d_km[disp], cls[disp], mod[disp], row_state[disp]
    gsz = np.bincount(row_state, minlength=len(states_sorted))
    if gsz.min() < 2:
        raise RuntimeError("배차 후보 2개 미만 상태 존재 — 표본 규약 위반")
    g_off = np.concatenate([[0], np.cumsum(gsz)])
    gid = np.repeat(np.arange(len(gsz)), gsz)
    tsum = np.bincount(gid, weights=target, minlength=len(gsz))
    if not np.all(np.bincount(gid, weights=chosen.astype(float), minlength=len(gsz)) == 1):
        raise RuntimeError("상태별 chosen 이 1 이 아님")
    target = target / tsum[gid]

    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "X.npy", Xa.astype(np.float32))
    np.save(CACHE / "target.npy", target.astype(np.float32))
    np.save(CACHE / "chosen.npy", chosen)
    np.save(CACHE / "gsz.npy", gsz.astype(np.int32))
    np.save(CACHE / "d_km.npy", d_km.astype(np.float32))
    np.save(CACHE / "cls.npy", cls)
    np.save(CACHE / "mode.npy", mod)
    np.save(CACHE / "dist_code.npy", dist_sorted.astype(np.int32))
    np.save(CACHE / "state_key.npy", sk[states_sorted])
    info = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "decisions": str(dec), "decisions_sha256": meta.get("output_sha256"),
        "model_dir": meta.get("model_dir"), "model_sha256": meta.get("model_sha256"),
        "manifest": meta.get("manifest"), "manifest_sha256": meta.get("manifest_sha256"),
        "static": str(STATIC),
        "sampling": {
            "protocol": ("시군구별 균형 비복원 추출 — 각 시군구 888 상태의 무작위 순열에서 "
                         "보류(dest=0) 아닌 상태를 앞에서부터 m 개. rng=default_rng(seed)."),
            "seed": args.seed, "m_per_district": args.m,
            "n_districts": int(len(codes)), "n_states_sampled": int(len(states_sorted)),
            "n_states_population": int(n_states),
            "frac_states": float(len(states_sorted) / n_states),
            "n_stay_states_skipped": int(n_stay),
            "n_candidate_rows": int(len(Xa)),
            "n_stay_rows_dropped": int(n_stay_rows),
            "mean_cand_per_state": float(len(Xa) / len(gsz)),
            "teacher_soft_mass_retained": float(np.mean(tsum)),
        },
        "features": names, "n_features": len(names),
        "derived_note": "max_capa=hospital_info.csv 수술실수; max_send=수술실수+병상수 검증 통과",
        "wall_sec": round(time.time() - t0, 1),
    }
    (CACHE / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(json.dumps(info["sampling"], ensure_ascii=False, indent=2))
    print(f"[sample] 캐시 {CACHE}  wall={(time.time()-t0)/60:.1f}분", flush=True)


# ---------------------------------------------------------------- 적합·지표
def load_cache(path: Path = CACHE):
    info = json.load(open(path / "info.json", encoding="utf-8"))
    d = {k: np.load(path / f"{k}.npy", allow_pickle=False)
         for k in ("X", "target", "chosen", "gsz", "d_km", "cls", "mode", "dist_code")}
    d["names"] = info["features"]
    d["info"] = info
    return d


def _group_stats(v, gsz, chosen, target=None):
    """그룹 softmax 지표. v=점수. 반환 per-set 로그우도·적중·CE."""
    offs = np.concatenate([[0], np.cumsum(gsz)])
    gid = np.repeat(np.arange(len(gsz)), gsz)
    mx = np.maximum.reduceat(v, offs[:-1])
    e = np.exp(v - mx[gid])
    den = np.add.reduceat(e, offs[:-1])
    logZ = mx + np.log(den)
    ll = v[chosen] - logZ
    # 적중: 그룹별 argmax
    best = np.zeros(len(gsz), int)
    for k in range(len(gsz)):
        a, b = int(offs[k]), int(offs[k + 1])
        best[k] = a + int(np.argmax(v[a:b]))
    hit = chosen[best].astype(float)
    out = {"ll": ll, "hit": hit, "ll0": -np.log(gsz.astype(float))}
    if target is not None:
        logp = v - logZ[gid]
        out["ce"] = -np.add.reduceat(target * logp, offs[:-1])
        tl = np.where(target > 0, target * np.log(np.maximum(target, 1e-300)), 0.0)
        out["ent"] = -np.add.reduceat(tl, offs[:-1])
    return out


def _metrics(v, gsz, chosen, target=None):
    s = _group_stats(v, gsz, chosen, target)
    m = {"recall": float(s["hit"].mean()),
         "ll_per_set": float(s["ll"].mean()),
         "pseudo_r2": float(1.0 - s["ll"].sum() / s["ll0"].sum()),
         "chance": float(np.mean(1.0 / gsz))}
    if target is not None:
        m["ce_per_set"] = float(s["ce"].mean())
        m["kl_per_set"] = float((s["ce"] - s["ent"]).mean())
    return m


def _prune(X, gsz, names, tol=1e-9, rmax=0.99995):
    """그룹 내 분산 0(=조건부 로짓에서 식별 불가) 및 완전 공선 열을 제거한다.

    조건부 로짓은 선택집합 안에서 상수인 모든 변수를 정확히 차분해 없앤다. 이는
    좌표 수준 원거리성·현장 맥락 같은 상태변수가 **주효과로는 식별되지 않는다**는 뜻이고,
    그래서 상호작용 항이 필요하다(DERIVED 의 red_x_·uav_x_·*_rho).
    """
    offs = np.concatenate([[0], np.cumsum(gsz)])
    gid = np.repeat(np.arange(len(gsz)), gsz)
    cnt = gsz.astype(float)
    Z = np.empty_like(X)
    for j in range(X.shape[1]):
        mu = np.add.reduceat(X[:, j], offs[:-1]) / cnt
        Z[:, j] = X[:, j] - mu[gid]
    wv = (Z ** 2).sum(0)
    scale = np.maximum(np.abs(X).max(0), 1e-12) ** 2 * len(X)
    keep = wv / scale > tol
    dropped_const = [names[j] for j in range(len(names)) if not keep[j]]
    idx = [j for j in range(len(names)) if keep[j]]
    # 완전 공선 제거(선행 열 우선)
    sd = Z[:, idx].std(0)
    sd[sd < 1e-12] = 1.0
    Zn = Z[:, idx] / sd
    sel, dropped_coll = [], []
    for p, j in enumerate(idx):
        if sel:
            r = np.abs(Zn[:, p] @ Zn[:, [idx.index(q) for q in sel]]) / len(Zn)
            if np.max(r) > rmax:
                dropped_coll.append(names[j])
                continue
        sel.append(j)
    return (np.asarray(sel, int), Z, dropped_const, dropped_coll)


_G: dict = {}


def _fit(cols):
    """cols(열 인덱스) 로 조건부 로짓을 적합하고 지표를 낸다."""
    X, gsz, chosen, target = _G["X"], _G["gsz"], _G["chosen"], _G["target"]
    cols = np.asarray(cols, int)
    Xs = np.ascontiguousarray(X[:, cols])
    beta, fun, _ = _cond_logit(Xs, chosen.astype(np.int8), gsz)
    v = Xs @ beta
    m = _metrics(v, gsz, chosen, target)
    m["beta"] = beta
    m["cols"] = cols
    return m


def _fit_job(cols):
    m = _fit(cols)
    return [int(c) for c in m["cols"]], m["recall"], m["pseudo_r2"], m["ll_per_set"], \
        m.get("kl_per_set"), [float(b) for b in m["beta"]]


def _pool(workers):
    from multiprocessing import Pool
    return Pool(workers)


# ------------------------------------------------------------------- 스테이지
def stage_sanity(d, out):
    """canonical 재현 — mine_national_lambda.json 의 λ·top-1 을 표본에서 다시 낸다.

    프레이밍 B(교사의 등급·수단 고정, 목적지만 선택)와 카드 2 특징 [d_km, load] 로
    `_lambda_fit`·`_topk_acc` 를 그대로 호출한다. 표본 추출 규약이 정본과 어긋나지
    않았는지 확인하는 유일한 외부 기준점이다.
    """
    names, X, gsz, chosen = d["names"], d["X"], d["gsz"], d["chosen"]
    offs = np.concatenate([[0], np.cumsum(gsz)])
    gid = np.repeat(np.arange(len(gsz)), gsz)
    tcls = d["cls"][chosen][gid]
    tmod = d["mode"][chosen][gid]
    sel = (d["cls"] == tcls) & (d["mode"] == tmod)
    g2 = np.bincount(gid[sel], minlength=len(gsz))
    ok = g2 >= 2
    rowok = sel & ok[gid]
    g2 = g2[ok]
    load = X[:, names.index("cand_occ")] + X[:, names.index("cand_in_flight")]
    Xd = np.column_stack([d["d_km"][rowok], load[rowok]])
    yd = chosen[rowok].astype(np.int8)
    lam, beta = _lambda_fit(Xd, yd, g2, LOGIT_FEATURES_CARD)
    acc, chance = _topk_acc(Xd, yd, g2, beta)
    canon = json.load(open(CARDS / "mine_national_lambda.json", encoding="utf-8"))["lambda"]
    res = {
        "framing": "B_destination_only (교사 등급·수단 고정)",
        "features": LOGIT_FEATURES_CARD,
        "n_choice_sets": int(len(g2)),
        "lambda_km_per_patient": float(lam),
        "top1_recall": float(acc), "chance": float(chance),
        "canonical": {"lambda": canon["all"], "top1_acc": canon["top1_acc"],
                      "chance": canon["chance"], "n_choice_sets": canon["n_choice_sets"]},
        "note": ("정본은 222,000 선택집합 전수, 여기는 균형표본. "
                 "λ·top-1 이 정본과 같은 자리수로 재현되면 표본 규약이 정본과 정합한다."),
    }
    out["sanity"] = res
    print(f"[sanity] λ={lam:.4f} (정본 {canon['all']:.4f}) · top1={acc:.4f} "
          f"(정본 {canon['top1_acc']:.4f}) · 집합 {len(g2)}", flush=True)


def stage_importance(d, out, workers, boot):
    """프레이밍 A' 전체 적합 + 표준화 계수 + leave-one-out 정보이득."""
    names = d["names"]
    sel, Z, dc, dcl = _prune(d["X"], d["gsz"], names)
    keep_names = [names[j] for j in sel]
    _G.update(X=d["X"], gsz=d["gsz"], chosen=d["chosen"], target=d["target"])
    full = _fit(sel)
    sd_glob = d["X"][:, sel].std(0)
    sd_within = Z[:, sel].std(0)
    beta = full["beta"]
    rows = []
    for i, nm in enumerate(keep_names):
        rows.append({"feature": nm, "beta": float(beta[i]),
                     "beta_std_within": float(beta[i] * sd_within[i]),
                     "beta_std_global": float(beta[i] * sd_glob[i])})
    # leave-one-out: 중요도(계수) 와 추가 설명력(Δ) 을 같은 표에 나란히
    jobs = [np.asarray([c for k, c in enumerate(sel) if k != i], int)
            for i in range(len(sel))]
    with _pool(min(workers, len(jobs))) as pool:
        loo = pool.map(_fit_job, jobs, chunksize=1)
    for i, r in enumerate(loo):
        rows[i]["loo_d_pseudo_r2"] = float(full["pseudo_r2"] - r[2])
        rows[i]["loo_d_recall"] = float(full["recall"] - r[1])
        rows[i]["loo_d_kl"] = float(r[4] - full["kl_per_set"])
    rows.sort(key=lambda r: -abs(r["beta_std_within"]))
    for k, r in enumerate(rows, 1):
        r["rank_beta"] = k
    by_gain = sorted(rows, key=lambda r: -r["loo_d_pseudo_r2"])
    for k, r in enumerate(by_gain, 1):
        r["rank_gain"] = k

    # 카드 변수집합 vs 전체 vs 계수 상위 k
    card_cols = [names.index(n) for n in CARD_SET if n in keep_names]
    card = _fit(np.asarray(card_cols, int))
    topk = {}
    for k in (3, 5, 7, 10, 15):
        cols = [names.index(r["feature"]) for r in rows[:k]]
        m = _fit(np.asarray(cols, int))
        topk[k] = {"features": [r["feature"] for r in rows[:k]],
                   "recall": m["recall"], "pseudo_r2": m["pseudo_r2"],
                   "kl_per_set": m["kl_per_set"]}

    res = {
        "framing": "A'_dispatch_all (보류 제외 전 배차후보: 등급 × 수단 × 목적지 동시)",
        "n_choice_sets": int(len(d["gsz"])),
        "n_candidate_rows": int(len(d["X"])),
        "n_features_offered": len(names),
        "n_features_identified": int(len(sel)),
        "dropped_not_identified": dc,
        "dropped_collinear": dcl,
        "identification_note": ("조건부 로짓은 선택집합 내 상수를 정확히 차분한다 → "
                                "상태 수준 변수(현장 인원·차량·시각·rho 등)는 주효과로 "
                                "식별되지 않는다. 이는 좌표 원거리성에 대한 완전 통제이기도 하다."),
        "full_model": {k: full[k] for k in
                       ("recall", "pseudo_r2", "ll_per_set", "chance", "ce_per_set", "kl_per_set")},
        "card_set": {"features": [n for n in CARD_SET if n in keep_names],
                     "missing": [n for n in CARD_SET if n not in keep_names],
                     **{k: card[k] for k in ("recall", "pseudo_r2", "kl_per_set")}},
        "top_k_by_beta": topk,
        "features": rows,
    }
    if boot:
        rng = np.random.default_rng(SEED)
        idx_of = None
        codes = d["dist_code"]
        gcode = codes
        vals = []
        for _ in range(boot):
            pick, idx_of = cluster_boot_index(gcode, rng, idx_of)
            m = np.zeros(len(d["gsz"]), bool)
            m[np.unique(pick)] = True          # 중복 클러스터는 1회만(보수적) — v17 규약
            rm = np.repeat(m, d["gsz"])
            b, _, _ = _cond_logit(np.ascontiguousarray(d["X"][rm][:, sel]),
                                  d["chosen"][rm].astype(np.int8), d["gsz"][m])
            vals.append(b * sd_within)
        V = np.asarray(vals)
        lo, hi = np.percentile(V, [2.5, 97.5], axis=0)
        for i, r in enumerate(rows):
            j = keep_names.index(r["feature"])
            r["beta_std_within_ci95"] = [float(lo[j]), float(hi[j])]
        res["bootstrap"] = {"n": int(boot), "cluster": "sigungu(250)",
                            "rule": "중복 클러스터 1회 반영(v17 report_silent 규약)"}
    out["importance"] = res
    print(f"[importance] 식별 {len(sel)}/{len(names)} · 전체 재현율 {full['recall']:.4f} "
          f"pseudoR2 {full['pseudo_r2']:.4f} · 카드7변수 {card['recall']:.4f}/"
          f"{card['pseudo_r2']:.4f}", flush=True)
    return sel, keep_names, Z, full


def stage_forward(d, out, sel, workers, kmax):
    """전진선택 — 중요도가 아니라 '증분 설명력' 순서."""
    _G.update(X=d["X"], gsz=d["gsz"], chosen=d["chosen"], target=d["target"])
    names = d["names"]
    cur, steps = [], []
    remain = list(map(int, sel))
    prev = {"pseudo_r2": 0.0, "recall": float(np.mean(1.0 / d["gsz"])), "kl": None}
    for step in range(kmax):
        jobs = [np.asarray(cur + [c], int) for c in remain]
        with _pool(min(workers, len(jobs))) as pool:
            res = pool.map(_fit_job, jobs, chunksize=1)
        best = int(np.argmax([r[2] for r in res]))
        c = remain[best]
        r = res[best]
        steps.append({"step": step + 1, "feature": names[c],
                      "pseudo_r2": r[2], "d_pseudo_r2": r[2] - prev["pseudo_r2"],
                      "recall": r[1], "d_recall": r[1] - prev["recall"],
                      "kl_per_set": r[4],
                      "d_kl": (prev["kl"] - r[4]) if prev["kl"] is not None else None})
        prev = {"pseudo_r2": r[2], "recall": r[1], "kl": r[4]}
        cur.append(c)
        remain.pop(best)
        print(f"  [fwd {step+1}] {names[c]:18s} R2={r[2]:.4f} recall={r[1]:.4f}", flush=True)
    out["forward"] = {"kmax": kmax, "criterion": "pseudo_r2 최대 증분", "steps": steps}


def stage_partial(d, out, sel):
    """부분상관 — 원거리성(ETA) 통제 전/후. v19 교훈의 직접 검사."""
    names, X, gsz, chosen = d["names"], d["X"], d["gsz"], d["chosen"]
    offs = np.concatenate([[0], np.cumsum(gsz)])
    gid = np.repeat(np.arange(len(gsz)), gsz)
    cnt = gsz.astype(float)

    def demean(v):
        mu = np.add.reduceat(v, offs[:-1]) / cnt
        return v - mu[gid]

    y = demean(chosen.astype(float))
    ctrl = [names.index(n) for n in ETA_CONTROLS]
    C = np.column_stack([demean(X[:, j]) for j in ctrl])
    C = np.column_stack([C, np.ones(len(C))])
    cy, *_ = np.linalg.lstsq(C, y, rcond=None)
    y_res = y - C @ cy
    rows = []
    for j in sel:
        x = demean(X[:, j])
        if x.std() < 1e-12:
            continue
        raw = float(np.corrcoef(x, y)[0, 1])
        cx, *_ = np.linalg.lstsq(C, x, rcond=None)
        xr = x - C @ cx
        par = float(np.corrcoef(xr, y_res)[0, 1]) if xr.std() > 1e-12 else float("nan")
        rows.append({"feature": names[j], "r_within": raw, "r_partial_eta": par,
                     "shrink": (abs(par) / abs(raw)) if abs(raw) > 1e-9 else float("nan")})
    rows.sort(key=lambda r: -abs(r["r_within"]))

    # ETA 를 모형에서 빼면 인프라 계수가 어떻게 커지는가 (누락변수 실증)
    _G.update(X=X, gsz=gsz, chosen=chosen, target=d["target"])
    infra = [n for n in ("is_tier3", "has_helipad", "max_capa", "max_send") if n in names]
    with_eta = _fit(np.asarray(sorted(set(list(sel))), int))
    no_eta_cols = np.asarray([c for c in sel if names[c] not in ETA_CONTROLS], int)
    no_eta = _fit(no_eta_cols)
    kn_w = [names[c] for c in sorted(set(list(sel)))]
    kn_n = [names[c] for c in no_eta_cols]
    Zsel = np.column_stack([demean(X[:, j]) for j in sel])
    sdw = {names[c]: Zsel[:, i].std() for i, c in enumerate(sel)}
    omit = []
    for nm in infra:
        if nm in kn_w and nm in kn_n:
            omit.append({"feature": nm,
                         "beta_std_with_eta": float(with_eta["beta"][kn_w.index(nm)] * sdw[nm]),
                         "beta_std_without_eta": float(no_eta["beta"][kn_n.index(nm)] * sdw[nm])})
    out["partial"] = {
        "control": ETA_CONTROLS,
        "note": ("상관은 모두 선택집합 내 평균차분 후 값이다. 차분 자체가 좌표 수준 "
                 "원거리성을 완전 제거하므로, 여기 남는 상관은 '같은 현장 안에서 후보 간' 차이다."),
        "omitted_variable_check": {"features": omit,
                                   "recall_with_eta": with_eta["recall"],
                                   "recall_without_eta": no_eta["recall"],
                                   "pseudo_r2_with_eta": with_eta["pseudo_r2"],
                                   "pseudo_r2_without_eta": no_eta["pseudo_r2"]},
        "rows": rows,
    }
    print(f"[partial] ETA 제거 시 재현율 {with_eta['recall']:.4f} → {no_eta['recall']:.4f}",
          flush=True)


def _arm_cols(names, sel, drop):
    ds = set(drop)
    return np.asarray([c for c in sel if names[c] not in ds], int)


def stage_info(d, out, sel, workers):
    """정보수준 절단 — 통신(census) 끊기 vs 자기 발송기록 끊기 비대칭."""
    names, gsz = d["names"], d["gsz"]
    arms = {
        "I3_ALL": [],
        "NO_CENSUS": BLK_CENSUS + BLK_MIXED,
        "NO_OWN": BLK_OWN + BLK_MIXED,
        "NO_BOTH": BLK_CENSUS + BLK_OWN + BLK_MIXED,
        "I1_FIELD": BLK_CENSUS + BLK_MIXED + BLK_TELEMETRY,
        "I0_MIN": BLK_CENSUS + BLK_MIXED + BLK_TELEMETRY + BLK_OWN + BLK_UNRESCUED,
    }
    _G.update(X=d["X"], gsz=gsz, chosen=d["chosen"], target=d["target"])
    fitted = {}
    for k, drop in arms.items():
        cols = _arm_cols(names, sel, drop)
        fitted[k] = _fit(cols)
        print(f"  [{k:10s}] p={len(cols):2d} recall={fitted[k]['recall']:.4f} "
              f"R2={fitted[k]['pseudo_r2']:.4f} KL={fitted[k]['kl_per_set']:.4f}", flush=True)

    # (지역 × 상태) 직사각 NLL 행렬 → v20 paired/wtl 커널 그대로.
    # NLL 은 낮을수록 좋다(PDR 과 같은 방향) → paired(a=후보, b=기준) 의 양수 = 후보 우세.
    m = d["info"]["sampling"]["m_per_district"]
    nd = d["info"]["sampling"]["n_districts"]
    code = d["dist_code"]
    order = np.lexsort((np.arange(len(code)), code))
    cube = {}
    for k, f in fitted.items():
        v = np.ascontiguousarray(d["X"][:, f["cols"]]) @ f["beta"]
        s = _group_stats(v, gsz, d["chosen"], d["target"])
        cube[k] = (-s["ll"])[order].reshape(nd, m)
    base = cube["I3_ALL"]
    cmp = {}
    for k in arms:
        if k == "I3_ALL":
            continue
        p = paired(cube[k], base)      # b − a = I3 − arm, 양수면 arm 이 더 나쁨→부호 반전 표기
        cmp[k] = {"cost_nll_vs_I3": -p["delta"], "ci95": p["ci95"],
                  "win": p["win"], "tie": p["tie"], "loss": p["loss"],
                  "wtl_note": "win=그 팔이 I3 보다 우세한 시군구 수"}
    out["info_levels"] = {
        "arms": {k: {"n_features": int(len(f["cols"])),
                     "features": [names[c] for c in f["cols"]],
                     **{q: f[q] for q in ("recall", "pseudo_r2", "ll_per_set", "kl_per_set")}}
                 for k, f in fitted.items()},
        "paired_vs_I3": cmp,
        "unit": "상태당 음의 로그우도(낮을수록 좋음). 시군구 250 × 상태 m 직사각, "
                "커널 tools/v20_threshold_report.paired/wtl.",
        "judgment_note": ("판정선 0.00053 은 PDR 단위라 여기 적용 불가. 팔 비교는 "
                          "자체 paired 95%CI 로만 판정한다."),
        "blocks": {"CENSUS": BLK_CENSUS, "OWN": BLK_OWN, "MIXED": BLK_MIXED,
                   "TELEMETRY": BLK_TELEMETRY},
    }


def stage_tree(d, out, args):
    """후보랭킹 GBDT — gain 중요도 vs 순열 Δ(홀드아웃). v17 G31 설정 재사용."""
    from lightgbm import LGBMRegressor
    names, X, gsz = d["names"], d["X"], d["gsz"]
    code = d["dist_code"]
    nd = int(code.max()) + 1
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(nd)
    test_d = set(perm[: max(1, nd // 5)].tolist())
    is_test = np.asarray([c in test_d for c in code])
    rmask = np.repeat(is_test, gsz)

    Xtr, Xte = X[~rmask], X[rmask]
    ytr = d["target"][~rmask]
    # v10/v17 결정로그 규약: 회귀 타깃 = 교사 softmax 확률, sample_weight 는 원본 weight.
    # 이 캐시는 weight 를 안 들고 있으므로 균등 가중으로 적합한다(차이는 gain 배분에만).
    model = LGBMRegressor(objective="regression_l2", num_leaves=31, learning_rate=0.04,
                          n_estimators=600, min_child_samples=40, subsample=0.85,
                          subsample_freq=1, colsample_bytree=0.90, reg_lambda=1.0,
                          random_state=0, n_jobs=args.lgbm_jobs, verbosity=-1,
                          deterministic=True, force_col_wise=True)
    t0 = time.time()
    model.fit(Xtr, ytr)
    gain = np.asarray(model.booster_.feature_importance("gain"), float)
    gain = gain / max(gain.sum(), 1e-12)

    gte = gsz[is_test]
    cte = d["chosen"][rmask]
    tte = d["target"][rmask]

    def rank_metrics(pred):
        offs = np.concatenate([[0], np.cumsum(gte)])
        hit = 0
        for k in range(len(gte)):
            a, b = int(offs[k]), int(offs[k + 1])
            hit += int(cte[a + int(np.argmax(pred[a:b]))])
        # soft 라벨 대비 L2 (그룹 무관)
        return hit / len(gte), float(np.mean((pred - tte) ** 2))

    base_pred = model.predict(Xte)
    base_rec, base_mse = rank_metrics(base_pred)
    rows = []
    rs = np.random.default_rng(SEED + 1)
    for j in range(X.shape[1]):
        drs, dms = [], []
        for _ in range(args.perm_reps):
            Xp = Xte.copy()
            Xp[:, j] = Xte[rs.permutation(len(Xte)), j]
            r, ms = rank_metrics(model.predict(Xp))
            drs.append(base_rec - r)
            dms.append(ms - base_mse)
        rows.append({"feature": names[j], "gain_share": float(gain[j]),
                     "perm_d_recall": float(np.mean(drs)),
                     "perm_d_mse": float(np.mean(dms))})
    rows.sort(key=lambda r: -r["gain_share"])
    for k, r in enumerate(rows, 1):
        r["rank_gain"] = k
    by_perm = sorted(rows, key=lambda r: -r["perm_d_recall"])
    for k, r in enumerate(by_perm, 1):
        r["rank_perm"] = k

    # gain 상위 k 만으로 재적합 → '중요도 질량' 대 '실제 재현율'
    nested = {}
    for k in (3, 5, 7, 10, 15, 20):
        cols = [names.index(r["feature"]) for r in rows[:k]]
        mm = LGBMRegressor(objective="regression_l2", num_leaves=31, learning_rate=0.04,
                           n_estimators=600, min_child_samples=40, subsample=0.85,
                           subsample_freq=1, colsample_bytree=0.90, reg_lambda=1.0,
                           random_state=0, n_jobs=args.lgbm_jobs, verbosity=-1,
                           deterministic=True, force_col_wise=True)
        mm.fit(Xtr[:, cols], ytr)
        r, ms = rank_metrics(mm.predict(Xte[:, cols]))
        nested[k] = {"gain_mass": float(sum(x["gain_share"] for x in rows[:k])),
                     "recall": r, "mse": ms,
                     "features": [x["feature"] for x in rows[:k]]}
    out["tree"] = {
        "model": "LGBMRegressor G31 (v17_feature_distill 설정) · target=교사 softmax 확률",
        "split": {"rule": "시군구 단위 5분할 중 1분할 홀드아웃", "seed": SEED,
                  "n_districts_test": len(test_d),
                  "n_sets_train": int((~is_test).sum()), "n_sets_test": int(is_test.sum())},
        "weight_note": "캐시가 원본 row weight 를 보관하지 않아 균등 가중 적합(gain 배분에만 영향)",
        "holdout": {"recall": base_rec, "mse": base_mse,
                    "chance": float(np.mean(1.0 / gte))},
        "perm_reps": args.perm_reps,
        "nested_topk_by_gain": nested,
        "features": rows,
        "fit_sec": round(time.time() - t0, 1),
    }
    print(f"[tree] 홀드아웃 재현율 {base_rec:.4f} (우연 {np.mean(1.0/gte):.4f}) "
          f"· {time.time()-t0:.0f}s", flush=True)


# ---------------------------------------------------------------------- 시도
def _sido_sample(dec: Path, m: int, seed: int):
    sk, off, teach = _lazy(dec, "state_key", "offsets", "teacher_action")
    sk = np.asarray([str(x) for x in sk])
    sig = np.asarray([k.rsplit("_", 2)[-2] for k in sk])
    dmap = {int(a): decode(int(a))[1] for a in np.unique(teach)}
    td = np.asarray([dmap[int(a)] for a in teach])
    rng = np.random.default_rng(seed)
    picked = []
    for c in np.unique(sig):
        idx = np.flatnonzero(sig == c)
        rng.shuffle(idx)
        got = [int(s) for s in idx if td[s] != 0][:m]
        picked.append(np.asarray(got, int))
    states = np.sort(np.concatenate(picked))
    keep = np.zeros(int(off[-1]), bool)
    for s in states:
        keep[int(off[s]):int(off[s + 1])] = True
    X = _stream(dec, "X.npy", keep).astype(np.float64)
    chosen = _stream(dec, "chosen.npy", keep)
    cact = _stream(dec, "cand_action.npy", keep)
    sizes = np.asarray([int(off[s + 1]) - int(off[s]) for s in states], int)
    row_state = np.repeat(np.arange(len(states)), sizes)
    dm = {int(a): decode(int(a)) for a in np.unique(cact)}
    dst = np.asarray([dm[int(a)][1] for a in cact], np.int16)
    disp = dst > 0
    X, chosen, row_state = X[disp], chosen[disp], row_state[disp]
    gsz = np.bincount(row_state, minlength=len(states))
    ok = gsz >= 2
    if not ok.all():
        remap = np.cumsum(ok) - 1
        keeprow = ok[row_state]
        X, chosen = X[keeprow], chosen[keeprow]
        gsz = gsz[ok]
    return X, chosen, gsz, sk[states]


def cmd_sido(args) -> None:
    from scipy.stats import kendalltau
    nat = json.load(open(OUTDIR / "link2_national.json", encoding="utf-8"))
    nat_rows = nat["importance"]["features"]
    nat_order = [r["feature"] for r in nat_rows]
    names_nat = nat["importance"]["features"]
    del names_nat
    feats_base = [str(x) for x in _lazy(DEC_NATIONAL, "feature_names")[0]]
    res = {"m_per_district": args.m, "seed": args.seed,
           "national_reference": {"order": nat_order},
           "note": ("시도 로그는 43 열 기본 특징만 사용한다 — 파생 특징(수술실수·hinge 등)은 "
                    "좌표별 hospital_info.csv 결합이 필요해 시도별 재계산 비용이 크고, "
                    "순위 안정성 질문은 기본 특징만으로도 닫힌다."),
           "sido": {}}
    files = sorted(CARDS.glob("dec_sido_*.npz"))
    # 전국 기준 순위를 기본 43 특징으로 다시 계산(같은 특징 공간에서 비교해야 τ 가 의미를 갖는다)
    d = load_cache()
    base_cols = [d["names"].index(n) for n in feats_base]
    sel, Z, _, _ = _prune(d["X"][:, base_cols], d["gsz"], feats_base)
    _G.update(X=np.ascontiguousarray(d["X"][:, base_cols]), gsz=d["gsz"],
              chosen=d["chosen"], target=d["target"])
    f = _fit(sel)
    sdw = Z[:, sel].std(0)
    nat_base = sorted([(abs(float(f["beta"][i] * sdw[i])), feats_base[c])
                       for i, c in enumerate(sel)], reverse=True)
    nat_names = [n for _, n in nat_base]
    nat_val = {n: v for v, n in nat_base}
    res["national_base43"] = {"order": nat_names,
                              "abs_beta_std_within": nat_val,
                              "recall": f["recall"], "pseudo_r2": f["pseudo_r2"],
                              "n_identified": int(len(sel))}
    for p in files:
        sido = p.name[len("dec_sido_"):-len(".npz")]
        t0 = time.time()
        X, chosen, gsz, _k = _sido_sample(p, args.m, args.seed)
        s2, Z2, _, _ = _prune(X, gsz, feats_base)
        _G.update(X=X, gsz=gsz, chosen=chosen, target=None)
        ff = _fit(s2)
        sw = Z2[:, s2].std(0)
        ranked = sorted([(abs(float(ff["beta"][i] * sw[i])), feats_base[c])
                         for i, c in enumerate(s2)], reverse=True)
        order = [n for _, n in ranked]
        common = [n for n in nat_names if n in set(order)]
        a = [nat_names.index(n) for n in common]
        b = [order.index(n) for n in common]
        tau, pv = kendalltau(a, b)
        res["sido"][sido] = {
            "n_choice_sets": int(len(gsz)), "n_identified": int(len(s2)),
            "recall": ff["recall"], "pseudo_r2": ff["pseudo_r2"],
            "order": order, "kendall_tau_vs_national": float(tau), "p": float(pv),
            "n_common": len(common),
            "top5": order[:5],
            "top5_overlap": len(set(order[:5]) & set(nat_names[:5])),
            "top8_overlap": len(set(order[:8]) & set(nat_names[:8])),
            "wall_sec": round(time.time() - t0, 1),
        }
        print(f"  [{sido:3s}] 집합 {len(gsz)} τ={tau:+.3f} top5∩={res['sido'][sido]['top5_overlap']}"
              f" recall={ff['recall']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    taus = [v["kendall_tau_vs_national"] for v in res["sido"].values()]
    o5 = [v["top5_overlap"] for v in res["sido"].values()]
    res["summary"] = {"n_sido": len(taus), "tau_mean": float(np.mean(taus)),
                      "tau_min": float(np.min(taus)), "tau_max": float(np.max(taus)),
                      "tau_ci95": ci(taus),
                      "top5_overlap_mean": float(np.mean(o5)),
                      "top5_overlap_min": int(np.min(o5))}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "link2_sido.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_meta(OUTDIR / "link2_sido.json", res["summary"])
    print(json.dumps(res["summary"], ensure_ascii=False, indent=2))


# ------------------------------------------------------------------- 출력
def _write_meta(path: Path, summary: dict) -> None:
    Path(str(path) + ".meta.json").write_text(json.dumps(
        {"schema_version": 1, "date": time.strftime("%Y-%m-%d %H:%M"),
         "tool": "tools/v22_link2_importance.py", "complete": True,
         "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")


def _csv(path: Path, header, rows) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _tables(out: dict) -> None:
    imp = out["importance"]["features"]
    _csv(OUTDIR / "tab_importance.csv",
         ["feature", "rank_beta", "beta_std_within", "beta_std_global",
          "rank_gain", "loo_d_pseudo_r2", "loo_d_recall", "loo_d_kl"],
         [[r["feature"], r["rank_beta"], f"{r['beta_std_within']:.6f}",
           f"{r['beta_std_global']:.6f}", r["rank_gain"],
           f"{r['loo_d_pseudo_r2']:.6f}", f"{r['loo_d_recall']:.6f}",
           f"{r['loo_d_kl']:.6f}"] for r in imp])
    _csv(OUTDIR / "tab_forward.csv",
         ["step", "feature", "pseudo_r2", "d_pseudo_r2", "recall", "d_recall", "kl_per_set"],
         [[s["step"], s["feature"], f"{s['pseudo_r2']:.6f}", f"{s['d_pseudo_r2']:.6f}",
           f"{s['recall']:.6f}", f"{s['d_recall']:.6f}", f"{s['kl_per_set']:.6f}"]
          for s in out["forward"]["steps"]])
    _csv(OUTDIR / "tab_partial.csv", ["feature", "r_within", "r_partial_eta", "shrink"],
         [[r["feature"], f"{r['r_within']:+.4f}", f"{r['r_partial_eta']:+.4f}",
           f"{r['shrink']:.3f}"] for r in out["partial"]["rows"]])
    a = out["info_levels"]["arms"]
    c = out["info_levels"]["paired_vs_I3"]
    _csv(OUTDIR / "tab_infolevel.csv",
         ["arm", "n_features", "recall", "pseudo_r2", "kl_per_set",
          "cost_nll_vs_I3", "ci95", "W", "T", "L"],
         [[k, a[k]["n_features"], f"{a[k]['recall']:.6f}", f"{a[k]['pseudo_r2']:.6f}",
           f"{a[k]['kl_per_set']:.6f}",
           f"{c[k]['cost_nll_vs_I3']:+.6f}" if k in c else "0",
           f"{c[k]['ci95']:.6f}" if k in c else "0",
           c[k]["win"] if k in c else "", c[k]["tie"] if k in c else "",
           c[k]["loss"] if k in c else ""] for k in a])
    if "tree" in out:
        tr = out["tree"]["features"]
        _csv(OUTDIR / "tab_tree.csv",
             ["feature", "rank_gain", "gain_share", "rank_perm", "perm_d_recall", "perm_d_mse"],
             [[r["feature"], r["rank_gain"], f"{r['gain_share']:.6f}", r["rank_perm"],
               f"{r['perm_d_recall']:+.6f}", f"{r['perm_d_mse']:+.3e}"] for r in tr])


def cmd_fit(args) -> None:
    t0 = time.time()
    d = load_cache()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = {"date": time.strftime("%Y-%m-%d %H:%M"), "tool": "tools/v22_link2_importance.py",
           "input": d["info"],
           "link2_statement": (
               "링크 2 는 카드에 넣을 **변수 집합**을 고르는 단계다. 숫자(임계값)는 폐루프가 정한다. "
               "교사 행동 재현율은 변수 선택의 증거로만 쓰며, 폐루프 성능의 근거로 쓰지 않는다 "
               "— v20 에서 재현율 순서가 폐루프 순서와 정확히 반대였다."),
           "workers": args.workers}
    stage_sanity(d, out)
    sel, keep_names, Z, full = stage_importance(d, out, args.workers, args.bootstrap)
    stage_forward(d, out, sel, args.workers, args.kmax)
    stage_partial(d, out, sel)
    stage_info(d, out, sel, args.workers)
    if not args.no_tree:
        stage_tree(d, out, args)
    out["wall_min"] = round((time.time() - t0) / 60, 2)
    (OUTDIR / "link2_national.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_meta(OUTDIR / "link2_national.json", {
        "recall_full": out["importance"]["full_model"]["recall"],
        "pseudo_r2_full": out["importance"]["full_model"]["pseudo_r2"],
        "card_set_recall": out["importance"]["card_set"]["recall"],
        "n_identified": out["importance"]["n_features_identified"]})
    _tables(out)
    print(f"[fit] 저장 {OUTDIR}  wall={out['wall_min']:.1f}분", flush=True)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="균형 표본 캐시 생성")
    s.add_argument("--decisions", default=str(DEC_NATIONAL))
    s.add_argument("--m", type=int, default=160, help="시군구별 상태 표본 수")
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--workers", type=int, default=24)
    f = sub.add_parser("fit", help="전 스테이지 적합")
    f.add_argument("--workers", type=int, default=16)
    f.add_argument("--kmax", type=int, default=10, help="전진선택 단계 수")
    f.add_argument("--bootstrap", type=int, default=0, help="시군구 클러스터 부트스트랩 반복")
    f.add_argument("--no_tree", action="store_true")
    f.add_argument("--lgbm_jobs", type=int, default=8)
    f.add_argument("--perm_reps", type=int, default=3)
    g = sub.add_parser("sido", help="17 시도 순위 안정성")
    g.add_argument("--m", type=int, default=160)
    g.add_argument("--seed", type=int, default=SEED)
    return p


def main() -> None:
    args = build_parser().parse_args()
    {"sample": cmd_sample, "fit": cmd_fit, "sido": cmd_sido}[args.cmd](args)


if __name__ == "__main__":
    main()
