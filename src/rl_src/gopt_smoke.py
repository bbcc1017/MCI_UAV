# -*- coding: utf-8 -*-
"""v12 GOPT 정책 스모크 7종 — 본 학습 전 게이트.

1) 구 경로 회귀   : v10 챔피언 zip 로드·예측 성공(pointer 경로 무회귀)
2) flatten 계약   : bilinear head 의 logit 인덱스가 프로젝트 정본 코덱
                    (loadbalance_heuristic._codec_from_mask) 과 일치
3) 순열등변       : 유효 병원 순열 시 dest logits 동일 순열, stay·class/mode 항 불변
4) 패딩 불변      : 패딩 행 특징 교란에 유효 dest logits·ctx 불변
5) save/load 왕복 : MaskablePPO.save → load 후 logits 비트 동일
6) 마스킹 정합    : apply_masking 후 불법 액션 확률 0
7) 파라미터 회계  : X1~X6 파라미터 표 + X5(용량 대조군) head_hidden 산출

실행: /home/ryu/anaconda3/envs/UAV/bin/python src/rl_src/gopt_smoke.py
"""
import os
import sys

import numpy as np
import torch as th

os.environ.setdefault("MCI_OBS_VARIANT", "essential+load+valid")
os.environ.setdefault("MCI_H_PAD", "47")
os.environ.setdefault("MCI_CAP_GATE", "occ")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = "/home/ryu/MCI_UAV"
sys.path.insert(0, os.path.join(REPO, "src/rl_src"))

import gymnasium as gym                                            # noqa: E402
from gymnasium import spaces                                       # noqa: E402
from sb3_contrib import MaskablePPO                                # noqa: E402

import pointer_policy                                              # noqa: F401,E402
import hospital_set_extractor                                      # noqa: F401,E402
from gopt_policy import (GoptBilinearActionNet, GoptMaskablePolicy,  # noqa: E402
                         GoptTokenExtractor, PointerPooledCriticMaskablePolicy,
                         build_demand_input, demand_input_dim)
from pointer_policy import PointerActionNet, PointerMaskablePolicy  # noqa: E402
from loadbalance_heuristic import _codec_from_mask                 # noqa: E402

H, F, G = 47, 8, 26
DIM = H * F + G
N_CLASS, N_MODE = 2, 2
N_ACT = N_CLASS * (H + 1) * N_MODE            # 192
VALID_COL = F - 1
OBS_SPACE = spaces.Box(-np.inf, np.inf, (DIM,), dtype=np.float32)
ACT_SPACE = spaces.Discrete(N_ACT)
fails = []


def make_obs(B=3, n_real=41, seed=1):
    """실병원 n_real 개 + 패딩. 전 병원 패딩 obs 는 MHA softmax(all -inf)=NaN 이므로 금지."""
    rng = np.random.default_rng(seed)
    ent = rng.normal(size=(B, H, F)).astype(np.float32)
    ent[:, :, VALID_COL] = 0.0
    ent[:, :n_real, VALID_COL] = 1.0
    glob = rng.normal(size=(B, G)).astype(np.float32)
    return ent, glob


def flat(ent, glob):
    return th.tensor(np.concatenate([ent.reshape(ent.shape[0], -1), glob], axis=1))


def build_policy(n_gopt_blocks=0, n_heads=4, embed=64, ctx=128, head_hidden=128,
                 ff_expansion=4, pooled_critic=False, gopt=True, n_attn_blocks=1, seed=0):
    th.manual_seed(seed)
    lr_sched = (lambda _: 3e-4)
    fe_kwargs = dict(n_hospitals=H, entity_f=F, global_dim=G, embed_dim=embed,
                     ctx_dim=ctx, n_attn_blocks=n_attn_blocks, n_heads=n_heads,
                     valid_col=VALID_COL)
    if gopt:
        fe_kwargs.update(n_gopt_blocks=n_gopt_blocks, ff_expansion=ff_expansion, dropout=0.0)
        return GoptMaskablePolicy(OBS_SPACE, ACT_SPACE, lr_sched,
                                  features_extractor_class=GoptTokenExtractor,
                                  features_extractor_kwargs=fe_kwargs,
                                  head_hidden=head_hidden, pooled_critic=pooled_critic).eval()
    cls = PointerPooledCriticMaskablePolicy if pooled_critic else PointerMaskablePolicy
    return cls(OBS_SPACE, ACT_SPACE, lr_sched,
               features_extractor_class=pointer_policy.HospitalTokenExtractor,
               features_extractor_kwargs=fe_kwargs, head_hidden=head_hidden).eval()


def logits_of(policy, ent, glob):
    with th.no_grad():
        feats = policy.extract_features(flat(ent, glob))
        if isinstance(feats, tuple):
            feats = feats[0]
        latent_pi = policy.mlp_extractor.forward_actor(feats)
        return policy.action_net(latent_pi)


# ---------- 1) 구 경로 회귀 ----------
try:
    zip_path = os.path.join(REPO, "results/rl/redesign/v10_random4_1000_pointer_s0/final_model.zip")
    m = MaskablePPO.load(zip_path, device="cpu")
    e0, g0 = make_obs(B=1, n_real=41, seed=3)
    obs = np.concatenate([e0.reshape(1, -1), g0], axis=1).astype(np.float32)
    a, _ = m.predict(obs, action_masks=np.ones((1, N_ACT), bool), deterministic=True)
    assert type(m.policy).__name__ == "PointerMaskablePolicy", type(m.policy).__name__
    print(f"[1] 구 경로 회귀 OK — v10 로드·예측 성공 action={int(a[0])} "
          f"policy={type(m.policy).__name__}")
except Exception as exc:                                            # noqa: BLE001
    fails.append(f"[1] {exc}")
    print(f"[1] FAIL {exc}")

# ---------- 2) flatten 계약 (정본 코덱과 교차검증) ----------
try:
    encode = _codec_from_mask(N_ACT, H)
    e = 64
    head = GoptBilinearActionNet(H, e, ctx_dim=128, n_class=N_CLASS, n_mode=N_MODE).eval()
    head.layer_1 = th.nn.Identity()
    head.layer_2 = th.nn.Identity()
    th.nn.init.zeros_(head.bias_cm.weight)
    th.nn.init.zeros_(head.bias_cm.bias)
    n_dst, n_dem = H + 1, N_CLASS * N_MODE
    rng = np.random.default_rng(0)
    bad = []
    for c, d, mm in [(0, 0, 0), (1, 0, 1), (0, 12, 1), (1, 47, 0), (1, 23, 1), (0, 5, 0)]:
        q_star = c * N_MODE + mm
        # S_target 를 dem=기저벡터 / dst=S열의 선형결합 으로 정확히 실현
        S_t = rng.normal(scale=0.1, size=(n_dem, n_dst)).astype(np.float32)
        S_t[q_star, d] = 10.0
        dem = th.zeros(1, n_dem, e)
        for q in range(n_dem):
            dem[0, q, q] = 1.0
        dst = th.zeros(1, n_dst, e)
        for dd in range(n_dst):
            for q in range(n_dem):
                dst[0, dd, q] = float(S_t[q, dd]) * np.sqrt(e)
        latent = th.cat([dst.reshape(1, -1), dem.reshape(1, -1), th.zeros(1, 128)], dim=1)
        with th.no_grad():
            lg = head(latent)
        got, want = int(lg.argmax().item()), int(encode(c, d, mm))
        if got != want:
            bad.append((c, d, mm, got, want))
        # 값 자체도 S_target 과 일치해야 한다
        if abs(float(lg[0, want].item()) - 10.0) > 1e-3:
            bad.append((c, d, mm, "value", float(lg[0, want].item())))
    assert not bad, f"인덱스/값 불일치 {bad}"
    print(f"[2] flatten 계약 OK — argmax 인덱스가 _codec_from_mask(c,d,m)와 6/6 일치 "
          f"(idx = c*{(H+1)*N_MODE} + d*{N_MODE} + m)")
except Exception as exc:                                            # noqa: BLE001
    fails.append(f"[2] {exc}")
    print(f"[2] FAIL {exc}")

# ---------- 3) 순열등변 / 4) 패딩 불변 ----------
for nb in (0, 1, 3):
    try:
        pol = build_policy(n_gopt_blocks=nb)
        ent, glob = make_obs(B=3, n_real=41, seed=5)
        base = logits_of(pol, ent, glob).reshape(-1, N_CLASS, H + 1, N_MODE)

        perm = np.arange(H)
        perm[:41] = np.random.default_rng(11).permutation(41)
        outp = logits_of(pol, ent[:, perm, :], glob).reshape(-1, N_CLASS, H + 1, N_MODE)
        # dest 0(stay)은 불변, dest 1..H 는 같은 순열
        d_stay = (outp[:, :, 0, :] - base[:, :, 0, :]).abs().max().item()
        want = base[:, :, 1:, :][:, :, perm, :]
        d_hos = (outp[:, :, 1:, :] - want).abs().max().item()
        assert d_stay < 5e-5, f"stay 불변 위반 {d_stay:.2e}"
        assert d_hos < 5e-5, f"dest 등변 위반 {d_hos:.2e}"

        entq = ent.copy()
        entq[:, 41:, :VALID_COL] += 50.0
        outq = logits_of(pol, entq, glob).reshape(-1, N_CLASS, H + 1, N_MODE)
        d_valid = (outq[:, :, :42, :] - base[:, :, :42, :]).abs().max().item()
        assert d_valid < 5e-5, f"유효 dest 패딩오염 {d_valid:.2e}"
        print(f"[3,4] n_gopt_blocks={nb} OK — perm(stay {d_stay:.1e} dest {d_hos:.1e}) "
              f"pad(유효dest {d_valid:.1e})")
    except Exception as exc:                                        # noqa: BLE001
        fails.append(f"[3,4] nb={nb} {exc}")
        print(f"[3,4] nb={nb} FAIL {exc}")


# ---------- 5) save/load 왕복 + 6) 마스킹 정합 ----------
class _StubEnv(gym.Env):
    """공간·마스크만 제공하는 더미(학습 안 함 — save/load·분포 계약 검증용)."""
    def __init__(self):
        self.observation_space = OBS_SPACE
        self.action_space = ACT_SPACE
        self._rng = np.random.default_rng(0)

    def action_masks(self):
        m = np.zeros(N_ACT, bool)
        m[self._rng.choice(N_ACT, size=30, replace=False)] = True
        return m

    def reset(self, *, seed=None, options=None):
        return np.zeros(DIM, np.float32), {}

    def step(self, a):
        return np.zeros(DIM, np.float32), 0.0, True, False, {}


for nb, tag in ((0, "X1"), (3, "X3")):
    try:
        fe_kwargs = dict(n_hospitals=H, entity_f=F, global_dim=G, embed_dim=64, ctx_dim=128,
                         n_attn_blocks=1, n_heads=4 if nb == 0 else 8, valid_col=VALID_COL,
                         n_gopt_blocks=nb, ff_expansion=4, dropout=0.0)
        model = MaskablePPO(GoptMaskablePolicy, _StubEnv(), device="cpu", seed=0, n_steps=8,
                            batch_size=8, policy_kwargs=dict(
                                features_extractor_class=GoptTokenExtractor,
                                features_extractor_kwargs=fe_kwargs,
                                head_hidden=128, pooled_critic=False))
        ent, glob = make_obs(B=2, n_real=41, seed=7)
        obs_np = np.concatenate([ent.reshape(2, -1), glob], axis=1).astype(np.float32)
        with th.no_grad():
            before = model.policy.get_distribution(th.tensor(obs_np)).distribution.logits.clone()
        p = f"/tmp/claude-1002/-home-ryu-MCI-UAV/gopt_smoke_{tag}.zip"
        model.save(p)
        del model
        m2 = MaskablePPO.load(p, device="cpu")
        with th.no_grad():
            after = m2.policy.get_distribution(th.tensor(obs_np)).distribution.logits
        d = (before - after).abs().max().item()
        assert d == 0.0, f"save/load logits 불일치 {d:.3e}"

        mask = np.zeros((2, N_ACT), bool)
        mask[:, [0, 5, 100, 191]] = True
        with th.no_grad():
            dist = m2.policy.get_distribution(th.tensor(obs_np), action_masks=mask)
            probs = dist.distribution.probs
        illegal = probs[~th.tensor(mask)].abs().max().item()
        legal_sum = probs[th.tensor(mask)].reshape(2, -1).sum(1)
        assert illegal == 0.0, f"불법 액션 확률 {illegal:.3e}"
        assert th.allclose(legal_sum, th.ones(2), atol=1e-5), f"합 {legal_sum}"
        os.remove(p)
        print(f"[5,6] {tag}(nb={nb}) OK — save/load Δ={d:.1e}, 불법확률={illegal:.1e}, "
              f"합법합={legal_sum.tolist()}")
    except Exception as exc:                                        # noqa: BLE001
        fails.append(f"[5,6] {tag} {exc}")
        print(f"[5,6] {tag} FAIL {exc}")

# ---------- 7) 파라미터 회계 + X5 head_hidden 산출 ----------
try:
    def npar(**kw):
        return sum(p.numel() for p in build_policy(**kw).parameters())

    rows = [
        ("v10(기준)", dict(gopt=False, head_hidden=128)),
        ("X1 bilinear", dict(gopt=True, n_gopt_blocks=0)),
        ("X2 xattn1", dict(gopt=True, n_gopt_blocks=1)),
        ("X3 gopt3", dict(gopt=True, n_gopt_blocks=3, n_heads=8)),
        ("X4 attn0", dict(gopt=False, n_attn_blocks=0, head_hidden=128)),
        ("X6 poolcritic", dict(gopt=False, pooled_critic=True, head_hidden=128)),
    ]
    counts = {}
    for name, kw in rows:
        counts[name] = npar(**kw)
    target = counts["X3 gopt3"]
    best = None
    for hh in range(128, 4097, 4):
        n = npar(gopt=False, head_hidden=hh)
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (hh, n)
        if n > target + 200000:
            break
    for name, kw in rows:
        print(f"[7] {name:16s} {counts[name]:10,}")
    print(f"[7] X5 용량대조군 : head_hidden={best[0]} → {best[1]:,} "
          f"(X3 {target:,} 대비 {best[1]-target:+,})")
except Exception as exc:                                            # noqa: BLE001
    fails.append(f"[7] {exc}")
    print(f"[7] FAIL {exc}")

print("\n=== SMOKE " + ("FAIL: " + " | ".join(fails) if fails else "ALL PASS") + " ===")
sys.exit(1 if fails else 0)
