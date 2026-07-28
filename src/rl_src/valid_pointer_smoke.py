# -*- coding: utf-8 -*-
"""v6 Track A-3 스모크 — obs valid 열(F=8)·포인터 마스크드 풀링·PadAwareVecNormalize 봉인.

세 축을 검증한다:
  T1 wrapper 실전 : essential+load+valid + MCI_H_PAD 로 자연-H(경기 실H=38) 시나리오를
                    obs 402=47*8+26 로 소비, valid 열(=[1]*38+[0]*9) 이 롤아웃 내내 불변.
  T2 extractor/head 수학: HospitalTokenExtractor(valid_col=7) 의 (a) 패딩 불변 (b) 순열등변
                    (c) all-valid=None 동치, (d) 기준 pointer 의 class 순위 공유 제약과
                    JointPointerActionNet 의 3원 상호작용 복원, (e) residual 3종의 기준선
                    0-init 동치·gradient 생존을 순수 torch 로 검증.
  T3 PadAware      : PadAwareVecNormalize 가 exempt 열을 정규화 면제·비면제 열은 변형,
                    save→VecNormalize.load 왕복 후 클래스·exempt_idx·동작 보존.

재사용 deps: env_factory.make_base_env → RewardRedesignWrapper → HospitalFeatureWrapper
(pad_smoke.py 의 단일 지역 체인과 동일) / pointer_policy.{HospitalTokenExtractor,
PointerActionNet} / pad_vecnorm.PadAwareVecNormalize.

실행(스레드 핀 필수): OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
  python src/rl_src/valid_pointer_smoke.py   (전 테스트 [OK] 여야 exit 0.)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import contextlib
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "sim_src"))

import numpy as np

# 자연-H 시도 시나리오: 경기(실H=38) — sido_natural_osrm_manifest 의 '경기' config.
CFG = ("/home/ryu/MCI_UAV/scenarios/exp_시도natural/osrm/경기_osrm/"
       "(37.2893,127.0535)/config_(37.2893,127.0535).yaml")


# ================================================================= T1 wrapper 실전
def build_wrapper():
    """essential+load+valid + MCI_H_PAD=47 체인 조립(생성 전 env var 설정)."""
    os.environ["MCI_OBS_VARIANT"] = "essential+load+valid"
    os.environ["MCI_H_PAD"] = "47"
    os.environ["MCI_CAP_GATE"] = "occ"
    from env_factory import make_base_env
    from reward_redesign_wrapper import RewardRedesignWrapper
    from hospital_feature_wrapper import HospitalFeatureWrapper
    base = make_base_env(CFG, seed=0, rule_test=False, eval_mode=True)
    return HospitalFeatureWrapper(RewardRedesignWrapper(base))


def t1_wrapper():
    out = []
    env = build_wrapper()
    assert env._H_real == 38, f"경기 실H {env._H_real} != 38"
    assert env.H == 47, f"H_pad {env.H} != 47"
    assert env._F == 8, f"F {env._F} != 8(valid 열 포함)"
    assert env.observation_space.shape[0] == 402, env.observation_space.shape
    assert env.action_space.n == 2 * 48 * 2, env.action_space  # 2×(H_pad+1)×2 = 192
    expect_valid = np.concatenate([np.ones(38, np.float32), np.zeros(9, np.float32)])
    rng = np.random.default_rng(11)
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        obs, _ = env.reset(seed=42)
        assert np.array_equal(obs[:47 * 8].reshape(47, 8)[:, 7], expect_valid), "reset valid 열 오염"
        checked = 0
        for step in range(30):
            ent = obs[:47 * 8].reshape(47, 8)
            assert np.array_equal(ent[:, 7], expect_valid), f"step {step}: valid 열 오염 {ent[:, 7]}"
            checked += 1
            mask = np.asarray(env.action_masks(), dtype=bool)
            # 패딩 dest(실H+1=39..47)는 마스크가 차단해야 함
            assert not mask.reshape(2, 48, 2)[:, 39:, :].any(), f"step {step}: 패딩 dest 마스크 True"
            a = int(rng.choice(np.flatnonzero(mask)))
            obs, _r, term, trunc, _ = env.step(a)
            if term or trunc:
                obs, _ = env.reset(seed=42 + step + 1)
    env.close()
    out.append(f"[OK] T1 wrapper 실전 (경기 실H=38, H_pad=47, obs=402, act=192, "
               f"valid 열+패딩 마스크 {checked}스텝 불변)")
    return out


# ================================================================= T2 extractor 수학
def t2_extractor():
    import torch as th
    from gymnasium.spaces import Box
    from pointer_policy import (ClassModeResidualPointerActionNet, HospitalTokenExtractor,
                                JointPointerActionNet, LowRankResidualPointerActionNet,
                                PointerActionNet)
    out = []
    H, F, G, e = 47, 8, 26, 32
    n_valid = 38
    obs_space = Box(-np.inf, np.inf, (H * F + G,), np.float32)
    th.manual_seed(0)
    ex = HospitalTokenExtractor(obs_space, n_hospitals=H, entity_f=F, global_dim=G,
                                embed_dim=e, ctx_dim=64, valid_col=7)
    head = PointerActionNet(H, e, 64)  # n_class=2, n_mode=2, hidden=64
    ex.eval(); head.eval()

    def split(feat):
        return feat[:, :H * e].reshape(feat.shape[0], H, e), feat[:, H * e:]

    B = 4
    th.manual_seed(1)
    ent = th.randn(B, H, F); ent[:, :, 7] = 0.0; ent[:, :n_valid, 7] = 1.0
    glob = th.randn(B, G)
    obs = th.cat([ent.reshape(B, -1), glob], dim=1)
    with th.no_grad():
        feat = ex(obs); log = head(feat)
    tok, ctx = split(feat)
    L = log.reshape(B, 2, H + 1, 2)  # (B,C,H+1,M): dest 0=stay,1..38=유효,39..47=패딩

    # (a) 패딩 불변: 패딩 행(38..46) 비-valid 특징 교란 → 유효 토큰·ctx·유효 dest 로짓 불변
    obs_p = obs.clone()
    ent_p = obs_p[:, :H * F].reshape(B, H, F)
    pert = th.randn(B, H - n_valid, F); pert[:, :, 7] = 0.0  # valid 열은 0 유지
    ent_p[:, n_valid:, :] = pert
    with th.no_grad():
        feat_p = ex(obs_p); log_p = head(feat_p)
    tok_p, ctx_p = split(feat_p)
    L_p = log_p.reshape(B, 2, H + 1, 2)
    d_tok = (tok[:, :n_valid] - tok_p[:, :n_valid]).abs().max().item()
    d_ctx = (ctx - ctx_p).abs().max().item()
    d_log = (L[:, :, :n_valid + 1, :] - L_p[:, :, :n_valid + 1, :]).abs().max().item()
    assert d_tok < 1e-6 and d_ctx < 1e-6 and d_log < 1e-6, \
        f"패딩 불변 위반 tok={d_tok} ctx={d_ctx} log={d_log}"
    d_padtok = (tok[:, n_valid:] - tok_p[:, n_valid:]).abs().max().item()
    assert d_padtok > 1e-3, f"패딩 교란 미반영(테스트 무효) {d_padtok}"
    out.append(f"[OK] T2(a) 패딩 불변 (유효 tok/ctx/로짓 Δ={max(d_tok, d_ctx, d_log):.1e}, "
               f"패딩행 토큰 Δ={d_padtok:.2f})")

    # (b) 순열등변: 유효 병원 순열 → 토큰·dest 1..38 순열, ctx·class/mode/s0 불변
    perm = th.randperm(n_valid)
    obs_pm = obs.clone()
    obs_pm[:, :H * F].reshape(B, H, F)[:, :n_valid, :] = ent[:, :n_valid, :][:, perm, :]
    with th.no_grad():
        feat_pm = ex(obs_pm); log_pm = head(feat_pm)
    tok_pm, ctx_pm = split(feat_pm)
    L_pm = log_pm.reshape(B, 2, H + 1, 2)
    d_tok_eq = (tok_pm[:, :n_valid] - tok[:, :n_valid][:, perm, :]).abs().max().item()
    d_ctx_eq = (ctx_pm - ctx).abs().max().item()
    d_stay = (L_pm[:, :, 0, :] - L[:, :, 0, :]).abs().max().item()
    d_dest_eq = (L_pm[:, :, 1:n_valid + 1, :]
                 - L[:, :, 1:n_valid + 1, :][:, :, perm, :]).abs().max().item()
    assert (d_tok_eq < 1e-5 and d_ctx_eq < 1e-5 and d_stay < 1e-5 and d_dest_eq < 1e-5), \
        f"순열등변 위반 tok={d_tok_eq} ctx={d_ctx_eq} stay={d_stay} dest={d_dest_eq}"
    out.append(f"[OK] T2(b) 순열등변 (토큰/dest축 순열 Δ={max(d_tok_eq, d_dest_eq):.1e}, "
               f"ctx/class/mode/s0 불변 Δ={max(d_ctx_eq, d_stay):.1e})")

    # (c) all-valid 동치: valid_col=None vs =7, valid 전부 1 → 마스크드 평균=일반 평균
    th.manual_seed(0)
    ex_none = HospitalTokenExtractor(obs_space, n_hospitals=H, entity_f=F, global_dim=G,
                                     embed_dim=e, ctx_dim=64, valid_col=None)
    ex_val = HospitalTokenExtractor(obs_space, n_hospitals=H, entity_f=F, global_dim=G,
                                    embed_dim=e, ctx_dim=64, valid_col=7)
    ex_val.load_state_dict(ex_none.state_dict())  # 파라미터 동일(valid_col 은 비파라미터)
    ex_none.eval(); ex_val.eval()
    ent_all = th.randn(B, H, F); ent_all[:, :, 7] = 1.0
    obs_all = th.cat([ent_all.reshape(B, -1), glob], dim=1)
    with th.no_grad():
        f_none = ex_none(obs_all); f_val = ex_val(obs_all)
    d_equiv = (f_none - f_val).abs().max().item()
    assert d_equiv < 1e-5, f"all-valid 동치 위반 Δ={d_equiv}"
    out.append(f"[OK] T2(c) all-valid 동치 (None vs valid_col=7 Δ={d_equiv:.1e} — "
               f"MHA 마스크 fast-path 반올림 오차)")

    # (d) 3원 상호작용: 기준선은 class 간 로짓 차이가 d,m 과 무관한 상수지만,
    # joint head 는 병원 토큰별 class×mode 점수를 가져 그 제약을 제거해야 한다.
    th.manual_seed(7)
    base_head = PointerActionNet(H, e, 64)
    joint_head = JointPointerActionNet(H, e, 64)
    base_head.eval(); joint_head.eval()
    with th.no_grad():
        LB = base_head(feat).reshape(B, 2, H + 1, 2)
        LJ = joint_head(feat).reshape(B, 2, H + 1, 2)
    base_gap = LB[:, 0, :n_valid + 1, :] - LB[:, 1, :n_valid + 1, :]
    joint_gap = LJ[:, 0, :n_valid + 1, :] - LJ[:, 1, :n_valid + 1, :]
    base_spread = (base_gap - base_gap[:, :1, :1]).abs().max().item()
    joint_spread = (joint_gap - joint_gap[:, :1, :1]).abs().max().item()
    assert base_spread < 1e-6, f"기준 pointer class 순위 공유식 위반 Δ={base_spread}"
    assert joint_spread > 1e-3, f"joint 3원 상호작용 미복원 Δ={joint_spread}"

    # joint head 도 병원 순열에 정확히 등변이어야 한다(가변 H 일반화 계약).
    with th.no_grad():
        LJ_pm = joint_head(feat_pm).reshape(B, 2, H + 1, 2)
    joint_eq = (LJ_pm[:, :, 1:n_valid + 1, :]
                - LJ[:, :, 1:n_valid + 1, :][:, :, perm, :]).abs().max().item()
    joint_stay = (LJ_pm[:, :, 0, :] - LJ[:, :, 0, :]).abs().max().item()
    assert joint_eq < 1e-5 and joint_stay < 1e-5, \
        f"joint 순열등변 위반 dest={joint_eq} stay={joint_stay}"
    out.append(f"[OK] T2(d) 3원 상호작용 (기준 class-gap spread={base_spread:.1e}, "
               f"joint={joint_spread:.2f}; joint 순열등변 Δ={max(joint_eq, joint_stay):.1e})")

    # (e) baseline 포함 residual: 기준 state_dict 이식 + residual 0-init이면 logits 정확히 동일,
    # 첫 역전파에서 0-init 마지막 층 gradient가 살아 있어 실제로 학습 가능해야 한다.
    th.manual_seed(13)
    base = PointerActionNet(H, e, 64)
    residuals = [
        ("cm", ClassModeResidualPointerActionNet(H, e, 64)),
        ("rank1", LowRankResidualPointerActionNet(H, e, 64, rank=1)),
        ("rank2", LowRankResidualPointerActionNet(H, e, 64, rank=2)),
    ]
    weight = th.randn(B, 2 * (H + 1) * 2)
    checks = []
    with th.no_grad():
        base_logits = base(feat)
    for name, rh in residuals:
        inc = rh.load_state_dict(base.state_dict(), strict=False)
        assert not inc.unexpected_keys, (name, inc)
        with th.no_grad():
            rlog = rh(feat)
        delta0 = (rlog - base_logits).abs().max().item()
        assert delta0 == 0.0, f"{name} 0-init baseline 동치 위반 Δ={delta0}"
        rh.zero_grad(set_to_none=True)
        loss = (rh(feat) * weight).mean()
        loss.backward()
        if name == "cm":
            grad = rh.r_cm.weight.grad.abs().max().item()
        else:
            grad = rh.r_v.weight.grad.abs().max().item()
        assert grad > 1e-8, f"{name} residual gradient 소실 {grad}"
        checks.append(f"{name}:Δ0={delta0:.0f},g={grad:.1e}")
    out.append("[OK] T2(e) residual baseline 포함·gradient 생존 (" + ", ".join(checks) + ")")
    return out


# ================================================================= T3 PadAwareVecNormalize
def t3_padvecnorm():
    import gymnasium as gym
    from gymnasium.spaces import Box, Discrete
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from pad_vecnorm import PadAwareVecNormalize
    out = []

    class TinyBoxEnv(gym.Env):
        def __init__(self, dim=6):
            self.observation_space = Box(-np.inf, np.inf, (dim,), np.float32)
            self.action_space = Discrete(2)
            self.dim = dim
            self._rng = np.random.default_rng(0)

        def _obs(self):
            o = self._rng.normal(5.0, 2.0, self.dim).astype(np.float32)
            o[-1] = 1.0  # valid-유사 상수 exempt 열(정규화 시 0 으로 뭉개짐 → 면제로 보존 확인)
            return o

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            return self._obs(), {}

        def step(self, a):
            return self._obs(), 0.0, False, False, {}

    dim = 6
    venv = DummyVecEnv([lambda: TinyBoxEnv(dim)])
    pv = PadAwareVecNormalize(venv, exempt_idx=[dim - 1], norm_obs=True, norm_reward=False,
                              clip_obs=10.0)
    pv.reset()
    for _ in range(200):  # obs_rms 통계 워밍(exempt 열도 통계엔 포함 — 사용처만 없음)
        pv.step(np.array([0]))

    raw = np.array([[10.0, 5, 5, 5, 5, 1.0]], dtype=np.float32)  # 배치 (1, dim)
    norm = pv.normalize_obs(raw)
    assert norm.shape == (1, dim), norm.shape
    assert norm[0, dim - 1] == 1.0, f"exempt 열 미보존 {norm[0, dim - 1]}"
    assert abs(norm[0, 0] - raw[0, 0]) > 1e-3, f"비면제 열 미변형 {norm[0, 0]}"
    norm_s = pv.normalize_obs(np.array([10.0, 5, 5, 5, 5, 1.0], dtype=np.float32))  # 단건 (dim,)
    assert norm_s.shape == (dim,), norm_s.shape
    assert norm_s[dim - 1] == 1.0, f"단건 exempt 미보존 {norm_s[dim - 1]}"
    out.append(f"[OK] T3(a) 면제 (exempt 열 1.0 보존, 비면제 {raw[0, 0]:.0f}→{norm[0, 0]:.2f} 변형, "
               f"배치·단건 지원)")

    path = os.path.join(tempfile.mkdtemp(), "pv.pkl")
    pv.save(path)
    venv2 = DummyVecEnv([lambda: TinyBoxEnv(dim)])
    pv2 = VecNormalize.load(path, venv2)  # pickle 이 PadAwareVecNormalize 자동 해석
    assert type(pv2).__name__ == "PadAwareVecNormalize", type(pv2)
    assert list(pv2.exempt_idx) == [dim - 1], pv2.exempt_idx
    norm2 = pv2.normalize_obs(raw)
    assert norm2[0, dim - 1] == 1.0 and abs(norm2[0, 0] - raw[0, 0]) > 1e-3
    assert np.allclose(norm2, norm), "load 후 정규화 동작 불일치"
    out.append(f"[OK] T3(b) save→load 왕복 (클래스 PadAwareVecNormalize·"
               f"exempt_idx={list(pv2.exempt_idx)}·동작 보존)")
    return out


if __name__ == "__main__":
    lines = []
    lines += t1_wrapper()
    lines += t2_extractor()
    lines += t3_padvecnorm()
    for ln in lines:  # sim print 억제 밖에서 결과 출력(레포 gotcha)
        print(ln)
    print("[PASS] valid_pointer_smoke 3/3 (T1·T2·T3)")
