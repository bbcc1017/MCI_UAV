# -*- coding: utf-8 -*-
"""MCI_H_PAD 런타임 병원 패딩(v6 Track A-1) 회귀 봉인 스모크.

hospital_feature_wrapper 의 H_pad 패딩이 (1) 미설정/H_pad=실H 에서 구 동작과
**비트동일**하고, (2) H_pad>실H 에서 obs/액션/마스크가 올바로 확장되며(패딩
dest 전부 차단, 정적 더미값 정합), (3) 실H>H_pad 는 명시 에러임을 봉인한다.

재사용 deps: env_factory.make_base_env → RewardRedesignWrapper → HospitalFeatureWrapper
(train_ppo_feature.FeatureMultiRegionEnv 의 단일 지역 체인과 동일). viper_distill 의
make_feature_env 는 프로세스당 캐시라 여기선 직접 조립한다.

실행: MCI_OBS_VARIANT=essential+load python src/rl_src/pad_smoke.py
(내부에서 essential+load 강제 설정 — 인자 불요. 3개 테스트 전부 [OK] 여야 exit 0.)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "sim_src"))

import contextlib
import numpy as np

CFG = ("/home/ryu/MCI_UAV/scenarios/exp_시군구/osrm/종로구_osrm/"
       "(37.599081,126.966781)/config_(37.599081,126.966781).yaml")
N_STEP = 100
EP_CAP = 5000  # 패딩 에피소드 완주 상한(무한루프 방어)

os.environ["MCI_OBS_VARIANT"] = "essential+load"
os.environ["MCI_CAP_GATE"] = "occ"


def build_env(h_pad):
    """실험 체인 조립(H_pad 설정/해제 포함). import 는 env var 반영과 무관하나
    wrapper 는 __init__ 에서 MCI_H_PAD 를 읽으므로 생성 전에 설정한다."""
    if h_pad is None:
        os.environ.pop("MCI_H_PAD", None)
    else:
        os.environ["MCI_H_PAD"] = str(h_pad)
    from env_factory import make_base_env
    from reward_redesign_wrapper import RewardRedesignWrapper
    from hospital_feature_wrapper import HospitalFeatureWrapper
    base = make_base_env(CFG, seed=0, rule_test=False, eval_mode=True)
    return HospitalFeatureWrapper(RewardRedesignWrapper(base))


def rollout_det(env, seed, n_step):
    """결정론 롤아웃: 매 스텝 마스크의 첫 True 액션. (obs,mask,reward) 궤적 반환."""
    traj = []
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        obs, _ = env.reset(seed=seed)
        for _ in range(n_step):
            mask = np.asarray(env.action_masks(), dtype=bool)
            a = int(np.flatnonzero(mask)[0])
            nobs, r, term, trunc, _info = env.step(a)
            traj.append((obs.copy(), mask.copy(), float(r)))
            obs = nobs
            if term or trunc:
                break
    return traj


def t1_bitidentical():
    """H_PAD 미설정 vs =실H(47): obs·mask·reward 비트동일."""
    e0 = build_env(None)
    assert e0._H_real == 47, f"종로구 실H {e0._H_real} != 47"
    tr0 = rollout_det(e0, seed=123, n_step=N_STEP)
    e0.close()
    e1 = build_env(47)
    tr1 = rollout_det(e1, seed=123, n_step=N_STEP)
    e1.close()
    assert len(tr0) == len(tr1), f"길이 불일치 {len(tr0)} vs {len(tr1)}"
    for i, ((o0, m0, r0), (o1, m1, r1)) in enumerate(zip(tr0, tr1)):
        assert np.array_equal(o0, o1), f"step {i}: obs 불일치"
        assert np.array_equal(m0, m1), f"step {i}: mask 불일치"
        assert r0 == r1, f"step {i}: reward 불일치 {r0} vs {r1}"
    print(f"[OK] 테스트1 비트동일 (steps={len(tr0)}, obs dim={tr0[0][0].shape[0]})")


def t2_padded():
    """H_PAD=60: 차원·마스크 차단·정적 더미·에피소드 완주."""
    env = build_env(60)
    F, Hp, Hr = env._F, env.H, env._H_real
    assert (Hp, Hr, F) == (60, 47, 7), f"차원 오류 Hp{Hp}/Hr{Hr}/F{F}"
    assert env.observation_space.shape[0] == 60 * 7 + 26, env.observation_space
    assert env.action_space.n == 2 * 61 * 2, env.action_space
    # 정적 더미
    assert np.all(env._eta_amb[47:] == 10.0) and np.all(env._eta_uav[47:] == 10.0)
    assert np.all(env._is_tier3[47:] == 0) and np.all(env._max_send[47:] == 0)
    assert env._helipad.shape[0] == 60 and np.all(env._helipad[47:] == 0)
    # 에피소드 완주(랜덤 마스크드 액션) — 전 스텝 패딩 dest 차단 확인
    rng = np.random.default_rng(7)
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        obs, _ = env.reset(seed=321)
        done, n = False, 0
        while not done and n < EP_CAP:
            mask = np.asarray(env.action_masks(), dtype=bool)
            m3 = mask.reshape(2, 61, 2)
            assert not m3[:, 48:, :].any(), f"step {n}: 패딩 dest 마스크 True"
            ent = obs[:60 * 7].reshape(60, 7)
            assert np.all(ent[47:, 2] == 10.0), f"step {n}: 패딩 eta_amb 오염"
            cand = np.flatnonzero(mask)
            a = int(rng.choice(cand))
            obs, _r, term, trunc, _ = env.step(a)
            done, n = (term or trunc), n + 1
    env.close()
    assert done, f"에피소드 미완주(cap {EP_CAP})"
    print(f"[OK] 테스트2 패딩 경로 (H_pad=60, obs 446, act 244, ep steps={n})")


def t3_error():
    """H_PAD=40 < 실H 47 → ValueError."""
    try:
        build_env(40)
    except ValueError as e:
        print(f"[OK] 테스트3 에러 경로 ({str(e)[:60]}...)")
        return
    raise AssertionError("실H>H_pad 인데 ValueError 미발생")


if __name__ == "__main__":
    t1_bitidentical()
    t2_padded()
    t3_error()
    os.environ.pop("MCI_H_PAD", None)
    print("[PASS] pad_smoke 3/3")
