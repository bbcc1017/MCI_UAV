# -*- coding: utf-8 -*-
"""플레이어에서 Python 으로 실제 도착하는 BEV 의 공간구조를 잰다.

점유 래스터라면 **가로로 인접한 픽셀**은 무작위 두 픽셀보다 훨씬 비슷해야 한다.
인덱싱이 어긋나 뒤섞이면 둘이 같아진다(구조 소멸). 구/신 플레이어를 같은 방식으로 잰다.
"""
import sys, os, numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

EXE, PORT = sys.argv[1], int(sys.argv[2])
cfg = EngineConfigurationChannel()
env = UnityEnvironment(file_name=EXE, no_graphics=True, base_port=PORT,
                       side_channels=[cfg], timeout_wait=180)
cfg.set_configuration_parameters(time_scale=10.0)
env.reset()
bn = list(env.behavior_specs)[0]
spec = env.behavior_specs[bn]
vis = [i for i, o in enumerate(spec.observation_specs) if len(o.shape) == 3]
print("behavior:", bn, "| visual obs idx:", vis,
      "| shape:", [spec.observation_specs[i].shape for i in vis])

frames = []
for step in range(40):
    ds, ts = env.get_steps(bn)
    if len(ds) == 0:
        env.step(); continue
    if step >= 10:
        frames.append(ds.obs[vis[0]][0].copy())
    n = len(ds)
    act = spec.action_spec.random_action(n)
    env.set_actions(bn, act)
    env.step()
env.close()

a = np.stack(frames)                       # (T, C, H, W)
occupied = float((a != 0).mean())
h_adj = np.abs(a[:, :, :, 1:] - a[:, :, :, :-1]).mean()      # 가로 이웃
v_adj = np.abs(a[:, :, 1:, :] - a[:, :, :-1, :]).mean()      # 세로 이웃
flat = a.reshape(a.shape[0], -1)
rng = np.random.default_rng(0)
i1, i2 = rng.integers(0, flat.shape[1], 200000), rng.integers(0, flat.shape[1], 200000)
rand = np.abs(flat[:, i1] - flat[:, i2]).mean()
print("frames %d | 비영 비율 %.4f" % (len(frames), occupied))
print("가로이웃 %.6f · 세로이웃 %.6f · 무작위쌍 %.6f" % (h_adj, v_adj, rand))
print("공간구조 지수(무작위/이웃) = %.2f배   (1.0 이면 구조 없음)"
      % (rand / max(1e-9, (h_adj + v_adj) / 2)))
