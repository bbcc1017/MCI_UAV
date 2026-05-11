"""sim_src를 import 가능하게 만들고 config → env 객체로 변환하는 헬퍼."""
import os
import sys
import yaml
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SIM_SRC = os.path.join(REPO_ROOT, "src", "sim_src")
if SIM_SRC not in sys.path:
    sys.path.insert(0, SIM_SRC)

from ScenarioManager import ScenarioManager  # noqa: E402
from MCIEnvironment_gymnasium import MCIEnvironment_gym  # noqa: E402


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_base_env(config_path: str, seed: int = 0,
                  rule_test: bool = False, eval_mode: bool = False) -> MCIEnvironment_gym:
    cfg = load_config(config_path)
    rng = np.random.default_rng(seed)
    sc = ScenarioManager(cfg, rng=rng)
    env = MCIEnvironment_gym(scenario=sc.scenario, rng=rng,
                             rule_test=rule_test, eval_mode=eval_mode)
    return env
