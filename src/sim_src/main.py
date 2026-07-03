import argparse
import yaml
import os
import sys
import json
import random
import numpy as np
import time
from datetime import datetime
from scipy.stats import t

from ScenarioManager import ScenarioManager
from RuleManager import RuleManager
from MCIEnvironment_gymnasium import MCIEnvironment_gym


class _Tee:
    """stdout 을 콘솔과 파일에 동시 기록."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except UnicodeEncodeError:
                # cp949 콘솔 호환을 위해 ASCII fallback
                st.write(s.encode('ascii', errors='replace').decode('ascii'))
    def flush(self):
        for st in self.streams:
            st.flush()


def _setup_logfile(config_path: str, log_root: str = "experiment_logs",
                   log_kind: str = "sim") -> 'tuple[str, object]':
    """experiment_logs/<kind>_<exp_indicator>_<timestamp>.log 파일 핸들 생성."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    exp_indicator = cfg['run_setting'].get('exp_indicator', 'unknown')
    os.makedirs(log_root, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 파일명에 좌표 그대로 포함 (괄호/콤마 OK on Windows)
    log_path = os.path.join(log_root, f"{log_kind}_{exp_indicator}_{ts}.log")
    return log_path, open(log_path, 'w', encoding='utf-8')


# parser 객체 생성 및 등록
parser = argparse.ArgumentParser(description='Run MCI_simulation')
parser.add_argument('--config_path', default="./config.yaml", help='configuration file(.yaml) 경로')
parser.add_argument('--trace', action='store_true', default=False,
                    help='Enable per-patient trace logging (saves trace JSON)')
parser.add_argument('--log_dir', default='experiment_logs',
                    help='시뮬레이션 로그 저장 디렉터리 (기본: experiment_logs/)')
parser.add_argument('--no_log', action='store_true', default=False,
                    help='로그 파일 저장 비활성 (콘솔만)')
args = parser.parse_args()


class RunManager():
    def __init__(self, args):
        # 0. configuration 파일(.yaml) 경로로 YAML 파일 불러오기
        config_path = args.config_path
        with open(config_path, 'r', encoding="utf-8") as f:
            configs = yaml.safe_load(f)

        # 1. Run setting
        cfg_run = configs['run_setting']
        totalSamples = cfg_run['totalSamples']  # number of samples
        self.init_random_seed = cfg_run['random_seed']
        if self.init_random_seed is not None:
            random.seed(self.init_random_seed) # python random seed 고정
            np.random.seed(self.init_random_seed) # numpy random seed 고정
            rng = np.random.default_rng(self.init_random_seed) # numpy random number generator seed 고정 및 사용
        else:
            rng = np.random.default_rng()

        output_path = cfg_run['output_path']
        exp_indicator = cfg_run['exp_indicator']
        output_path = os.path.join(output_path, exp_indicator)
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True) # 덮어쓰기 가능
        save_info = cfg_run['save_info']
        rule_test = cfg_run['rule_test']
        eval_mode = cfg_run['eval_mode']

        # 1. 데이터 읽어서 시나리오 생성
        self.s_manager = ScenarioManager(configs, rng=rng)
        scenario = self.s_manager.scenario

        # Enable trace logging if requested
        self.enable_trace = getattr(args, 'trace', False)
        if self.enable_trace:
            scenario['EventManager'].enable_trace = True

        # 2. Rule 생성
        # configs: rule_name
        self.r_manager = RuleManager(configs['rule_info'], scenario=scenario, rng=rng) # Rule 객체 모음 return?
        self.rules = self.r_manager.rules

        # 4. 환경 생성
        self.env = MCIEnvironment_gym(scenario=scenario,
                                      rng=rng,
                                      rule_test=rule_test,
                                      eval_mode=eval_mode)

        # 5. 시뮬레이션 실행
        output, output_stat = self.run(self.env, self.rules, totalSamples)
        # MCI_CAP_GATE=psent 일 때 occ 결과를 덮어쓰지 않도록 파일명에 _psent 접미사.
        _cap_sfx = "" if os.environ.get("MCI_CAP_GATE", "occ").strip().lower() != "psent" else "_psent"
        np.savetxt(os.path.join(output_path, "results_{0}{1}.txt".format(exp_indicator, _cap_sfx)), output, fmt='%s', delimiter="  ")
        np.savetxt(os.path.join(output_path, "results_{0}{1}_stat.txt".format(exp_indicator, _cap_sfx)), output_stat, fmt='%s', delimiter="  ")

        # Save trace data if enabled
        if self.enable_trace and hasattr(self, '_all_traces') and self._all_traces:
            trace_path = os.path.join(output_path, "trace_{0}.json".format(exp_indicator))
            with open(trace_path, 'w', encoding='utf-8') as f:
                json.dump(self._all_traces, f, indent=2, default=str)
            print(f"Trace saved to {trace_path}")

    def set_random_seed(self, seed):
        seed += self.init_random_seed
        random.seed(seed)  # python random seed 고정
        np.random.seed(seed)  # numpy random seed 고정
        rng = np.random.default_rng(seed)  # numpy random number generator seed 고정 및 사용

        self.s_manager.set_seed(rng) # self.s_manager.scenario.ev_manager.set_seed(rng)
        for r in self.rules:
            r.set_seed(rng)
        self.env.set_seed(rng)

    def run(self, env, rules, totalSamples):
        def safe_pdr(saved_reward, preventable_reward):
            if preventable_reward <= 0:
                return 0.0
            return 1 - saved_reward / preventable_reward

        results_rew = np.zeros((len(rules), totalSamples), dtype=float)
        results_time = np.zeros((len(rules), totalSamples), dtype=float)
        results_pdr = np.zeros((len(rules), totalSamples), dtype=float)
        results_rewWOG = np.zeros((len(rules), totalSamples), dtype=float)
        results_pdrWOG = np.zeros((len(rules), totalSamples), dtype=float)
        # results_preventable = np.zeros((len(rules), totalSamples), dtype=float)

        rule_names = np.array([rule.rule_name for rule in rules])

        action_logs = []
        self._all_traces = {}
        for iter in range(1, totalSamples + 1):
            print("Iter :", iter)
            action_log = {'red_uav': 0, 'red_amb': 0, 'yellow_uav': 0, 'yellow_amb': 0, 'green': 0}
            for r_idx in range(len(rules)):
                print(rules[r_idx].rule_name)
                self.set_random_seed(iter)
                obs, _ = env.reset()
                done = False
                cumul_reward = 0
                cumul_r_woG = 0.0  # 정확 woG — env info['r_woG'] 누적 (2026-07-03 수정)
                count = 0
                while not done:
                    action = rules[r_idx].select(obs)
                    # print(env.state)
                    # print(action)
                    # print(obs['num_amb'], obs['num_uav'])
                    # if obs['num_amb'] > 0 and obs['num_uav'] > 0:
                    #     print("Observation: ", obs, "Action: ", action)

                    if action[1] != 0:  # 현장에서 머무르거나 돌아오는게 아니고 환자 이송인 경우
                        pat_level = action[0]
                        trans_veh = action[2]
                        if pat_level == 0:  # red
                            if trans_veh == 0:  # amb
                                action_log['red_amb'] += 1
                            else:
                                action_log['red_uav'] += 1
                        elif pat_level == 1:  # yellow
                            if trans_veh == 0:  # amb
                                action_log['yellow_amb'] += 1
                            else:
                                action_log['yellow_uav'] += 1
                        else:  # green
                            action_log['green'] += 1
                    # action = env.action_space.sample()
                    obs, reward, done, truncated, info = env.step(action)
                    done = done or truncated  # OVERTIME 은 truncated 로 옴 (2026-07-03)

                    cumul_reward += reward
                    cumul_r_woG += info.get('r_woG', 0.0)
                    if (r_idx == 0) & (reward < 0.0):
                        print('지금')
                    # print(obs['num_amb'],obs['num_uav'])
                    # print("이번 의사결정시점 생존율 합: ", reward, "현재 시각", info['time'])
                action_logs.append(action_log)
                # Collect trace if enabled
                if self.enable_trace:
                    trace_key = f"{rules[r_idx].rule_name}__iter{iter}"
                    self._all_traces[trace_key] = env.ev_manager.get_trace()
                results_rew[r_idx, iter - 1] = cumul_reward
                results_time[r_idx, iter - 1] = info['time']
                results_pdr[r_idx, iter - 1] = safe_pdr(cumul_reward, env.preventable)
                # 정확 woG(info['r_woG'] 누적)·preventable_woG 사용 — 기존 근사식
                # (cumul-total_Green)은 미처치 Green 이 있으면(절단·과부하) 과소산출.
                # 정상 종료(전원 처치)에선 값이 동일해 기존 결과와 정합. (2026-07-03)
                results_rewWOG[r_idx, iter - 1] = cumul_r_woG
                results_pdrWOG[r_idx, iter - 1] = safe_pdr(cumul_r_woG, env.preventable_woG)
                # results_preventable[r_idx, iter - 1] = env.preventable

        stat_rew = np.zeros((len(rules), 3), dtype=float)
        stat_time = np.zeros((len(rules), 3), dtype=float)
        stat_pdr = np.zeros((len(rules), 3), dtype=float)
        stat_rewWOG = np.zeros((len(rules), 3), dtype=float)
        stat_pdrWOG = np.zeros((len(rules), 3), dtype=float)
        def get_CI(data):
            n_sample = len(data)
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            std_err = std / (n_sample ** 0.5)
            CI = t.interval(confidence=0.95, df = n_sample-1, loc = mean, scale = std_err)
            return mean, std, (CI[1] - CI[0])/2

        for r_idx in range(len(rules)):
            stat_rew[r_idx][:] = get_CI(results_rew[r_idx])
            stat_time[r_idx][:] = get_CI(results_time[r_idx])
            stat_pdr[r_idx][:] = get_CI(results_pdr[r_idx])
            stat_rewWOG[r_idx][:] = get_CI(results_rewWOG[r_idx])
            stat_pdrWOG[r_idx][:] = get_CI(results_pdrWOG[r_idx])
        stat_print_rew = np.column_stack((rule_names, stat_rew))
        stat_print_time = np.column_stack((rule_names, stat_time))
        stat_print_pdr = np.column_stack((rule_names, stat_pdr))
        stat_print_rewWOG = np.column_stack((rule_names, stat_rewWOG))
        stat_print_pdrWOG = np.column_stack((rule_names, stat_pdrWOG))

        outcomes_rew = np.column_stack((rule_names, results_rew))
        outcomes_time = np.column_stack((rule_names, results_time))
        outcomes_pdr = np.column_stack((rule_names, results_pdr))
        outcomes_rewWOG = np.column_stack((rule_names, results_rewWOG))
        outcomes_pdrWOG = np.column_stack((rule_names, results_pdrWOG))
        # outcomes_preventable = np.column_stack((rule_names, results_preventable))

        output = np.concatenate((outcomes_rew, outcomes_time, outcomes_pdr, outcomes_rewWOG, outcomes_pdrWOG), axis=0)
        output_stat = np.concatenate((stat_print_rew, stat_print_time, stat_print_pdr, stat_print_rewWOG, stat_print_pdrWOG), axis=0)

        return output, output_stat

if __name__ == '__main__':
    log_handle = None
    if not args.no_log:
        try:
            log_path, log_handle = _setup_logfile(args.config_path, log_root=args.log_dir, log_kind="sim")
            sys.stdout = _Tee(sys.__stdout__, log_handle)
            print(f"[LOG] simulation log -> {log_path}")
        except Exception as e:
            print(f"[LOG] 로그 파일 설정 실패 (콘솔만 출력): {e}")
            log_handle = None

    try:
        start_t = time.time()
        run_model = RunManager(args)
        print(f"Computation time(s): {time.time() - start_t}")
    finally:
        if log_handle is not None:
            sys.stdout = sys.__stdout__
            log_handle.close()
