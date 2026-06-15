# -*- coding: utf-8 -*-
"""시나리오 1개에 대해 시뮬을 --trace 로 돌리고 trace_flat.json 까지 생성하는 래퍼.

sim_src/main.py 의 까다로운 호출(상대경로 cwd=repo, PYTHONPATH=src/sim_src, totalSamples 축소)을
캡슐화해서 Unity 오케스트레이터(에디터)나 사람이 한 줄로 부를 수 있게 한다.

흐름:
  1) 시나리오 coord 폴더의 config_*.yaml 을 복사 → totalSamples 축소한 임시 config
  2) main.py --config_path <temp> --trace --no_log  (cwd=repo, PYTHONPATH=repo/src/sim_src)
  3) results/{exp}/{coord}/trace_{coord}.json 생성 확인
  4) trace_export.py → <coord 폴더>/trace_flat.json

사용:
  python run_sim_trace.py --scenario "Y:/scenarios/exp_plan1_서울_dep_202605261530" [--samples 1] [--repo Y:/]
"""
from __future__ import annotations
import argparse, glob, os, re, subprocess, sys, tempfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

UAV_PY = r"C:/Users/User/anaconda3/envs/UAV/python.exe"
TOOLS = os.path.dirname(os.path.abspath(__file__))


def find_coord_dir(scenario, coord):
    if glob.glob(os.path.join(scenario, "config_*.yaml")):
        return scenario
    subs = [d for d in glob.glob(os.path.join(scenario, "(*)")) if os.path.isdir(d)]
    if coord:
        return os.path.join(scenario, coord)
    if len(subs) == 1:
        return subs[0]
    raise SystemExit(f"[ERROR] 좌표 폴더 다수 → --coord 필요: {[os.path.basename(s) for s in subs]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--coord", default=None)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--repo", default=None, help="시뮬 상대경로 기준 repo 루트(기본=시나리오의 드라이브 루트 추정)")
    ap.add_argument("--uav_py", default=UAV_PY)
    args = ap.parse_args()

    coord_dir = find_coord_dir(args.scenario, args.coord)
    coord = os.path.basename(coord_dir.rstrip("/\\"))
    exp = os.path.basename(os.path.dirname(coord_dir.rstrip("/\\")))
    cfg = glob.glob(os.path.join(coord_dir, "config_*.yaml"))
    if not cfg:
        raise SystemExit(f"[ERROR] config_*.yaml 없음: {coord_dir}")

    # repo 루트 추정: scenario 가 <repo>/scenarios/<exp>/<coord> 구조
    repo = args.repo or os.path.dirname(os.path.dirname(os.path.dirname(coord_dir.rstrip("/\\"))))
    repo = repo.rstrip("/\\") + os.sep
    main_py = os.path.join(repo, "src", "sim_src", "main.py")
    if not os.path.exists(main_py):
        raise SystemExit(f"[ERROR] main.py 없음: {main_py} (--repo 확인)")

    # 임시 config (totalSamples 축소)
    txt = open(cfg[0], encoding="utf-8").read()
    txt = re.sub(r"totalSamples:\s*\d+", f"totalSamples: {args.samples}", txt)
    tmp = os.path.join(tempfile.gettempdir(), f"_simcfg_{coord}.yaml")
    open(tmp, "w", encoding="utf-8").write(txt)

    env = dict(os.environ); env["PYTHONPATH"] = os.path.join(repo, "src", "sim_src")
    print(f"[RUN] sim --trace  exp={exp} coord={coord} samples={args.samples} repo={repo}")
    r = subprocess.run([args.uav_py, main_py, "--config_path", tmp, "--trace", "--no_log"],
                       cwd=repo, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-1500:]); print("STDERR:", r.stderr[-1500:]); raise SystemExit("[ERROR] 시뮬 실패")

    trace = os.path.join(repo, "results", exp, coord, f"trace_{coord}.json")
    if not os.path.exists(trace):
        print(r.stdout[-1000:]); raise SystemExit(f"[ERROR] trace 미생성: {trace}")
    print(f"[OK] trace: {trace}")

    out_flat = os.path.join(coord_dir, "trace_flat.json")
    r2 = subprocess.run([args.uav_py, os.path.join(TOOLS, "trace_export.py"),
                         "--trace", trace, "--out", out_flat],
                        capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r2.stdout.strip())
    if r2.returncode != 0:
        print("STDERR:", r2.stderr[-800:]); raise SystemExit("[ERROR] trace_export 실패")
    print(f"[DONE] {out_flat}")


if __name__ == "__main__":
    main()
