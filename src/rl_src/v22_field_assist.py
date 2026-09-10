# -*- coding: utf-8 -*-
"""v22 현장 지휘 보조 파이프라인 — 상황 숫자 몇 개 → 채워진 규칙집 1장.

MCI 현장 지휘관이 t=0 에 아는 숫자(사고 좌표·환자 총원·가용 AMB/UAV 대수 등)를 넣으면
**그 상황에서 폐루프 시뮬을 돌려** CARD 규칙집의 파라미터(λ·red_gain·yhold 또는 무튜닝
S족의 wait_scale)를 골라 준다. 지휘차 서버 16~32 코어·3~5분 예산을 가정한다.

연구용 전수평가(`v17_rule_eval.py`)와 목적이 다르다. 전수평가는 **여러 좌표**에서 고정
파라미터의 일반화를 재고, 여기서는 **한 좌표**에서 파라미터를 고른다. 그래서 배관도 다르다.

── ★ 왜 새 드라이버가 필요한가: 기존 하니스는 현장 사례를 병렬화하지 못한다 ─────────
`v17_rule_eval.main()` 의 `Pool` 은 **지역(좌표) 단위로 샤딩**한다
(`jobs = [(key, cfg, specs, n_eps, seed0) for key in keys]`). 워커 안에서는 에피소드 ×
정책이 직렬이다. 좌표가 750개면 워커 56개가 꽉 차지만, 현장 사례는 **좌표가 1개**라
잡이 1개 = 워커 1개 = 병렬도 0 이다. 56코어 서버에서 1코어만 쓰는 셈이다.

이 드라이버는 **(후보 × 시드) 격자를 샤딩**한다. env 생성이 에피소드보다 훨씬 비싸므로
(cold 수백 ms ~ 수 초 vs 에피소드 수십 ms) 워커마다 그 좌표의 env 를 **한 번만** 만들어
프로세스 전역에 캐시하고, 자기 샤드의 (후보, 시드) 조합만 돌린다. `Pool` 은 라운드마다
새로 만들지 않고 **지속**시켜서 successive halving 의 2라운드 이후는 warm env 를 재사용한다.
구조 선례는 `score_cma.py`(CEM + CRN paired, 워커당 env 1회 빌드)다.

**샤딩이 결과를 바꾸지 않는 근거**: `MCIEnvironment_gym.reset(seed)` 이 sim 동역학 rng
(`ev_manager`)까지 전부 재시드하므로 한 프로세스 안의 롤아웃 순서는 결과에 무관하다.
env **생성** 시드는 `ScenarioManager` 를 거쳐 `EventManager` 로만 흘러가고 reset 이 그걸
덮어쓰지만, 구조적 결정성을 위해 모든 워커가 **같은 warm_seed 로 env 를 생성**한다.
그래도 이건 코드 독해에 기반한 논증이므로 `--gate workers` 로 워커 1 vs N 동일성을 실증한다.

── 단계 (각각 독립 타이머) ────────────────────────────────────────────────────
  T1 입력 파싱·검증      상황 숫자 → 런타임 노브 사전. 상속 `MCI_*` 전량 purge.
  T2 시나리오 확보        좌표를 사전생성 원장에서 최근접 스냅. **현장에서 OSRM 라우팅
                          안 함**(네트워크 의존 제거)이 설계 의도다. 스냅 거리 기록.
  T3 env 생성            워커당 1회. cold 비용 별도 계측.
  T4 후보 집합 생성       파라미터 격자(Q·P·S족) 또는 halving 초기 집단.
  T5 폐루프 롤아웃        본체. (후보 × 시드) 샤딩.
  T6 선택 + 카드 렌더     규칙집 1장 + 예상 PDR_woG + CI + 계산시간 내역.

── ★ 통계 계약 (누수 방지) ────────────────────────────────────────────────────
1. **내부 시드(선택용) ∩ 외부 시드(평가용) = ∅.** 기본 내부 `[0,K)`, 외부 `[1000,1000+M)`.
   현장 지휘관은 실현될 난수를 알 수 없다. 내부 시드로 고른 후보를 같은 내부 시드로
   평가하면 **승자의 저주**(winner's curse)가 붙어 기대성능을 낙관한다.
2. **CRN**: 한 실행 안의 모든 후보가 **같은 내부 시드 집합**을 공유한다. 후보 간 차이의
   분산을 줄이는 유일한 장치다. 캐시가 (후보,시드) 단위라 halving 라운드 간에도 유지된다.
3. 산출에 **승자의 저주 크기**를 명시한다 — 선택 후보의 `내부 PDR` vs `외부 PDR` 차이.
4. 판정선을 하드코딩하지 않는다. paired 차이의 95%CI 를 실제로 계산한다. 통계 함수는
   `tools/v20_threshold_report.py` 의 `ci`/`cube`/`paired` 를 **재사용**한다(재구현하면
   W/T/L 정의가 갈린다). 좌표가 1개이므로 **paired 단위는 시드**다 — `cube()` 가 주는
   지역×시드 행렬을 전치해서 넘긴다(자세한 이유는 `_paired_over_seeds` 주석).

── 실행시간 계측 규격 ─────────────────────────────────────────────────────────
방법론 표준은 `src/sim_src_upgrade/bench/bench_core.py:1-17` 을 그대로 승계한다.
1차 지표 = `time.process_time()` CPU 시간, 2차 = `time.perf_counter()` 벽시계, **둘 다** 기록.
자식 CPU 는 부모의 `process_time` 에 안 잡히므로 `os.times()` 의 children 항과 워커가
돌려준 `cpu_ms` 합을 함께 남긴다. BLAS/OpenMP 4종은 numpy import 전에 1로 핀하고
**상속값과 유효값을 둘 다** 메타에 적는다 — 과거에 스레드 수 차이(1 vs 128)를 코어 차이로
오인해 허위 32.5× 가 나온 사건이 있다(`src/sim_src_upgrade/README.md`).
`os.getloadavg()` 를 시작·종료에 기록한다. **공유 노드이므로 여기서 나온 벽시계는
"부하 하 참고값"이며 성능 실측이 아니다.**

── 예산 산술의 입력 (이미 측정 완료, `results/field/timing/n_scaling.json`) ──────
원본 코어 1코어 median cpu ms/ep: N50 36.1 · N100 66.6 · N200 137.5 · N300 209.6 · N500 343.1.
비용은 사고규모 N 에 **정확히 선형**(로그-로그 기울기 0.989), 결정당 1.67~1.77 ms 일정,
정책 계산은 총비용의 27~29%뿐(나머지가 sim). 지형(도시/농촌) 영향은 약 2%로 무의미.
→ 32코어 × 300초 예산이면 N500 에서 약 28,000 에피소드가 상한이다. 기본 격자
(39후보 × 내부 10시드 = 390 ep)는 그 1.4% 이므로 격자·시드를 크게 늘릴 여유가 있다.
프로세스 기동·import(약 2.3초)는 에피소드 비용에 포함되지 않으므로 별도 항목으로 남긴다.

── 사용 예 ────────────────────────────────────────────────────────────────────
  python src/rl_src/v22_field_assist.py \
      --lat 37.6489 --lon 127.5054 --patients 200 --amb 20 --uav 10 \
      --search halving --inner_seeds 10 --outer_seeds 10 --workers 16 \
      --budget_sec 240 --tag gapyeong_n200

  # 상황을 JSON 으로 (CLI 인자가 JSON 을 덮어쓴다)
  python src/rl_src/v22_field_assist.py --situation sit.json --search grid

  # 배선 게이트 (워커 1 vs 4 동일성 · 결정론 · grid vs halving 일치)
  python src/rl_src/v22_field_assist.py --lat .. --lon .. --gate all
"""
from __future__ import annotations

import os
import time

# ── 프로세스 기동 시각 — numpy/pandas import 전에 잡아야 import 비용이 분리된다.
_T_MODULE_WALL0 = time.perf_counter()
_T_MODULE_CPU0 = time.process_time()

# BLAS/OpenMP 는 numpy import **전에** 1로 핀한다. `setdefault` 가 아니라 강제 대입이다 —
# 상속된 스레드 수가 부동소수 결과를 바꾸면 같은 인자로 돌린 카드가 실행마다 달라진다.
# 상속값은 메타에 남기므로 조용히 지워지지 않는다.
BLAS_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
BLAS_INHERITED = {v: os.environ.get(v) for v in BLAS_VARS}
for _v in BLAS_VARS:
    os.environ[_v] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool, Value  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src/sim_src"))

# 좌표 원장 — 사전생성 시나리오(거리행렬 포함)가 존재하는 7,500 점.
POINTS = REPO / "scenarios/manifests/sigungu30_points.json"
MANIFEST = REPO / "scenarios/manifests/sigungu30_manifest.json"
OUT_DIR = REPO / "results/field/assist"

# 연구 분할 — 스냅된 점이 어느 분할에 속하는지 산출에 기록하고, 판정셋은 기본 차단한다.
SPLIT_MANIFESTS = {
    "budget750": REPO / "scenarios/manifests/sigungu30_budget750_manifest.json",
    "test750": REPO / "scenarios/manifests/sigungu30_test750_manifest.json",
    "train6000": REPO / "scenarios/manifests/sigungu30_train6000_manifest.json",
}
# AGENTS.md 계약 1 — 판정셋으로 튜닝하지 않는다. 개발 중 이 도구가 test750 좌표에
# 스냅되면 판정 좌표의 PDR 을 읽게 되므로 기본은 거부다. 실제 현장 배치에서만
# `--allow_test750` 로 푼다(현장 사고는 어디서든 날 수 있으므로 도구 자체의 결함은 아니다).
FORBIDDEN_SPLIT = "test750"

# ── 상황 입력 → 런타임 노브. 값은 문자열로 export 하며 타입은 ScenarioManager 규약을 따른다.
#    미지정 노브는 **건드리지 않는다**(unset 유지 = config 기본값).
KNOB_MAP = {
    "patients": ("MCI_INCIDENT_SIZE", int),
    "amb": ("MCI_AMB_NUM", int),
    "uav": ("MCI_UAV_NUM", int),
    "treat_scale": ("MCI_TREAT_SCALE", float),
    "capa_scale": ("MCI_CAPA_SCALE", float),
    "amb_kmh": ("MCI_AMB_VELOCITY", float),
    "uav_kmh": ("MCI_UAV_VELOCITY", float),
    "amb_handover": ("MCI_AMB_HANDOVER", float),
    "uav_handover": ("MCI_UAV_HANDOVER", float),
}
# 평가기 규약 노브 — `v17_rule_eval.worker` 와 **같은 값**이어야 규칙 평가 배관이 동일하다.
FIXED_KNOBS = {
    "MCI_CAP_GATE": "occ",
    "MCI_OBS_VARIANT": "essential+load+valid",
    "MCI_H_PAD": "47",
    "MCI_REWARD_MODE": "woG",
}

# ── 후보 격자 기본값 (코드 상수로 명시; CLI 로 덮어쓸 수 있다) ──────────────────
# λ 는 v20/v21 에서 Q18 이 최강 교사와 실용적 동률이었던 지점을 가운데 두고 로그 간격.
DEFAULT_LAM = (8.0, 12.0, 18.0, 26.0, 36.0)
DEFAULT_WAIT_SCALE = (0.4, 1.0, 1.25)      # S족 — 1.0 이 이론값
DEFAULT_RED_GAIN = (6.6,)                  # v20 채택값. 수단 축은 이번 기본에서 고정
DEFAULT_YHOLD = (0.0, 2.0, 8.0)            # 등급 축 — v20 에서 0 이 아님이 확인된 축
DEFAULT_FAMILIES = ("Q", "P", "S")
FAMILY_DESC = {
    "Q": "시간축 CARD-T · 부하 = 입원 census + 이송 중 (병원 통신 필요)",
    "P": "시간축 CARD-T · 부하 = 내가 보낸 누적 − 수술실수 (통신 불요)",
    "S": "무튜닝 CARD-S · 치료개시 시각의 생존확률 직접 최대화",
}
# 기준 카드 — AGENTS.md 권장(P18, 통신 불요). 외부 시드에서 선택 결과와 paired 비교한다.
DEFAULT_REFERENCE = "P18=cardt:18,6.6,0,hingerate_psent"

DEFAULT_INNER_SEED0 = 0
DEFAULT_OUTER_SEED0 = 1000
SNAP_WARN_KM = 5.0


# ════════════════════════════════════════════════════════════════════ 계측 유틸
def _times_children() -> float:
    """자식 프로세스 CPU 시간(초). `os.times()` 는 **reap 된** 자식만 센다.

    부모의 `time.process_time()` 에는 워커 CPU 가 안 잡히므로 T5 의 실제 계산비용은
    이 값(또는 워커가 돌려준 `cpu_ms` 합)으로만 읽을 수 있다.
    """
    t = os.times()
    return float(t.children_user + t.children_system)


def _proc_age_s() -> float | None:
    """프로세스 나이(초) — 인터프리터 기동까지 포함한 진짜 시작 이후 경과.

    모듈 최상단 타이머는 "첫 파이썬 줄" 이후만 잡으므로 기동 자체가 빠진다.
    Linux `/proc` 으로 그걸 메운다. 실패하면 None(추정 불가로 남긴다).
    """
    try:
        with open("/proc/self/stat", "rb") as fh:
            fields = fh.read().rsplit(b")", 1)[1].split()
        start_ticks = float(fields[19])                    # field 22 (1-based)
        hz = float(os.sysconf("SC_CLK_TCK"))
        with open("/proc/uptime", encoding="ascii") as fh:
            uptime = float(fh.read().split()[0])
        return uptime - start_ticks / hz
    except Exception:
        return None


class Timers:
    """단계별(T1~T6) 독립 타이머. 벽시계·부모 CPU·자식 CPU 를 모두 잡는다."""

    def __init__(self) -> None:
        self.stages: dict[str, dict] = {}
        self._order: list[str] = []

    def stage(self, name: str):
        return _StageCtx(self, name)

    def _record(self, name: str, wall: float, cpu: float, ch: float) -> None:
        if name in self.stages:                            # 재진입(halving 라운드 등) 누적
            s = self.stages[name]
            s["wall_s"] += wall
            s["cpu_s"] += cpu
            s["cpu_children_s"] += ch
            s["n_calls"] += 1
        else:
            self.stages[name] = {"wall_s": wall, "cpu_s": cpu,
                                 "cpu_children_s": ch, "n_calls": 1}
            self._order.append(name)

    def dump(self) -> dict:
        return {k: self.stages[k] for k in self._order}


class _StageCtx:
    def __init__(self, owner: Timers, name: str) -> None:
        self.owner, self.name = owner, name

    def __enter__(self):
        self.w0, self.c0, self.h0 = time.perf_counter(), time.process_time(), _times_children()
        return self

    def __exit__(self, *exc):
        self.owner._record(self.name, time.perf_counter() - self.w0,
                           time.process_time() - self.c0, _times_children() - self.h0)
        return False


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout.strip()


def _git_provenance() -> dict:
    try:
        sha = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain")
        return {
            "sha": sha,
            "short_sha": sha[:9],
            "dirty": bool(dirty),
            "dirty_paths": [ln[3:] for ln in dirty.splitlines()][:40],
            "evaluator_path": "src/rl_src/v17_rule_eval.py",
            "evaluator_commit": _git("log", "-1", "--format=%H", "--",
                                     "src/rl_src/v17_rule_eval.py"),
        }
    except Exception as exc:                               # git 없는 환경에서도 죽지 않는다
        return {"error": str(exc)[:200]}


# ═══════════════════════════════════════════════════════════ T1 입력 파싱·검증
class Situation:
    """t=0 에 현장 지휘소가 아는 값만. 미지정 필드는 None = 노브 unset(config 기본값)."""

    FIELDS = tuple(KNOB_MAP) + ("lat", "lon")

    def __init__(self, **kw) -> None:
        for f in self.FIELDS:
            setattr(self, f, kw.get(f))

    def as_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.FIELDS}

    def knobs(self) -> dict[str, str]:
        """지정된 것만 노브로 번역한다. 미지정은 아예 키를 만들지 않는다."""
        out = {}
        for field, (var, cast) in KNOB_MAP.items():
            val = getattr(self, field)
            if val is None:
                continue
            out[var] = str(cast(val)) if cast is int else repr(float(val))
        return out


def parse_situation(args) -> Situation:
    """JSON 파일 + CLI 인자 → Situation. CLI 가 JSON 을 덮어쓴다."""
    data: dict = {}
    if args.situation:
        path = Path(args.situation)
        if not path.exists():
            raise SystemExit(f"[T1] 상황 JSON 없음: {path}")
        raw = json.load(open(path, encoding="utf-8"))
        if not isinstance(raw, dict):
            raise SystemExit("[T1] 상황 JSON 은 객체(dict)여야 한다")
        unknown = set(raw) - set(Situation.FIELDS)
        if unknown:
            raise SystemExit(f"[T1] 상황 JSON 미지 키: {sorted(unknown)} "
                             f"(허용: {sorted(Situation.FIELDS)})")
        data.update(raw)
    for f in Situation.FIELDS:
        v = getattr(args, f, None)
        if v is not None:
            data[f] = v

    # 등급 분포 노브는 시뮬에 아직 없다 — 조용히 무시하면 "넣었는데 반영 안 된" 카드가 나온다.
    if getattr(args, "grade_ratio", None) is not None:
        raise SystemExit(
            "[T1] --grade_ratio 미구현: 환자 등급 비율은 patient_info.csv 의 `ratio` 열에서만"
            " 오고 ScenarioManager 에 런타임 오버라이드 노브가 없다(MCI_* 목록 확인)."
            " 등급 분포를 흔들려면 시뮬에 노브를 먼저 추가해야 한다 — 이번 범위 밖.")

    if data.get("lat") is None or data.get("lon") is None:
        raise SystemExit("[T1] --lat/--lon (또는 상황 JSON 의 lat/lon) 필수")
    lat, lon = float(data["lat"]), float(data["lon"])
    if not (32.0 <= lat <= 40.0 and 124.0 <= lon <= 132.5):
        raise SystemExit(f"[T1] 좌표가 한국 범위 밖: ({lat}, {lon}) — 위/경도 뒤바뀜 여부 확인")
    for f in ("patients", "amb", "uav"):
        if data.get(f) is not None:
            v = int(data[f])
            if v < 0:
                raise SystemExit(f"[T1] --{f} 는 음수 불가: {v}")
            if f == "patients" and v == 0:
                raise SystemExit("[T1] --patients 0 은 시뮬이 정의되지 않는다")
            data[f] = v
    for f in ("treat_scale", "capa_scale", "amb_kmh", "uav_kmh"):
        if data.get(f) is not None and float(data[f]) <= 0:
            raise SystemExit(f"[T1] --{f} 는 0보다 커야 한다: {data[f]}")
    for f in ("amb_handover", "uav_handover"):
        if data.get(f) is not None and float(data[f]) < 0:
            raise SystemExit(f"[T1] --{f} 는 음수 불가: {data[f]}")
    if data.get("amb") == 0 and data.get("uav") == 0:
        raise SystemExit("[T1] AMB·UAV 둘 다 0 이면 이송 수단이 없다")
    return Situation(**{k: data.get(k) for k in Situation.FIELDS})


def apply_knobs(knobs: dict[str, str]) -> dict:
    """관리 대상 `MCI_*` **전량 unset** 후 지정분만 export.

    `agent_docs/operations.md` 경고 — "실행 전 의도치 않은 상속 물리 노브가 없는지
    확인한다". 상속 오염은 조용하다(예: 셸에 남은 `MCI_TREAT_SCALE=4` 가 카드를 바꾼다).
    그래서 확인이 아니라 **삭제**하고, 무엇을 지웠는지 산출에 남긴다.
    """
    dropped = {k: v for k, v in os.environ.items() if k.startswith("MCI_")}
    for k in list(dropped):
        del os.environ[k]
    os.environ.update(FIXED_KNOBS)
    os.environ.update(knobs)
    return dropped


# ═════════════════════════════════════════════════════════════ T2 시나리오 확보
def _haversine_km(lat1: float, lon1: float, lat2, lon2) -> np.ndarray:
    """구면 대권거리(km). 벡터화 — 원장 7,500 점을 한 번에 잰다."""
    r = 6371.0088
    p1, p2 = math.radians(lat1), np.radians(np.asarray(lat2, float))
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2, float) - lon1)
    a = np.sin(dp / 2.0) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _split_of(key: str, cache: dict = {}) -> str:
    """스냅된 점이 속한 연구 분할 이름(없으면 'none')."""
    if not cache:
        for name, path in SPLIT_MANIFESTS.items():
            cache[name] = set(json.load(open(path, encoding="utf-8"))) if path.exists() else set()
    return next((n for n, keys in cache.items() if key in keys), "none")


def snap_point(lat: float, lon: float, allow_test750: bool,
               ledger: Path = POINTS, manifest: Path = MANIFEST) -> dict:
    """사고 좌표를 사전생성 좌표 원장의 **최근접 점**으로 스냅.

    ★ 설계 의도: 현장에서 OSRM 라우팅을 하지 않는다. 거리행렬 생성은 좌표당 수백~수천
    번의 라우팅 호출이 필요해 네트워크·수 분이 든다. 원장의 7,500 점은 그 행렬을 이미
    갖고 있으므로 스냅 한 번으로 T2 가 밀리초로 끝나고 **오프라인 지휘차에서도 돈다**.
    대가는 스냅 오차이므로 거리(km)를 산출에 반드시 기록하고 임계 초과 시 경고한다.

    ⚠️ 원장 좌표는 시나리오 생성 시 OSRM 이 도로로 한 번 더 스냅한 적이 있다
    (`snap_m`). AMB 는 그 스냅점, UAV 는 원좌표를 쓰므로 UAV 이득이 약간 과소평가된다
    (기지 결함, 2026-08 기록). 그 값도 같이 남겨 사후에 보정할 수 있게 한다.
    """
    pts = json.load(open(ledger, encoding="utf-8"))
    man = json.load(open(manifest, encoding="utf-8"))
    keys = [k for k in pts if k in man]
    if not keys:
        raise SystemExit(f"[T2] 원장과 매니페스트의 교집합이 비었다: {ledger}, {manifest}")
    lats = np.array([pts[k]["lat"] for k in keys], float)
    lons = np.array([pts[k]["lon"] for k in keys], float)
    d = _haversine_km(lat, lon, lats, lons)
    i = int(np.argmin(d))
    key, snap_km = keys[i], float(d[i])
    split = _split_of(key)
    if split == FORBIDDEN_SPLIT and not allow_test750:
        raise SystemExit(
            f"[T2] 최근접 점 {key} 가 판정셋({FORBIDDEN_SPLIT})이다 — AGENTS.md 계약 1에 따라"
            " 기본 차단한다(판정 좌표의 PDR 을 읽으면 그 자체가 선택누수).\n"
            "      개발/검증이면 budget750 좌표를 쓰고, 실제 현장 배치라면"
            " --allow_test750 을 명시하라.")
    cfg = man[key]
    if not Path(cfg).exists():
        # ★ Pool(initializer=...) 는 초기화가 실패하면 워커를 무한 재생성하며 매달린다.
        #    워커에서 죽을 원인은 부모에서 미리 걸러야 한다.
        raise SystemExit(f"[T2] 시나리오 config 없음: {cfg} (원장 키 {key})")
    return {
        "query_lat": lat, "query_lon": lon,
        "key": key, "name": pts[key].get("name"), "sido": pts[key].get("sido"),
        "point_lat": float(pts[key]["lat"]), "point_lon": float(pts[key]["lon"]),
        "snap_km": snap_km,
        "snap_warn": snap_km > SNAP_WARN_KM,
        "snap_warn_km": SNAP_WARN_KM,
        "ledger_osrm_snap_m": pts[key].get("snap_m"),
        "research_split": split,
        "config": cfg,
        "ledger": str(ledger), "manifest": str(manifest),
        "n_ledger_points": len(keys),
    }


# ══════════════════════════════════════════════════════════ T4 후보 집합 생성
def _num(x: float) -> str:
    return f"{x:g}"


def build_candidates(families, lams, wait_scales, red_gains, yholds) -> list[tuple[str, str]]:
    """(이름, 스펙) 목록. 스펙 문법은 `v17_rule_eval.build_rule_policies` 정본을 그대로 쓴다.

      Q족 `cardt:<λ>,<red_gain>,<yhold>,hingerate`          (부하 = occ + in_flight)
      P족 `cardt:<λ>,<red_gain>,<yhold>,hingerate_psent`    (통신 불요)
      S족 `cards:<w>,<red_gain>,<yhold>`                    (무튜닝)

    2임계 밴드(`cardt2:`)는 별도 작업 소유이므로 여기서 건드리지 않는다.
    이름은 가능하면 `tools/v20_threshold_report.ARM_RE`(`^[KTHLQS][0-9.]+$`) 와 같은
    꼴로 만든다(Q18·P26·S1.25). 등급/수단 축을 움직이면 접미사가 붙어 그 정규식 밖이 된다.
    """
    fams = [f.strip().upper() for f in families]
    bad = [f for f in fams if f not in ("Q", "P", "S")]
    if bad:
        raise SystemExit(f"[T4] 미지 가족: {bad} (허용 Q·P·S)")
    out: list[tuple[str, str]] = []
    for fam in fams:
        firsts = wait_scales if fam == "S" else lams
        for a in firsts:
            for rg in red_gains:
                for yh in yholds:
                    name = f"{fam}{_num(a)}"
                    if rg != DEFAULT_RED_GAIN[0]:
                        name += f"G{_num(rg)}"
                    if yh != 0.0:
                        name += f"Y{_num(yh)}"
                    body = (f"cards:{_num(a)},{_num(rg)},{_num(yh)}" if fam == "S"
                            else f"cardt:{_num(a)},{_num(rg)},{_num(yh)},"
                                 f"{'hingerate_psent' if fam == 'P' else 'hingerate'}")
                    out.append((name, body))
    dup = {n for n, _ in out if [x for x, _ in out].count(n) > 1}
    if dup:
        raise SystemExit(f"[T4] 후보 이름 충돌: {sorted(dup)} — 격자 값이 중복됐다")
    return out


def parse_reference(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise SystemExit(f"[T4] --reference 는 '이름=스펙' 꼴: {spec}")
    name, body = spec.split("=", 1)
    return name, body


# ═════════════════════════════════════════════════════════ T3·T5 워커 (샤딩 본체)
_W: dict = {}          # 워커 프로세스 전역 캐시 — env 는 프로세스당 1회만 만든다


def _worker_init(cfg: str, knobs: dict, specs: dict, warm_seed: int,
                 ready=None) -> None:
    """워커 1회 초기화 = T3. env cold 비용을 여기서 물고 `_W` 에 기록해 첫 결과와 함께 올린다.

    ★ 노브 re-assert: `Pool` 은 fork 라 부모의 환경변수가 상속되지만, 상속을 신뢰하지 않고
    워커 안에서도 purge + 재설정한다. `ScenarioManager` 는 **env 생성 시점**에 노브를 읽으므로
    이 순서(노브 → import → env 생성)가 뒤집히면 노브가 조용히 무시된다.

    `ready` 는 준비 완료 카운터(`multiprocessing.Value`)다. 부모가 이걸 폴링해서 T3 를
    끝내므로 cold 비용이 T5 벽시계로 새지 않는다. **워커를 블록시키지 않는다** —
    `Barrier` 를 쓰면 초기화가 실패한 워커 때문에 `Pool` 이 재생성 루프에 매달린다.
    """
    w0, c0 = time.perf_counter(), time.process_time()
    for k in [k for k in os.environ if k.startswith("MCI_")]:
        del os.environ[k]
    os.environ.update(FIXED_KNOBS)
    os.environ.update(knobs)
    for v in BLAS_VARS:
        os.environ[v] = "1"

    from v17_rule_eval import build_rule_policies, rollout
    from viper_distill import _suppress_stdout, make_feature_env

    with _suppress_stdout():
        factory = make_feature_env(cfg, None)
        factory(seed=warm_seed)                 # env 실제 빌드(cold). 모든 워커가 같은 시드로
        pols = dict(build_rule_policies([f"{n}={b}" for n, b in specs.items()]))
    _W.update(factory=factory, policies=pols, rollout=rollout,
              suppress=_suppress_stdout, cold_wall_s=time.perf_counter() - w0,
              cold_cpu_s=time.process_time() - c0, cold_reported=False,
              knob_sig=json.dumps(knobs, sort_keys=True), pid=os.getpid())
    if ready is not None:
        try:
            with ready.get_lock():
                ready.value += 1
        except Exception:                                  # 계측 실패가 실행을 막지 않는다
            pass


def _worker(job):
    """샤드 1개 = [(후보이름, 시드), ...] 를 그 워커의 warm env 로 돌린다."""
    shard_id, pairs, knob_sig = job
    try:
        if not _W:
            raise RuntimeError("워커 초기화 안 됨(initializer 누락)")
        if _W["knob_sig"] != knob_sig:
            raise RuntimeError(f"노브 불일치: worker={_W['knob_sig']} master={knob_sig}")
        rows = []
        with _W["suppress"]():
            for name, seed in pairs:
                pol = _W["policies"][name]
                (reward, pdr, sim_time, n_dec, ms,
                 wall_ms, cpu_ms) = _W["rollout"](_W["factory"], pol, int(seed))
                rows.append({"policy": name, "episode": int(seed), "seed": int(seed),
                             "reward_woG": reward, "pdr_woG": pdr, "sim_time": sim_time,
                             "n_decisions": n_dec, "ms_per_decision": ms,
                             "wall_ms": wall_ms, "cpu_ms": cpu_ms})
        cold = None
        if not _W["cold_reported"]:                        # 워커당 한 번만 올린다
            cold = {"pid": _W["pid"], "wall_s": _W["cold_wall_s"], "cpu_s": _W["cold_cpu_s"]}
            _W["cold_reported"] = True
        return {"ok": True, "shard": shard_id, "rows": rows, "cold": cold}
    except Exception as exc:
        import traceback
        return {"ok": False, "shard": shard_id,
                "err": (str(exc) + traceback.format_exc())[:1500]}


def shard_pairs(pairs: list[tuple[str, int]], n_shards: int) -> list[list]:
    """(후보, 시드) 격자를 라운드로빈으로 n_shards 등분.

    ★ 지역 단위가 아니라 이 격자를 쪼개는 것이 이 드라이버의 존재 이유다. 라운드로빈은
    후보·시드를 샤드에 골고루 흩어 뒤쪽 후보만 무거운 편중을 막는다. 샤드 수는
    `min(workers, len(pairs))` 이고, 라운드마다 **워커당 정확히 1잡**을 던져
    (persistent Pool 이므로) env 재빌드가 라운드 수만큼 늘지 않게 한다.
    """
    n = max(1, min(n_shards, len(pairs)))
    buckets: list[list] = [[] for _ in range(n)]
    for i, pr in enumerate(pairs):
        buckets[i % n].append(pr)
    return [b for b in buckets if b]


class Runner:
    """(후보 × 시드) 평가기 — CRN 캐시 + 예산 감시 + 샤딩 실행을 한 곳에 묶는다."""

    def __init__(self, pool, n_workers: int, knob_sig: str, timers: Timers,
                 budget_sec: float | None, t_start: float) -> None:
        self.pool, self.n_workers, self.knob_sig = pool, n_workers, knob_sig
        self.timers, self.budget_sec, self.t_start = timers, budget_sec, t_start
        self.cache: dict[tuple[str, int], dict] = {}       # ★ CRN — (후보,시드) 결과 재사용
        self.cold: list[dict] = []
        self.episodes = 0
        self.budget_exhausted = False
        self.workers_ready = 0

    def warm(self, ready, timeout: float = 900.0) -> None:
        """T3 배리어 — **모든 워커의 초기화(env 생성) 완료**를 준비 카운터로 기다린다.

        `Pool(initializer=...)` 은 프로세스 기동과 함께 초기화를 시작하지만 `Pool()` 호출
        자체는 즉시 돌아온다. 배리어 없이 재면 T3 가 거의 0 이 되고 env cold 비용이
        첫 T5 라운드 벽시계에 섞인다.

        빈 잡을 던지는 방식은 **틀린다** — 먼저 준비된 워커가 즉시 끝나는 빈 잡을 전부
        가로채므로(실측: 4워커 중 1워커만 응답) 나머지 워커의 cold 를 기다리지 못한다.
        그래서 워커가 초기화 끝에 올리는 카운터를 부모가 폴링한다. 워커를 블록시키지
        않으므로 초기화 실패 시 `Pool` 재생성 루프에 매달리는 위험도 없다.
        cold 비용 자체는 각 워커의 첫 **실제** 잡에 실려 T5 에서 전량 수거된다.
        """
        t0 = time.perf_counter()
        while int(ready.value) < self.n_workers and time.perf_counter() - t0 < timeout:
            time.sleep(0.005)
        self.workers_ready = int(ready.value)
        if self.workers_ready < self.n_workers:
            print(f"  ⚠️ [T3] 준비된 워커 {self.workers_ready}/{self.n_workers}"
                  f" (timeout {timeout}s) — cold 일부가 T5 벽시계에 섞인다", flush=True)

    def elapsed(self) -> float:
        return time.perf_counter() - self.t_start

    def over_budget(self) -> bool:
        if self.budget_sec is None:
            return False
        if self.elapsed() >= self.budget_sec:
            self.budget_exhausted = True
        return self.budget_exhausted

    def run(self, names: list[str], seeds: list[int], stage: str = "T5_rollout") -> int:
        """캐시에 없는 (후보,시드) 만 평가. 반환 = 새로 돌린 에피소드 수."""
        pairs = [(n, s) for n in names for s in seeds if (n, s) not in self.cache]
        if not pairs:
            return 0
        with self.timers.stage(stage):
            jobs = [(i, shard, self.knob_sig)
                    for i, shard in enumerate(shard_pairs(pairs, self.n_workers))]
            for res in self.pool.imap_unordered(_worker, jobs, chunksize=1):
                if not res["ok"]:
                    raise RuntimeError(f"샤드 {res['shard']} 실패: {res['err']}")
                if res["cold"]:
                    self.cold.append(res["cold"])
                for row in res["rows"]:
                    pdr = float(row["pdr_woG"])
                    if not np.isfinite(pdr) or not 0.0 <= pdr <= 1.0:
                        raise RuntimeError(f"PDR 오류: {row['policy']}/{row['seed']}={pdr}")
                    self.cache[(row["policy"], int(row["seed"]))] = row
        self.episodes += len(pairs)
        return len(pairs)

    def mean(self, name: str, seeds: list[int]) -> float:
        return float(np.mean([self.cache[(name, s)]["pdr_woG"] for s in seeds]))

    def vec(self, name: str, seeds: list[int]) -> np.ndarray:
        return np.array([self.cache[(name, s)]["pdr_woG"] for s in seeds], float)

    def have(self, name: str, seeds: list[int]) -> bool:
        return all((name, s) in self.cache for s in seeds)

    def rows(self, region: str, seeds: list[int] | None = None) -> list[dict]:
        """v17 스키마(13열) 그대로.

        내부/외부 시드가 서로소라 `seed` 열만으로 국면(선택용/평가용)이 구분되므로
        열을 새로 만들지 않는다 — 기존 v17~v21 전수평가 CSV 와 그대로 조인된다.
        `episode` 는 v17 에서 0..n_eps-1 인덱스지만 여기서는 시드 집합이 연속이 아니라
        **시드 값 자체**를 넣는다(같은 (region,policy,episode,seed) 유일성은 유지).
        """
        keep = None if seeds is None else set(seeds)
        out = []
        for (name, seed), row in sorted(self.cache.items()):
            if keep is not None and seed not in keep:
                continue
            out.append({"region": region, "info_level": "RULE", "complexity": "-", **row})
        return out


# ═════════════════════════════════════════════════════════════ 탐색 알고리즘
def search_grid(runner: Runner, names: list[str], seeds: list[int],
                seed_block: int) -> dict:
    """전 후보 × 내부 시드 전수. 시드 블록 단위로 돌려 anytime 성질을 확보한다.

    블록이 **시드 축**인 이유: 중간에 예산이 끊겨도 "모든 후보가 같은 시드 집합을 공유"
    (CRN)라는 직사각 구조가 깨지지 않는다. 후보 축으로 잘랐다면 일부 후보만 시드가
    많아져 paired 비교가 성립하지 않는다.
    """
    blocks = [seeds[i:i + max(1, seed_block)] for i in range(0, len(seeds), max(1, seed_block))]
    used: list[int] = []
    log = []
    for bi, blk in enumerate(blocks):
        if runner.over_budget():
            break
        ep = runner.run(names, blk)
        used.extend(blk)
        log.append({"block": bi, "seeds": list(blk), "episodes": ep,
                    "elapsed_s": runner.elapsed()})
    if not used:
        raise RuntimeError("예산이 첫 블록도 못 돌 만큼 작다 — --budget_sec 를 늘려라")
    return {"mode": "grid", "seeds_used": used, "alive": list(names), "log": log}


def search_halving(runner: Runner, names: list[str], seeds: list[int],
                   init_seeds: int) -> dict:
    """CRN paired successive halving.

    라운드마다 시드를 2배로 늘리고 하위 절반을 탈락시킨다. 모든 후보가 같은 시드 접두사를
    공유하므로(CRN) 라운드 안의 순위 비교는 paired 다. 캐시가 (후보,시드) 단위라
    라운드가 넘어가도 이전 시드는 다시 돌지 않는다.

    `init_seeds == len(seeds)` 면 1라운드에서 이미 전 시드를 쓰므로 순위가 grid 와
    같아지고 **같은 최선**이 나온다(배선 게이트가 이 성질을 쓴다).
    """
    alive = list(names)
    s = max(1, min(int(init_seeds), len(seeds)))
    rounds = []
    while True:
        if runner.over_budget():
            break
        prefix = seeds[:s]
        ep = runner.run(alive, prefix)
        scored = sorted(alive, key=lambda n: (runner.mean(n, prefix), n))
        keep = scored if len(alive) == 1 else scored[:max(1, len(alive) // 2)]
        rounds.append({"round": len(rounds), "n_alive": len(alive), "n_seeds": len(prefix),
                       "episodes": ep, "elapsed_s": runner.elapsed(),
                       "best": scored[0], "best_pdr": runner.mean(scored[0], prefix),
                       "eliminated": [n for n in scored if n not in keep]})
        alive = keep
        if len(alive) == 1:
            if s >= len(seeds):
                break
            s = len(seeds)                                 # 생존자 1명 → 전 시드로 확정 평가
            continue
        s = min(len(seeds), s * 2)                         # 시드 2배 · 후보 절반
    if not rounds:
        raise RuntimeError("예산이 1라운드도 못 돌 만큼 작다 — --budget_sec 를 늘려라")
    return {"mode": "halving", "seeds_used": seeds[:s], "alive": alive, "log": rounds}


# ══════════════════════════════════════════════════════════════ T6 선택·통계·카드
def _paired_over_seeds(rows: list[dict], region: str, a_name: str, b_name: str) -> dict:
    """`tools/v20_threshold_report.py` 의 `cube`/`paired` 재사용. **paired 단위 = 시드**.

    그 모듈의 규약은 지역×시드 행렬이고 `paired` 는 행(=지역)마다 시드평균을 낸 뒤
    지역 축으로 CI 를 낸다. 현장 사례는 **좌표가 1개**라 그대로 쓰면 표본크기 1 →
    `ci()` 가 0.0 을 돌려준다(무의미). 여기서는 좌표가 고정이고 불확실성의 원천이
    시드이므로 **행렬을 전치**해 시드를 행으로 넘긴다. 그러면
      `paired` 의 delta = 시드별 차이의 평균, ci95 = 그 차이의 95%CI,
      `wtl` 의 W/T/L = 시드별 부호 계수(행 길이 1이라 `ci()`=0 → 부호 그대로)
    가 되고, 정의는 원 모듈 함수 그 자체이므로 W/T/L 정의가 갈리지 않는다.
    """
    from v20_threshold_report import cube, paired
    df = pd.DataFrame(rows)
    a = cube(df, a_name)
    b = cube(df, b_name)
    if a.shape[0] != 1 or b.shape[0] != 1:
        raise RuntimeError(f"현장 어시스트는 좌표 1개 전제 (region={region}, shape={a.shape})")
    res = paired(a.T, b.T)
    res["unit"] = "seed"
    res["n_seeds"] = int(a.shape[1])
    res["note"] = ("b−a. a=후보, b=기준. CI 는 시드별 paired 차이의 95%CI 이며 "
                   "판정선은 하드코딩하지 않았다(0 을 포함하면 동률).")
    return res


def render_card(name: str, spec: str, situation: Situation, snap: dict,
                stats: dict) -> str:
    """채워진 규칙집 1장. 3단 구조는 `v17_field_rules.py` 의 정본 docstring 을 따른다."""
    fam = name[0]
    toks = spec.split(":", 1)[1].split(",")
    a, rg, yh = float(toks[0]), float(toks[1]), float(toks[2])
    load_desc = {"Q": "입원 census + 이송 중 (병원 통신 필요)",
                 "P": "내가 그 병원으로 보낸 누적 − 수술실수, 0 하한 (통신 불요)"}.get(fam)

    if fam == "S":
        dest = ("② 목적지 — 후보 병원마다 **치료개시 시각**을 추정해 생존확률이 가장 큰 곳.\n"
                f"      치료개시 = 지금 + 이송(분) + 인계(분) + 대기(분)\n"
                f"      대기 = {a:g} × 치료시간 × max(0, 부하+1−수술실수) ÷ 수술실수"
                f"{'   (1.0 = 이론값)' if a == 1.0 else ''}")
    else:
        dest = ("② 목적지 — 확정된 수단으로 다음이 **가장 작은** 병원.\n"
                f"      도달시간(분) + {a:g} × 부하(명)\n"
                f"      부하 = {load_desc}\n"
                f"      → 환자 1명이 도달시간 {a:g}분과 맞교환된다는 뜻")

    hold = ("Yellow 가 현장에 **한 명도 없고**" if yh == 0
            else f"현장 대기 Yellow 가 **{yh:g}명 이하**이고")
    sit = situation.as_dict()
    known = ", ".join(f"{k}={v}" for k, v in sit.items() if v is not None and k not in ("lat", "lon"))
    lines = [
        "┌─ 현장 규칙집 (MCI 환자 1명마다 이 순서로 결정) ─────────────────────────",
        f"│ 사고지점  {snap['query_lat']:.5f}, {snap['query_lon']:.5f}"
        f"  → 시나리오 {snap['key']} ({snap['name']}, {snap['sido']})",
        f"│ 스냅거리  {snap['snap_km']:.2f} km"
        + ("  ⚠️ 임계 초과 — 시나리오 인프라가 현장과 다를 수 있다" if snap["snap_warn"] else ""),
        f"│ 상황입력  {known or '(전부 시나리오 기본값)'}",
        f"│ 카드      {name}   [{FAMILY_DESC[fam]}]",
        "├──────────────────────────────────────────────────────────────────────",
        f"│ ① 등급 — {hold} UAV 가 현장에 있으면 **Red 를 먼저** 보낸다.",
        "│           그 밖에는 Yellow 를 먼저 보낸다.",
        "│ " + "\n│ ".join(
            (f"③ 수단 — 현장에 한 종류만 대기 중이면 그것. 둘 다 대기 중이면\n"
             f"      Red 는 (AMB 도달 − UAV 도달) > {rg:g}분 일 때 UAV, 아니면 AMB.\n"
             "      Yellow 는 항상 AMB.").splitlines()),
        "│ " + "\n│ ".join(dest.splitlines()),
        "│           (계산 순서는 등급 → 수단 → 목적지다. 목적지가 수단에 의존한다)",
        "├──────────────────────────────────────────────────────────────────────",
        f"│ 예상 PDR_woG  {stats['outer_mean']:.5f} ± {stats['outer_ci']:.5f}"
        f"   (외부 시드 {stats['n_outer']}개, 선택에 쓰지 않은 난수)",
        f"│ 기준 {stats['ref_name']}    {stats['ref_mean']:.5f} ± {stats['ref_ci']:.5f}"
        f"   → (기준−선택) {stats['delta']:+.5f} ± {stats['delta_ci']:.5f} "
        f"({stats['verdict']}, 시드 W/T/L {stats['wtl']})",
        f"│ 승자의 저주   내부 {stats['inner_mean']:.5f} → 외부 {stats['outer_mean']:.5f}"
        f"  ({stats['curse']:+.5f})",
        f"│ 계산          후보 {stats['n_cand']} · 에피소드 {stats['n_episodes']}"
        f" · 벽시계 {stats['wall_s']:.1f}s · 워커 {stats['n_workers']}"
        f" · loadavg {stats['loadavg']:.1f}",
        "└──────────────────────────────────────────────────────────────────────",
    ]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════ 게이트
GATES = ("workers", "determinism", "search", "knobs")


def run_gates(args) -> int:
    """배선 게이트 — 소규모 전용. 성능 판정이 아니라 **배관이 맞는지** 만 본다."""
    import copy
    wanted = GATES if args.gate == "all" else (args.gate,)
    base = copy.copy(args)
    base.gate, base.tag = None, None
    base.families, base.lam, base.wait_scale = ["Q", "P"], [12.0, 18.0], [1.0]
    base.red_gain, base.yhold = [6.6], [0.0]
    base.inner_seeds = base.outer_seeds = 2
    base.search, base.workers, base.seed_block = "grid", 1, 1
    base.quiet = True
    base.out_dir = str(Path(args.out_dir) / "gate")        # 본 산출 디렉터리를 안 더럽힌다

    def once(**kw):
        a = copy.copy(base)
        for k, v in kw.items():
            setattr(a, k, v)
        return run_once(a)

    print("[gate] 소규모 배선 게이트 — 후보 4 · 내부 2 · 외부 2", flush=True)
    fails = []
    r_w1 = once(workers=1)
    if "workers" in wanted:
        r_w4 = once(workers=4)
        same = (r_w1["selected"]["name"] == r_w4["selected"]["name"]
                and _cands_equal(r_w1, r_w4))
        print(f"[gate] 워커 1 vs 4 동일성: {'PASS' if same else 'FAIL'}"
              f"  ({r_w1['selected']['name']} / {r_w4['selected']['name']})", flush=True)
        if not same:
            fails.append("workers")
            for c1, c4 in zip(r_w1["candidates"], r_w4["candidates"]):
                if c1["inner_mean"] != c4["inner_mean"]:
                    print(f"         {c1['name']}: {c1['inner_mean']!r} vs {c4['inner_mean']!r}")
    if "determinism" in wanted:
        r_b = once(workers=1)
        same = r_b["selected"]["name"] == r_w1["selected"]["name"] and _cands_equal(r_w1, r_b)
        print(f"[gate] 결정론(같은 인자 2회): {'PASS' if same else 'FAIL'}", flush=True)
        if not same:
            fails.append("determinism")
    if "search" in wanted:
        # init_seeds = 전 시드 → halving 1라운드가 전수와 같은 데이터로 순위를 낸다
        r_h = once(search="halving", halving_init_seeds=base.inner_seeds, workers=4)
        same = r_h["selected"]["name"] == r_w1["selected"]["name"]
        print(f"[gate] grid vs halving 최선 일치: {'PASS' if same else 'FAIL'}"
              f"  (grid={r_w1['selected']['name']} halving={r_h['selected']['name']})", flush=True)
        if not same:
            fails.append("search")
    if "knobs" in wanted:
        r_k = once(patients=200, workers=2)
        knobs = r_k["scenario_knobs"]
        ok = knobs.get("MCI_INCIDENT_SIZE") == "200"
        nd = float(np.mean([c["n_decisions_mean"] for c in r_k["candidates"]]))
        nd0 = float(np.mean([c["n_decisions_mean"] for c in r_w1["candidates"]]))
        print(f"[gate] 노브 적용(--patients 200): {'PASS' if ok else 'FAIL'}"
              f"  scenario_knobs={knobs}  결정수 {nd0:.0f} → {nd:.0f}", flush=True)
        if not ok:
            fails.append("knobs")
        if not nd > nd0:
            print("         ⚠️ 환자를 늘렸는데 결정 수가 안 늘었다 — 노브가 물리에 안 닿았을 수 있다")
    print(f"[gate] {'전부 PASS' if not fails else 'FAIL: ' + ','.join(fails)}", flush=True)
    return 1 if fails else 0


def _cands_equal(a: dict, b: dict) -> bool:
    ka = {c["name"]: (c["inner_mean"], c["n_inner_seeds"]) for c in a["candidates"]}
    kb = {c["name"]: (c["inner_mean"], c["n_inner_seeds"]) for c in b["candidates"]}
    return ka == kb


# ═════════════════════════════════════════════════════════════════════ 파이프라인
def run_once(args) -> dict:
    t_start = time.perf_counter()
    cpu_start, ch_start = time.process_time(), _times_children()
    loadavg_start = list(os.getloadavg())
    timers = Timers()
    quiet = getattr(args, "quiet", False)

    def say(msg: str) -> None:
        if not quiet:
            print(msg, flush=True)

    # ── T1 ────────────────────────────────────────────────────────────────
    with timers.stage("T1_parse"):
        situation = parse_situation(args)
        knobs = situation.knobs()
        dropped = apply_knobs(knobs)
    say(f"[T1] 상황 {situation.as_dict()}\n     노브 {knobs}"
        + (f"\n     ⚠️ 상속 MCI_* {len(dropped)}개 제거: {sorted(dropped)}" if dropped else ""))

    # ── T2 ────────────────────────────────────────────────────────────────
    with timers.stage("T2_snap"):
        snap = snap_point(float(situation.lat), float(situation.lon),
                          allow_test750=bool(getattr(args, "allow_test750", False)))
    say(f"[T2] 스냅 {snap['key']} ({snap['name']}) · {snap['snap_km']:.3f} km"
        f" · 분할 {snap['research_split']}"
        + ("  ⚠️ 스냅 거리 임계 초과" if snap["snap_warn"] else ""))

    # ── T4 (후보) — T3 는 워커 초기화라서 T5 직전에 온다 ────────────────────
    with timers.stage("T4_candidates"):
        cands = build_candidates(args.families, args.lam, args.wait_scale,
                                 args.red_gain, args.yhold)
        ref_name, ref_body = parse_reference(args.reference)
        spec_map = dict(cands)
        if ref_name in spec_map and spec_map[ref_name] != ref_body:
            raise SystemExit(f"[T4] 기준 이름 {ref_name} 이 격자 후보와 스펙 충돌")
        spec_map[ref_name] = ref_body
        inner = list(range(args.inner_seed0, args.inner_seed0 + args.inner_seeds))
        outer = list(range(args.outer_seed0, args.outer_seed0 + args.outer_seeds))
        if set(inner) & set(outer):                        # ★ 통계 계약 1
            raise SystemExit(f"[T4] 내부·외부 시드가 겹친다: {sorted(set(inner) & set(outer))}"
                             " — 승자의 저주가 측정 불가능해진다")
        # `build_rule_policies` 로 스펙 문법을 부모에서 먼저 검증(워커에서 죽지 않게)
        from v17_rule_eval import build_rule_policies
        build_rule_policies([f"{n}={b}" for n, b in spec_map.items()])
    names = [n for n, _ in cands]
    est = len(names) * len(inner) + 2 * len(outer)          # grid 상한(halving 은 이보다 작다)
    say(f"[T4] 후보 {len(names)} · 내부시드 {inner[0]}..{inner[-1]} · "
        f"외부시드 {outer[0]}..{outer[-1]} · 기준 {ref_name} · 최대 {est} ep")
    if getattr(args, "plan_only", False):
        return {"plan_only": True, "n_candidates": len(names), "candidates": names,
                "episodes_max": est, "snap": snap, "scenario_knobs": dict(
                    sorted({k: v for k, v in os.environ.items()
                            if k.startswith("MCI_")}.items()))}

    n_workers = max(1, min(int(args.workers), len(names) * len(inner)))
    knob_sig = json.dumps(knobs, sort_keys=True)
    region = snap["key"]
    ready = Value("i", 0)
    pool = Pool(n_workers, initializer=_worker_init,
                initargs=(snap["config"], knobs, spec_map, args.inner_seed0, ready))
    try:
        runner = Runner(pool, n_workers, knob_sig, timers, args.budget_sec, t_start)

        # ── T3 (워커당 env 1회 — 배리어까지 이 단계에 넣어야 cold 가 T5 에 안 섞인다) ──
        with timers.stage("T3_env"):
            runner.warm(ready)
        say(f"[T3] 워커 {runner.workers_ready}/{n_workers} 준비 · "
            f"cold {timers.stages['T3_env']['wall_s']:.2f}s (env 생성 + 정책 빌드)")

        # ── T5 ────────────────────────────────────────────────────────────
        if args.search == "grid":
            search = search_grid(runner, names, inner, args.seed_block)
        else:
            init = args.halving_init_seeds or max(1, len(inner) // 4)
            search = search_halving(runner, names, inner, init)
        ep_search = runner.episodes
        seeds_sel = search["seeds_used"]
        say(f"[T5] 탐색 {search['mode']} · 에피소드 {ep_search}"
            f" · 선택시드 {len(seeds_sel)} · {runner.elapsed():.1f}s"
            + ("  ⚠️ 예산 소진" if runner.budget_exhausted else ""))

        # ── T6 ────────────────────────────────────────────────────────────
        with timers.stage("T6_select"):
            # 예산이 끊겼으면 **모든 생존 후보가 공통으로 가진** 시드 접두사에서만 고른다
            # (일부 후보만 시드가 많은 상태로 비교하면 CRN paired 가 깨진다).
            alive = [n for n in search["alive"] if runner.have(n, seeds_sel[:1])]
            common = [s for s in seeds_sel if all(runner.have(n, [s]) for n in alive)]
            if not common:
                raise RuntimeError("공통 시드가 없다 — 예산을 늘려라")
            ranked = sorted(alive, key=lambda n: (runner.mean(n, common), n))
            best = ranked[0]
            # 선택 후보의 내부 추정은 **전 내부 시드**로 맞춘다(승자의 저주 비교의 짝).
            # 예산이 끊긴 상태면 이 보강은 건너뛴다 — 있으면 좋은 진단이지 필수가 아니다.
            # (반대로 외부 시드 평가는 건너뛸 수 없다. 그게 보고하는 기대성능 자체다.
            #  즉 --budget_sec 는 **T5 탐색**을 제한하고, 후보 1~2개 × M 시드의 최종
            #  평가는 예산 밖의 고정비용이다.)
            ep_topup = 0 if runner.budget_exhausted else runner.run(
                [best], inner, stage="T6_topup")
            inner_seeds_best = inner if runner.have(best, inner) else common
            inner_mean = runner.mean(best, inner_seeds_best)

            # 외부 시드 = 평가 전용. 선택에 절대 쓰지 않는다.
            ep_outer = runner.run(sorted({best, ref_name}), outer, stage="T6_outer")
            rows = runner.rows(region)
            ov = runner.vec(best, outer)
            rv = runner.vec(ref_name, outer)
            from v20_threshold_report import ci as _ci
            # ⚠️ `cube()` 는 결측이 있으면 거부한다. 기준 카드가 격자 밖이면 내부 시드가
            #    없으므로 **외부 시드 행만** 넘겨야 직사각 행렬이 된다.
            pair = (_paired_over_seeds(runner.rows(region, outer), region, best, ref_name)
                    if best != ref_name else None)
            curse = float(ov.mean()) - inner_mean
            stats = {
                "outer_mean": float(ov.mean()), "outer_ci": _ci(ov), "n_outer": len(outer),
                "inner_mean": inner_mean, "curse": curse,
                "ref_name": ref_name, "ref_mean": float(rv.mean()), "ref_ci": _ci(rv),
                "delta": (pair or {}).get("delta", 0.0),
                "delta_ci": (pair or {}).get("ci95", 0.0),
                "wtl": (f"{pair['win']}/{pair['tie']}/{pair['loss']}" if pair else "-"),
                "verdict": ("기준과 동일 카드" if pair is None else
                            "유의하게 나음" if pair["delta"] > pair["ci95"] else
                            "유의하게 나쁨" if pair["delta"] < -pair["ci95"] else "동률"),
                "n_cand": len(names), "n_episodes": runner.episodes,
                "wall_s": runner.elapsed(), "n_workers": n_workers,
                "loadavg": os.getloadavg()[0],
            }
            card = render_card(best, spec_map[best], situation, snap, stats)
    finally:
        pool.close()
        pool.join()

    cold_w = [c["wall_s"] for c in runner.cold]
    cold_c = [c["cpu_s"] for c in runner.cold]
    cand_rows = []
    for n in names:
        have = [s for s in inner if runner.have(n, [s])]
        if not have:
            continue
        v = runner.vec(n, have)
        nd = [runner.cache[(n, s)]["n_decisions"] for s in have]
        cand_rows.append({"name": n, "spec": spec_map[n], "n_inner_seeds": len(have),
                          "inner_mean": float(v.mean()), "inner_ci": _ci(v),
                          "n_decisions_mean": float(np.mean(nd)),
                          "eliminated_early": len(have) < len(inner)})
    cand_rows.sort(key=lambda r: (r["inner_mean"], r["name"]))

    result = {
        "schema_version": 1,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "MCI 현장 지휘 보조 — 상황 입력에서 CARD 파라미터를 폐루프로 선택",
        "git": _git_provenance(),
        "situation": situation.as_dict(),
        "scenario_knobs": dict(sorted({k: v for k, v in os.environ.items()
                                       if k.startswith("MCI_")}.items())),
        "inherited_mci_dropped": dropped,
        "snap": snap,
        "search": {
            "mode": search["mode"],
            "families": list(args.families), "lam": list(args.lam),
            "wait_scale": list(args.wait_scale), "red_gain": list(args.red_gain),
            "yhold": list(args.yhold),
            "n_candidates": len(names), "candidate_names": names,
            "inner_seeds": inner, "outer_seeds": outer,
            "seed_disjoint": True,
            "seeds_used_for_selection": common,
            "seed_block": args.seed_block,
            "halving_init_seeds": getattr(args, "halving_init_seeds", None),
            "episodes_search": ep_search, "episodes_topup": ep_topup,
            "episodes_outer": ep_outer, "episodes_total": runner.episodes,
            "budget_sec": args.budget_sec, "budget_exhausted": runner.budget_exhausted,
            "budget_note": ("--budget_sec 는 T5 탐색의 벽시계 상한이다. 소진되면 그 시점의"
                            " 공통 시드(모든 생존 후보가 가진 접두사)에서 최선을 고르고"
                            " 내부 보강을 건너뛴다. 외부 시드 최종평가는 후보 1~2개 × M"
                            " 시드의 고정비용이라 예산 밖에서 반드시 수행한다."),
            "log": search["log"],
            "crn_note": "한 실행의 모든 후보가 같은 내부 시드 집합을 공유(캐시가 (후보,시드) 단위)",
        },
        "candidates": cand_rows,
        "selected": {
            "name": best, "spec": spec_map[best],
            "inner_mean": inner_mean, "n_inner_seeds": len(inner_seeds_best),
            "outer_mean": stats["outer_mean"], "outer_ci": stats["outer_ci"],
            "n_outer_seeds": len(outer),
            "rank_runner_up": ranked[1] if len(ranked) > 1 else None,
        },
        "winners_curse": {
            "inner_mean": inner_mean, "outer_mean": stats["outer_mean"], "delta": curse,
            "note": ("외부 − 내부. 내부 시드로 고른 후보를 같은 시드로 평가하면 낙관되므로"
                     " 서로소 외부 시드로 다시 잰 차이다. 양수 = 내부 추정이 낙관적이었다."),
        },
        "reference": {"name": ref_name, "spec": ref_body,
                      "outer_mean": stats["ref_mean"], "outer_ci": stats["ref_ci"]},
        "paired_vs_reference_outer": pair,
        "environment": {
            "hostname": platform.node(), "python": sys.executable,
            "python_version": platform.python_version(),
            "n_logical_cores": os.cpu_count(),
            "n_workers": n_workers,
            "blas_threads_effective": {v: os.environ.get(v) for v in BLAS_VARS},
            "blas_threads_inherited": BLAS_INHERITED,
            "loadavg_start": loadavg_start, "loadavg_end": list(os.getloadavg()),
            "load_note": ("공유 노드. 아래 timing 의 벽시계는 **부하 하 참고값**이며 성능"
                          " 실측이 아니다. 코어 스케일링은 노드가 빈 뒤 따로 재야 한다."),
        },
        "timing": {
            "estimator": ("1차 = time.process_time() CPU 시간, 2차 = time.perf_counter()"
                          " 벽시계. 자식 CPU 는 os.times() children + 워커 cpu_ms 합."),
            "reference": "src/sim_src_upgrade/bench/bench_core.py:1-17",
            "process_age_s_at_main": _PROC_AGE_AT_MAIN,
            "import_s": _IMPORT_WALL_S,
            "stages": timers.dump(),
            "stages_note": ("`cpu_children_s` 는 `os.times()` 기반이라 **reap 된** 자식만"
                            " 센다. Pool 이 살아 있는 동안(T3·T5·T6)은 0 으로 나오므로"
                            " 워커 CPU 는 `rollout_cpu_ms_sum` 과 아래"
                            " `total_cpu_children_s`(pool.join 이후)로 읽어라."),
            "env_cold": {
                "n_workers_ready_at_T3": runner.workers_ready,
                "n_workers_reporting": len(cold_w),
                "wall_s": cold_w, "cpu_s": cold_c,
                "mean_wall_s": float(np.mean(cold_w)) if cold_w else None,
                "mean_cpu_s": float(np.mean(cold_c)) if cold_c else None,
                "note": ("워커당 1회 = env 생성 + 정책 빌드. Pool 을 지속시키므로 halving"
                         " 라운드가 늘어도 재지불하지 않는다. T3_env 단계는 이 초기화가"
                         " 전 워커에서 끝나는 것을 빈 잡 배리어로 기다린 시간이다."),
            },
            "rollout_cpu_ms_sum": float(sum(r["cpu_ms"] for r in rows)),
            "rollout_wall_ms_sum": float(sum(r["wall_ms"] for r in rows)),
            "rollout_cpu_ms_median": float(np.median([r["cpu_ms"] for r in rows])),
            "total_wall_s": time.perf_counter() - t_start,
            "total_cpu_s": time.process_time() - cpu_start,
            "total_cpu_children_s": _times_children() - ch_start,
        },
        "card_text": card,
    }

    tag = args.tag or _auto_tag(situation, snap, args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{tag}.json"
    if args.save_rows:
        out_csv = out_dir / f"{tag}.csv"
        from v17_rule_eval import COLS
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        result["rows_csv"] = str(out_csv)
        result["rows_csv_columns"] = list(COLS)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    result["output"] = str(out_json)
    say(card)
    say(f"[T6] → {out_json}")
    return result


def _auto_tag(situation: Situation, snap: dict, args) -> str:
    """결정적 태그 — 같은 인자로 두 번 돌리면 같은 파일에 덮어쓴다(카드가 흔들리면 안 된다)."""
    payload = json.dumps({"sit": situation.as_dict(), "search": args.search,
                          "fam": list(args.families), "lam": list(args.lam),
                          "ws": list(args.wait_scale), "rg": list(args.red_gain),
                          "yh": list(args.yhold), "in": args.inner_seeds,
                          "out": args.outer_seeds, "in0": args.inner_seed0,
                          "out0": args.outer_seed0}, sort_keys=True)
    h = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"assist_{snap['key']}_{args.search}_{h}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="v22 현장 지휘 보조 — 상황 숫자 → 채워진 CARD 규칙집 1장",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("상황 입력 (t=0 에 현장 지휘소가 아는 값만)")
    g.add_argument("--situation", help="상황 JSON 경로 (CLI 인자가 덮어쓴다)")
    g.add_argument("--lat", type=float, help="사고 위도")
    g.add_argument("--lon", type=float, help="사고 경도")
    g.add_argument("--patients", type=int, help="환자 총원 → MCI_INCIDENT_SIZE")
    g.add_argument("--amb", type=int, help="가용 AMB 대수 → MCI_AMB_NUM")
    g.add_argument("--uav", type=int, help="가용 UAV 대수 → MCI_UAV_NUM")
    g.add_argument("--treat_scale", type=float, help="병원 치료 지체 배수 → MCI_TREAT_SCALE")
    g.add_argument("--capa_scale", type=float, help="용량 배수 → MCI_CAPA_SCALE")
    g.add_argument("--amb_kmh", type=float, help="→ MCI_AMB_VELOCITY")
    g.add_argument("--uav_kmh", type=float, help="→ MCI_UAV_VELOCITY")
    g.add_argument("--amb_handover", type=float, help="→ MCI_AMB_HANDOVER")
    g.add_argument("--uav_handover", type=float, help="→ MCI_UAV_HANDOVER")
    g.add_argument("--grade_ratio", help="환자 등급 비율 — **미구현**(자리만 있다)")
    g.add_argument("--allow_test750", action="store_true",
                   help="스냅 결과가 판정셋이어도 진행(실제 현장 배치 전용)")

    c = p.add_argument_group("후보 공간")
    c.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES),
                   help=f"Q·P·S 부분집합 (기본 {list(DEFAULT_FAMILIES)})")
    c.add_argument("--lam", nargs="+", type=float, default=list(DEFAULT_LAM),
                   help=f"Q·P족 교환율 λ(분/명) 격자 (기본 {list(DEFAULT_LAM)})")
    c.add_argument("--wait_scale", nargs="+", type=float, default=list(DEFAULT_WAIT_SCALE),
                   help=f"S족 대기 배율 (기본 {list(DEFAULT_WAIT_SCALE)})")
    c.add_argument("--red_gain", nargs="+", type=float, default=list(DEFAULT_RED_GAIN),
                   help=f"UAV 전환 시간이득 임계(분) (기본 {list(DEFAULT_RED_GAIN)})")
    c.add_argument("--yhold", nargs="+", type=float, default=list(DEFAULT_YHOLD),
                   help=f"Red 우선 전환의 Yellow 대기 상한(명) (기본 {list(DEFAULT_YHOLD)})")
    c.add_argument("--reference", default=DEFAULT_REFERENCE,
                   help=f"외부 시드에서 비교할 기준 카드 (기본 {DEFAULT_REFERENCE})")

    s = p.add_argument_group("탐색·통계")
    s.add_argument("--search", choices=("grid", "halving"), default="grid")
    s.add_argument("--inner_seeds", type=int, default=10, help="선택용 시드 수 K")
    s.add_argument("--outer_seeds", type=int, default=10, help="평가용 시드 수 M")
    s.add_argument("--inner_seed0", type=int, default=DEFAULT_INNER_SEED0)
    s.add_argument("--outer_seed0", type=int, default=DEFAULT_OUTER_SEED0)
    s.add_argument("--seed_block", type=int, default=1,
                   help="grid 의 시드 블록 크기(예산 확인 주기)")
    s.add_argument("--halving_init_seeds", type=int, default=None,
                   help="halving 1라운드 시드 수 (기본 K//4). K 로 주면 grid 와 같은 순위")
    s.add_argument("--budget_sec", type=float, default=None,
                   help="벽시계 예산(초). 넘기면 안전 중단 후 그 시점 최선을 반환")

    r = p.add_argument_group("실행·산출")
    r.add_argument("--workers", type=int, default=8, help="Pool 워커 수 (샤드 수)")
    r.add_argument("--tag", default=None, help="산출 파일명(기본 = 인자 해시로 결정적 생성)")
    r.add_argument("--out_dir", default=str(OUT_DIR))
    r.add_argument("--save_rows", action="store_true", default=True,
                   help="에피소드 CSV(v17 13열 스키마)도 저장")
    r.add_argument("--no_save_rows", dest="save_rows", action="store_false")
    r.add_argument("--plan_only", action="store_true", help="후보·에피소드 수만 찍고 종료")
    r.add_argument("--quiet", action="store_true")
    r.add_argument("--gate", choices=GATES + ("all",), default=None,
                   help="소규모 배선 게이트만 실행")
    return p


def main() -> None:
    global _PROC_AGE_AT_MAIN, _IMPORT_WALL_S
    _PROC_AGE_AT_MAIN = _proc_age_s()
    _IMPORT_WALL_S = _T_MAIN_WALL - _T_MODULE_WALL0
    args = build_parser().parse_args()
    if args.gate:
        raise SystemExit(run_gates(args))
    run_once(args)


_T_MAIN_WALL = time.perf_counter()      # import 완료 시각 — main() 진입 전에 확정
_PROC_AGE_AT_MAIN: float | None = None
_IMPORT_WALL_S: float = 0.0

if __name__ == "__main__":
    main()
