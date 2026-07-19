"""v6 Track A5 스모크: FeatureMultiRegionEnv overrides(빌드시 노브 주입) + 랜덤화 매니페스트 배선 봉인.

학습·대규모 실행 없음 — env 몇 개만 순차 빌드해 배선을 검증한다(워커 불요). sim/wrapper print 억제.

체크:
  ① 전 지역 obs/action 차원 동일(essential+load+valid, H_pad47 → 402/192)
  ② 변형 env 의 unwrapped incident_size/amb_num/uav_num 이 overrides 와 일치 + 비오버라이드 축 보존
  ③ 빌드 후 os.environ 원상복구(주입 키 삭제) / ③b 변형 뒤 plain 지역 누수 카나리 /
     ③c 사전 존재 키의 복구(삭제 아님)
  ④ 구 스키마(문자열 값)만 빌드 시 기존과 동일 동작(402/192)
  ⑤ reset 반복 → 지역 샘플링 정상
  ⑥ gen_randomized_manifest.build_randomized: base500 → 1000 엔트리·스키마·재현성·기본조합 배제
  ⑦ 화이트리스트 밖 overrides 키 → ValueError / ⑧ amb=0 override → ValueError

실행:
  MCI_OBS_VARIANT=essential+load+valid MCI_H_PAD=47 MCI_CAP_GATE=occ \\
    /home/ryu/anaconda3/envs/UAV/bin/python src/rl_src/rand_env_smoke.py
"""
import contextlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

import gen_randomized_manifest as grm          # noqa: E402
from train_ppo_feature import FeatureMultiRegionEnv  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
BASE_MANIFEST = os.path.join(REPO, "scenarios/manifests/sigungu_mixed47_manifest.json")
OVERRIDE_KEYS = ["MCI_INCIDENT_SIZE", "MCI_AMB_NUM", "MCI_UAV_NUM", "MCI_CAPA_SCALE"]
EXP_OBS, EXP_ACT = (402,), 192


@contextlib.contextmanager
def _quiet():
    """sim/wrapper 의 stdout print 억제(체크 결과는 반드시 블록 밖에서 출력)."""
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        yield


def _write_tmp(d: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".json", prefix="rand_smoke_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    return path


def _build_raises(manifest: dict) -> bool:
    """미니 매니페스트 빌드가 ValueError 를 내면 True."""
    p = _write_tmp(manifest)
    try:
        with _quiet():
            FeatureMultiRegionEnv(p, seed=0)
        return False
    except ValueError:
        return True
    finally:
        os.remove(p)


def main():
    assert os.environ.get("MCI_OBS_VARIANT") == "essential+load+valid", \
        "MCI_OBS_VARIANT=essential+load+valid 필요"
    assert os.environ.get("MCI_H_PAD") == "47", "MCI_H_PAD=47 필요"
    print(f"[smoke] env: MCI_OBS_VARIANT={os.environ.get('MCI_OBS_VARIANT')} "
          f"MCI_H_PAD={os.environ.get('MCI_H_PAD')} "
          f"MCI_CAP_GATE={os.environ.get('MCI_CAP_GATE', '(occ default)')}")

    with open(BASE_MANIFEST, encoding="utf-8") as f:
        base = json.load(f)
    fixed_k = next(k for k in base if not k.endswith("_nat"))
    nat_k = next(k for k in base if k.endswith("_nat"))
    fixed_p, nat_p = base[fixed_k], base[nat_k]
    print(f"[smoke] fixed={fixed_k}  nat={nat_k}  (base {len(base)} 엔트리)")

    # 미니 매니페스트: plain(구 스키마) + 변형2. nat 을 변형 뒤에 둬 env 복구 누수 카나리로 사용.
    mini = {
        fixed_k: fixed_p,                                                       # plain (built 1st)
        "fixed_vA": {"path": fixed_p, "overrides": {"MCI_INCIDENT_SIZE": 200}},  # 변형 (2nd)
        "fixed_vB": {"path": fixed_p,
                     "overrides": {"MCI_AMB_NUM": 10, "MCI_UAV_NUM": 5, "MCI_CAPA_SCALE": 0.5}},  # 변형 (3rd)
        nat_k: nat_p,                                                            # plain (built LAST)
    }
    mini_path = _write_tmp(mini)

    # ③ 사전: 주입 키가 원래 환경에 없음 → 빌드 후에도 없어야 복구 증명.
    before = {k: os.environ.get(k) for k in OVERRIDE_KEYS}
    assert all(v is None for v in before.values()), f"스모크 전제 위반: 주입 키가 이미 설정됨 {before}"

    with _quiet():
        menv = FeatureMultiRegionEnv(mini_path, seed=0)
        pairs = list(zip(menv.regions, menv._envs))
        obs_dims = {r: tuple(e.observation_space.shape) for r, e in pairs}
        act_dims = {r: int(e.action_space.n) for r, e in pairs}
        knobs = {r: (int(e.unwrapped.incident_size), int(e.unwrapped.amb_num),
                     int(e.unwrapped.uav_num)) for r, e in pairs}
        seen, reset_ok = [], True
        for _ in range(20):
            obs, _info = menv.reset()
            seen.append(menv.current_region)
            if tuple(obs.shape) != EXP_OBS:
                reset_ok = False
                break
        menv.close()
    after = {k: os.environ.get(k) for k in OVERRIDE_KEYS}
    os.remove(mini_path)

    ok = True

    # ① 차원 동일 402/192
    d_ok = (set(obs_dims.values()) == {EXP_OBS} and set(act_dims.values()) == {EXP_ACT})
    ok &= d_ok
    print(f"{'[OK]' if d_ok else '[FAIL]'} ① 차원 동일: obs={set(obs_dims.values())} "
          f"act={set(act_dims.values())} (기대 {{{EXP_OBS}}}/{{{EXP_ACT}}})")

    # ② 변형 노브 일치 + 비오버라이드 축 보존
    base_inc, base_amb, base_uav = knobs[fixed_k]
    vA, vB = knobs["fixed_vA"], knobs["fixed_vB"]
    a_ok = (vA == (200, base_amb, base_uav))                       # incident 만 200, amb/uav 보존
    b_ok = (vB[0] == base_inc and vB[1] == 10 and vB[2] == 5)      # amb10/uav5, incident 보존
    ok &= a_ok and b_ok
    print(f"{'[OK]' if a_ok else '[FAIL]'} ②A incident 200: base(fixed)={knobs[fixed_k]} "
          f"vA={vA} (기대 (200,{base_amb},{base_uav}))")
    print(f"{'[OK]' if b_ok else '[FAIL]'} ②B amb10/uav5: vB={vB} (기대 ({base_inc},10,5))")

    # ③ 복구: 주입 키가 빌드 후 다시 없음
    r_ok = (after == before)
    ok &= r_ok
    print(f"{'[OK]' if r_ok else '[FAIL]'} ③ env 복구(주입키 삭제): after={after}")

    # ③b 변형 뒤 빌드된 plain nat 누수 카나리: incident=100(매니페스트 base 불변값)
    leak_ok = (knobs[nat_k][0] == 100)
    ok &= leak_ok
    print(f"{'[OK]' if leak_ok else '[FAIL]'} ③b 누수 카나리 nat incident={knobs[nat_k][0]} (기대 100)")

    # ⑤ reset 지역 샘플링
    s_ok = reset_ok and len(set(seen)) >= 2 and set(seen) <= set(mini.keys())
    ok &= s_ok
    print(f"{'[OK]' if s_ok else '[FAIL]'} ⑤ reset 샘플링: obs{EXP_OBS}={reset_ok}, "
          f"distinct={len(set(seen))}/{len(mini)} {sorted(set(seen))}")

    # ③c 사전 존재 키의 복구(삭제 아님) 브랜치
    os.environ["MCI_INCIDENT_SIZE"] = "111"
    p1 = _write_tmp({"v": {"path": fixed_p, "overrides": {"MCI_INCIDENT_SIZE": 200}}})
    with _quiet():
        FeatureMultiRegionEnv(p1, seed=0).close()
    c_ok = (os.environ.get("MCI_INCIDENT_SIZE") == "111")
    ok &= c_ok
    print(f"{'[OK]' if c_ok else '[FAIL]'} ③c 사전존재키 복구(삭제 아님): "
          f"MCI_INCIDENT_SIZE={os.environ.get('MCI_INCIDENT_SIZE')} (기대 111)")
    os.environ.pop("MCI_INCIDENT_SIZE", None)
    os.remove(p1)

    # ④ 구 스키마 전용 회귀
    p_old = _write_tmp({fixed_k: fixed_p, nat_k: nat_p})
    with _quiet():
        mo = FeatureMultiRegionEnv(p_old, seed=0)
        od = {tuple(e.observation_space.shape) for e in mo._envs}
        oa = {int(e.action_space.n) for e in mo._envs}
        mo.close()
    o_ok = (od == {EXP_OBS} and oa == {EXP_ACT})
    ok &= o_ok
    print(f"{'[OK]' if o_ok else '[FAIL]'} ④ 구 스키마 회귀: obs={od} act={oa}")
    os.remove(p_old)

    # ⑦ 화이트리스트 밖 키 → ValueError
    w_ok = _build_raises({"v": {"path": fixed_p, "overrides": {"MCI_BOGUS": 1}}})
    ok &= w_ok
    print(f"{'[OK]' if w_ok else '[FAIL]'} ⑦ 화이트리스트 밖 overrides 키 → ValueError={w_ok}")

    # ⑧ amb=0 → ValueError
    z_ok = _build_raises({"v": {"path": fixed_p, "overrides": {"MCI_AMB_NUM": 0}}})
    ok &= z_ok
    print(f"{'[OK]' if z_ok else '[FAIL]'} ⑧ amb=0 override → ValueError={z_ok}")

    # ⑥ gen_randomized_manifest: base500 → 1000·스키마·재현성·기본조합 배제
    out0 = grm.build_randomized(base, 1, 0)
    out0b = grm.build_randomized(base, 1, 0)
    try:
        grm._validate(base, out0, 1)
        val_ok = True
    except AssertionError as e:
        val_ok = False
        print(f"  _validate 실패: {e}")
    g_ok = val_ok and len(out0) == 2 * len(base) == 1000 and out0 == out0b
    ok &= g_ok
    print(f"{'[OK]' if g_ok else '[FAIL]'} ⑥ gen: base {len(base)} → {len(out0)} 엔트리"
          f"(변형 {len(out0) - len(base)}), 재현성={out0 == out0b}, 스키마검증={val_ok}")

    print("\n" + ("[ALL OK] v6 A5 배선 스모크 전부 통과"
                  if ok else "[FAILED] 일부 체크 실패"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
