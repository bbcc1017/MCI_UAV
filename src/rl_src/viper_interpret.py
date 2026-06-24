"""VIPER 트리 → 사람이 읽는 해석본 export.

viper_distill.py 가 남기는 viper_*.pkl 의 sklearn 트리는 노드가
`feature_190 <= -0.91`, 잎이 `class: 38` 형태(원시 피처idx + VecNorm 정규화 임계값 +
flatten Discrete 액션)라 사람이 못 읽는다. 이 스크립트가 후처리로:

  1) **피처명 복원** — HospitalFeatureWrapper essential obs 레이아웃
     (entity H×4 [is_tier3, cap_remain, eta_amb, eta_uav] + global 21)을 인덱스→이름 매핑.
  2) **임계값 역정규화** — VecNormalize(obs) 통계로 raw = norm*std + mean (원단위 복원).
  3) **액션 디코드** — Discrete a → (class[R/Y/G], dest[현장잔류/병원k], mode[AMB/UAV]).

원본 시나리오/모델 미수정(read-only). 출력: <pkl경로>_interpreted.txt (+ .json).

예:
  python src/rl_src/viper_interpret.py \\
    --pkl results/viper/sigungu_nat_occ/viper_loggap_d8.pkl \\
    --vecnorm results/rl/sigungu_nat/ds_ess_woG_occ_s0/vecnormalize.pkl \\
    --gate occ --label "시군구 전국 occ"
"""
import argparse
import json
import os
import pickle

import numpy as np

# essential 병원당 특징 순서 (hospital_feature_wrapper._ESSENTIAL_COLS 와 동일해야 함)
ESSENTIAL_COLS = ["is_tier3", "cap_remain", "eta_amb", "eta_uav"]
# 글로벌(21): 환자 R/Y 2등급×5단계(10) + AMB/UAV fleet 5개씩(10) + time(1)
PATIENT_STAGES = ["미구조", "현장대기", "이송중", "병원도착", "완료"]
PATIENT_CLASSES = ["Red", "Yellow"]            # patient_agg[:10] = R/Y 만
FLEET_COLS = ["n_avail", "n_busy", "min_t", "mean_t", "n_crit"]
CLASS_LABEL = {0: "Red", 1: "Yellow", 2: "Green"}
MODE_LABEL = {0: "AMB", 1: "UAV"}


def build_feature_names(H, cols=ESSENTIAL_COLS):
    """flat obs 인덱스 → 사람이 읽는 이름 (entity H×F + global 21)."""
    names = []
    for h in range(H):
        for c in cols:
            names.append(f"H{h:02d}.{c}")
    for cls in PATIENT_CLASSES:
        for st in PATIENT_STAGES:
            names.append(f"환자.{cls}.{st}")
    for veh in ("AMB", "UAV"):
        for fc in FLEET_COLS:
            names.append(f"{veh}.{fc}")
    names.append("time")
    return names


def decode_action(a, H):
    """Discrete a → (class, dest, mode). amb+uav 둘 다 있는 3D 평탄화(mode 자유) 기준."""
    n_dest, n_mode = H + 1, 2
    a = int(a)
    c = a // (n_dest * n_mode)
    rem = a % (n_dest * n_mode)
    d, m = rem // n_mode, rem % n_mode
    dest = "현장잔류" if d == 0 else f"병원{d - 1}"
    return f"{CLASS_LABEL.get(c, c)}→{dest}/{MODE_LABEL.get(m, m)}"


def load_vecnorm(path):
    with open(path, "rb") as f:
        vn = pickle.load(f)
    rms = vn.obs_rms
    mean = np.asarray(rms.mean, dtype=np.float64)
    std = np.sqrt(np.asarray(rms.var, dtype=np.float64) + vn.epsilon)
    return mean, std


def walk_tree(tree, names, mean, std, H, max_print_depth=None):
    """sklearn 트리를 사람이 읽는 들여쓰기 규칙으로 변환 (임계값 역정규화 + 액션 디코드)."""
    t = tree.tree_
    classes = tree.classes_
    lines = []

    def denorm(feat, thr):
        if mean is not None and feat < len(mean):
            return thr * std[feat] + mean[feat]
        return thr  # 정규화 미적용 학습(이론상)

    def recurse(node, depth):
        if max_print_depth is not None and depth > max_print_depth:
            return
        indent = "│  " * depth
        left, right = t.children_left[node], t.children_right[node]
        if left == right:  # leaf
            val = t.value[node][0]
            a = int(classes[int(np.argmax(val))])
            n = int(t.n_node_samples[node])
            lines.append(f"{indent}└─▶ {decode_action(a, H)}   (n={n})")
            return
        feat = t.feature[node]
        thr_raw = denorm(feat, t.threshold[node])
        fname = names[feat] if feat < len(names) else f"feature_{feat}"
        lines.append(f"{indent}├─ {fname} ≤ {thr_raw:.3g} ?")
        lines.append(f"{indent}│  [예]")
        recurse(left, depth + 1)
        lines.append(f"{indent}│  [아니오]")
        recurse(right, depth + 1)

    recurse(0, 0)
    return lines


def feature_importance(tree, names, topk=20):
    imp = tree.feature_importances_
    order = np.argsort(-imp)
    rows = []
    for i in order[:topk]:
        if imp[i] <= 0:
            break
        rows.append((names[i] if i < len(names) else f"feature_{i}", float(imp[i])))
    return rows


def leaf_action_summary(tree, H):
    """잎들이 내리는 결정 분포 (어떤 class/mode 로 주로 보내는가)."""
    t = tree.tree_
    classes = tree.classes_
    from collections import Counter
    cls_c, mode_c, dest_stay = Counter(), Counter(), 0
    n_leaves = 0
    for node in range(t.node_count):
        if t.children_left[node] != t.children_right[node]:
            continue
        n_leaves += 1
        a = int(classes[int(np.argmax(t.value[node][0]))])
        n_dest, n_mode = H + 1, 2
        c = a // (n_dest * n_mode)
        rem = a % (n_dest * n_mode)
        d, m = rem // n_mode, rem % n_mode
        cls_c[CLASS_LABEL.get(c, c)] += 1
        mode_c[MODE_LABEL.get(m, m)] += 1
        if d == 0:
            dest_stay += 1
    return n_leaves, dict(cls_c), dict(mode_c), dest_stay


def main():
    ap = argparse.ArgumentParser(description="VIPER 트리 해석본 export")
    ap.add_argument("--pkl", required=True, help="viper_*.pkl (viper_distill 출력)")
    ap.add_argument("--vecnorm", default=None, help="vecnormalize.pkl (역정규화용; 없으면 정규화공간 임계값)")
    ap.add_argument("--hos_num", type=int, default=None, help="병원 수 H (미지정시 n_features 로 추정)")
    ap.add_argument("--gate", default="?", help="occ|psent (라벨용)")
    ap.add_argument("--label", default="", help="사람용 제목(예: '시군구 전국 occ')")
    ap.add_argument("--max_print_depth", type=int, default=None, help="규칙 출력 최대 깊이")
    ap.add_argument("--out", default=None, help="출력 txt (기본: <pkl>_interpreted.txt)")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        d = pickle.load(f)
    tree = d["tree"]
    n_feat = int(tree.n_features_in_)
    H = args.hos_num if args.hos_num else (n_feat - 21) // 4
    if H * 4 + 21 != n_feat:
        print(f"⚠️ n_features={n_feat} 가 H*4+21 와 불일치 — essential obs 아닐 수 있음. H={H} 가정 진행.")

    names = build_feature_names(H)
    mean, std = (load_vecnorm(args.vecnorm) if args.vecnorm and os.path.exists(args.vecnorm)
                 else (None, None))

    rules = walk_tree(tree, names, mean, std, H, args.max_print_depth)
    imp = feature_importance(tree, names)
    n_leaves, cls_c, mode_c, dest_stay = leaf_action_summary(tree, H)

    hist = d.get("history", [])
    best_fid = max((h.get("fidelity", 0) for h in hist), default=None)
    best_rew = max((h.get("reward", -1e9) for h in hist), default=None)

    out = args.out or (os.path.splitext(args.pkl)[0] + "_interpreted.txt")
    L = []
    L.append(f"╔══ VIPER 해석본: {args.label or args.pkl} ══╗")
    L.append(f"crit={d.get('crit','?')} | gate={args.gate} | H(병원수)={H} | obs차원={n_feat} "
             f"| depth={tree.get_depth()} | leaves={tree.get_n_leaves()}")
    if best_fid is not None:
        L.append(f"오라클 충실도(fidelity) best={best_fid:.3f} | 트리 보상 best={best_rew:.2f}")
    L.append(f"역정규화: {'적용(원단위 임계값)' if mean is not None else '미적용(정규화공간 임계값)'}")
    L.append("")
    L.append("── 피처 중요도 Top (트리가 실제로 쓰는 변수) ──")
    for nm, v in imp:
        L.append(f"   {v:6.3f}  {nm}")
    L.append("")
    L.append(f"── 잎 결정 분포 (총 {n_leaves} 잎) ──")
    L.append(f"   class: {cls_c}")
    L.append(f"   mode : {mode_c}")
    L.append(f"   현장잔류(dest=0) 잎: {dest_stay}")
    L.append("")
    L.append("── 결정 규칙 (임계값=원단위, 잎=디코드된 액션) ──")
    L.extend(rules)
    txt = "\n".join(L)
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    # 머신용 json
    jout = os.path.splitext(out)[0] + ".json"
    with open(jout, "w", encoding="utf-8") as f:
        json.dump({"label": args.label, "gate": args.gate, "H": H, "n_feat": n_feat,
                   "depth": int(tree.get_depth()), "leaves": int(tree.get_n_leaves()),
                   "best_fidelity": best_fid, "best_reward": best_rew,
                   "feature_importance_top": imp,
                   "leaf_class_dist": cls_c, "leaf_mode_dist": mode_c,
                   "leaf_stay": dest_stay}, f, ensure_ascii=False, indent=2)
    print(f"[저장] {out}\n[저장] {jout}")
    print("\n".join(L[:14]))


if __name__ == "__main__":
    main()
