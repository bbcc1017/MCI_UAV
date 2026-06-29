"""VIPER 증류 트리 시각화 (matplotlib plot_tree) — 피처명·역정규화 임계값·액션 디코드.

증류 트리는 노드가 raw feature_idx + VecNorm 정규화 임계값이라 그대로는 못 읽는다. 이 스크립트가:
  · feature_N → 사람이 읽는 피처명(병원i×[is_tier3,cap_remain,eta_amb,eta_uav] + global 21)
  · 정규화 임계값 → 원단위 역정규화(vecnorm mean/std 로 threshold*std+mean, in-place)
  · 잎 class(Discrete) → (class[R/Y/G], dest[현장/병원k], mode[AMB/UAV]) 디코드
상위 N단계만 그려 가독성 확보(깊은 트리는 top-3~4가 핵심 결정). 한글 NanumGothic.

예: python src/rl_src/viper_plot_tree.py --pkl results/viper/sido/서울_occ/viper_loggap_d12.pkl \
       --vecnorm results/rl/sido/서울_ds_ess_woG_occ_s0/vecnormalize.pkl --plot_depth 3 --label "시도 서울 occ"
"""
import argparse, copy, os, pickle, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.tree import plot_tree
import viper_interpret as VI

# 한글 폰트
_fp = "/home/ryu/.fonts/NanumGothic-Regular.ttf"
if os.path.exists(_fp):
    font_manager.fontManager.addfont(_fp)
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--vecnorm", default=None)
    ap.add_argument("--plot_depth", type=int, default=3, help="그릴 상위 단계 수")
    ap.add_argument("--hos_num", type=int, default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = pickle.load(open(args.pkl, "rb"))
    tree = copy.deepcopy(d["tree"])
    n_feat = int(tree.n_features_in_)
    H = args.hos_num or (n_feat - 21) // 4
    names = VI.build_feature_names(H)
    class_names = [VI.decode_action(int(c), H) for c in tree.classes_]

    # 임계값 역정규화 (in-place; plot_tree 가 원단위로 표시되게)
    if args.vecnorm and os.path.exists(args.vecnorm):
        mean, std = VI.load_vecnorm(args.vecnorm)
        thr = tree.tree_.threshold
        feat = tree.tree_.feature
        for n in range(tree.tree_.node_count):
            f = feat[n]
            if f >= 0:  # 내부 노드
                thr[n] = thr[n] * std[f] + mean[f]
        denorm = True
    else:
        denorm = False

    import re
    eff_depth = min(args.plot_depth, tree.get_depth())
    n_leaves_shown = 2 ** eff_depth
    fig, ax = plt.subplots(figsize=(max(n_leaves_shown * 2.4, 14), 2.5 + 2.6 * eff_depth))
    plot_tree(tree, max_depth=args.plot_depth, feature_names=names, class_names=class_names,
              filled=True, rounded=True, impurity=False, proportion=True, fontsize=10, ax=ax,
              precision=2)
    # 확률 배열(proportion=True 라 "value=" 접두사 없이 다중행 숫자배열) 제거.
    # 화이트리스트: 분기조건(<=)·samples·class 줄만 남기고, 숫자/괄호/콤마뿐인 배열조각 제거.
    def clean(txt):
        out = []
        for ln in txt.split("\n"):
            s = ln.strip()
            if not s:
                continue
            if s.startswith("value"):                      # "value = [..]" (단일행 배열)
                continue
            if re.fullmatch(r"[\d\.\,\s\[\]eE+\-]+", s):    # 줄바꿈된 배열 조각
                continue
            out.append(s)
        return "\n".join(out)
    for t in ax.texts:
        t.set_text(clean(t.get_text()))
    shown = "전체" if args.plot_depth >= tree.get_depth() else f"상위 {args.plot_depth}단계"
    title = (f"VIPER 증류 트리 — {args.label}   (depth={tree.get_depth()}, 잎={tree.get_n_leaves()}, {shown} 표시)\n"
             f"내부노드=분기조건(피처 {'원단위' if denorm else '정규화'} 임계값) · 잎=결정 액션(분류→목적지/모드)\n"
             f"표기: H##=근접순위 병원특징(00=최근접) · 환자.등급.단계 · AMB/UAV.fleet · 병원##=이송 목적지 병원")
    ax.set_title(title, fontsize=11)
    out = args.out or (os.path.splitext(args.pkl)[0] + f"_treeplot_d{args.plot_depth}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[저장] {out}  (depth={tree.get_depth()}, leaves={tree.get_n_leaves()}, classes={len(tree.classes_)})")


if __name__ == "__main__":
    main()
