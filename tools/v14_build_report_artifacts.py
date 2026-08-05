# -*- coding: utf-8 -*-
"""v14 사후규칙 분석을 재현 가능한 노트북과 통합 HTML 보고서로 묶는다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import markdown
import nbformat as nbf
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
V14 = REPO / "results/scoreboard/v14"
COMPARISON = V14 / "policy_rule_comparison"
GUIDELINE = V14 / "guideline_results"


def build_notebook(path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# [철회된 탐색] PPO·최종교사 규칙과 LB-T3 결합

> **사용 금지:** `GUIDE_*_LBT3`는 UAV 확보용 원거리 헬기장병원을 AMB 발송상한 풀에 포함하는 도메인 부정합이 확인되어 최종 연구 주장에서는 철회했다. LB-T3는 별도 기준선으로만 유지한다.

## TL;DR

- 최종교사는 같은 상태의 PPO 행동을 46.5% 바꿨고, 대부분은 **병원 목적지 변경(45.4%)**이었다.
- Yellow 현장 대기가 커질수록 Yellow 선택 비중이 증가했고, UAV는 대체로 **AMB보다 약 12분 이상 빠를 때** 선택됐다.
- 이 패턴을 깊이 3 class·mode tree로 규칙화하고 LB-T3 병원선택과 결합한 정책은 대표점 250개×seed 0–29에서 **PDR 0.16361**을 기록했다.
- LB-T3 0.16863 대비 **2.98% 감소**, 지역 paired bootstrap CI가 0을 넘고 Holm 보정 Wilcoxon도 유의했다.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

분석학습은 random4 p0–p2 750좌표, 행동규칙 재현성 확인은 p3 250좌표, 폐루프 최종평가는 규칙 적합에 사용하지 않은 대표점 250좌표다. PPO와 최종교사의 실제 궤적은 기술통계로만 비교하고, 최종교사 상태에서 계산한 PPO 행동과 최종행동만 동일상태 보정으로 해석한다.
"""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image

here = Path.cwd().resolve()
repo = next(
    p for p in (here, *here.parents)
    if (p / 'results/scoreboard/v14').is_dir()
)
v14 = repo / 'results/scoreboard/v14'
comparison = v14 / 'policy_rule_comparison'
guideline = v14 / 'guideline_results'

dq_behavior = json.loads((comparison / 'data_quality.json').read_text())
dq_guideline = json.loads((guideline / 'data_quality.json').read_text())
assert dq_behavior['status'] == 'pass'
assert dq_guideline['status'] == 'pass'
print('데이터 품질:', dq_behavior['status'], '/', dq_guideline['status'])
print('폐루프 완전격자:', dq_guideline['rows'], 'rows')
"""
        ),
        nbf.v4.new_markdown_cell("## 행동 사후분석"),
        nbf.v4.new_code_cell(
            """rates = pd.read_csv(comparison / 'policy_behavior_rates.csv')
bins = pd.read_csv(comparison / 'conditional_action_bins.csv')
corrections = pd.read_csv(comparison / 'matched_correction_rates.csv')

display(rates[rates.dataset.eq('validation')][
    ['policy','metric','rate_equal_region','boot_lo','boot_hi','n_decisions']
].reset_index(drop=True))
display(corrections[corrections.dataset.eq('validation')][
    ['change_type','rate_equal_region','boot_lo','boot_hi','n_decisions']
].reset_index(drop=True))
display(bins[
    bins.dataset.eq('validation') & bins.condition.eq('현장 Yellow 대기자수')
][['policy','stratum','rate_equal_region','n_decisions']].reset_index(drop=True))
"""
        ),
        nbf.v4.new_code_cell(
            """display(Image(filename=str(comparison / 'class_priority_policy_comparison.png')))
display(Image(filename=str(comparison / 'uav_advantage_policy_comparison.png')))
display(Image(filename=str(comparison / 'hospital_choice_coefficients.png')))
"""
        ),
        nbf.v4.new_markdown_cell("## 폐루프 성능 검증"),
        nbf.v4.new_code_cell(
            """score = pd.read_csv(guideline / 'guideline_scoreboard.csv')
pair = pd.read_csv(guideline / 'guideline_pairwise.csv')
ablation = pd.read_csv(guideline / 'guideline_component_ablation.csv')

display(score[['display_name','pdr_wog_mean','bootstrap_lo','bootstrap_hi']])
display(pair[
    pair.reference.eq('LB_T3')
][['candidate','improvement_ref_minus_candidate','improvement_boot_lo',
   'improvement_boot_hi','wilcoxon_holm_p','W','T','L']])
display(ablation[['ablation','improvement_ref_minus_candidate',
                  'improvement_boot_lo','improvement_boot_hi',
                  'wilcoxon_holm_p','significant_after_holm_0_05']])
"""
        ),
        nbf.v4.new_code_cell(
            """display(Image(filename=str(guideline / 'guideline_scoreboard.png')))
display(Image(filename=str(guideline / 'guideline_component_ablation.png')))
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. **중증도:** Red-first 고정규칙보다 Yellow 적체를 반영한 조건부 우선순위가 좋았다.
2. **수단:** UAV는 절대 우선이 아니라 AMB 대비 시간절감이 충분할 때 선택하는 보완수단이다.
3. **병원:** 교사의 병원 선택은 가까움과 점유를 함께 반영했지만, 간단한 회귀점수의 지역별 우세는 확정되지 않았다. 현재 폐루프 최선 규칙은 LB-T3 발송상한 골격이다.
4. **플래너:** 보정의 중심은 mode가 아니라 destination이며, 고부하일수록 목적지 재검토율이 커졌다.

## Limitations

행동 빈도는 인과효과가 아니며, class tree는 설명을 위한 축소모형이다. 7개 후보 탐색을 보정했고 대표점은 규칙 적합에서 제외했지만, 최종 논문 전에는 규칙과 임계값을 동결한 외부좌표 재검증이 필요하다.
"""
        ),
    ]
    nbf.write(nb, path)


def build_markdown() -> str:
    score = pd.read_csv(GUIDELINE / "guideline_scoreboard.csv").set_index("method")
    pairs = pd.read_csv(GUIDELINE / "guideline_pairwise.csv")
    pair = pairs[
        (pairs.reference == "LB_T3") & (pairs.candidate == "GUIDE_FINAL_LBT3")
    ].iloc[0]
    correction = pd.read_csv(COMPARISON / "matched_correction_rates.csv")
    correction = correction[correction.dataset == "validation"].set_index("change_type")
    lb = score.loc["LB_T3", "pdr_wog_mean"]
    guide = score.loc["GUIDE_FINAL_LBT3", "pdr_wog_mean"]
    return f"""# [철회된 탐색] PPO·NCRP·MILP 사후규칙과 LB-T3 결합

> **중요:** 이 v14 보고서의 `GUIDE_*_LBT3` 결합은 UAV 운용을 위해 추가된 원거리 헬기장병원을 AMB 발송상한 풀에도 포함한다는 도메인 부정합이 확인되어 최종 연구 주장에서는 철회했다. LB-T3는 별도 휴리스틱으로만 유지한다. 아래 결과는 감사 추적용이다.

## 결론 먼저

최종교사는 동일 상태의 PPO 행동을 **{correction.loc['행동 전체','rate_equal_region']:.1%}** 수정했으며, 핵심은 병원 목적지 변경 **{correction.loc['병원','rate_equal_region']:.1%}**였다. 이 행동을 class·mode 축의 깊이 3 트리로 축약하고 LB-T3 병원당 3명 발송상한 골격에 결합한 `GUIDE_FINAL_LBT3`는 대표점 250개×공통 seed 0–29에서 PDR_woG **{guide:.6f}**를 기록했다. LB-T3 **{lb:.6f}**보다 **{100*(lb-guide)/lb:.2f}%** 낮고, 지역 paired bootstrap 95% CI **[{pair.improvement_boot_lo:.6f}, {pair.improvement_boot_hi:.6f}]**, Holm 보정 p={pair.wilcoxon_holm_p:.3g}였다.

![전체 성능](guideline_results/guideline_scoreboard.png)

## 어떤 조건에서 어떤 행동을 했는가

### 1. 환자등급: Yellow 적체를 방치하지 않는다

Red와 Yellow가 모두 선택 가능한 상태에서 Yellow 대기자가 늘수록 Red 선택률이 감소했다. 최종교사는 Yellow 0–9명에서 Red를 약 36.6% 선택했지만, 20명 이상에서는 약 3.8%만 선택했다. 즉 단순 Red-first가 아니라 **Yellow 적체가 커지면 Yellow 이송 비중을 높이는 조건부 우선순위**다.

![환자등급 조건](policy_rule_comparison/class_priority_policy_comparison.png)

### 2. 이송수단: UAV는 시간절감이 충분할 때 쓴다

수단 선택 축소트리의 첫 분기는 최종교사에서 `UAV 시간절감 > 12.24분`이었다. 단, 두 수단이 같은 목적지에 모두 유효한 p3 결정은 323건이므로, 임계값은 폐루프 성능과 함께 해석한다.

![UAV 조건](policy_rule_comparison/uav_advantage_policy_comparison.png)

### 3. 병원: 가까움과 부하분산을 함께 본다

조건부 병원선택 모형에서 ETA와 점유비 계수는 두 정책 모두 음수였다. 그러나 ETA·점유 선형점수를 LB-T3 골격으로 바꾼 평균 이득은 지역 순위 검정에서 Holm 보정 후 유의하지 않았다. 따라서 현재 근거는 **병원당 발송상한 규칙이 폐루프 평균에 유망하다**는 수준이며, 인과적 병원 권고로 과장하지 않는다.

![병원 조건](policy_rule_comparison/hospital_choice_coefficients.png)

### 4. 플래너 보정: 고부하일수록 병원을 다시 본다

같은 상태에서 class 변경은 6.6%, mode 변경은 1.9%였지만 destination 변경은 45.4%였다. 이송·복귀 중 차량 13대부터 전체 행동 보정률이 급격히 증가했다. 즉 NCRP·MILP의 실무적 시사점은 “UAV를 더 많이 써라”보다 **자원 압박이 커질 때 PPO가 고른 병원을 재검토하라**에 가깝다.

## 규칙화와 폐루프 검증

사전고정한 일곱 후보를 대표점에서 재시뮬레이션했다. 최선 규칙은 최종교사의 class·mode tree와 LB-T3 병원선택 결합이다. Red-first 대비 class tree의 개선이 가장 컸고, 단순 UAV 12.2분 임계값을 mode tree로 바꾼 이득은 작지만 유의했다.

![구성요소 기여](guideline_results/guideline_component_ablation.png)

## 학술적 의미

이 결과는 “블랙박스가 좋은 행동을 했다”에서 끝나지 않는다. **교사 행동에서 반복 조건을 추출하고, 명시적 규칙으로 재구성하고, 독립 좌표의 폐루프 시뮬레이션에서 기존 강한 휴리스틱을 유의하게 개선**했다. 따라서 논문의 메시지는 모델 배포가 아니라 다음 세 단계로 정리할 수 있다.

1. 전국 MCI에서 환자등급·병원·AMB/UAV를 통합 결정하는 고성능 교사정책을 구축한다.
2. 교사가 어떤 상황에서 어떤 결정을 하는지 통계적으로 설명한다.
3. 설명에서 끝내지 않고 규칙을 재시뮬레이션해 기존 휴리스틱의 성능 개선으로 환원한다.

## 한계와 다음 검증

- 행동 패턴은 인과효과가 아니며 임상적 안전성을 직접 증명하지 않는다.
- 7개 후보 탐색은 Holm 보정했지만, 최종 규칙을 완전히 동결한 외부좌표 검증이 남아 있다.
- 병원 선형점수의 지역별 우세는 불확실하다. 이 구성은 추가 근거 없이 최종 가이드라인으로 승격하지 않는다.
- 지역별 UAV 효용은 행정구역 유형이 아니라 AMB–UAV 시간절감, 헬리패드·Tier3 접근성, 병원 용량을 직접 사용해 이질성을 분석해야 한다.

## 재현 자료

- 행동 비교: `results/scoreboard/v14/policy_rule_comparison/`
- 규칙 정책: `results/scoreboard/v14/guideline_policies/`
- 폐루프 원자료: `results/scoreboard/v14/guideline_eval250_seed0_29.csv`
- 성능·검정: `results/scoreboard/v14/guideline_results/`
"""


def build_html(markdown_text: str, path: Path) -> None:
    body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code", "toc"])
    html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MCI UAV 사후규칙 분석</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Noto Sans KR','NanumGothic',sans-serif;max-width:1080px;margin:0 auto;padding:40px 28px;color:#1f2933;line-height:1.72}}
h1,h2,h3{{line-height:1.3;color:#102a43}} h1{{border-bottom:3px solid #2b6f9f;padding-bottom:16px}}
img{{max-width:100%;height:auto;border:1px solid #d9dee3;border-radius:8px;margin:12px 0 24px}}
code{{background:#f2f4f6;padding:2px 5px;border-radius:4px}} table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #d9dee3;padding:8px;text-align:left}} th{{background:#eef4f8}}
</style></head><body>{body}</body></html>"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=V14)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    notebook = out / "v14_policy_guideline_analysis.ipynb"
    report_md = out / "v14_policy_guideline_report.md"
    report_html = out / "v14_policy_guideline_report.html"
    build_notebook(notebook)
    text = build_markdown()
    report_md.write_text(text, encoding="utf-8")
    build_html(text, report_html)
    print(json.dumps({
        "notebook": str(notebook), "markdown": str(report_md), "html": str(report_html),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
