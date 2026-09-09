# -*- coding: utf-8 -*-
"""자율주행 학습 tfevents 에서 **평균보상 말고 실제 위반 지표**를 뽑는다.

평균보상만 보면 커리큘럼이 난이도를 올릴 때 값이 내려가는지 정책이 나빠진 건지 구분이 안 된다.
특히 L2(terminate_on_red 활성)부터는 `Road/RedViolation` 이 **카메라 탐지만으로 신호를 읽을 수
있는가**의 판정 지표다 — 여기가 안 떨어지면 탐지 게이트가 너무 빡빡한 것이다.

실행: python road_train_report.py <run_id> [bin_steps]
"""
import glob, sys, os
from tensorboard.backend.event_processing import event_accumulator as ea

RUN = sys.argv[1] if len(sys.argv) > 1 else 'korea_drive_v6'
BIN = int(sys.argv[2]) if len(sys.argv) > 2 else 400000
ROOT = os.environ.get('ROAD_DRIVE_ROOT', '/home/ryu/roaddrive')

files = sorted(glob.glob('%s/results/%s/RoadDriving/*.tfevents*' % (ROOT, RUN)))
if not files:
    print('tfevents 없음:', RUN); raise SystemExit(1)

series = {}
for f in files:                      # 재개하면 파일이 여러 개다 — 전부 합친다
    a = ea.EventAccumulator(f, size_guidance={ea.SCALARS: 0}); a.Reload()
    for t in a.Tags()['scalars']:
        series.setdefault(t, []).extend((e.step, e.value) for e in a.Scalars(t))
for t in series:
    series[t] = sorted(set(series[t]))

# ⚠집계 방식이 지표마다 다르다 — 섞어 읽으면 오독한다(실측으로 한 번 틀렸다).
#   Sum    = summary_freq(10,000 스텝) 구간 **합계**  : RedViolation·PhysicsFault·Speeding·BusLane·WrongWay·BlockBox
#   Average= **에피소드 평균**                        : Success·Collision·OffLane·RouteCompletion·ShieldIntervene·RedViolationEpisode
#   → Sum 지표는 `× 에피길이 / 10000` 으로 에피소드당으로 환산해야 Average 지표와 나란히 볼 수 있다.
#   적신호는 `RedViolationEpisode`(위반이 있었던 에피소드 **비율**)가 가장 해석하기 쉽다.
COLS = [('보상', 'Environment/Cumulative Reward', 'avg'),
        ('에피길이', 'Environment/Episode Length', 'avg'),
        ('성공', 'Road/Success', 'avg'),
        ('경로완주', 'Road/RouteCompletion', 'avg'),
        ('적신호에피율', 'Road/RedViolationEpisode', 'avg'),
        ('적신호/에피', 'Road/RedViolation', 'sum2ep'),
        ('충돌', 'Road/Collision', 'avg'),
        ('차선이탈', 'Road/OffLane', 'avg'),
        ('셸개입', 'Road/ShieldIntervene', 'avg')]
EPLEN = 'Environment/Episode Length'
LESSON = 'Environment/Lesson Number/terminate_on_red'

maxstep = max(s for v in series.values() for s, _ in v)
print('run=%s  최대 step=%s' % (RUN, format(maxstep, ',')))
print('%-11s %5s ' % ('step', 'L') + ' '.join('%12s' % c for c, _, _ in COLS))
lo = 0
while lo < maxstep:
    hi = lo + BIN
    row, has = [], False
    eps = [v for s, v in series.get(EPLEN, []) if lo <= s < hi]
    eplen = sum(eps) / len(eps) if eps else 0.0
    for _, tag, mode in COLS:
        vals = [v for s, v in series.get(tag, []) if lo <= s < hi]
        if not vals:
            row.append('%12s' % '-'); continue
        m = sum(vals) / len(vals)
        if mode == 'sum2ep':          # 10k 스텝 합계 -> 에피소드당
            m = m * eplen / 10000.0
        row.append('%12.4f' % m)
        has = True
    les = [v for s, v in series.get(LESSON, []) if lo <= s < hi]
    if has:
        print('%-11s %5s ' % (format(hi, ','), '%d' % les[-1] if les else '-') + ' '.join(row))
    lo = hi
