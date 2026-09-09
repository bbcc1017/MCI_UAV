#!/usr/bin/env bash
# 학습 진행 스냅샷을 서버에서 기록한다. 로컬(윈도우) 폴링이 메모리 압박으로 두 번 죽어서 옮긴 것 —
# ssh 세션을 붙들지 않으므로 로컬 자원을 전혀 안 쓴다. 읽기: tail Logs/pg_monitor.log
set -u
R=/home/ryu/roaddrive
while sleep 900; do
  ft=$(grep -oE 'Step: [0-9]+' "$R/Logs/korea_drive_pg_ft_launcher.log" 2>/dev/null | tail -1)
  fr=$(grep -oE 'Mean Reward: [-0-9.]+' "$R/Logs/korea_drive_pg_ft_launcher.log" 2>/dev/null | tail -1)
  fl=$(grep -aoE "lesson 'L[0-9][A-Za-z_]*" "$R/Logs/korea_drive_pg_ft_launcher.log" 2>/dev/null | tail -1 | tr -d "'")
  mn=$(grep -oE 'Step: [0-9]+' "$R/Logs/korea_drive_pg_launcher.log" 2>/dev/null | tail -1)
  mr=$(grep -oE 'Mean Reward: [-0-9.]+' "$R/Logs/korea_drive_pg_launcher.log" 2>/dev/null | tail -1)
  w1=$(grep -c '죽음 감지' "$R/Logs/korea_drive_pg_ft_watchdog.log" 2>/dev/null)
  w2=$(grep -c '죽음 감지' "$R/Logs/korea_drive_pg_watchdog.log" 2>/dev/null)
  echo "$(date '+%m-%d %H:%M') | ft $ft $fr ${fl:-?} 재시작$w1 | main $mn $mr 재시작$w2 | load $(cut -d' ' -f1 /proc/loadavg)" >> "$R/Logs/pg_monitor.log"
done
