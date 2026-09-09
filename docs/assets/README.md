# README 미디어

루트 README는 SVG 배너와 짧은 GIF 미리보기만 임베드한다.
전체 GIF는 링크로 열도록 유지한다. 접힌 `<details>` 안의 이미지도 브라우저가 미리 요청할 수
있으므로, 토글만으로 로딩 비용이 사라진다고 가정하지 않는다.

| 파일 | 용도 |
|---|---|
| `readme-overview.svg` | 밝은 도시 지도·사고 구역·UAV 경로를 그린 벡터 배너. 외부 폰트·스크립트·리소스 없음 |
| `unity-uav-preview.gif` | UAV 원본의 4~10초, 320×180 / 6fps / 64색 |
| `unity-driving-preview.gif` | ADS 원본의 12~18초, 320×180 / 6fps / 64색 |
| `KoreaDigitalTwin_UAV.gif` | 전체 자율비행 영상, 원본 보존 |
| `KoreaDigitalTwin_ADS.gif` | 전체 자율주행 영상, 원본 보존 |

2026-09-09 기준 전체 GIF 합계51,955,027 bytes → README 임베드 미디어 합계1,703,002 bytes
(배너 포함, 약96.7% 감소). 실제 페이지 로드시간은 GitHub·네트워크·브라우저에 따라 달라진다.
두 미리보기는 원본 구간을 실시간 속도로 재생하며 프레임 수·색상 수를 줄였다.

## 미리보기 재생성

FFmpeg가 설치된 환경에서 저장소 루트 기준으로 실행한다.
원본 GIF와 구간·길이를 바꿀 때는 README의 설명도 함께 갱신한다.

```bash
ffmpeg -hide_banner -loglevel error -y -threads 1 \
  -ss 4 -t 6 -i docs/assets/KoreaDigitalTwin_UAV.gif \
  -filter_complex_threads 1 \
  -filter_complex 'fps=6,scale=320:-1:flags=lanczos,hqdn3d=3:3:6:6,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3' \
  -loop 0 docs/assets/unity-uav-preview.gif

ffmpeg -hide_banner -loglevel error -y -threads 1 \
  -ss 12 -t 6 -i docs/assets/KoreaDigitalTwin_ADS.gif \
  -filter_complex_threads 1 \
  -filter_complex 'fps=6,scale=320:-1:flags=lanczos,hqdn3d=3:3:6:6,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3' \
  -loop 0 docs/assets/unity-driving-preview.gif
```

확인 항목: GIF 재생·전체 원본 링크·토글·모바일 폭·다크 모드.
최적화 때문에 전체 GIF를 README에 다시 직접 삽입하거나 같은 미디어를 중복 표시하지 않는다.
