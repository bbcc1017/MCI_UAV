# CLAUDE.unity.md — Unity 디지털트윈 · CAR_test · GIS 파이프라인

**로컬-C(Windows) 전용.** 학습박스 `aigpu0617` 에는 `external/ml-agents` 가 빈 서브모듈이라
여기 적힌 것 중 실행 가능한 게 없다. 그래서 2026-09-06 에 `CLAUDE.md` 본문(46%·78KB)에서
이 파일로 분리했다 — **Unity 씬·ML-Agents·GIS 수집 작업을 시작하기 전에 이 파일을 Read 할 것.**

`tools/` 의 Unity/GIS 스크립트는 **추적하지 않는다**(gitignore). 1차는 같은 날 GIS 수집도구
30개(`osm_*.py` `v2_*.py` `vworld_*` `terrain_*` `hdmap_fetch.py` `scene_export.py` `trace_export.py`
`run_sim_trace.py` `nationwide/` `seoul_pilot/`), **2차는 2026-09-09 에 전면 확대** — 학습·배치
드라이버(`exp_drivers/road_*` `run_{road,kdt,hd,pg,ortho,dtm}_*` `sync_roaddrive.sh` `unity_*`),
에셋 생성기(`blender*`), 지형·정사·정밀도로지도 검증기(`dtm_*` `ortho_*` `v2_hd_*` 등) 53개를
**푸시 전에** 뺐다. 원격엔 Unity 코드가 없고 파일은 각 박스 워킹트리에만 있다.
⚠️ignore 목록은 **접두 glob** 이다 — 1차가 파일명 열거였던 탓에 새로 만든 도구가 목록에 없어
3일 만에 53개가 다시 추적됐다. **새 Unity 도구는 위 접두사 중 하나를 쓸 것.**

⚠️ **Windows 박스가 이 커밋을 pull 하면 그 37개가 사라진다.** `.gitignore` 는 *미추적* 경로에만
적용되는데, Windows HEAD 기준으로는 아직 tracked 라 체크아웃이 그냥 지운다(실측: 수정 안 한
파일은 조용히 삭제, 수정한 파일이 있으면 `Please commit your changes or stash them` 으로 pull 자체가
중단). 절차:

```bash
cp -r tools ../tools_backup_20260906          # 1. 먼저 백업 (Windows 로컬 수정분 보존)
git pull                                       # 2. 중단되면 git checkout -- tools/ 후 재시도
git restore --source=9f4d4fc --worktree -- tools/   # 3. 워킹트리만 복원
diff -r ../tools_backup_20260906 tools         # 4. 백업본에만 있던 수정분 확인
```

`git checkout 9f4d4fc -- tools/` **는 쓰지 말 것** — 인덱스까지 갱신해 37개를 다시 스테이지하므로
다음 커밋에서 재추적된다. `--worktree` 로 복원하면 인덱스가 안 변해 ignore 가 그대로 먹는다
(실측: `git status` 무변경). 개별 파일은 `git show 9f4d4fc:tools/<파일> > tools/<파일>`.
RL/sim 이 공유하는 OSRM 배관(`osrm_*.ps1/.sh`, `build_distance_matrix_osrm.py`)은 추적 유지이며
설명도 지침 정본 `AGENTS.md` 쪽에 남겼다. 2차 53개는 푸시 전에 뺀 것이라 다른 박스엔
tracked 이력이 없다 — 위 pull 삭제 함정은 1차에만 해당한다.

---
## Unity digital-twin architecture (big picture)

Project root: `external/ml-agents/UAV_test/` — **이 섹션의 `Assets/...` 경로는 전부 이 프로젝트 기준**. ⚠️자율주행 프로젝트 `CAR_test/` 도 `Assets/Scripts/V2/` 를 갖고 이름이 겹치는 파일이 있으니(예: 트윈 v2 타일 vs `LaneGraphV2.cs`) 열기 전에 **프로젝트 루트부터 확인**할 것. Pipeline: **GIS/OSM fetch (tools) → Editor importers bake meshes into Region scenes → runtime additively loads needed scenes and plays the sim trace.**

- **Coordinate system**: `Assets/Scripts/Geo/RegionRegistry.cs` holds a per-시군구 EPSG:5186 "frame". `TryWorld(lat,lon)`/`TryWorldIn(frame,...)` convert WGS84→Unity world (meters). All importers and runtime spawning go through this. `tools/nationwide/sgg.json` is the source of the 255 districts (name/kor/frame/bbox/rings). **`KoreaGeo.LatLonToWorld`/`TM5186ToWorld` are hardcoded to the legacy Sudogwon frame — don't use them per-region; use `TryWorldIn(frame,…)`.** `TryWorld` now adds terrain height (`TerrainHeight.Ground`) to y; `TryWorldIn` stays pure (importers depend on this). `KoreaGeo.TM5186ToLatLon` is the inverse projection.
- **Scene structure**: `Assets/Scenes/SampleScene.unity` is the 3MB entry scene (only one in Build Settings) holding `MapVersionSelector`. `Assets/Scenes/Regions/<name>.unity` are 255 시군구 scenes (each ~50-80MB) with meshes embedded under a `Vworld_<name>` root: `_Ortho`, `_Buildings`, `_Roads`, `_Features`(traffic signals), `_Areas`(park/water), `_POI`(hospital/school/fire). There is no global terrain — districts are loaded **additively** on demand. (Legacy manual-build scenes were retired/deleted 2026-06; don't expect them.)
- **Import pipeline** (`Assets/Editor/VworldRegionImporter.cs`): reads `tools/nationwide/{roads2,feat,area,poi}/<name>.txt` and the vworld building/ortho data, then bakes ribbon/polygon/marker meshes into each Region scene under the `Vworld_<name>_*` roots. Menus under `Tools/MCI/...`. For mass (255-scene) imports use `StartBackgroundImport(kind)` (runs one scene per `EditorApplication.update` tick) — see Unity MCP notes below. Realism passes (menus `Tools/MCI/Realism/...`): kinds `roads2` (re-drape roads to terrain), `bldg-collider` (MeshCollider + "Building" layer), `water` (SimpleWater material + sea plane), `sea-reclip` (clip sea to below-sea-level cells); each idempotent via a marker (`RegionRoadNetwork.terrainDraped`, `__bldg_collider`, `__water_applied`, `__sea_clipped`).
- **Runtime playback** (`Assets/Scripts/Sim/`): `MapVersionSelector` (IMGUI shown on Play) lets the user pick a scenario; `ScenarioSceneResolver` decides which Region scenes the trace actually traverses; selected scenes load additively; then `TracePlayer` animates `scene.json`+`trace_flat.json` (AMB/UAV dispatch, hospital `HospitalFacility`, camera modes) and spawns `TrafficManager` (NPC cars on `RegionRoadNetwork`, yield to emergency, stop at red `TrafficSignal`) and `PedestrianManager`. `FreePilotController` is the alternative free-drive/fly mode.
- **★노면 높이 불변식**: 임포터는 아스팔트 리본을 `Ribbon(road, half, ROAD_CLEAR)` = **드레이프 y + 0.65m** 에 굽는다. 지상 차량/프롭 접지는 반드시 **`TracePlayer.ROAD_CLEAR` 공용 상수**를 쓸 것 — 이 값을 런타임에 손으로 복제하면 반드시 어긋난다(ROAD_CLEAR 0.50→0.65 상향 때 `TrafficManager` 는 +0.52, `TracePlayer` 는 +0.05 로 남아 **구급차가 바퀴까지 노면에 파묻힌 채** 달렸다, 2026-07-14).
- **도로 리본 = 건물 발자국에 클리핑**(DRAPE_VER 10, `RoadBuf.BldgMask` 1m 격자): OSM 골목 중심선이 실측보다 수 m 어긋나(건물·정사는 정합) 리본이 집 위로 올라탄다 → **지상(strct==0) 도로만 좌/우 반폭을 독립으로** 건물에 안 닿을 때까지 축소(최소 반폭 1.2m; 교량/터널 제외). `CLASS_W` 는 이면도로 하향 + OSM `lanes`(좁은 골목에도 2가 흔함)를 클래스 상한×1.15 로 클램프.
- **런타임 독점/싱글턴 규칙**: `Camera.main`(MainCamera 태그)은 `TracePlayer.EnsureCam`(MCI_ViewCamera) / `FreePilotController.SetupCamera`(FreePilot_Camera) 가 **독점**하고 FreePilot 은 `OnDestroy` 에서 자기 카메라·차량을 치운다 — 유령 카메라가 태그를 쥐면 NPC LOD·가로수 인스턴싱이 엉뚱한 위치를 컬링(차 버벅임/가로수 실종). `DayNightController` 는 **싱글턴**(시나리오 전환·자유조종 진입마다 Create 호출 → 예전엔 5개가 태양/NightFactor 를 서로 덮어썼다).
- **병원 내부·보행 모션**: `Resources/Hospital/ER_INTERIOR_KR`(R/Y/G 트리아지 슬롯) + `Hospital/SF/*`(가구 19종)를 `HospitalFacility.Build` 가 돌하우스 안에 앉히고 병상 슬롯을 룸 마커로 우선 매핑(천장 `ER_Ceil`·`ER_Light` 는 끈다). 배회 인원(구조대/구경꾼/의료진)의 걷는 모션은 **`WanderAgent.Init`** 이 붙인다(`ProceduralGait`=리그드, `StaticPedestrianGait`=포토스캔 정적메시).
- **ML-Agents semantic layer**: `SceneObjectMeta` (category/width/lanes/oneway/speed/height) is attached to roads/areas/POI/signals so future ML-Agents observations can read each object's characteristics. `RegionRoadNetwork` stores drivable centerlines + oneway/width + per-point terrain-draped `y[]` (NPCs terrain-follow). Building chunks carry MeshColliders on the "Building" layer for camera-occlusion/NPC-stop raycasts; UAVs land on the real GIS roof via `TracePlayer.CityRoofAtCached`/`HospitalRoof`. **장애물 인식**: 건물=`Physics.OverlapSphere(LayerMask "Building")`로 ML UAM 경로가 회피; 지형(산)=`TerrainHeight`(DEM) 높이함수로 `TerrainClearanceAt` 최소이격(물리 Ground 콜라이더는 `FreePilotController`만 런타임 부착).
- **★UAM 캐빈 내부 + C키 시점 전환 (2026-08-01)**: eVTOL 에어앰뷸런스가 **무인기**라는 설정에 맞춰 조종사 없이 **의사·환자·구급대원 3인**만 태운 캐빈을 Blender 로 만들어 붙였다. `Assets/Scripts/Sim/UamCabin.cs`(내장재+탑승자 조립·카메라 앵커 4종·흉부압박/호흡 모션) + `UamCabinCamera.cs`(**C 키**: 외부→무인칵핏→의사→환자→캐빈광각 순환. `DefaultExecutionOrder(600)` 로 TracePlayer LateUpdate **뒤에** 카메라를 잡고, 외부 모드에선 아무것도 안 건드려 기존 시점이 그대로다. 1인칭일 땐 자기 몸을 숨긴다). TracePlayer 가 UAV 기체를 스폰할 때 `UamCabin.Attach(model)` 자동 부착.
  - **★자율비행 씬에도 배선(2026-08-03)**: 캐빈이 `TracePlayer`(재난 재생) 경로에만 붙어 있어서 **`UavKoreaAutonomyFlight` 씬에서 Play 하면 칵핏이 아예 없었다**(사용자 신고). 그 씬은 `UavGangnamRouteBootstrap` 이 기체를 스폰하고 **관전 HUD `UavFpvViewerHUD` 가 이미 C 키를 자기 카메라 순환에 쓰며** "콕핏" 모드는 기체 밖 고정점을 잡는 가짜였다 → ⓐ부트스트랩이 `!training && i==0` 일 때 `UamCabin.Attach(vis)`(학습 중엔 미부착) ⓑHUD `CameraMode` 를 체이스/**무인칵핏·의사·환자·캐빈광각**/오비트/탑다운/시네마 8종으로 확장해 캐빈 앵커를 직접 쓰게 하고, 캐빈이 없으면 구 콕핏으로 폴백. 실내 진입 시 `nearClipPlane` 0.02(기체 스케일 0.68 이라 계기판이 잘린다).
  - 에셋(Blender 헤드리스 → GLB → `Resources/SeoulPilot/`): `UAM_CabinInterior`(쉘 204면 + 의장품 1,100면 — 무인 칵핏 콘솔·3면 디스플레이·페데스탈·대시, 들것/매트/스트랩, 벽걸이 환자모니터·제세동기·흡인기·약품서랍, 산소통 2본, IV 폴, 소화기, 그랩레일·LED), `CHAR_UAM_Doctor/Medic/Patient`(각 ~1.9k정점 — 의사=가운+장갑+청진기·흉부압박 자세, 대원=형광조끼+태블릿, 환자=산소마스크+담요+ECG패드).
  - **★함정 5건(전부 실측으로 물림)**: ①Unity 규약(y-up, +z 전방)으로 모델링하고 `export_yup` 하면 축이 뒤바뀐다 → 메시에 X+90° 적용 후 export. ②`bpy.ops.object.join()` 은 **활성 오브젝트의 원점**을 유지 → 인물이 원점에서 밀린다(3D 커서로 `origin_set`). ③`transform_apply` 는 선택 상태·다중유저에 좌우돼 **두 오브젝트 중 하나만** 변환됐다 → `obj.data.transform(Matrix.Rotation(...))` 로 메시를 직접 회전. ④좌석 인물은 **발밑이 원점** — 좌석 높이(0.44)에 놓으면 47cm 공중에 뜬다. ⑤기체 원본 인테리어(`*_Interior_*` 여객 좌석 2열)를 안 끄면 들것을 관통하고, 캐빈 바닥이 낮으면 **기체 외피(노란 도장) 안쪽**이 바닥으로 비친다 → `HideStockInterior()` + `floorLift 0.66`.
- **★캐빈 v2 — 실척 월드 리그 + 실동작 계기 (2026-08-03)**: v1 캐빈은 **0.68배로 축소된 기체 모델의 자식**이라 실내고가 0.91m 밖에 안 나왔고(카메라 넣으면 관짝), 기체 형상 안에 갇혀 외피가 계속 비쳤다. v2 는 구조를 바꿨다 — 캐빈은 **씬 루트의 스케일 1 리그**(`UAM_CabinRig`)로 띄우고 기체의 위치·자세만 추종(`attitudeFollow 0.45`, 피치/롤 완화), **실내 시점에서만 활성**한다. 결과: ⓐ외형은 전혀 안 건드림(외부 시점엔 캐빈이 아예 없다) ⓑ실내 **2.40W × 1.96H × 5.73L** 실척 ⓒ창이 진짜 유리라 **밖의 실제 도심이 보인다**(리그가 기체 위치에 있으므로).
  - 에셋 재제작(`scratchpad/uam_cabin_v2.py`·`uam_people_v2.py` → `Assets/KRAssets/UamCabinV2/` → `Resources/SeoulPilot/*V2.prefab`): 쉘 4,772면(측벽 4창×2·창틀·코빙·오버헤드 로커 10문·천장 조명 트로프·그랩레일·시트트랙·후방 도어) + 의장 2,370면(들것·의료랙 4서랍·제세동기·흡인기·모니터암·산소 2본·IV폴·소화기·샤프스·키트백) + **화면 판 5개**(개별 노드). 인물은 **부위 분리**(`Doc_Torso`/`Doc_ArmL`/`Doc_ArmR`/`Doc_Head`/`Doc_Legs` 등)로 내보내 Unity 가 계층 재결합 후 관절처럼 돌린다 — 흉부압박 시 상체+양팔이 실제로 눌리고, 환자는 흉곽이 오르내린다.
  - **계기 = 자율주행차식 인지 화면 `UamCabinDisplays.cs`**(2026-08-03 2차 전면 교체 — 구 EFIS 아날로그 계기판 폐기): **Perception**(실제 LiDAR `UavPredictiveSensorRig.Lidar` 72방위×5고도·400m 로 만든 **자유공간 폴리곤 + 점유 윤곽** 조감도, 거리 링·흐르는 격자·건물 블록/이름·항로 리드라인·자기기체) · **Autonomy**(큰 활자 카드 — 상태/속도/고도/다음 조작(경로 유지·우선회·상승·회피·착륙 접근)/ETA/신뢰도/경로·배터리 바) · **DepthCam**(실제 깊이 카메라 `UavPredictiveDepthRenderer.Distances` 64×48·350m 를 컬러 램프로 + 전방 여유·회피 경고) · **Mission** · **Vitals**(ECG·SpO2 파형). 조감도·깊이맵은 UI 요소 대신 **소프트웨어 래스터**(`Disc/Ring/Line/Fan` → `Texture2D`)로 그린다. 센서가 없는 씬(재난 재생)은 36방위 간이 LiDAR 로 자동 폴백.
  - **★함정 3건 추가 (2026-08-03, 자율비행 씬 실측)**: ⑥**`Renderer.bounds`(월드 AABB)로 동체를 재면 안 된다** — 기체가 롤/피치하는 순간 AABB 가 부풀어 `min.y` 가 실제 바닥보다 수십 cm 내려가고 캐빈이 선체 밑으로 가라앉는다(호버만 하는 재난 재생 씬에선 안 드러나 멀쩡해 보였다). `localBounds` 8코너를 호스트 로컬로 옮겨 재는 `BodyLocalBounds()`/`ToLocal()` 로 교체. ⑦**라이너로 외피를 막는 건 일반해가 없다** — 동체 배가 기수로 갈수록 안쪽으로 솟아 평평한 라이너 바닥을 뚫는다(칵핏 시점 하단 좌우가 통째로 노란 도장). 동체 실측 바운즈로 만드는 절차 라이너(`BuildLiner`, 바닥·측벽·앞뒤격벽, 창턱 위는 개방)는 **보조**고, 확실한 해법은 **실내 시점 동안 외피 렌더러를 끄는 `SetExteriorVisible(false)`**(로터는 유지 — 창밖 회전이 보이는 편이 낫다). ⑧앵커는 사람과 겹치기 쉽다 — 캐빈 광각을 의사석 x 에 두면 **의사 머릿속**, 콘솔까지 z 를 밀면 디스플레이 뒷면이 하단을 덮는다. 환자 시점 요는 62°가 아니라 **136°**(들것 −x 에서 의사석 +x 를 보는 실제 방위).
  - **★함정 7건 추가 (2026-08-03 2차, v2 제작 중 전부 실측)**: ⑨**Blender `primitive_cube_add(size=1)` 은 한 변이 1** — 헬퍼가 `scale=s/2` 를 주고 있어 **모든 박스가 절반 크기**였다(v1 캐빈이 외피를 비친 진짜 원인: 벽 패널이 절반이라 사이가 뚫려 있었다). 전체 바운즈는 위치가 정확해 맞아떨어져서 눈치채기 어렵다 — **파생값(개별 박스 dimensions)을 실측**할 것. ⑩**uGUI 월드 캔버스는 −Z 쪽에서 읽힌다** — 캔버스 +Z 가 보는 사람을 향하면 글자가 **거울상**이 된다(UI 셰이더는 Cull Off 라 "안 보임"이 아니라 뒤집혀 보인다). 그리고 화면 판 노드의 임포트 자세는 화면마다 보정 방향이 달라 일반해가 없다 → **캐빈 로컬 자세를 코드에서 명시**. ⑪**Blender 색은 리니어, Unity 표시는 감마** — 매트 0.40 이 화면엔 0.67 로 뜬다. 흰 매트+흰 가운으로 **환자가 통째로 안 보였다**; 팔레트는 리니어 값이 아니라 **표시값 기준**으로 잡을 것. ⑫들것 등받이를 세우면 **환자 머리 앞을 가로막아** 광각에서 환자가 사라진다(CPR 중엔 평와위가 맞다). ⑬1인칭에서 오브젝트를 통째로 끄면 **자기 손까지 사라진다** → 머리·몸통·다리 **렌더러만** 끄고 팔은 남길 것. ⑭관전용 **예측궤적 리본·비행 트레일이 캐빈을 관통**한다(기체 위치에서 시작) → 실내 동안 에이전트 루트 전체 + `UamFlightPathVisualizer` 까지 소등. ⑮같은 파트 이름으로 두 번 `part()` 하면 앞서 만든 오브젝트가 그룹에서 떨어져 **join 없이 `Cube`/`Sphere` 로 개별 export** 된다(리스트를 `setdefault` 로).
  - **★함정 2건 추가 (2026-08-03 3차)**: ⑯**화면이 깜빡인 진짜 원인은 z-fighting** — 캔버스를 화면 판 중심에서 8mm 앞에 뒀는데 판 두께가 12mm 라 실제 여유가 **2mm** 뿐이었다. 25mm 로 벌려 해결. 그 위에 **의도된** 미세 이동·휘도 변동·드문 글리치(`ApplyInstability`, 기체 각속도에 연동)를 얹어 "진동하는 기내 모니터"로 보이게 한다 — 버그로서의 깜빡임과 연출로서의 흔들림을 구분할 것. ⑱**계기 캔버스를 진동 연출로 움직이면 화면이 물리 모니터 밖으로 삐져나온다** — 캔버스 자체가 아니라 **안쪽 `Content` 홀더**만 흔들고 캔버스에 `RectMask2D` 를 걸어 경계에서 자를 것(실측: 기체가 흔들릴 때 표출 화면이 베젤을 벗어나 떠다녔다). ⑲**실내 시점 near clip 을 0.03 까지 당기지 말 것** — far 6000 과의 비가 20만:1 이라 깊이 정밀도가 무너져 0.3~0.8m 앞 인물의 겹친 면이 지글거린다(0.06 이면 캐빈에 충분). ⑰**IMGUI HUD 오버레이는 렌더러 소등으로 안 사라진다** — 실내 시점인데 벽 너머 건물 이름표가 화면에 떴다(`DrawStructureLabels`). 월드 오브젝트가 아니라 화면 좌표에 그리는 라벨이므로 `IsCabinView` 로 그리기 자체를 막아야 한다.
  - **★인물 = Sketchfab 실사 에셋 (2026-08-03 4차)**: 절차 인물은 한계라 **의사·환자를 실사 스캔 모델로 교체**(대원은 원거리라 절차 v3 유지). Blender MCP 의 `search/download_sketchfab_models` 명령은 **애드온 핸들러 버전 불일치로 실패**(`Unknown command type`) → 애드온 프리퍼런스의 `sketchfab_api_key` 를 읽어 **REST API 직접 호출**(`/v3/search`, `/v3/models/{uid}/download`)로 우회. 파이프라인 `scratchpad/sf_fetch.py`(검색·다운로드·조사) + `sf_pose.py`(자세·정규화·데시메이트·텍스처 축소·GLB). 산출 `Assets/KRAssets/UamCabinV2/CHAR_SF_{Doctor,Patient}.glb`(34k/30k면, 1.9/0.9MB). **전부 CC BY 4.0 — 표시 의무**, 출처는 같은 폴더 `ATTRIBUTION.md`. 환자 원본은 나체 T포즈라 Mixamo 본으로 앙와위를 만들고 런타임에 구조담요(`BuildPatientBlanket`)를 덮는다. ⚠단일 메시라 부위 관절이 없어 흉부압박·호흡은 **전신 모션**으로 표현(`_doctorRoot`/`_patientRoot`).
    - **3인 전원 실사(2026-08-03 5차)**: 대원도 `Nurse Surgical Rigged`(bachelorgkv, CC BY)로 교체. ⚠**착석 자세는 포기**했다 — 리그마다 본 로컬축이 달라 허벅지·무릎 회전이 뒤틀렸다(실측) → 팔만 내린 **서 있는 자세**로 후방에 배치(실내고 1.96m 라 성인이 선다). 텍스처는 원거리라 512(7.4MB).
    - **실내 진동 금지(사용자 결정)**: 원인이 **셋**이었다 — ⓐ계기 캔버스 흔들기(`screenShake=false`) ⓑ캐빈 리그가 기체 자세를 따름(`attitudeFollow 0`, 수평 유지) ⓒ**카메라가 앵커로 Lerp**(`UamCabinCamera`)+**리그 위치도 Lerp** → 카메라와 캐빈 사이에 상대 운동이 생겨 실내가 계속 흔들렸다. 카메라는 `SetPositionAndRotation` 으로 **강체 고정**, 리그 위치는 **스냅**, 대신 기체 `Rigidbody.interpolation=Interpolate` 를 켜 물리 스텝 계단을 없앤다. 실측: **카메라↔앵커 상대변위 최대 0.00000m**(120프레임).
    - **★한글 계기 글자가 "떨리며 깨진" 진짜 원인 = 레거시 uGUI `Text` 의 동적 폰트 아틀라스**(진동이 아니었다). 5개 캔버스가 서로 다른 크기로 CJK 글리프를 요구하면 아틀라스가 재구성되며 이미 그린 글자의 글리프가 빠진다 → **TMP(`TextMeshProUGUI`)** 로 전면 교체. ⚠`TMP_FontAsset.CreateFontAsset(Font)` 에 **OS 동적 폰트**(`Font.CreateDynamicFontFromOSFont`)를 넘기면 글리프 소스가 없어 한글이 전부 **두부(□)** 로 나온다 — 반드시 **TTF 경로** 오버로드(`CreateFontAsset(path, 0, 90, 9, SDFAA, 2048, 2048)`)로 만들 것(`Font.GetPathsToOSFonts()` 에서 `malgun.ttf` 검색). ⚠TMP 로 바꿔도 **동적 아틀라스를 런타임에 채우면 여러 캔버스가 동시에 새 글리프를 요청하는 프레임에 글자가 빠진다** → 쓰는 글자를 상수(`PRELOAD`)로 모아 `TryAddCharacters` 로 **사전 굽기**(실측 202자 1아틀라스). 새 문구를 추가하면 그 글자도 PRELOAD 에 넣을 것.
    - **화면 구성 변경(사용자 요청)**: 깊이 카메라는 저해상 회색 그라디언트라 "가짜 화면 같다"는 지적 → **미니맵**(HUD 와 같은 `UavOsmTileMap` 타일 + 자기기체/목적지 마커)으로 교체. **속도·고도는 페데스탈(가운데 아래)** 로 이동, 자율비행 카드는 조작·목적지·경로·배터리에 집중. ⚠페데스탈은 눕혀 있어 칵핏 시점에서 **아래 절반이 안 보인다** → 판독값을 화면 위쪽에 모을 것.
    - **초기 화면 "No cameras rendering"**: 지역 선택 패널이 뜨는 동안 씬에 카메라가 하나도 없다(FPV 카메라는 지역 확정 후 생성) → 부트스트랩이 `Boot_Camera`(cullingMask 0, 단색 배경)를 먼저 세우고 HUD 생성 시 제거.
    - **실내 시점은 2종만**(사용자 결정): `계기판`(무인 콘솔 앞) · `캐빈 후방`(계기판 뒤에서 후방 조망). 의사/환자 1인칭은 프레이밍이 불안정해 폐기 — `CameraMode`·`UamCabinCamera.View` 양쪽에서 제거.
    - **구조담요 = 드레이프 메시**: 박스를 얹으면 "뭔가 덮여 있다"로만 보인다 → `blanket_build.py` 가 **환자 메시에 레이캐스트**를 쏴 상면 높이장을 뜨고(평활 3회) 그 위 2.8cm 를 따라가는 곡면 + 가장자리 스커트 + 발끝/쇄골 접힘단을 굽는다(`UAM_Blanket.glb`, 12.7k면). 배치는 **환자와 동일 위치·요**(환자 로컬 좌표계에서 구웠으므로).
    - **★함정 3건**: ⓐ`sf_pose.py` 가 부모 엠프티를 지우기 전에 `parent_clear(CLEAR_KEEP_TRANSFORM)` 을 안 하면 glTF 축 보정이 통째로 사라져 **인물이 거꾸로 매달린다**(실측). ⓑ`obj.bound_box` 는 캐시라 `data.transform()` 직후 갱신 안 됨 → **정점에서 직접** bbox 계산. ⓒ원본 텍스처가 8192² 두 장이면 GLB 가 42MB → 1024 로 축소. ⓓ**상하 자동판별 휴리스틱(단면 폭 비교)은 정상 모델을 뒤집는다**(대원이 물구나무) — ⓐ만 지키면 애초에 바로 서므로 그런 안전망을 넣지 말 것. ⓔ**깊이 배열 행 0 은 화면 위쪽 레이인데 `Texture2D` 행 0 은 아래** → 그대로 넣으면 깊이 화면이 상하 반전된다(읽을 때 행 뒤집기).
  - **인물 v3(2026-08-03)**: `uam_people_v3.py` — 몸통·머리·사지에 **Bevel+Subsurf**(각진 상자 제거), 얼굴(코·눈두덩·눈·홍채·눈썹·귀·턱·입선), **손(손바닥+손가락4+엄지, `hand()` curl 파라미터)**, 의복 디테일(칼라·라펠·소매단·밑단·주머니·단추·벨트·견장·반사밴드). 파트 이름/원점은 v2 규약 그대로라 **Unity 무수정**. 면수 의사 15.1k·대원 13.5k·환자 12.3k.
  - **★캐빈 v3 — 실사 의료장비 + 실제 임무 데이터 (2026-08-04)**: 사용자 신고 3건(①계기 목적지가 `"SNU 병원 헬리패드"` 로 **하드코딩** ②실내 떨림 잔존 ③내장 고도화)을 처리했다.
    - **하드코딩 제거**: `UamCabinDisplays.State` 의 `wpName/fromName/toName` 초기값·`socPct 78` 을 전부 비우고, `UamMissionSetupPanel` 이 확정 임무의 **실제 지점명·좌표**를 정적으로 발행(`MissionOriginName/DestinationName/…Lat/Lon`, 프리셋 라벨·Kakao 장소명·OSM 헬리패드 번호 순, 좌표를 손으로 고치면 라벨 무효). 이름은 임무 이력(PlayerPrefs)에도 저장돼 "최근 임무로 바로 시작" 에서 복원된다. **계기가 거짓말하지 않도록** 실제 `navAgent.Goal` 과 임무 좌표가 200m 이상 어긋나면(=플래너가 무작위 폴백) 이름을 버리고 좌표를 띄운다. 재난 재생(`TracePlayer`)은 구간별로 주둔 헬기장→현장→배정 병원 실명을 매 틱 갱신(`UpdateCabinMission`).
    - **배터리**: 상수 감소(78%에서 초당 0.05%)를 폐기하고 **거리 기반 소비**로 교체(`packKWh 140`·`cruiseKW 175`/`45m/s` = **1.08 kWh/km** + 대기 12kW). ⚠순수 시간적분(호버 580kW)은 물리적으로 맞지만 **씬의 기체가 실기보다 훨씬 느려** 15분이면 0%가 된다(실측). 잔여 항속거리·비행시간·순간 소요전력을 같이 표시.
    - **떨림**: 카메라는 이미 앵커에 강체 고정이라 남은 원인은 **리그가 기체 위치·요 지터를 그대로 받아 창밖 세계가 떠는 것**이었다 → 리그에 **2차 임계감쇠 필터**(`SmoothDamp` 위치 0.28s·요 0.45s, `maxLagM 2.5`/`maxLagDeg 18` 로 클램프). 동일 함수로 실측: 프레임간 위치 흔들림 **17.4→0.40mm(−97.7%)**, 요 **0.508→0.022°(−95.6%)**.
    - **실사 장비(Sketchfab CC BY 7종)**: `sf_kit.py`(검색·다운로드·실척 정규화·데시메이트·텍스처 축소 → `KRAssets/UamCabinV2/EQ_*.glb` → `Resources/SeoulPilot/EQ_*.prefab`, 바닥정렬). 캐빈 GLB(`uam_cabin_v3.py` → **`Resources/SeoulPilot/UAM_CabinInteriorV3.glb`**, 프리팹 아님 = 재빌드 drop-in)는 해당 절차 장비를 **빈 앵커**(`Anch_Defib/Ventilator/IvStand/O2_0/O2_1/Extinguisher`)로 바꿨고, 런타임이 거기에 얹는다(천장 관통 가드 포함). 출처·라이선스는 `KRAssets/UamCabinV2/ATTRIBUTION.md` — **표시 의무**.
    - **누락 현실성 보강**: 의료가스 아웃렛(산소·흡인 포트)·창 롤러 블라인드·척추고정판·비상 탈출 손잡이/도어창 + **한글 안전표지 6종**(판은 GLB, 글자는 런타임 TMP: 비상구/산소/소화기/제세동기/의료폐기물/기내 안내) + **장비↔환자 연결선**(ECG 3리드+패드·SpO2 프로브·벽 산소포트→마스크·수액→팔, 처지는 베지어 튜브) + 실사 산소마스크를 **환자 메시 실측 바운즈**로 얼굴에 정렬 + 처치등(스포트)·**주야 반응 조명**(`_MCI_NightFactor` → 실내등↑·계기 디밍) + **CPR–ECG 동기화**(압박 위상이 파형 기저동요로 실린다).
    - **★함정 6건(전부 실측)**: ⓐ**셸이 그림자를 안 던지면 태양광이 캐빈을 관통**해 실내가 하얗게 날아간다(담요 albedo 0.34인데 화면 휘도 0.82·포화 38%) → `Cabin_Shell` 만 `ShadowCastingMode.On`. ⓑ glTF 재질을 **URP/Lit 로 교체하면 단면이 된다** → 한 겹 드레이프(담요)가 백페이스 컬링으로 **통째로 사라지고 환자 나체가 노출**됐다. `_Cull 0` 필수. ⓒ구운 담요 GLB는 실사 환자와 정렬이 안 맞는다 → **환자 바운즈 실측으로 런타임 생성**(크라운을 몸 높이와 같게 잡으면 격자 사이로 몸이 뚫고 나온다 → +6cm). ⓓ**머리 쪽 축 방향 시선에선 맨 가슴이 덮인 다리를 가린다** → 드레이프를 목까지. ⓔ장비 프리팹은 바닥정렬이라 얼굴 앵커에 그대로 놓으면 **마스크가 얼굴 위에 뜬다** → 렌더러 중심을 앵커에 맞출 것. ⓕ**MCP 가 두 Unity 인스턴스(UAV_test·CAR_test) 사이를 오간다** — `mcpforunity://instances` + `set_active_instance` 로 고정하지 않으면 에셋이 "없다"고 나오고 `Resources.Load` 가 null 을 뱉는다. 또 `execute_code` 가 **codedom 으로 폴백하면 Unity 메인스레드 API 가 전부 null/빈값**을 반환하니 결과를 믿지 말 것(`compiler:"roslyn"` 확인).
- **Blender 에셋→Unity 배선** (2026-06): GLB는 `_AssetStaging/`(Unity 옆·비트래킹; 세션 scratchpad는 정리돼 소실) 보관→`Assets/KRAssets/` 복사→glTFast(6.19) 자동임포트→`execute_code`로 `Resources/SeoulPilot/*.prefab`(바닥정렬 `child.position-=min.y`·**기존 리소스명 drop-in**=코드 무수정; AMB=`Ambulance_no_damage`·UAV=`PP_Drone`·`PoliceCar`·`FireTruck`·`Doctor`·`Patient`·`HospitalBed`…). 차량 스피닝=`WheelSpin`/`MultiRotorSpin`(eVTOL 분산로터) prefab 베이크. 한국 가로수 4종=`BuildStreetFurniture` 절차생성 종 분기(합본메시, prefab 인스턴싱 회피). 상세는 auto-memory `reference_asset_mcp.md`.
  - **★Blender MCP 도구가 세션에 없어도 헤드리스 CLI 로 다 된다 (2026-07-30)**: `"/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" --background --python <script.py> -- <outdir>` → `bpy` 로 메시 생성 → `bpy.ops.export_scene.gltf(export_format='GLB', export_yup=True, export_apply=True)` → Unity **`mcp__UnityMCP__import_model_file`** 로 임포트 → `execute_code` 로 프리팹화(바닥정렬 동일 규약). `-b` 모드에선 blender-mcp 애드온이 "cannot start server in background" 경고를 내지만 **스크립트 실행엔 무관**. ⚠️세션 **도구 목록에 없다 ≠ 미연결** — `claude mcp list` 로 확인할 것(이번에 `blender: ✔ Connected` 인데 "미연결"로 두 번 오보했다). 실제 산출 예: `Resources/SeaLife/{KR_FishingBoat,KR_NavBuoy}.prefab`(126정점/89면, 110/88).
- **이미 구현됨 — 재구현 전 확인**: NPC/AMB 차선준수(`TrafficManager.LaneOffset`+`TracePlayer.laneOffsetM`, 우측차선+일방통행), 도로 클래스 폭(`CLASS_W` 14등급), 황색 중앙선/백색 차선점선, 교량 데크+터널 포털(roads2 `struct`), `NightCityLights` 창문 발광(글로벌 `_MCI_NightFactor`→`BuildingTriplanar`), `ProceduralGait`(절차적 NPC 보행 — 보행자는 Animator 없음). **DEM `.bin` 소스(Copernicus GLO-30, `Scenes/Regions/terrain/`)는 깨끗**(2026-06 255 전수검증: 고립스파이크 2파일×1픽셀뿐, 나머지 편차≤33m). 과거 '스파이크'는 **정사영상 메시** 아티팩트(despike됨)·DEM 소스 아님. 건물/도로는 `GroundCeil`→`TerrainHeight.Ground`(깨끗 DEM)에 이미 정합 → ML 지형회피 정확. **블라인드 DEM despike=실제 산(문학산 216·마니산 467m) 훼손이라 금지**; 특정 해안 씬 정사메시 과평탄화 시 그 씬만 `RedrapeOrthoOne` 검증.
- **★전국 균질성 — "서울 전용 아님"(2026-07-28 감사)**: v1 트윈은 **255 시군구 전부** 씬·데이터·현실성 패스가 서 있다(마커 실측 `__terrain_rebased`/`__grass_applied`/`__ortho_despiked`/`__sidewalk_v1`/`__water_applied`/`__bldg_collider`/`__furnver7` = 255/255, OSM roads2·feat·area·poi·infra·footways·ngiiroads 255/255, hdmap 13레이어×255). 그럼에도 **서울 편중 결함 4건**이 있었고 전부 수정했다: ①DEM 스무딩 서울 25구만(→전국 255, 위 DEM 항목) ②건물높이 인덱스가 강남 1곳뿐이라 `UamCityLife` 가 `Ensure` 실패 시 장애등·옥상프롭을 **전부 `SetActive(false)`**(목포 실측: 장애등 0개) → ⓐ`BuildingHeightIndex.EnsureAny` + **메시 프로브 폴백**(Building 레이어 격자 하향 레이캐스트, 조건별 캐시 슬롯, 뷰어 120m 이동 시에만 재프로브; 최초 26ms·캐시 0ms) ⓑ인덱스 자체를 전국 255 수집(위 DEM 항목). 목포 장애등 0 → 프로브 5(무명) → 인덱스 14(실명·층수) ③`GeoPointerHUD`·`GeoLocateWindow` 가 **수도권 프레임 고정** `KoreaGeo.LatLonToWorld` 를 써 타 지역 좌표 지시가 어긋남 → `RegionRegistry.TryWorld` 로 교체(CLAUDE.md 좌표계 항목의 경고를 코드가 어기고 있었다) ④`TerrainSmoothBatch` 메뉴 서울 하드코딩. **로드 UI**: `MapVersionSelector` 에 `Step.SidoPick`(시도 통째 로드 + 시군구 수·용량 GB 표시, 4GB↑ ⚠) 추가, 자유조종은 프리셋 9개 광역시 + **시도→시군구 선택기**(좌표 몰라도 255개 진입). 여전히 강남 전용인 것 = **v2 타일(정사·건물 베이크)뿐**.
- **디지털트윈 v2 재구축(진행 중·로컬-C)**: v1(255 시군구 씬, 백업 보존) 위에 **1km² 타일 k-ring 스트리밍 + HD맵 + 콜라이더** 트윈을 신규 구축(파일럿=강남). 베이커 `Assets/Editor/V2/TileBakerV2.cs`(`BAKE_VER` bump=전량 재베이크, 1타일/틱 BG드라이버 멱등; 레이어=지형드레이프·정사4096·건물압출·**HD 차도면(_RoadSurf)**·OSM 이면도로리본·HD노면선/보도·신호/방지턱/가드레일·**B3노면표시/C6지주/B1표지**·MeshCollider → `Assets/TilesV2/<region>/tile_x_z.prefab` + `_manifest.asset`), 런타임 `Assets/Scripts/V2/{TileIndex(5186격자↔world/tile-local·k-ring), TileManifestV2(프리팹목록 SO), TileStreamerV2(뷰어중심 k-ring 로드/해제)}.cs`. 좌표=EPSG:5186 1km 전역격자(프레임무관), 배치만 `TileIndex.TryOriginWorld(frame,tile)`. 데이터 버킷터(**origin 추적**) `tools/v2_{ortho_warp,buildings_bucket,roads_bucket,hd_bucket,hd_bucket3c,hd_bucket_a3}.py` — ⚠️`v2_ortho_warp.py`만 `qgis_batch` env(GDAL, 3857→5186 워프), 나머지 `UAV` env(pyproj·Liang-Barsky 타일클립; **면 클립은 shapely 교차** — SH는 오목폴리곤서 500m+ 브리지 삼각형 유발로 기각). 타일 산출물(`tools/nationwide_v2/`·`Assets/TilesV2/`)은 gitignore. 계획/진행=`~/.claude/plans/uav-test-peppy-dream.md` + auto-memory `project_twin_v2_rebuild.md`.
- **★도로 3D 전환 + 파사드 v2 (2026-07-11, BAKE_VER 8→9)**: vWorld WFS 가 **정밀도로지도 전 레이어를 z(고도) 포함**으로 서빙(`hdmap_fetch.py` HD_LAYERS 13종: 기존 5종 + `lt_c_a3drivewaysection` 차도면·`lt_l_a2link` 주행링크(laneno/r_l_linkid/from/to 위상)·`lt_p_a1node`·`lt_c_b3surfacemark`·`lt_p_b1safetysign`·`lt_l_c5heightbarrier`·`lt_p_c6postpoint`·`lt_c_a5parkinglot`). **도로면=A3 폴리곤을 데이터 z로 직접 삼각분할**(고가/교량 실높이+데크스커트, DEM 드레이프 폐기), 노면선/가드레일/신호도 소스 z(점선 dash='D' 구현), **지형은 도로에 카빙**(`SurfSampler.CarvedGround`: 면내 |도로z−DEM|≤8m→도로z−0.06, 밖 18m 사면 블렌드; 초과=교량/터널 비카빙 — GLO-30 DSM 도시 바이어스 평균 −2.1m 실측), **OSM 리본은 A3 커버 60%+ 조각 제외**(HD/OSM 이중 소스 = 차선 어긋남 주범 제거). 파사드 v2: `v2_buildings_bucket.py` 가 `높이 유형 lon lat…`(유형=usability 코드+층수 폴백: 1아파트/2업무/3근생/4주택) 방출 → 베이커가 **버텍스컬러 r=유형*32·g=바닥고도*2** 인코딩 + 옥상 프롭(옥탑·물탱크, centroid PIP 가드) → `BuildingTriplanar` 셰이더가 유형 분기(아파트 발코니슬래브/파스텔·업무 커튼월·근생 1~2층 통유리+간판밴드·주택 벽돌; 무컬러=v1 해시 폴백·상대높이 ly 로 1층 정렬). 실사 3D 건물 외부 데이터는 전 경로 막힘(vworld XDO 폐쇄/네이버·카카오 미제공/Google 3D Tiles 한국 미커버) → **NGII 3차원 가시화모델·S-Map 신청 병행**(가이드 `docs/3d_building_data_guide.md`).

- **★v1 시군구 씬 도로 = 정밀도로지도 기준 전면 재임포트 (2026-08-16, `Assets/Editor/V2/HdRoadImporterV1.cs`)**: v1 도로는 OSM 중심선을 DEM 에 드레이프한 리본이라 **수평**(OSM 간선이 실제 차도면에서 p90 4~6m 이탈)·**수직**(GLO-30 DSM 이 실노면보다 높은 구간 65.8%·중앙값 1.33m·p05 7.21m) 둘 다 틀렸다. 255 시군구 씬을 1씬/틱 배치로 돌며 ①정사 지형메시를 **8m 격자 재세분**(마커 `__ortho_hd8_v1`) ②구 OSM 리본 렌더러 소등(**`RegionRoadNetwork` 데이터는 보존** — NPC/RL 소비자) 후 A2 차선리본 노면·B2 노면선·A4 보도+연석을 **데이터 z 그대로** 신규 빌드(마커 `__hdroads_v1`, 루트 `Vworld_<region>_HdRoads`) ③HD 차선 기준 지형 카빙. 메뉴 `Tools/MCI/HD Roads/*`.
  - **골목 예외처리**: 정밀도로지도는 이면·골목 미수록 → OSM 도로 정점의 **과반이 HD 차선 4.5m 안**이면 중복으로 버리고, 아니면 골목으로 보고 기존 기하를 새 아스팔트 재질로만 다시 굽는다(강남 유지 4,319/제외 1,044).
  - **정점 예산**: 무단순화 강남 5.0M 정점 → 3D Douglas–Peucker(노면 0.12m·도색 0.05m·보도 0.15m)로 **노면 6.8배·노면선 4.9배** 감소, 이어서 `Densify(60m)` 로 다시 쪼갠다(**PhysX "두 정점 간 500 단위 초과" 경고**가 콜라이더 안정성을 깎기 때문). 씬 255→496MB(강남).
  - **카빙 규약**: 완전반경 12m → 기울기 0.12 사면 → 페더 60m, 한 정점 최대 하강 15m(터널·산 보호). ⚠️좁고 가파른 사면(9m·0.26·30m)은 **눈높이에서 지형이 도로 위 벽으로 남는다** — DSM 이라 도로 *사이* 지면이 통째로 부푼 게 원인이라 넓고 완만해야 한다. 강남 실측: 회랑 내 지형정점의 **3.8%**만 노면 위 잔여(평균 +52m = 구룡산·대모산 실제 산).
  - **★★프레임을 좌표로 역추정하지 말 것**: `RegionRegistry.TryWorld(lat,lon,out _,out frame)` 는 점이 속한 광역 폴리곤을 주는데 경계 지역은 씬이 구워진 프레임과 다르다 — **부산 금정구 실측 TryWorld="Gyeongnam" vs 씬="Busan" → 도로가 453km 밖**(1차 255씬 전량 폐기). 반드시 **`sgg.json`/`region_index.json` 의 `frame` 필드**를 읽고, **차선점의 30% 미만이 정사 bounds(여유 2km) 안이면 예외로 실패**시켜 조용한 저장을 막는다.
  - ⚠️**`ORTHO_MAX_N` 상한 재발**: 400 이면 정사가 **8.1km 단일 타일**인 지역(부산 금정/북구 등)에서 목표 8m 가 실제 **20.4m** 로 구워져 카빙이 헛돈다. `TerrainImporter.OrthoMaxN`·`OrthoTargetSpacingM` 을 static 으로 열고 임포터가 **지역 정점예산 2.5M** 기준으로 계산해 주입(160,801→1,042,441 정점). **마커만 믿고 스킵하지 말고 실측 간격을 재서** 판정한다.
  - **재질(v1·v2 공유)** `Assets/Environment/HdRoad_{Asphalt,Paver,Curb,Mark{W,Y,B}}.mat` ← Blender 로 구운 타일링 텍스처 `Assets/TilesV2/_shared/tex/{AsphaltV2,PaverV2,GraniteV2}_{alb,nrm}.png`(2048²). **이음매 없음의 열쇠 = UV 를 4D 토러스로 매핑**(cos/sin 2πu, cos/sin 2πv → Noise/Voronoi 4D). 노멀맵은 알베도 휘도 → wrap Sobel(`tools/v2_tex_normals.py`). ⚠️Blender EMIT 베이크는 대상 이미지 노드를 **selected + active 둘 다** 해야 한다(active 만 하면 두 번째 베이크가 통째로 검게 나온다).
  - **v2 `TileBakerV2` 도 동반 수정(BAKE_VER 26→27)**: 같은 재질 사용 + **`SaveSub` 가 UV 미지정 시 타일로컬 XZ 평면 UV 자동 생성**(구 동작=전 정점 UV (0,0) → 텍스처가 단색 한 픽셀로 뭉개짐 = "회색 판때기 도시"의 진범) + `_Sidewalks` 를 `COLLIDER_LAYERS` 에 추가(보이지만 못 밟는 면이었다).
  - **전국 버킷 드라이버**(재개 가능 스탬프 `tools/nationwide_v2/.bucket/`): `tools/exp_drivers/run_hd_bucket_all.sh`(a3+lanes+3c) · `run_roads_bucket_all.sh`(OSM 골목 폴백) · `run_hd_lanes_rerun.sh`. 결과 **hd_links 251/255 지역·49,620 타일**(미수록 4곳=`busan_{namgu,suyeonggu}`·`gyeongbuk_ulleunggun`·`incheon_ganghwagun`), roads_tiles 255/255.
  - ⚠️**원천 z 쓰레기값**: A4 보도에 `z=-3.7e306`(강남 302,960점 중 3개) → float 로 −Infinity → 메시 bounds NaN → 청크 소멸. `v2_hd_bucket.py` 에 `sane_z`+`repair_isolated_z` 추가(3c/a3 는 이미 적용돼 있었고 **이 스크립트만 빠져 있었다**), C# 쪽에도 방어 가드.
  - ⚠️**수직 하향 카메라는 `LookRotation(Vector3.down, Vector3.up)` 이 퇴화**한다 — 임의 방향을 잡아 "지면이 갈색 단색"으로 찍혀 씬 버그로 2회 오진했다. `Vector3.forward` 를 up 으로 줄 것. 레이어 범인 찾기에 **평균색 비교 금지**(정사 0.355 vs 회색 블랭킷 0.341 — 눈엔 완전 다른데 평균은 같다).

- **★v1 씬 층(고가/지하) 분리 + 교각·방호벽 + NPC 차선주행 + 지면 재접지 (2026-08-18)**: 위 재임포트가 도로 *평면*은 맞췄지만 **층을 몰랐다** — 고가 아래 지형이 파이고, 고가 밑 골목이 "중복"으로 지워지고, 지하차도가 지상 노면과 한 덩어리로 취급됐다(사용자 신고 "다른 길인데 같은 좌표라 하나로 취급").
  - **데이터**: `v2_hd_bucket_a3.py` 가 `hd_links` head 에 **`roadtype` 을 8번째 토큰**으로 추가(`laneno linktype roadrank from to rlink llink roadtype`). 소비측은 `tok[7]` 에 `.` 이 있으면 구 7토큰 파일로 보고 지표로 간주(하위호환). `--only a2` 로 A2 만 재버킷 가능(4.2s/지역). 전국 255/255 재버킷 완료(드라이버 `tools/exp_drivers/run_hd_links_roadtype.sh`, 스탬프 `.bucket/<region>.a2rt.ok`).
  - **★★roadtype 코드만 믿으면 안 된다(전국 실측)**: 같은 XY 근방 지표도로 대비 Δz 중앙 — roadtype 3 이 **강남 +7.59m 인데 춘천 +1.25m · 제주 +0.93m · 부산 해운대 +1.55m**. 같은 코드가 지역마다 다르게 쓰였다. 반대로 **강남에서 실제로 다른 도로 위를 지나는 상판 1,497개 중 1,101개가 roadtype 1**이다. ⚠️구 문서의 "A3 roadtype 1 지표 / 3 고가 / 4 지하차도"는 **강남 한정 관측**이었다 — 전국 규약이 아니다. 신뢰 가능한 건 **2 터널 · 4 지하차도**(전 지역 음수)뿐.
  - **판정 = 기하 1순위**(`HdRoadImporterV1.ClassifyLayers`): 같은 XY 7m 격자에 **3.5m 이상 아래로 다른 차선**이 지나가면(점의 30% 이상) 상판. ⚠️비교 기준 격자에서 **터널/지하차도(roadtype 2·4)를 반드시 뺀다** — 안 빼면 "지하차도 위를 지나는 평범한 지표도로"가 상판으로 오분류돼 교각이 서고 카빙에서 빠진다(인천 연수구 실측 Δz 중앙 2m 미만). 아래에 도로가 없는 **하천 교량**은 roadtype 3/5 + DEM 대비 4m 이상일 때만 보조 인정(제주처럼 지표에 붙은 roadtype 3 을 통째로 띄우지 않기 위한 안전장치).
  - **층별 산출**: `_HdRoadSurf`(지표, 콜라이더) / `_HdElevated`(상판 위+하면) · `_HdPiers` · `_HdGuardrails` / `_HdUnderground`(**콜라이더 없음** — 지상 물리와 섞이면 안 된다). **카빙·골목 중복판정은 지표 차선만** 기준(고가 밑 골목이 살아난다: 강남 유지 4,319→4,572).
  - **교각/방호벽**: 바깥 차선 식별은 `r_linkid`/`l_linkid == "-"` (그쪽에만 방호벽·거더 스커트). 교각은 **중앙분리대측(왼쪽 이웃 없음) 차선에만** 30m 간격, **아래 지상 차선 5.5m 안이면 종방향 ±10m 로 피하고 그래도 걸리면 미설치**(밑 도로 통행 보장). 성토 구간(상판 하면−지면 < 2.5m)은 생략. 강남 실측 **교각 2,432 설치 · 125 회피**.
  - **NPC 차선주행**: `BuildTrafficNetwork` 가 A2 링크를 `tonode→fromnode` **양방향·긴 링크 우선**으로 이어 `_HdTrafficNetwork`(`RegionRoadNetwork`, `laneCenterlines=true`)에 굽는다. ⚠️**150m 미만 경로는 등록 안 한다** — `TrafficManager` 는 도로 길이로 위치를 순환시켜(`car.s -= rd.length`) 짧은 교차로 연결차선에선 차가 5초마다 출발점으로 되감긴다. 강남 실측 등록 1,779경로·중앙 465m·**연장 91.0% 커버**. `TrafficManager` 는 씬에 `laneCenterlines` 망이 있으면 **OSM 중심선망을 통째로 무시**하고, `LaneOffset` 도 0 을 준다(폴리라인 자체가 차로 중심).
  - **지면 재접지 `GroundSnapPass`**(마커 `__ground_snap_v1`): 건물이 뜬 원인은 DEM 해상도가 아니라 **기준면 불일치** — 건물/도로/공원은 `RebaseGroundLayers` 가 **DEM json** 으로 앉혔는데 정사 메시만 그 뒤 8m 재세분+카빙으로 내려갔다. DEM 을 다시 읽지 않고 **정사 메시 정점에서 직접 높이장**(4m 격자)을 만들어 델타를 잰다. 건물은 청크 메시를 **동 단위 union-find**(삼각형 연결 + 5cm 위치 일치)로 갈라 **강체 평행이동**(지붕이 기울지 않게) 후 **바닥 정점만 지형 아래로 내려 경사 스커트**. 리본/합본 프롭(`_Roads/_Areas/_Sidewalks/_Furniture/_Infra`)은 정점별 델타, 공유 프리미티브(`_POI/_Features`)는 **트랜스폼 y**(정점 수정 금지). 강남 실측 **32,138동 재접지(평균 0.91m·최대 14.91m) · 리본정점 1,526,736 · 프롭 2,005**.
  - **NPC 바퀴 회전 제거**(사용자 결정): 시민차 프리팹은 바퀴가 차체와 한 메시라 이름 노드가 0개 → 형상 추정이 차축·측면 패널을 바퀴로 오인한다("바퀴축이 도는" 신고). `TrafficManager.SpawnModel` 이 인스턴스의 `WheelSpin` 을 제거한다. 히어로 차량(구급차, 덩어리 분리로 실측 검증됨)은 유지.
  - **철로 = 지하철 필터**: `osm_infra.py` 는 railway=rail/**subway**/light_rail/tram 을 전부 `rail` 로 묶어 강남 149개가 전부 지하철인데 지상에 검은 궤도가 깔렸다. 재페치 없이 **정밀도로지도 지표 차선**(반경 6m)을 기준으로, 점의 과반이 차도면을 따라가면 지하로 보고 미표시(`HdRoadImporterV1.GroundLaneProbe`). 같은 판정기로 **가로수·가로등이 실제 차로 위에 서던 문제**도 막는다(`BuildStreetFurniture` 거부 그리드에 추가, 마커 `__furnver9`→**`__furnver10`**).
  - **실행 통로 `HdSceneRebuildBatch`**: 두 패스는 **순서가 의미를 갖는다**(카빙으로 지면이 내려간 뒤에 접지해야 한다) → `ImportOne` → `SnapOne` → 저장을 1씬/틱으로 돌린다. 에디터가 열려 있어 batchmode CLI 를 못 쓸 때를 위해 **요청 파일** `Temp/hd_rebuild.request`(`skipDone=0 region=<name>` 또는 `shot=<region> lat= lon= alt= pitch= yaw=`)를 **2초 주기로 폴링**하고 진행은 `Temp/hd_rebuild.log` 에 append. 스크린샷은 `Temp/shots/*.png`.
  - **★재발 함정 3건**: ⓐ**요청 감시를 `delayCall` 한 번으로 두면 안 된다** — 에디터가 포커스를 못 받으면 Auto Refresh 가 안 돌아 새 요청을 영영 못 집는다(실측 6분 무반응) → `EditorApplication.update` 주기 폴링. ⓑ**이미 포커스인 창에 `AppActivate` 하면 focus 이벤트가 안 뜬다** — 스크립트 변경을 컴파일시키려면 **다른 창을 거쳤다가** Unity 를 활성화해야 한다. ⓒ**검증용 런타임 카메라엔 `UniversalAdditionalCameraData` 가 없어** opaque/depth 텍스처가 안 만들어져 물·굴절이 단색으로 뭉개진다 → 씬 버그로 오진하기 전에 카메라에 붙일 것.
  - 검증 `tools/v2_hd_layer_check.py`(층 분리 Δz · 주행망 체이닝 길이/커버율 · 교각 회피율을 C# 과 같은 규칙으로 재현). 6개 대표 지역 통과. ⚠️제주처럼 상판이 전부 도로 바로 위인 지역은 **교각 0 이 정상**이다(그 구간은 실제로도 교각이 도로 밖 성토부에 있다).

- **★고가 구조물 물리화 + 진입로 교각 + 차도 위 건물 제거 + 자유비행 센서 (2026-08-19, 마커 `__hdroads_v2`/`__ground_snap_v3`)**
  - **구조물이 장애물이 아니었다**: HD 루트 이름이 `Vworld_<region>_HdRoads` 라 `PhysicsSceneSetup.IsGroundLike` 의 "Roads" 규칙에 걸려 **교각·방호벽까지 전부 Ground 레이어**가 됐고, UAV 장애물 인식(`Building` 마스크)이 교각을 통째로 못 봤다. `HD_STRUCTURE` 토큰(`_HdPiers`/`_HdGuardrails`/`hdpier`/`hdrail`/`hddeckunder`)으로 먼저 걸러 **Building** 으로 보낸다 — 상판 윗면(`hddeck`)은 주행면이라 Ground 유지. 난간·상판 하면·교각은 임포터에서 **콜라이더 on**.
  - **추락방지벽은 id 로 세우면 안 된다**: `r_linkid`/`l_linkid == "-"` 기준은 강남 고가 540링크 중 **180개(33%)가 "양쪽 이웃 있음"** 으로 나와 난간이 안 섰다(그 이웃이 상판이 아닌 경우가 섞인다). **기하 판정**(`OuterEdges` + `PointGrid3`)으로 교체 — 옆으로 차로폭(3.6m)만큼 밀어 **같은 높이(±2.2m)의 상판 점**이 실제로 있는지 본다. 강남 방호벽 **2,672개** 생성.
  - **★교각은 공간 중복제거가 필수**: 바깥 가장자리 차선마다 세우면 **기둥 숲**이 된다(강남 8,304개 — 고가 아래가 통째로 막혔다). `PierSet` 으로 13m 안(높이차 6m 이내)의 중복을 걸러 **3,303개**로 정리했다. 실제 교량 지간이 30~50m 인데 나란한 차선마다 기둥을 세우는 건 애초에 틀린 모델이다.
  - **★진입로에 다리가 없던 이유**: 고가 **본선**은 아래에 도로가 있어 기하 판정에 잡히는데, 그리로 올라가는 램프는 아래에 아무것도 없어 지표로 남았다(사용자 신고 "대교 진입로에 다리 미반영"). `ClassifyLayers` 끝에 **연결성 전파** 추가 — 고가 링크와 노드로 이어지고 **DEM 대비 2.5m 이상 떠 있으면** 같은 구조물로 승격, 끊길 때까지 반복. 강남 고가 1,877→**2,510**, 교각 2,432→**8,304**.
  - **아스팔트 위로 올라온 지형**: 카빙에 걸린 15m 상한이 원인이었다. 카빙 장을 **지표 차선만**으로 만든 뒤로는 차도 바로 위에 솟은 지형이 전부 오차이므로, `CarveField.core`(코어 반경 안 표식)를 두고 **코어는 상한 없이** 깎는다. 코어 밖은 산·절토부 보호를 위해 상한 유지.
  - **차도 위 건물 제거**(`GroundSnapPass`): 교차로 한복판에 남던 동은 대개 **지하상가/지하도 shp** 가 DEM 오차로 지상에 압출된 것이다. 동 단위 union-find 결과별로 **기초 정점의 60% 이상이 정밀도로지도 지표 차선 2.6m 안**이면 그 동의 삼각형을 제거한다. ⚠️바운즈 면적 20,000㎡ 초과 덩어리는 **여러 동이 붙은 것**이라 제외(줄집 병합 오제거 방지). 강남 **69동 제거**.
  - **자유비행에 자율비행 센서 스택**(`FreeFlightSensorRig`): LiDAR(72×5·400m)+깊이(64×48)+GPS 관측을 붙이고 10Hz 로 틱한다. ⚠️`ResetSensorDomain` 은 `Academy.Instance` 를 만지므로 학습 밖에선 부르면 안 된다 — `ScanLidar`/`TickNavigation`/`RenderDepth` 는 Academy 무관이라 그대로 쓴다. HUD 는 **`J`** 토글(폴라 스코프·깊이 컬러맵·최근접/TTC/GPS).
  - **★재발 함정 2건**: ⓐ**Play 중에는 에셋 새로고침이 `NoUpdateAssetOptions` 로 돌아 스크립트가 컴파일되지 않는다** — 포커스를 아무리 줘도 소용없다. 외부에서 진행하려면 `WScript.Shell.SendKeys('^{p}')` 로 Play 를 빠져나와야 한다(Editor.log 에 `FreePilotController:Update` 가 보이면 Play 중). ⓑ요청 파일에 `import=`/`shot=` 같은 **새 키워드를 넣어도 구 어셈블리는 그걸 모른다** — 파싱에서 안 걸리고 기본 분기(전체 배치)로 흘러 엉뚱한 작업이 돈다. 새 키워드는 반드시 **재컴파일 확인 후** 투입할 것.

- **★NPC 현실성: 뷰어 스트리밍 · ITS 비례 보행자 · 지상 신호 · 가로 프롭 (2026-08-19, `__hdroads_v4`)**
  - **"도심이 휑한" 원인은 밀도가 아니라 배치 범위였다**: `TrafficManager` 가 시작 시 **도시 전역**에 900대를 뿌려서, 시군구 하나가 수십 km² 인 탓에 눈앞은 텅 비고 자유비행으로 이동한 새 지역엔 차가 0대였다(보행자 80명도 같은 구조). **뷰어 중심 스트리밍**으로 전환 — 차 420m 유지/620m 회수 · 보행자 220/340 · 신호 190/260 · 가로 프롭 150/210. 회수는 **화면 밖일 때만** 한다(눈앞에서 사라지면 안 된다). ⚠️근접판정을 **도로 중점**으로 하면 안 된다 — HD 주행경로는 최장 16km 라 중점이 8km 밖이어도 발밑을 지난다. 도로 표본을 200m 격자에 올려 뽑는다.
  - **`LiveTrafficService`**(신규): ITS 실시간 소통정보(`openapi.its.go.kr:9443/trafficInfo`, 키=env **`ITS_API_KEY`**)를 **뷰어 위경도 ±0.02°** 로만 조회한다 — 자유비행은 전국을 돌아다니므로 고정 bbox 로는 못 쓴다. 90초 주기 + 1.5km 이동 게이트로 호출을 억제하고, 키가 없거나 실패하면 **시간대 곡선으로 폴백**(출퇴근 0.85 / 새벽 0.05)해 씬이 비지 않게 한다. 차 대수 = `MAX_CARS×(0.55+0.45·혼잡)`.
  - ⚠️**유동인구 API 는 국내에 공개된 게 없다** → 보행자 수도 이 교통 혼잡도에 비례시킨다(120~460명, 사용자 결정). 정확한 유동인구는 아니지만 "차가 막히는 시간·장소면 사람도 많다"는 상관은 강하다.
  - **지상 신호체계**: 정밀도로지도 C1(`hd_signals`, 255지역)을 **`RegionSignalSet`(위치·방위·현시그룹)으로만** 굽고 `SignalStreamer` 가 뷰어 근처만 `KR_MastSignal` 프리팹으로 세운다 — 강남만 6,720개라 씬에 심으면 씬도 프레임도 못 버틴다. HD 신호가 있는 지역은 **구 OSM 신호 그룹(`_Features/Signals`)을 끈다**(이중 배치 방지). 현시는 접근 방위를 90° 양자화한 0/1 그룹으로 **반주기 오프셋** → 같은 교차로의 직교 방향이 번갈아 녹색이 된다. NPC 는 기존 `TrafficSignal` 로직 그대로 적색에 선다.
  - ⚠️**신호 지주를 최근접 차선에서 한 번만 오프셋하면 안 된다** — 다차로 도로에선 **바로 옆 차로 한복판**에 선다(실측 44%가 차도 위였다). 차로폭(3.25m)씩 밀며 어떤 지표 차선에도 안 닿는 자리를 찾고, 좌우 모두 실패하면 그 신호는 포기한다 → 강남·부산중구·제주 **차도 침범 0%**. 검증 `tools/v2_hd_signal_check.py`.
  - **가로 프롭 킷**(Blender 헤드리스 `tools/blender_kr_cityprops.py` → `Resources/SeoulPilot/KR_CityProps.glb`, **8품목 254면**): 상가 돌출간판 2종·입간판·현수막·에어컨 실외기·자판기·화단·볼라드 — 한국 거리를 한국처럼 보이게 하는 요소가 통째로 없었다. `CityPropStreamer` 가 인도 띠에서 바깥으로 레이를 쏴 **벽(Building 레이어)에 맞으면 벽걸이(간판·실외기), 못 맞히면 지상 프롭**을 놓는다. ⚠️프리팹을 따로 만들지 않고 **GLB 루트의 자식을 이름으로 복제**한다(프리팹 생성 단계를 없앤 배선).

- **★차도 배제 프로브 · NPC 전량 물리 · 보행자 자유배회 · 도로걸침 건물 띄우기 (2026-08-19, `__ground_snap_v5`)**
  - **간판이 차도 한복판에 서던 원인**: 인도 위치를 "차선 중심 + 5.2m" 로만 잡았다 — 교차로·회전반경·중앙분리대에서는 그 자리가 그대로 차도다. **`RoadSurfaceProbe`**(신설)로 판정을 바꿨다: 하향 레이캐스트가 맞은 **콜라이더 이름**이 노면 태그(`hdsurf`/`hdalley`/`hddeck`)면 차도다 — 기하 근사보다 정확하고 교차로에서도 안 틀린다. 프롭은 **중심 + 네 귀퉁이** 전부 통과해야 놓는다. 보행자 배회 목표·스크린샷 미리보기도 같은 규칙을 쓴다.
  - ⚠️**검증장비가 오해를 만들었다**: `HdSceneRebuildBatch.Shot` 의 미리보기가 프롭을 **신호 위치 옆에 그냥** 놓고 있어서, 실제 배치 로직과 무관하게 스크린샷에 "차도 위 간판"이 찍혔다. 미리보기도 실제 규칙(`Walkable`)을 쓰도록 고쳤다 — **검증 코드가 본 코드와 다른 규칙을 쓰면 없느니만 못하다**.
  - **NPC 물리 인식**: `NPC_BODY_MAX = 320` 이라 나머지 NPC 는 콜라이더가 없었다(도시 전역 900대 시절의 절충). 뷰어 반경 유지로 바뀐 뒤로는 전량 부착이 가능해 제한을 없앴다 — 이제 모든 시민차가 박스 콜라이더 + kinematic Rigidbody(Vehicle 레이어)를 갖는다. 보행자는 원래부터 캡슐 콜라이더가 있었다.
  - **보행자 자유 배회**: 인도 폴리라인만 따라 걸어서 광장·공개공지·상가 앞마당에 사람이 한 명도 없었다. `WANDER_FRAC 0.45` 가 폴리라인을 벗어나 배회한다(목표 4~34m, 9초 홀드, 지면 0.4초 캐시). **차도로 들어서면 즉시 목표를 재설정**한다.
  - **★도로를 걸치는 건물은 지우는 게 아니라 띄운다**(사용자 지시 "다리가 받쳐주는 형태로"). 지하철역 상부·복합환승센터·육교형 건물은 발자국이 도로에 **일부만** 걸치는데, 지우면 도시가 비고 두면 차도를 막는다. 기초 정점의 차도 겹침 비율 f 로 3분기 — **f≥0.6 제거**(지하상가 shp 가 DEM 오차로 압출된 것) / **0.12≤f<0.6 → 노면 최고고 +5.2m 로 띄우고 도로 밖 기초 자리에 기둥**(7m 간격·동당 14개 상한·콜라이더 포함, `Vworld_<region>_BuildingSupports`) / **f<0.12 유지**. 강남 실측 **제거 291동 · 띄움 114동 · 기둥 147개**.

### ★KDT 통합 + 경계 소유권 정합 (2026-09-01, `__hdroads_v5` / `__ground_snap_v6`)

세 모드(자율비행·자율주행·자유모드)를 하나의 프로젝트로 묶고, ML-Agents 전이학습 전에
기하·물리 정합성을 바로잡은 패스. 사용자 신고 5건 중 **③아스팔트/흰선/중앙선 부정확 ④건물
부양 ⑤아스팔트 매몰·이격**은 각각의 버그가 아니라 **하나의 뿌리**를 공유했다.

- **엔트리 통합** — `UavKoreaAutonomyFlight.unity` → **`KoreaDigitalTwin.unity`**(3모드 통합
  엔트리, `UamMissionSetupPanel` 이 모드·카카오 장소검색·시군구 선택·F9 복귀를 전부 갖는다),
  `KoreaAutonomousDrive.unity` → **`KDT_DriveTraining.unity`**(모드가 아니라 ML-Agents 학습
  하네스). 개명은 `AssetDatabase.RenameAsset`(GUID 유지 → Build Settings 자동 추종).
  경로는 `Assets/Editor/KdtScenes.cs` 한 곳 + 구 이름 폴백. 메뉴 `Tools/KDT/World/*`.
  `MapVersionSelector` 의 **중복 3모드 화면 222줄 삭제**(자유모드 스폰 버그가 이쪽에만 있었다) —
  SampleScene 은 이제 MCI 재난 재생 + KDT 링크뿐.
- **★경계 중복이 ③④⑤의 공통 뿌리** — `hdmap_fetch.py`·`vworld_buildings_batch.py` 는 시군구
  **bbox** 로 페치한다(폴리곤이 아니다). 실측: 강남 hd_links **26,165 중 17,616(67%)이 서초
  버킷에도** 있고, 강남 건물 인덱스 첫 줄이 **송파구 롯데월드타워**다. 지역 씬은 3×3 으로 겹쳐
  로드되니 같은 데이터 z 의 리본이 2~3장 겹쳐 **z-fighting** 이 난다(강남역 250m 실측: 두 씬의
  도색 정점이 **2,988 / 2,988 로 완전 동일**). 정사만 `OrthoClipPass` 로 소유권을 자르고
  도로·도색·보도·건물은 아무도 안 잘랐다.
  → `OrthoClipPass.OwnerAt(lat,lon)` 를 public 으로 노출하고 **같은 중재 규칙**을
  `HdRoadImporterV1.ReadPolys`(모든 HD 레이어의 단일 깔때기)와 `GroundSnapPass` 건물 루프에 적용.
  폴리라인 **중점** 하나로 통째 귀속시킨다(점 단위로 자르면 경계에 이음새가 생긴다).
  ⚠️카빙장·골목 중복판정·교각 회피·도색 높이 기준·상판 이웃 판정은 **거르기 전 전체 점**을 쓴다.
  ⚠️**정밀도로지도가 없는 구**(`busan_{namgu,suyeonggu}`·`gyeongbuk_ulleunggun`·`incheon_ganghwagun`
  버킷 4곳이 빈다)가 소유권을 이기면 아무도 안 그려 도로가 **사라진다** → `HasHdData` 가드.
  검증: `tools/v2_hd_owner_check.py`(C# 과 같은 규칙, Unity 없이 실행) — 지역별 제외율이 C# 로그와
  **소수점까지 일치**(금정 25.9% · 서산 35.2% · 강남 54.2%), 11개 지역 표본 **도로 손실 0건**.
- **갓길(apron)** — 차선 리본은 반폭 1.9m 에서 뚝 끊기고 그 바깥은 카빙 지형이라 도로마다
  `SURF_CLEAR+CARVE_CLEAR` 만큼의 **턱이 실선처럼 따라 그어졌다**(구 35cm = "아스팔트가 지형 위에
  안 깔리고 떠 있다"). `AppendApron` 이 바깥 변(`OuterEdges` + 지표 `PointGrid3`)에서 카빙 지면까지
  1.2m 경사면을 잇는다. ⚠️카빙 안 걸린 자리에선 지면이 노면보다 높을 수 있어 **안쪽보다 낮게 클램프**.
  ★**갓길이 단차를 덮으므로 카빙 깊이는 공짜다** — `CARVE_CLEAR` 를 0.25→0.12 로 줄였더니 8m 격자
  보간이 노면을 뚫어 "지형이 아스팔트 위" 가 0.3%→5.8% 로 **악화**했다. 되돌리고 `SURF_CLEAR` 만 0.06.
- **도색을 노면 z 에 얹는다** — 노면=A2 링크 z, 도색=B2 선 z 로 **원천이 달라** 인접 차선 z 차이
  (p90 12cm·p99 26cm)가 도색 여유를 넘어 흰선·중앙선이 묻히거나 떴다. `LaneTopProbe`(반경 4m
  최고 y)로 도색을 아스팔트 윗면 +5cm 에 얹는다. ⚠️`|markZ−laneZ|>2m` 면 고가/지하 도색이므로
  원래 z 유지. 정지선 색은 `hd_lanes` **전국 255 재버킷**으로 정정(`YELLOW_KIND` 에서 530 제거 —
  강남 황색 12,933→10,180줄). ⚠️버킷 재생성은 **베이크 전에 전국을 끝내야 한다** — 파일럿만
  돌리고 전국 베이크를 걸었다가 248개 구가 구 버킷(황색 정지선)으로 구워져 다시 돌렸다.
  구 스탬프 `.bucket/<r>.lanez.ok` 는 08-16 산출물 기준이라 재사용 금지(새 스탬프 `.lanew.ok`).
- **스폰을 물리 기준으로** — `FreePilotController` 가 `TerrainHeight`(DEM .bin)로 스폰 y 를 정했는데
  **보이고 밟히는 지면은 DEM 이 아니다**(8m 재세분 + 카빙된 정사 + HD 노면). 카빙 구간에서 DEM 은
  노면보다 중앙 +1.33m·p05 +7.21m 높다 → 허공 낙하/지하 스폰. `RoadSurfaceProbe` 레이캐스트 우선
  + DEM 폴백으로 교체하고, **스폰 지점에 실제로 레이가 맞을 때까지** 대기(루트만 보고 진행하면
  정사가 아직 안 올라온 프레임에 스폰돼 무한낙하). 자율주행의 `PHYSX_SAFE_ABS` 프레임 검사도
  **씬 로드 전**으로 옮겼다(구 코드는 500MB 씬을 다 연 뒤에야 에러를 냈다).
- **스트리밍 지역 물리** — `RegionSceneStreamer` 가 `EnsureGroundColliders` 만 불렀다 →
  `EnsureBuildingColliders` 추가 + 새 지역마다 `onRegionReady` 로 OSM 신호/횡단보도 소등(구 코드는
  최초 1개 지역만 정리했다) + **빌드 플레이어 경로**(`CanStreamedLevelBeLoaded`) 추가(구 코드는
  통째로 `#if UNITY_EDITOR` 라 학습 빌드에서 지역 스트리밍이 아예 없었다).

#### 검증 (스크린샷으로 판정하지 않는다)
- **요청 파일 `audit=<지역들> lat= lon= r=`**(`HdSceneRebuildBatch`) — 여러 구를 동시에 열고
  ①경계 중복(씬끼리 **0.15m 안 겹친 정점**) ②노면−지형 ③도색−노면 을 잰다.
  강남역 250m 실측 **전/후**: 도색 겹침 2,988쌍→**30** · 도색 매몰 10.9%→**1.2%** · 지형이 노면 위
  0.3%→**0.0%**. 서대문역(침수 신고 지점) **노면 위 0.0%**(구 실측 DEM−노면 중앙 +2.65m·최대 +7.44m).
- ⚠️**지표를 잘못 짜서 두 번 오진했다** — ⓐ`_HdRoadSurf` 에 **갓길 정점**이 섞여 "지형이 노면 위"
  가 5.8% 로 부풀었다(기준을 갓길 없는 **차선 중심선**으로 교체하니 0.0%). ⓑ건물 접지를 격자
  최저 y 로 재면 **건물 내부 셀엔 기초 정점이 없어 지붕 높이**가 잡히고 정사 최저 y 는 **카빙 도랑**을
  집어 "53% 부양" 이 나왔다 — 실제 잔차는 중앙 **+0.15m**. 건물 지표는 감사에서 빼고
  **`GroundSnapPass` 자신이 자기 기준면으로 남기는 기초 잔차 로그**를 정본으로 삼는다
  (베이크마다 255지역 전부에 자동으로 찍힌다). **검증 코드가 본 코드와 다른 기준을 쓰면 없느니만 못하다.**
- **`접지실패(NaN)` 신규 계수** — 정사가 구를 못 덮는 지역이 드러났다(강남·중구 **0동** vs
  `busan_geumjeonggu` 25,412 · `chungnam_seosansi` 18,587). 이건 v1 정사 커버리지 자체의 결함이지
  이번 패스의 회귀가 아니다(구 코드는 조용히 `continue` 했다).
- **Unity 없이 C# 전수 컴파일** `tools/exp_drivers/csc_check.sh`(30초). ⚠️참조에
  `Data/Managed/UnityEditor.dll` 을 **넣지 말 것** — `Managed/UnityEngine/UnityEditor.CoreModule.dll`
  과 타입이 중복돼 CS0433 이 116건 쏟아진다. 모듈 DLL 만으로 MenuItem·EditorApplication 전부 해결된다.
- 실행 통로 `tools/exp_drivers/run_kdt_rebuild.sh {pilot|all|audit|raw}`.
  파일럿 7곳(경계 쌍 포함) ≈ 4분, 전국 255 ≈ 2.3시간.


#### 전국 반영 결과 (2026-09-01, 255/255 실패 0)
- **소유권 제외 전국 49.3%** — 총 링크 2,627,274 중 **1,295,802**. 기존 씬들은 도로 기하의 절반이
  이웃 구 중복이었다. 갓길 1,124,018 변 생성.
- **건물 접지(스냅 자가검증)**: 255지역 평균 잔차 중앙 **+0.081m** · 부양(0.5m 초과) 평균 **0.08%** ·
  부양 5% 초과 지역 **0개**.
- **경계 감사 4곳**: 부산(해운대∩금정∩동래)·제주(제주시∩서귀포) **겹침 0** · 강남역 61(노면 31/도색 30,
  노면 정점의 2%) · 서대문역 99. 남는 건 **경계를 횡단하는 폴리라인**이라 중점 귀속으로는 못 없앤다
  (점 단위로 자르면 이음새가 생긴다 — 그쪽이 더 나쁘다).
- **노면 위 지형 0.0%**(4곳 전부) · **도색 매몰 0.0~1.8%**.
- **세 모드 무인 플레이테스트**(`run_kdt_playtest.sh`) — 전부 `badBounds 0 · errors 0 · exceptions 0`:

  | 모드 | 조종체 | 스폰 지면여유 | 콜라이더 | NPC 차/보행 |
  |---|---|---|---|---|
  | 자율주행 | `EgoAMB` | **+1.50m** | 1,660 | (CarDrive 스택) |
  | 자유모드 | `FreePilot_Vehicle` | **+80.00m**(헬기 규약) | 1,048 | 134 / 257 |
  | 자율비행 | `UamPredictiveAgent_0` | **+2.65m** | 1,534 | 645 / 257 |

  자율주행 주행품질: 차선 횡오차 평균 **0.111m**·p95 0.299m, 조향 반전 1.41/s, 요레이트 RMS 0.028.
- ⚠️**Play 진입은 `EditorApplication.delayCall` 한 번으로는 조용히 무시된다** — 요청은 소비되고 로그도
  남는데 Play 가 안 켜졌다(실측 2회). `EditorApplication.update` 로 **20초간 재시도 + 실패 시 사유 로그**
  (compiling/updating/dirtyScenes)로 바꿨다. `SendKeys('^{p}')` 는 포커스가 확실할 때만 먹으므로 보조수단이다.

#### 2026-09-01 (2차) — 플레이 피드백 6건: 터널·옥상 보행자·지형 절벽·차도 위 구조물

사용자가 실제로 주행해 보고 낸 신고를 **전부 원인까지 추적**했다. 스크린샷은 증상만 주므로
`what=`(그 자리 렌더러를 그룹별로 조회) 진단을 새로 넣어 정체를 확정했다.

- **★지상 씬 절벽 = `CarvedGround` 의 하드 임계**. 절토량이 `CARVE_MAX`(15m)를 넘으면 카빙을
  **통째로 포기**해서 그 경계에서 지형이 15m 수직으로 튀었다. 임계를 없애고 **chamfer 전파**로
  허용고를 전 래스터에 퍼뜨린다(원거리 절토 사면 0.7≈35°, 국내 토사 절토 1:1.5 관행).
  두 번의 스윕이라 O(셀수) — 반경을 넓히는 것(비용 ∝ 반경²)과 달리 사실상 공짜다.
  실측: 회랑 내 지형 잔여 융기 **3.8% → 전국 평균 0.03%**(0.5% 초과 지역 255 중 2).
  ⚠️`CARVE_CLEAR` 를 0.25→0.12 로 줄여 이격을 없애려 했더니 8m 격자 보간이 노면을 뚫어
  "지형이 아스팔트 위" 가 0.3→5.8% 로 **악화**했다. **갓길이 단차를 덮으므로 카빙 깊이는 공짜다** —
  여유는 되돌리고(0.25) `SURF_CLEAR` 만 0.06 으로 낮춘다.
- **★터널 = 콜라이더도 벽도 천장도 없었다**(데이터 z 리본만). 들어갈 수도, 들어가서 볼 것도 없었다.
  `BuildBore` 신설: 지반이 천장보다 위인 **덮인 구간**에만 벽+천장+조명 띠(자체발광)를 세우고,
  얕은 구간은 카빙 대상에 넣어 **절토 포털**로 연다(전 구간에 천장을 씌우면 입구 밖 허공에
  콘크리트 상자가 뜬다). 터널 노면엔 콜라이더를, 굴착 단면은 `HD_STRUCTURE` 로 **Building 레이어**
  (차가 벽을 통과하지 않고 UAV 센서에도 잡힌다). 전국 **14,292 구간**.
- **★보행자가 옥상을 걸어다닌 이유** — `RoadSurfaceProbe.Ground` 가 `DriveMask`(Ground|Building)로
  하향 레이를 쏴서 **건물 지붕이 '지면'** 이 됐다. `GroundMask` 만 쓰고, `Walkable` 은 건물 하부도
  배제한다. 지붕에 올라갈 일이 있는 소비자(UAV 착륙)는 자기 경로를 따로 쓴다.
- **★"도로를 막는 건물" 의 정체는 응급실 캐노피**였다(`ER_ENTRANCE_KR`). `ErEntrancePlacer` 가 병원
  POI 에서 도로 쪽으로 (벽까지 거리 + 5.375m) 밀어내 **차도 한복판에 착지**했다. 노면 메시 마스크로
  1m 씩 후퇴시키고, 끝내 못 피하면 세우지 않는다(전국 6,343개 중 33개 생략). `_ER` 을
  `GroundSnapPass` 재접지 대상에 추가(빠져 있어 DEM 높이에 떠 있었다).
- **차도 판정을 노면 메시로** — 구 `GroundLaneProbe(2.6m)` 는 **교차로 안쪽을 못 덮어**(어느 차선
  중심선에서도 2.6m 보다 멀다) 교차로 한복판 건물이 겹침 0% 로 살아남았다. `GroundSnapPass.RoadMask`
  (노면 삼각형 1.5m 래스터)로 교체. 도로 걸침 건물은 **띄우지 않고 제거**(사용자 지시 변경 —
  구 정책의 상판이 눈높이에서 도로를 덮고 기둥이 차도에 섰다). 전국 354,476동(접지실패 포함
  분모로 ~10% 이하, 대덕구 스크린샷으로 도시 온전 확인).
- **갓길은 양쪽에 상시** — `OuterEdges` 는 링크 전체를 8점 **다수결**로 판정해 분기·합류 구간에서
  갓길이 빠지고 리본 사이에 쐐기 틈이 벌어졌다. 정점 3배(강남 22만→34만)는 씬 15MB 값이다.
- **고가 오분류 강등** — 연결성 전파(램프 승격)가 지표 링크까지 끌고 올라가 **차도에 방호벽 0.95m**
  가 섰다. 지반 대비 여유 `ELEV_MIN_CLEAR`(1.5m) 미만이면 지표로 되돌린다. 전국 6,029건.
- ⚠️**★고아 정점이 통계를 통째로 왜곡한다**: 앞선 패스가 삼각형만 지우고 정점은 남기므로
  재실행마다 단독 컴포넌트가 쌓인다. 강남 실측 "차도 겹침 [0.9,1.0] 8,770개" 가 **전부 고아**였고
  "차도 위 건물 2,734동 제거" 라는 가짜 통계를 만들었다. `triCnt < 4` 로 제외하니 실제로 겹침
  0.2 이상인 동은 **0개**였다. 컴포넌트를 셀 땐 **삼각형 유무**를 먼저 볼 것.
- ⚠️**비율은 분모를 확인하고 읽을 것**: "대덕구 90.4% 제거" 는 분모에 **접지실패(정사 미커버)
  27,517동**이 빠진 값이었다. 실제 10.9% 이고 스크린샷상 도시는 온전하다.
- ⚠️**Unity 재컴파일은 `AppActivate` 로 안 된다**(포커스가 안 옮겨간 채 성공을 반환 — 실측 10분 무반응).
  Win32 `SetForegroundWindow` 로 포그라운드를 **확인한 뒤** Ctrl+R 을 보낼 것:
  `tools/exp_drivers/unity_recompile.sh`. Play 진입도 `EditorApplication.delayCall` 한 번으로는
  조용히 무시된다 → `EditorApplication.update` 로 20초 재시도 + 실패 사유 로그.

##### 최종 검증 (전국 255/255 실패 0)
- 카빙 잔여 융기 **0.03%** 평균 · 건물 기초 잔차 중앙 **+0.081m**·부양 0.02%(1% 초과 지역 0)
- 경계 감사 4곳: 부산·경기/송파 **겹침 0**, 강남역 86·서대문역 146(노면 정점의 2% — **경계 횡단
  폴리라인**이라 중점 귀속으로는 못 없앤다). 지형이 노면 위 **0.0%** 전 지점, 도색 매몰 1.1~1.6%.
- 세 모드 플레이테스트(badBounds/exceptions 0): 자율주행 `EgoAMB` 스폰 **+1.50m**(차선 횡오차 평균
  **0.069m**·p95 0.200m) · 자유모드 **+80.00m**(NPC 72/보행 292) · 자율비행 **+2.65m**(NPC 524/보행 290).

#### 2026-09-02 — 세션 원점(floating origin): 자율주행 차단 135/255 해소

사용자 콘솔 오류 `[KoreaDrive] daegu_bukgu 은 월드 좌표가 원점에서 너무 멀다 (133135, -708256)`.

- **★차단 규모는 절반이었다** — `RegionRegistry.Regions` 표가 광역 프레임을 월드에 **격자로 흩어**
  놓아(Gangwon `offZ = −850,000`) 전국 255구 중 **135구(53%)가 |좌표| 25만 초과**로
  `PHYSX_SAFE_ABS` 가드에 막혔다. **경북·전남·경남·강원·전북은 100% 차단.**
  ⚠️**격자 오프셋이 원인이지 지리가 아니다** — 전국 단일 앵커였다면 최대 |좌표| 는 (255k, 283k) 다.
- **해법 = 세션 원점**(`Assets/Scripts/Geo/RegionRebase.cs` 신설 + `RegionRegistry.WorldShift`).
  모드 시작 시 스폰 지점의 원시 월드좌표를 원점으로 잡고 그만큼 월드를 평행이동한다.
  소비자 24곳이 전부 `TryWorld*`/`WorldToLatLon` 을 지나므로 **한 곳만 고치면 전부 따라온다**.
  ⚠️**y 는 안 옮긴다**(고도는 절대값). ⚠️에디터 패스는 shift 를 설정하지 않으므로 0 으로 남아
  베이크 좌표계가 그대로다 — 런타임 부트스트랩만 설정한다.
- **★트랜스폼만 옮기면 안 된다.** 씬에는 월드좌표를 **숫자로 구워 든 블롭**이 4종 있고
  (`RegionRoadNetwork`·`RegionFootwayNetwork`·`RegionSignalSet`·`RegionRoadGraph`), NPC 주행·보행·
  신호·경로는 트랜스폼을 거치지 않고 그 배열을 **직접** 읽는다. 메시만 옮기면 NPC 가 옛 좌표에서
  달린다. `RegionRebase.ShiftBlobs` 가 같이 옮긴다. 순서: **씬 로드 → 즉시 리베이스 → 그 다음 전부**.
- **모드 전환 시 반드시 `RegionRebase.Reset()`** — static 은 씬 로드로 안 사라진다. 자율비행·자유모드
  (`UavGangnamRouteBootstrap`)와 MCI 재생(`MapVersionSelector.StartWith`)에서 해제한다. 안 하면
  자율주행에서 넘어온 원점이 남아 좌표가 통째로 어긋난다.
- `RegionSceneStreamer.rebase` 플래그 — 나중에 붙는 시군구도 **콜라이더를 세우기 전에** 같은 원점으로.
- `PHYSX_SAFE_ABS` 가드는 **사후 확인**으로 남긴다(리베이스가 안 걸리면 조용히 크래시하는 것보다 낫다).

##### 실측 (4개 프레임 × 자율주행, 전부 badBounds/errors/exceptions 0)
| 지역(프레임) | 리베이스 전 | ego 좌표 | 차선 횡오차 평균 / p95 |
|---|---|---|---|
| 강남(Sudogwon) | 동작(|좌표| 13만) | (593, 162) | **0.014 / 0.072 m** (구 0.069/0.200) |
| 부산 중구(Busan) | 동작 | (−558, 247) | 0.046 / 0.184 m |
| 대구 북구(Gyeongbuk) | **차단** | (−1036, 1006) | **0.006 / 0.026 m** |
| 강릉(Gangwon, offZ −850k) | **차단** | (966, −388) | **0.013 / 0.031 m** |

정밀도 개선은 부수효과가 아니라 원인이 같다 — 원점에서 70만 m 면 float32 ULP 가 **8cm** 라
휠 접촉이 뭉개진다. 원점 근처면 0.1mm 이하다. **가장 멀던 강원이 이제 정밀도 최상**이다.
자율비행·자유모드는 원점을 쓰지 않으며(Reset) 결과 불변.

#### 2026-09-04 — 인물 에셋 전면 교체 · NPC 스폰 규칙 · DTM 전국 적용

사용자 신고 5건("사람 모션이 저급" / "사람이 도로 한복판" / "차·버스가 무작정 스폰" /
"지형이 아스팔트 위로 돌출" / "DEM 과 건물 부조화"). **전부 수치로 판정**했다.

- **★인물 = Blender 절차 생성 + 실측 보행 클립**(`tools/blender/gen_people_kr.py`, 24종·각 ~2.5k tris).
  구 풀은 15슬롯 중 **9개가 애니메이션 클립 없는 정적 스캔**이라 `ProceduralGait` 가 본을 사인파로
  흔드는 가짜 보행이었다. 이제 전 인물이 **동일 제네릭 리그**(Mixamo 이름)를 공유하고 GLB 안에
  `Walk`/`Idle` 클립을 갖는다. 관절각은 **Winter 정상보행**(고관절 ±23°·무릎 이상성 5→18→65°·
  족관절 저굴/배굴)을 그대로 쓰고, 골반 3축·체간 역회전·팔 반대위상까지 넣는다.
  - **★발미끄러짐 제거 = 속도 매칭**. 생성기가 **접지 프레임의 양 발끝 간격을 실측**해
    `PEOPLE_META.json` 에 보폭을 쓰고(`PED_KR_Suit_M` 1.31m/주기 = 1.31 m/s), 런타임
    `PedestrianAnim` 이 `animator.speed = v / clipWalkSpeed`. 실측 `pedSlip` **0.000 m/s**.
  - **★골반 상하 진폭은 사인파로 주지 않는다** — `apply_pose` 가 **지지발 밑창을 z=0 에 붙이도록**
    루트를 내린다(4점 밑창 캐시). 다리 기하에서 나오는 진짜 진폭(L·(1−cosθ))이 공짜로 나오고
    발이 바닥을 뚫거나 뜨는 일이 원리적으로 없다(IK 불요).
  - ⚠️**관절각은 부모 상대**로 합성해야 한다. rest 회전에 절대 회전을 곱하면(첫 구현) 무릎이 대퇴
    각도를 무시해 다리가 안 굽고 보폭이 0.73m 로 반토막 났다. `D[bone]=D[parent]@rot` 누적.
  - ⚠️**반지름 표는 실측에서 역산할 것**. 눈대중으로 넣었더니 어깨 61cm·머리 27cm 짜리 술통이
    나왔다(실제 39cm·16cm). 전체 바운즈는 맞아떨어져 눈치채기 어렵다.
  - 설치 `Assets/Editor/PeopleAssetInstaller.cs`(클립 복제+loopTime, `speed` 1D 컨트롤러, 프리팹).
    **구 이름은 별칭 프리팹으로 유지**(`PED_Walk`·`Doctor`·`Patient_*` — 소비자 코드 무수정).
    `ProceduralGait.cs`·`StaticPedestrianGait.cs` **삭제**.
- **보행자**: 구 코드는 **스폰 때만** `Walkable()` 을 봤고 **걷는 중엔 차도 판정이 없었다**.
  이제 지면 프로브마다 차도면이면 인도 쪽으로 밀고(5회 실패 시 회수), **횡단 허가 상태**
  (`crossOkAt`)를 둬 정상 횡단자와 구분한다. 무단횡단 비율(`JAYWALK_FRAC` 0.10, 문헌 위반율
  3.8~62% 중 하단)·**교착 탈출 20초**(SUMO jamtime 계열) 추가. 실측 `pedStray` **0**.
- **★차량 0대의 진범 3중**(해운대 `npcImpactCars:0`):
  ①`hasHd` 판정이 **씬 전역**이라 HD 망이 있는 **이웃 구**가 뷰어가 선 구의 OSM 망 3,728경로를
  통째로 죽였다 ②HD 주행망은 지역 편차가 크다(강남 1,779경로 vs 해운대 **38**)
  ③`AddRoad` 의 **200m 미만 제외**에 OSM way 조각이 대부분 걸렸다.
  → ⓐ판정을 임포터의 골목 규칙과 같은 **HD 커버리지 격자**(12m, 과반 표본)로 교체
  ⓑOSM 을 **끝점 체이닝**(1.5m 일치·가장 직진, `ChainOsm`)으로 먼저 잇는다. 실측 0 → **428대**.
- **버스는 정류장이 있는 길에만**. HD 주행망은 위계 정보가 없으므로(`width=3.2`·`oneway=1` 상수)
  씬에 이미 구워져 있는 **OSM 정류장**(`_Features/BusStops`)을 노선 대리지표로 쓴다(2곳 이상 통과).
  구 풀은 20슬롯 중 2개가 버스라 폭 3m 이면도로에도 시내버스가 굴러다녔다. 실측 노선도로 69·버스 16.
  스폰 팝인 방지 = **화면 안이면 150m 밖**, 도로 위계 가중 = 경로 길이(`clamp(len/1200, .28, 1)`).
- **★DEM = GLO-30 DSM 을 DTM 으로 교체**(`tools/dtm_build.py --all`, **218/218 실패 0**, 지역당 ~22s).
  차선 홀드아웃 잔차 **p95 10.49m → 0.47m**(중앙 3.26 → 0.02). 감사 실측(종로 250m):
  **지형이 노면 위 0개(0.0%)** · 경계 중복 0 · 도색 매몰 3.1%. 건물은 재접지가 평균 **5~9m**
  움직였다(구 DSM 이 그만큼 틀렸다) — 기초 잔차 중앙 **+0.00m**·p05 −0.11m.
  ⚠️"경사지에서 평균 기준면이 1층을 묻는다"는 가설로 상위 백분위수(p75)를 넣었다가 **되돌렸다** —
  잔차가 매몰을 전혀 안 보여줬다. 근거 없는 통계 교체는 하지 말 것.
- **가로 프롭**: `tools/blender/sf_fetch_props.py`(Sketchfab REST 직접 호출, CC0/CC-BY 만, 실척
  정규화·데시메이트·텍스처 축소·`ATTRIBUTION.md` 자동 기록). 10종 중 **5종 채택**(벤치·버스쉘터·
  자판기·쓰레기통·킥보드), 3종은 검색 품질 불량으로 폐기. `CityPropStreamer` 가 같은 품목의 절차
  프롭을 **덮어쓰고**(`HQ_OVERRIDE`), 버스쉘터는 **실제 OSM 정류장 좌표**에 세운다.
  ⚠️`location` 을 **대입**하면 안 된다 — glTF 임포터가 부모 변환을 자식 로컬로 내려 둬서 이미 0 이
  아니다(실측: 자판기가 +10.76m 공중). 월드 바운즈 보정량을 **더할 것**.
- **신호 데이터**: `v2_hd_bucket3c.py` 를 **255/255 재버킷**(`lat lon z type linkid`) — 보행등
  (type 11)이 서초 외 전 지역에서 비어 있어 `pedSignals:0` 이었다. 씬 반영은 HD v7 재임포트 필요.
- 검증 지표 신설(`ModeSelfTest`): `pedTotal/pedOnRoad/`**`pedStray`**`/pedAnim/pedSlip/busNpc/busRouteRoads`.
  ⚠️`pedOnRoad` 는 **정상 횡단자까지 세므로 0 이 될 수 없다** — 결함 지표는 `pedStray`.
- 자율주행 영향 없음(역삼역 drive): 차선 횡오차 평균 **0.008m**·p95 0.048m(기준선 0.014/0.072).
- **지하철 출입구**: `tools/osm_subway_entrances.py`(OSM `railway=subway_entrance`, **255/255 수집**) →
  런타임 `CityPropStreamer` 가 사이드카를 직접 읽어 뷰어 주변에만 세운다(**씬 재베이크 불요**).
  에셋 `tools/blender/gen_subway_entrance.py`(난간벽·계단·핸드레일·유리캐노피·표지 토템, 532 삼각형).

#### ★★2026-09-04 회귀: 세션 원점이 베이크를 오염시켜 **52개 지역 건물 전멸** (원인·복구·재발방지)

전국 재구축 도중 처리된 83개 중 **52개 지역의 건물이 0동**이 됐다(서울 중구 17,297동, 인천·광주·
대전·세종 전역). 규모가 큰 만큼 규약으로 남긴다.

- **원인**: 자율주행 모드 Play 가 설정한 `RegionRegistry.WorldShift`(세션 원점, [[RegionRebase]])는
  **Play 를 나가도 static 으로 살아남는다**. 그 상태에서 에디터 베이크 패스가 `WorldToLatLon` 을
  부르면 씬에 구워진(shift 이전) 좌표가 통째로 어긋나고, `GroundSnapPass` 의 소유권 판정
  (`OrthoClipPass.OwnerAt`)이 **자기 구 건물을 전부 "이웃구 소유"로 보고 삭제**한다.
  타임라인이 정확히 일치했다 — 파일럿 3개(플레이테스트 **이전**)만 멀쩡했다.
- **재발방지 3중**: ①`RegionRebase` 에 `playModeStateChanged → EnteredEditMode` 시 `Reset()`
  ②`GroundSnapPass.SnapOne`·`HdRoadImporterV1.ImportOne` 진입에서 `Reset()` 강제
  ③**결과 검산 게이트** — 소유권 판정이 후보의 **과반**을 이웃구로 보면 좌표계 이상으로 판단해
  **아무것도 지우지 않고 `LogError`**. 원인만 막고 검산을 안 넣으면 다음에 또 조용히 지운다.
- **복구 `Assets/Editor/V2/BuildingTriangleRepair.cs`**: `GroundSnapPass` 는 `mesh.triangles` 만
  비우고 **정점은 남긴다**. `VworldRegionImporter.Extrude` 의 배치가 결정적이라(링 n 점 →
  벽 4정점×n + 지붕 n = **5n 정점**) 정점만으로 경계를 되찾아 삼각형을 복원할 수 있다.
  실측 **51씬·1,336청크·3,532만 삼각형** 복원, 삼각형/정점 비율 0.53~0.54(건강 지역과 동일),
  지붕 법선 표본 위 2,771/아래 0. ⚠️판정은 **XZ 로만** — 스커트가 기초 y 를 개별로 낮춰
  "밑변 y 가 같다" 규칙은 깨진다. 지붕은 **원본과 같은 ear clipping·같은 와인딩**이어야 한다.
- ⚠️**씬 임베디드 메시를 고쳐도 씬이 dirty 로 표시되지 않는다** — `EditorSceneManager.SaveOpenScenes()`
  가 조용히 아무것도 저장하지 않았다(1차 복구가 로그엔 3,532만 삼각형인데 씬 파일은 그대로 0).
  `EditorUtility.SetDirty(mesh)` + `EditorSceneManager.MarkSceneDirty(scene)` 필수.
  **판정은 로그가 아니라 씬 파일 mtime·재열람으로** 할 것.
- ⚠️**플레이테스트와 베이크 배치를 같은 세션에서 번갈아 돌릴 때**는 이 오염을 항상 의심할 것.
  베이크 전 `RegionRegistry.WorldShift` 가 0 인지 확인하는 게 가장 싸다.

#### 2026-09-08 — 시민차 NPC 전면 교체 · 교차로 규칙 · "정체" 원인 규명

사용자 신고 3건("차량 NPC 가 저수준" / "차가 겹치고 사고나서 정체" / "아스팔트가 울퉁불퉁해서
전진을 못 한다"). **전부 수치로 판정**했다 — `TrafficManager.TrafficJson()` 신설(원인별 정지 대수·
차체 겹침 쌍·노면 프로브·요철), `ModeSelfTest` 가 결과 JSON 의 `traffic` 블록으로 싣는다.

- ★**`Assets/Resources/TrafficVehicles` 프리팹 10종이 UAV_test 에선 통째로 비어 있었다.**
  원본 GLB(guid `814bb07e…`)가 **CAR_test 에만** 있어 렌더러 21개인데 **삼각형 0·바운즈 0** —
  즉 **자율주행(drive) 모드의 시민차가 보이지 않는 콜라이더**로 돌고 있었다(`TrafficManagerV2`
  는 `Resources.LoadAll("TrafficVehicles")` 를 쓴다). CAR_test 의 `Assets/ThirdParty` 를 **`.meta`
  째로** 복사해 GUID 를 살리자 10종 전부 복원(LOD0 6,022~7,950 tris). **프로젝트 간 이식 때
  `Resources` 의 프리팹만 따라오고 `ThirdParty` 원본이 빠지는** 전형적 사고 — 포팅 후엔
  `AssetDatabase.GUIDToAssetPath` 로 참조가 풀리는지부터 확인할 것(빈 문자열이면 원본 없음).
- ★**시민 승용차 = Comrade1280 "Generic passenger car pack"(CC BY 4.0)** 로 전면 교체
  (`TrafficManager.PREFABS`). 구 절차생성 `VEH_KR_*` 는 **형상 4종을 크기만 바꾼 2,680 삼각형
  상자**였고 폭이 2.12~2.31m 로 과대(실차 1.8~2.0m)라 3.2m 차로에서 겹쳐 보였다. 새 풀은
  10차종·LOD 3단·실척 3.26~5.18m 이고 **자율주행 모드가 쓰던 바로 그 에셋**이라 두 모드의
  차량 품질이 처음으로 일치한다. **시내버스(`BUS_KR_*`)는 사용자 지시로 유지**(정류장 2곳 이상
  지나는 길에서만, `BUS_SHARE` 0.09). 원장 `Assets/ThirdParty/Vehicles/vehicle_assets.json`,
  표기 `Assets/_RawModels/CREDITS.md` + `Licenses/GenericPassengerCarPack_CC-BY-4.0.md`.
- ⚠️**도장은 MPB 가 아니라 런타임 재질 복제로**, 그리고 **`baseColorTexture` 를 떼고**
  `baseColorFactor` 로 칠해야 한다. 팩 차체는 **색이 텍스처에 구워져** 있어(공유 아틀라스의 색
  영역을 UV 로 집는다) factor 만 바꾸면 텍스처 색과 곱해진다 — 흰색 지정이 **검정**으로 나왔다(실측).
  금속/거칠기 맵은 남기므로 광택은 유지된다. 색 인덱스별 1회 복제 캐시라 배칭도 유지.
- ⚠️**팩 바퀴는 노드가 차 원점에 있다**(`Sedan_Wheel_1.localPosition = (0,0,0)`, 변환이 정점에
  베이크됨). 그대로 `Rotate` 하면 제자리 회전이 아니라 **차 원점 궤도운동**이 된다 —
  2026-09-04 "바퀴축이 도는" 신고와 같은 형태. `NpcWheelRoll` 이 **Awake 에서 노드를 바퀴
  중심으로 옮기고 자식 메시를 반대로 밀어** 한 번에 교정한다(이름 규약도 `WHEEL_*` 와
  `<차종>_Wheel_n` 둘 다 수용, LOD0 만).
- ★★**정체의 진범은 DTM 이 아니라 `ScanConflicts` 의 가상 리더에 진행방향 필터가 없던 것**이었다.
  구 코드는 전방 원뿔(16m×±2m) 안의 차를 **방향과 무관하게** 앞차로 삼았다 → 교차로에서
  직교로 지나가거나 교차 도로 정지선에 선 차가 내 '앞차'가 되고, 그 차도 같은 이유로 서 있어
  고리가 닫혔다. 기준선 실측 **3초 이상 정지 95대 중 77대(81%)가 이 경로**(신호 3·건물 0).
  수정 3건: ①원뿔에 `dot(f, otherFwd) > 0.5` 필터(교차류는 `CrossYield` 가 전담)
  ②`CrossYield` 가 **2초 넘게 서 있는 차에는 양보하지 않는다**(3중 고리 파훼 — 구 2.5m 예외는
  교차점 코앞에 굳은 차를 못 걸러낸다) ③12초 교착 탈출 서행(SUMO jamtime 계열).
- **교통규칙 추가**: **황색 딜레마존**(편안한 감속으로 정지선에 못 서면 통과 — 구 코드는 황색에
  급정지해 **교차로 한복판에 서서** 직교류를 막았다) · **교차로 물기 금지**(출구 여유 10m 없으면
  진입 안 함) · **안전속도 5030**(간선 38~50 / 이면도로 22~30km/h, 구 값은 이면도로에도 52) ·
  **횡단보도 보행자 양보**(도로교통법 제27조).
- ⚠️**보행자 양보는 횡단보도 좌표로 하지 않는다** — `PedestrianManager.PedOnRoadAhead` 가 **차도 위
  보행자 위치만** 모아 두고 차량이 전방 11m·반폭 2.4m 를 조회한다. "차도에 사람이 있으면 선다"가
  정상 횡단자·무단횡단자를 모두 덮고 데이터도 필요 없다. **콜라이더 스피어캐스트로 하면 안 된다** —
  좁은 골목에서 **인도 위** 보행자까지 물어 차가 계속 선다(보행자는 이미 `RoadSurfaceProbe` 로
  차도 여부를 매 프로브 판정하고 있으므로 그 결과를 재사용하는 쪽이 정확하고 싸다).
- ⚠️**트라이 수는 품질의 대리변수가 아니다**(2차 정정). 1차에 `TRUCK_KR_Porter`(2,242면)·
  `SCOOTER_KR_Delivery` 를 "저수준"으로 묶어 풀에서 뺐는데, 실제로 렌더해 보니 **한국 번호판·미러·
  적재함 포장까지 있는 제대로 된 1톤 트럭**이었고 팩엔 대응 차종이 없다(Pickup 은 미국식).
  승용차만 남기면 한국 도로에서 가장 흔한 상용차가 통째로 사라진다 → 등록 통계 비율
  (트럭 10%·이륜 5%)로 되돌렸다. **상용차·버스는 저작 도장 유지**(도장 교체는 팩 차량만).
- ★**판정(강남역 freecar 100s, 시민차 ~495대)**

  | 지표 | 기준선 | 수정 후(범위) | 최종 구성 |
  |---|---|---|---|
  | 주행 중 | 376 (76%) | 380~423 | **423 (86%)** |
  | 3초 이상 정지 | 95 | 39~70 | **39** |
  | **15초 이상 정지** | **48** | **4** (8회 반복 전부 4) | **4** |
  | 차체 겹침 쌍 | 6 | 1~6 | **1** |
  | 원인 conflict | 77 | 25~49 | 25 (전부 인접차로, 교차 0) |
  | 보행자 양보 발동 | — | — | **1,245회/100s** |

  ⚠️**양보 규칙은 '3초 이상 정지' 집계로 검증할 수 없다** — 정상 양보는 짧아서 `cause.ped` 가 계속
  0 이고, 그러면 "동작한다"와 "한 번도 안 걸린다"가 구분이 안 된다. 누적 발동수(`pedYields`)를
  따로 세야 한다(실측 1,245회인데 3초+ 정지 0 = 양보 후 곧바로 재출발 = 의도한 동작).
  ⚠️**`TRUCK_KR_Porter` 는 실척이 아니었다**(2.59×2.61m, 실차 1.74×1.99+미러). 되돌린 직후
  `overlapPairs` 가 6 으로 뛰었다 → `Instantiate` 직후 `localScale (0.80, 0.85, 1)` 로 보정
  (헤드램프 위치·콜라이더가 전부 바운즈에서 나오므로 **스케일은 반드시 그 전에** 준다).

- ⚠️**`minFps` 는 지역 씬 additive 로드 히칭을 그대로 문다**(실측 0.1~0.2). 이걸로 성능을 판단하면
  안 된다 — 25초 이후 구간의 `minFpsLate`/`meanFpsLate` 를 신설했다. **정상 상태 평균 50~67fps**
  (시민차 ~493 + 보행자 137, 같은 런 안에서도 시야에 따라 47~60 을 오간다). 참고로 CAR_test
  기준 측정치가 28~43fps 였다.
- **"아스팔트가 울퉁불퉁"은 실재하지만 정체 원인이 아니다**(NPC 는 폴리라인 y 를 따르므로 지형에
  걸리지 않는다). 분모를 갈라 재니 **HD 차선망 구간과 OSM 골목 구간이 완전히 다른 이야기**였다:

  | | HD(정밀도로지도) | OSM 골목 |
  |---|---|---|
  | 차 대수 | 321 | 79 |
  | 노면 메시 밖 | **0~1 (0.3%)** | **22~25 (28~32%)** |
  | 2m 구간 높이차 p50/p95 | 0.010 / 0.11m | 0.03 / 0.13~0.17m |

  ★★**OSM 골목의 진범 — 차가 지면에서 중앙 2.50m·p95 4.46m 떠 있었다**(`osmAboveP50`). 두 가지가
  겹쳤다: ①구 OSM 아스팔트 리본은 HD 재임포트 때 **렌더러 소등 + 콜라이더 제거**됐는데(서초 실측
  `_Roads` rend 15 / **enabled 0** / meshCol 0) `TrafficManager` 는 여전히 `ROAD_CLEAR 0.65m` 를
  더하고 있었고 ②**폴리라인 `y[]` 가 구 DSM 드레이프**라 2026-09-04 DTM 교체 뒤로 스테일하다
  (DSM 은 건물·수목 상면 = 실지면보다 그만큼 높다. 같은 교체에서 건물 재접지가 평균 5~9m
  움직였던 것과 같은 원인). ②가 지배적이다. → OSM 도로에 한해 **실제 지면 콜라이더에 스냅**
  (하향 8m·상향 2m 창 = 고가·교량 낙하 방지, 히트 없으면 손대지 않음, 시간 평활 + **첫 프레임 즉시**).
  실측 `osmAboveP50` **2.50 → 0.06m**, 전체 `aboveP95` **3.16 → 0.09m**, 지면 미검출 3 → 0.
  ⚠️**창을 좁게 잡으면 지표가 아예 안 움직인다** — 1차 시도의 ±1.6m 는 2.5m 부양에 닿지도 못했고
  수치가 그대로라 "원래 그런가" 로 오독할 뻔했다. 보정 창은 **먼저 부양량 분포를 재고** 정할 것.
  남은 HD 구간 요철 p95 11cm 은 교차로에서 차선 리본이 겹쳐 쌓인 것으로, 고치려면 재베이크가 필요하다.
- ⚠️**헤드램프 발광 렌즈(`VehicleHeadlights._glowMat`)가 낮에도 항상 흰색 발광**이었다. 구 저폴리
  차에서는 헤드램프로 보였지만 실차형 팩에 붙으니 **범퍼에 붙은 흰 탁구공**이다 → 공유 런타임
  재질 하나를 야간계수로 흔들고(0.4s 주기 1회) 렌즈 크기도 실차값으로 줄였다. 구급차·버스 포함
  모든 차량이 같이 고쳐진다.
- ⚠️`HD_SURF_CLEAR` 가 **0.10 인데 `HdRoadImporterV1.SURF_CLEAR` 는 0.06** 이었다("반드시 동일"이라고
  주석에 써 놓고 어긋나 있었다) → 차가 노면 위 4cm 를 떠 있었다. 동기화 후 프로브 중앙값이
  0.06 → **0.02**(=의도한 접지 여유)로 떨어져 수치로 확인된다.
- 재현: `Temp/playtest.txt` 에 `freecar 37.4979 127.0276 100 13 clear` → Play → `Temp/selftest_freecar.json`.

#### 2026-09-08 (2차) — 노면 기준면(`__hdroads_v8`) · 미니맵 전국화 · 센서 누수 · 열원

사용자 신고 묶음: ①결정론적 평가 모드가 왜 있는지 모르겠다 ②타이틀 글씨가 배경에 잠긴다
③자율주행 미니맵에 타일이 안 나온다 ④HUD 가 차 내부와 겹친다 ⑤흰선·중앙선이 부정확하고
중간중간 잘린다 ⑥교차로 아스팔트가 요철이다 ⑦NPC 가 적외선에 안 잡힌다 ⑧F3 가상선을
전방 카메라가 본다 ⑨정사가 아스팔트 위로 돌출한다. **⑤⑥⑨는 하나의 뿌리를 공유한다.**

- ★★**뿌리 = 링크마다 자기 z 로 리본을 굽는 것**. A2 주행링크는 차로별로 z 를 따로 갖는데
  (미추홀 원자료 실측: 0.75m 격자에 겹친 링크점의 z 폭 p95 **0.020m**·2.0% 가 3cm 초과)
  리본은 나란한 차로끼리 0.55m, 교차로에선 수십 겹이 겹친다 → 그 차이가 **단차**로 남고
  MeshCollider 가 같은 메시라 자율주행 차가 물리적으로 밟는다. 도색(B2)은 원천이 또 달라
  구 코드가 반경 4m **최고점**(`LaneTopProbe`)에 얹었는데, 옆에 조금이라도 높은 링크가 있으면
  도색만 그만큼 떠올랐다 — 실측 도색−노면 p95 **+0.61m**·최대 +0.90m·묻힘 1.7%.
  → **`HdRoadImporterV1.SurfaceField`**(2m 희소 격자·이중선형·빈 노드 3회 확장·박스 2회 평활)를
  지표 링크점 전체로 굽고 **`allLinks[i].pts` 를 그 면으로 제자리 스냅**(±0.40m 클램프).
  **한 곳만 고치면 리본·갓길·도색·카빙·NPC 주행망·신호 기준면이 전부 같은 면을 본다** —
  소비처마다 따로 보정하면 반드시 어긋난다(`ROAD_CLEAR` 사고와 같은 종류). 소유권 필터 **앞**에
  둔다(경계 도로를 이웃 구가 그려도 같은 면). 미추홀 실측 보정 66,862점·평균 0.021m·최대 0.400m.
- ★**B3 노면표시 면(`hd_marks`)을 v1 씬에도 굽는다.** 구 코드는 이 버킷을 v2 타일
  (`TileBakerV2.BuildMarks`)에서만 읽어, **KDT 세 모드가 실제로 쓰는 v1 시군구 씬에는 화살표·
  정지선이 아예 없었다**. ⚠️**면적 게이트 필수** — B3 에는 글리프(화살표 4.5×3.9m·7점)와
  **구역**(횡단보도 11.4×12.5m·20점, 도류대, 정차금지지대)이 섞여 있어 구역을 통짜로 칠하면
  얼룩말 무늬가 아니라 **흰 판때기**가 된다. 8개 지역 표본으로 분리 규칙을 실측했다 —
  ①**한 변 6m 초과 제외**(화살표류 5371/5372/5373/5381/5382/5391/543x 오분류 1~6%,
  정차금지지대 5392·안전지대 534 는 96~99% 가 구역으로 걸린다) ②⚠️**횡단보도 종류 5321 은
  크기와 무관하게 제외** — 89% 는 11×12m 구역이지만 좁은 이면도로 횡단보도가 **11% 나 6m 이하**라
  크기만으로는 못 거르고, 그리는 순간 WLK2 줄무늬와 **이중 도색**이 된다.
  `hd_marks` 보유 지역은 **251/255**(HD 자체가 없는 4곳과 동일).
- ★**카빙 사면 거리는 셀 인덱스 차로 쟀다** — 점과 셀중심이 각자 반대각(8m 격자 → 5.66m)만큼
  어긋나 실제보다 짧게 나오고, 그만큼 사면이 일찍 올라와 **회랑 가장자리에서 지형이 아스팔트 위로
  최대 +0.23m** 솟았다(⑨의 정체). `allowOff` 계산에서 반대각을 빼면 끝 — 더 깊이 파는 쪽은
  갓길이 덮으므로 공짜다. ⚠️`CarveToLanes` 자체 로그는 "노면 위 잔여 0개"라고 **정상을 보고한다**
  (자기 기준인 `raster[cell]` 로 재기 때문). **검증 지표는 실제 리본 정점을 기준으로** 잡을 것.
- **횡단보도 z**: `CrosswalkInfrastructureV2` 는 줄무늬 전부를 **횡단보도 중심 y 하나**에 12m 통째로
  얹고 `markingLift` 가 **0.16m**(HD 도색은 +0.05m)였다 → 편경사에서 한쪽이 묻히고 반대쪽이 뜬다.
  줄무늬마다 `RoadSurfaceProbe.Ground` 로 **밟히는 콜라이더**를 다시 재고 lift 0.05 로 낮췄다.
  기준을 차선그래프 z 로 두면 안 된다 — 노면 기준면 도입으로 둘이 최대 0.4m 갈린다.
- ⚠️**감사 지표를 두 번 더 잘못 짰다**(전례와 같은 함정). ⓐ③(도색−노면)이 `Deltas`= **최근접**
  노면 정점이라 일부러 내려가는 **갓길**을 집어 도색이 뜬 것처럼 부풀었다 → 반경 1.5m **최고 y**
  기준으로 교체. ⓑ그 기준면에 **고가 상판·지하차도**를 안 넣어 고가 도색이 지상 노면과 비교되며
  +0.4~0.7m 로 잡혔다 → 모든 아스팔트 층을 기준면에 넣는다. ⓒ⑤(겹침 단차)에 **골목**(DEM 드레이프
  +0.35m, 원래 층이 다름)이 섞이면 HD 리본끼리의 단차를 못 본다 → HD 노면만.
  새 지표 `⑤ 노면 겹침 단차`(차선 중심선 0.75m 안 노면 y 폭)·`⑥ 정사 돌출`(HD/골목 분리)은
  `HdSceneRebuildBatch.Audit` 에 상설. **최악 표본 좌표를 함께 찍는다** — "어디"를 모르면 원인을
  계속 추측하게 된다(이번에 골목이라고 두 번 헛짚었다).
- ★**미니맵이 강남 밖에서 비어 있던 이유 = 번들 타일이 218장뿐**(`Resources/OSMTiles`, z16 49 +
  z17 169 = 37.45N/127.03E 한 블록). 255 구를 다 넣으면 5만 장이라 빌드가 못 버틴다 →
  **`OsmTileCache`**(번들 → 디스크 캐시 → 온디맨드 다운로드, 자율비행 HUD `UavOsmTileMap` 과
  **같은 캐시 폴더**)로 일원화하고 `MinimapOSM`·`NavigationPlannerV2` 가 함께 쓴다. 코루틴 없이
  `SendWebRequest().completed` 콜백(정적 클래스라 MonoBehaviour 가 없다), 연결 실패 1회로 오프라인
  판정 후 번들만. 미니맵 하단에 **타일 상태**를 찍어 "좌표 문제냐 다운로드 문제냐"가 즉시 갈린다.
  축척 바 위도가 강남(37.4998)으로 박혀 있던 것도 에고 위도로 교체(제주·강원에서 몇 % 틀렸다).
  **정합 자가검증**: `MinimapOSM.VerifyAlignment` 가 에고 월드 → 위경도 → `RegionRegistry` 역변환
  잔차를 로그로 남긴다(2m 초과면 에러). 화면으로는 "대충 맞아 보여서" 못 잡는 종류다.
- ★**HUD 겹침의 뿌리 = 좌표계가 둘이었다.** 주행 HUD 는 UI Toolkit `ScaleWithScreenSize`
  (1920×1080·match 0.5)로 스스로 줄어드는데 IMGUI 오버레이(미니맵·건물 카드)는 **원시 스크린
  픽셀**이라 게임뷰가 작아질수록 배치 비율이 갈라져 파고든다(1024×576 계산: 센서 컬럼 하단 357px
  vs 미니맵 상단 336px = 21px 겹침). → `HudTheme.BeginScaled()/VirtualW/VirtualH/ToVirtual` 로
  IMGUI 도 같은 기준 좌표에서 그린다(카메라 투영값은 `UiScale` 로 환산해 넘길 것). 추가로
  **운전석 시점에서 계기판을 우측으로 물린다**(하단 중앙 = 실제 핸들·대시보드 자리) + 센서 컬럼
  `maxHeight` 로 미니맵 구역 침범 차단 + `AutonomySensorHud.VerifyLayout` 이 패널 사각형을
  전부 교차 검사해 콘솔에 좌표와 함께 남긴다(겹치면 에러).
- ★**F3 가상 차선선·경로 리본·목표 비콘이 Default 레이어**라 전방 RGB·열화상·깊이 카메라에
  그대로 찍혔다 — **비전 정책이 자기 정답을 화면에서 읽는 누수**다. 세 카메라는 이미
  `SensorDebug` 를 컬링하고 있었는데 정작 오버레이가 그 레이어에 없었다 →
  `ResearchOverlay.HideFromSensors`(`LaneLineRenderer`·`RouteManager` 3곳).
- ★**NPC 온도(`ThermalHeatSource`)**. 열화상 센서는 **가시광 휘도**를 팔레트로 칠할 뿐이라 검은 옷
  보행자·짙은 차체가 배경과 구분되지 않았다. 열화상 카메라는 `Camera.Render()` 로 **명시 렌더**
  하므로 그 호출 앞뒤에서만 등록된 열원의 머티리얼을 밝기=온도인 Unlit 로 갈아끼웠다 되돌린다
  (URP 는 `SetReplacementShader` 가 안 먹고 RendererFeature 런타임 삽입은 과하다).
  ⚠️**MaterialPropertyBlock 을 쓰면 복원 후에도 남아 평상시 렌더까지 물든다** → 온도 16단계
  **공유 머티리얼 표**. 보행자 0.88 / 차량 0.58(±지터). 레이어(Vehicle/Pedestrian)·콜라이더는
  원래 있었으므로 LiDAR·레이더·카메라 탐지는 이미 NPC 를 객체로 보고 있었다 — 빠진 건 LWIR 뿐.
- **결정론적 평가 모드 제거**: `FlightMode.Evaluation` · `EvaluationOffered` · 모드 카드(UXML/IMGUI)
  삭제, `-batchmode` 전용으로 내렸다(`evaluation = runDeterministicEvaluation && Application.isBatchMode`).
  씬 `KoreaDigitalTwin.unity` 의 `runDeterministicEvaluation: 1` → 0.
- **타이틀 가시성**: 타이틀 화면만 `.dim` 없이 **밝은 하늘**(안개 rgb 0.55/0.65/0.78) 위에 글자를
  얹는데 본문색이 rgb(240,245,250)이라 명도차가 거의 없었다 → 화면을 통째로 덮는 대신
  **글자 뭉치 뒤 스크림 카드**(`.title-scrim`)와 모서리 라벨 알약 배경. 기체 위치도 스크림 밖으로.
- **재베이크 필요**: 위 기하 수정(노면 기준면·B3 표시면·카빙 사면)은 전부 임포터 산출물이라
  `Marker` 를 `__hdroads_v8` 로 올렸다. 한 구 ≈ 24초, 전국 255 ≈ 2.3시간
  (`tools/exp_drivers/run_kdt_rebuild.sh all` 또는 요청파일 `skipDone=0`).

#### 2026-09-08 (3차) — 미니맵 상하거울 · 도색 이음 · UI/UX 전면 재설계 초안

- ★★**미니맵이 세로로 거울이었다**(헤딩업·노스업 **둘 다** 진행방향과 반대였던 원인).
  지도는 RT 에 `GL.LoadPixelMatrix(0,size,size,0)`(y-down)으로 그린 뒤 `GUI.DrawTexture` 앞에서
  `ScaleAroundPivot(1,-1)` 로 한 번 더 뒤집는데, 그 뒤집기가 **RT 저장방향 보정이 아니라 덧붙은
  거울**이었다. 부호가 두 번 뒤집혀 **추론으로는 확정이 안 됐다**(양쪽 결론이 다 성립한다) →
  지도와 **같은 변환**으로 '전방 60m'·'우측 60m' 표식을 찍어 캡처했더니 전방이 화살표 **아래**,
  우측은 오른쪽 = 순수 상하거울. `display = FLIP∘drawn` 이므로 `drawn = FLIP∘ROT` 로 그리도록
  회전행렬 앞에 중심 기준 flipY 를 곱해 해결(OSM 한글 라벨이 정상 판독되는 것으로 교차 확인).
  영구 가드로 `ScreenBearingOf()` 가 매 5초 **화면상 전방/우측 방위**를 재고 8° 넘게 벗어나면 에러.
- **도색 이음**: B2 노면선이 한 차선당 중앙값 11m 조각으로 들어와 조각마다 대시 위상이 0 에서
  다시 시작했다 → `StitchLines`(끝점 0.6m·접선 30°)로 먼저 이어 붙인 뒤 대시(미추홀 16,989→7,275).
  도색 y 는 `Sample` 이 아니라 **리본과 같은 `Snap`**(±0.40m 클램프)으로 얹는다 — 클램프가 걸리는
  자리에서 리본과 최대 0.4m 벌어지던 것을 없앤다.
- ⚠️**감사 지표를 또 잘못 짰다(세 번째)**. "도색 렌더 커버리지"를 원자료 1m 점 ↔ **렌더 정점** 근접으로
  재서 **10.2%** 가 나왔다 — 직선 실선은 단순화로 50m 에 정점이 2개뿐이라 애초에 맞을 수 없는 기준이다.
  리본은 폭이 일정하므로 **삼각형 면적 ÷ 폭 = 길이**로 바꾸니 **101.1%**(원자료 1,526m ↔ 렌더 1,543m)
  = 도색은 처음부터 온전했다. 운전시점 캡처로도 확인(흰 실선·3m/5m 점선·황색 복선·도류대 정상).
- **판정 수치(미추홀 주안 남측 간선 r=120m)**: 겹침 단차 중앙/p95 **0.000m**(3cm 초과 0%) ·
  정사 돌출 최대 **+0.028m** · 도색−아스팔트 중앙 **+0.044m** · 도색 길이 **101.1%**.
- **HUD 겹침 0 확인**: `AutonomySensorHud.VerifyLayout` 이 NAV∩신호 8px 겹침을 잡아내 신호 위젯을
  96→112px(칵핏 202→218)로 내렸고, 이후 자가검증이 `겹침 0 ✔` 를 출력한다.
- ★**UI/UX 전면 재설계 초안 = Figma `KDT UI/UX Redesign`**
  (`https://www.figma.com/design/y0lPuZ3ec4S4LhAamd7PEf`). 페이지: Foundations(변수 KDT/Color·
  KDT/Space) · Shell(타이틀/모드/지역) · Drive HUD · Flight HUD · **Spec(토큰↔`Kdt.uss`/`HudTheme`,
  블록↔소유 스크립트 매핑)**. 기능·데이터는 그대로 두고 표현만 교체하는 것이 전제다.
  Drive HUD 배경은 실제 미추홀 운전시점 캡처를 얹어 대비를 판정할 수 있게 했다.

#### 2026-09-09 — HUD 2세대 병존(F6) · 테슬라식 인지 · PFD 계기 · 시점 연동

Figma 초안(`KDT UI/UX Redesign`)을 Unity 에 반영. 사용자 지시 3건: ①주행·비행 **둘 다 v1/v2 둘 다**
올릴 것 ②**시점 변경 시 둘 다** 따라올 것 ③숫자 위주를 벗어나 직관적으로.

- ★**세대 전환 = `KdtHud`(F6)**. 두 세대를 **동시에 만들어 두고 표시만 토글**한다 —
  컴포넌트를 따로 두면 데이터 배선이 두 벌이 되고 그 순간 한쪽이 낡는다(같은 원천을 읽게 강제).
  상태는 `PlayerPrefs` 에 남겨 모드를 오가도(주행↔비행) 세대가 유지되고 도메인 리로드도 넘긴다.
  **주행·비행이 같은 키(F6)** 를 쓴다 — 두 화면에서 손이 달라지면 안 된다.
- ★**주행 v2 = 테슬라식 인지 시각화**(`PerceptionViewElement`, UI Toolkit `Painter2D`).
  색 규약은 Tesla FSD 관례 그대로: 차선 백색 점선/황색 중앙선 · **도로 가장자리 적색**(주행가능
  영역 경계) · **계획 경로 청색 리본**(감속 시 옅게) · 차량 회색/진회색(선행·제동등)/**청색(경로상)**/
  적색(경고) · 보행자 황색 · **자차 밑 청색 네온 글로우 = 자율주행 작동 중**(FSD v14 규약).
  데이터는 전부 실측: `CameraDetectionSensor.Detections`(자차 로컬좌표) · `LaneGraphV2.NearestLane`
  + `LeftOf/RightOf` 로 차로 수 · `RouteManager.TryPointAtS` 로 경로. **연출 상수 없음.**
  ⚠️`halfWidthM` 22 → **9 m**: 22 면 3.25m 차로 7개가 화면 폭에 들어가 차선이 한 줄로 뭉친다.
  원근 압축도 0.14 → 0.30(지평선에서 차량이 점이 됐다).
- ★**비행 v2 = PFD**(`PfdElements.cs` + `UavPfdHud`). 항공 표준 배치: 중앙 인공수평의(피치 사다리
  ±30°·롤 스케일·뱅크 포인터·플라이트디렉터) · 좌 대기속도 테이프(트렌드 벡터·녹황적 속도대) ·
  우 고도 테이프(선택고도 버그·지면 밴드) + 승강계 · 하단 HSI(러버라인·헤딩버그·트랙 다이아·
  WP 방위 지침) · 풍향 벡터 · 모드 어넌시에이터 · 에너지/환자.
  ⚠️**IMGUI 로 만들지 않았다** — 인공수평의는 회전+클리핑이 동시에 필요한데 `GUI.matrix` 는 클립이
  스크린 좌표로 확정된 뒤 적용돼 내용이 창 밖으로 새어나온다(미니맵에서 이미 물린 함정).
  ⚠️**HSI 나침반 카드를 프레임 회전으로 돌리지 말 것** — 회전 중심이 좌상단이라 어긋난다.
  각도에 침로를 반영해 눈금을 직접 놓는다(Figma 초안에서도 같은 실수를 한 번 했다).
- ★**시점 연동**: 주행은 `ApplyGraphicCameraLayout(cockpit)` — 운전석에서 룸미러(상단 중앙)와
  실제 대시보드(하단 중앙)를 피해 기동 카드를 내리고 속도 다이얼을 우측으로 물린다.
  비행은 `SetCabinView(cabin)` — 캐빈/칵핏에서 ADI 660→520·HSI 350→250 으로 줄이고 좌우 하단
  카드를 숨긴다(페데스탈이 눕혀 있어 아래 절반이 안 보인다는 기존 실측 반영).
- **거짓 수치 금지**: 비행 v2 의 에너지·환자는 `UamCabinDisplays.S`(캐빈 계기와 **같은 원천**,
  `UamCabin.UpdateBattery` 가 적분한 실 소비량)를 읽고, 모델이 안 돌고 있으면 "배터리 모델 미가동"
  으로 **비워 둔다**. 남은거리/항속으로 그럴싸한 값을 만들지 않는다.
- **v2 에서 구 IMGUI 카드 소등**: `BuildingInfoOverlayV2` 의 측위·랜드마크 카드는 v1 전용
  (월드 건물 라벨은 두 세대 공통 — v2 에서도 유용). 안 끄면 그림 위에 계측 표가 겹쳐 둘 다 못 읽는다.
- **무인 검증 배관**: `ModeSelfTest` flags 에 `hud1|hud2` 와 `cockpit|cabin|chase` 추가 →
  결과 JSON 에 `hudVersion`·`view` 를 남긴다. **어느 조합의 캡처인지 사후에 알 수 있어야 한다.**
  `AutonomySensorHud.VerifyLayout` 도 세대별로 검사 대상을 바꿔 v2 에서 `겹침 0` 을 계속 보증한다.
- ⚠️**청색 노면 도색은 버그가 아니다** — 미추홀 `hd_lanes` 에 색코드 `B` 가 849줄(버스전용차로
  표시)이고 임포터가 `MarkMat('B')` 로 파랗게 굽는다. 화면의 파란 선을 경로 리본으로 오독하지 말 것.

##### 2026-09-09 (이어서) — 캡처 검증에서 드러난 결함 4건과 근본수정

v1/v2 × 시점 캡처를 **실제로 눈으로 열어 보면서** 나온 것들이다. 앞 절의 `errors 0`·`겹침 0 ✔`
는 전부 참이었는데도 화면은 망가져 있었다 — **검증 대상이 아니었던 것들이 망가져 있었다.**

- ★**캡처 파일명에 조합이 안 들어가 있었다**(`play_<mode>_h13.png`, `h13`=**시각 13시**이지 HUD 세대가
  아니다). 조합별 캡처가 같은 이름으로 덮이고, 사후에 파일만 보고는 어느 조합인지 알 수 없어
  **"v1 캡처"라고 이름 붙인 파일이 실제로는 v2 캡처의 복사본**이었다(md5 동일로 발각).
  → `ModeSelfTest` 태그를 `_h13_v2_cabin` 처럼 **세대·시점 포함**으로 바꿨다. `run_kdt_playtest.sh`
  에 5번째 인자 `flags` 추가(`"hud2,cabin"`). **이름이 조합을 말하지 않으면 캡처는 증거가 아니다.**
- ★★**IMGUI 오버레이 2개가 원시 픽셀로 그려져 좌상단·좌하단을 밟고 있었다.**
  `RoadDrivingDemoHud`(Rect(10,10,330,132))가 조작안내 카드(18,16,470×50)+경고줄(18,78) 위에
  겹쳐 **좌상단에 글자 4겹**(v1·v2 둘 다), `ModeHome` 칩(`Screen.height-34`)이 네 모드 하단 카드를
  전부 밟았다(주행 v1 AUTONOMY · v2 인지패널 · 비행 v1 깊이/LiDAR · v2 에너지). 원인은 미니맵과
  **똑같다** — UI Toolkit 은 1920×1080 기준, IMGUI 는 원시 픽셀. → 둘 다 `HudTheme.BeginScaled()`
  안으로. 데모 HUD 는 속도·경로진행·남은거리(메인 HUD 중복)를 버리고 **한 줄**로 접어 예약 밴드
  (18,104,470×24)에, F9 칩은 네 모드 공통으로 비는 하단 밴드(480, VirtualH−28)로.
  → **두 Rect 를 `VerifyLayout` 예약구역에 등록**(미니맵과 같은 방식)해 앞으로는 수치로 잡힌다.
  ⚠️`VerifyLayout` 이 UI Toolkit 트리만 훑고 있었던 것이 이 결함을 놓친 이유다 —
  **트리 밖 IMGUI 는 스스로 등록하지 않으면 검증에 안 잡힌다.**
- **배경 없는 Lo 색 키 힌트가 밝은 노면에서 앞머리째 묻혔다**(실측: "F6 계"가 안 보였다) →
  주행 v2·비행 PFD 힌트에 알약 배경. 사용자가 초기에 지적한 가시성 문제와 같은 종류다.
- ★★**야간 관전이 흰 안개막에 덮여 있었다**(강남 야간, `nightF=1.0`). `nofx` A/B 로 후처리 소산
  확정. 중앙 60% 휘도 실측: nofx **p50 54·p95 242** vs 구 후처리 **p50 74(+37%)·p95 184(−24%)**
  = 클리핑이 아니라 **대비 붕괴**. 분해하면 야간 노출 보정 `+0.65 EV`(구 `0.35+0.3·nf`)가 안 켜진
  면·하늘을 회색으로 들어올린 것이 본체, Neutral 톤매핑이 창문 하이라이트를 눌러 붙인 것이 둘째,
  블룸(0.38·임계 0.95)이 셋째. → 채택값 **밤 노출 보정 0**(`0.35−0.35·nf`) + **밤 블룸 0**
  (낮과 동일). 낮(`nf=0`) 경로는 구 값과 **비트동일**이다(한낮 백화 방지 튜닝 보존).
  ⚠️**블룸은 3단으로 줄여 보고서야 0 이 답이라는 게 나왔다** — 0.38→0.16(임계 1.25): 안개막은
  걷혔으나 지름 수백 px **흰 원반 3개** 잔존 → 0.10(임계 1.9): 좌측 대형 광원 번짐 잔존 →
  **0: 완전 소거**. 창문 발광이 HDR 로 임계를 훨씬 넘어 **세기를 줄여도 산포 반경이 안 줄기**
  때문이다. 2026-09-03 에 야간 가로등 번짐용으로 넣은 항인데, 발광 레벨이 정리되기 전엔 못 켠다.
- ⚠️**휘도 지표를 프레임 간 비교하면 오진한다**(이번에 1회 적중: 수정 후 p50 이 오히려 92 로
  올라 "더 나빠졌다"고 읽혔는데, 기체가 움직여 크롭 안의 장면 자체가 달랐다). 노출 A/B 는
  같은 프레임이 아니면 **절대 휘도로 판정하지 말 것** — 안개막 여부는 p50↑ ∧ p95↓ 의 **동시 발생**
  으로 보고, 최종 확인은 그림으로 한다.
- **거짓 수치 금지 재확인(정상 동작)**: 체이스 시점의 `BATT`·에너지 게이지는 `— / 배터리 모델
  미가동`, 캐빈 시점은 `100%`. 캐빈에서만 `UamCabin.UpdateBattery` 가 도는 구조라 맞는 동작이다.


## CAR_test 자율주행 씬 (`external/ml-agents/CAR_test/`, 로컬-C 전용)

파일럿 씬 `Assets/Scenes/GangnamLaneDrive.unity` = **강남 110타일(201~211×539~548, EPSG:5186 11×10km) k-ring 스트리밍**. 씬엔 `DriveBootstrap` + `TilesV2_Streamer`(+ 조명/카메라)뿐이고 `Lane/GangnamDriveBootstrap.cs` 가 Play 시 전부 조립한다 — **씬 직렬화 값이 스크립트 default 를 이긴다**, 기본값을 바꿨으면 씬 컴포넌트도 같이 고칠 것.

- **데이터 소비**: `V2/LaneGraphV2.cs` 가 `tools/nationwide_v2/lanegraph/seoul_gangnamgu.bin`(LGV2) + `.walk.bin`(WLK2) + `.stdlink.bin`(STDL) 직독.
- **★3D 복원(2026-07-30) — 평탄화 폐기**: 예전엔 지형고도를 통째로 버린 평탄 씬(ground y=0·건물 base −0.25·도로 0.08·모든 프로브 ±3m 밴드)이었다. 이제 **정밀도로지도 차선 z 가 지면의 단일 출처**다.
  - `LaneGraphV2` 가 elev 를 보존(`_wy`)해 `NearestLane/SampleLane/LinkPoint/GroupPos/SignalPos` 가 3D 를 돌려주고, 링크 문맥이 없는 소비자를 위해 **`SurfaceY(pos, maxDist, fallback)`**(근접 차선 중 **최저** — 머리 위 고가 회피)를 제공한다. **레이캐스트 지면 프로브는 전부 제거**(`RoadMarkingsV2`·`CrosswalkInfrastructureV2`·`StreetFurnitureV2`·`TrafficControlObstacleInfrastructureV2`) — 타일 스트리밍이라 Build 시점엔 먼 타일 콜라이더가 아예 없어 프로브가 원리적으로 불가능하다.
  - 보행자는 **WLK2 3번째 성분(표고)**을 쓴다(강남 10.3~86.2m, 결측 0%). 건물 라벨(`BuildingInfoOverlayV2`)은 인덱스가 2D(y=0)라 `SurfaceY` 로 지면고를 채운다.
  - 주행면 = 타일 **`_RoadSurf`**(HD A2 차선 리본 + A3 차도면, 정사 텍스처). CAR_test 가 굽던 `LaneRoads_V2` 리본은 폐기(`LaneRoadBuilderV2` 는 미사용 보존). 스트리머가 `_LaneMarks/_Marks/_Signals/_Posts/_Signs/_Guardrails/_Props/_Roads` 를 **로드 시마다 끈다**(CAR_test 가 같은 걸 런타임 생성 — 구 평탄 씬은 프리팹 인스턴스 오버라이드로 껐지만 스트리밍은 매번 새 인스턴스라 코드가 꺼야 한다).
  - 논리 구역(차선쿼리·NPC·보행자·도색)은 부트스트랩 `boundsMinEN/MaxEN`(현재 5×5km=25타일)로 제한, **타일은 경계와 무관하게 110장 전부 스트리밍**된다.
  - 실측(강남역 tile_203_544, 차선점 13,198): 차선z − 지형y **p50 +0.186m** / p05 +0.093 / p95 +0.316(고가는 p99 +5.3). Play 실측 = 4/4 휠이 `_RoadSurf` 접지(3.2kN/휠), 에고 Δ 0.06m, NPC 1,016대 평균 +0.05m, 보행자 136/140 이 ±1m.
- **환경 구축 완료 상태(2026-07-27)**: 신호(`TrafficSignalDirectorV2`)·신호등 메시(`SignalHeadRenderer`)·물리 지주(`TrafficControlObstacleInfrastructureV2`)·횡단보도(`CrosswalkInfrastructureV2`)·**노면도색(`RoadMarkingsV2`)**·NPC(`TrafficManagerV2`+ITS 실시간 `TrafficVolumeService`)·보행자(`PedestrianManagerV2`)·**등화(`VehicleLightRig`)**·**카메라 리그(`DriveCameraRig`)**·**HUD(`Autonomy/AutonomySensorHud`)**·미니맵(`MinimapOSM`) + **건물 실측 HUD(`BuildingInfoOverlayV2`+`CarBuildingIndex`, B 키)** — vWorld GIS건물통합정보 인덱스(`tools/nationwide/bldgindex/<region>.txt`)로 이름·높이·층수 라벨 + 측위 카드.
- **NPC 교차로 통행 규칙(2026-07-28)**: ①**정지선 미수록 링크는 링크 끝을 정지 기준**으로(LGV2 stopS 합격선이 85% 라 구 코드는 그 접근로 신호를 통째 무시 → 적색 진입) ②황색 딜레마존(편안한 감속으로 못 서면 통과) ③**교차 통행권=선착순**(두 진행 직선의 교차점까지 거리 비교, **먼 쪽이 양보** — 비대칭이라 교착 없음. 멀리 정지한 차에는 양보 안 함, 안 그러면 모든 교차로가 잠긴다) ④**교차로 물기 금지**(출구 여유 16m 없으면 정지선 앞 대기). 검증 지표 = 적색 정지선 통과 대수·차체 OBB 실제 겹침쌍(944대 기준 각각 0).
- **차량 물리** `V2/RoadVehicleDynamics.cs`(WheelCollider+ABS/TCS/ESC) + **`V2/Powertrain.cs`**(엔진 토크곡선→7단 기어→종감속, RPM/기어 노출·절차적 엔진음). 조향은 속도감응(고속서 최대각 32%).
- **센서 스택** `Autonomy/`: LiDAR16·Radar77GHz·GNSS·IMU·열화상 + **휠오도(`WheelOdometrySensor`)·초음파 12구(`UltrasonicSensor`)·V2X SPaT(`V2xSpatSensor`, 전방 신호 현시/잔여초/GLOSA/딜레마존)·전방 RGB(`RgbCameraSensor`)** + **(2026-07-28) 스테레오 깊이(`DepthCameraSensor`)·GNSS/INS 통합측위(`GnssInsFusionSensor`)**. 전부 `AutonomySensorRig` 가 조립·시드리셋·기상열화 전달.
  - **Depth 가 필요한 이유**: LiDAR 는 세로 16층뿐이라 연석·방지턱·낙하물·보행자 다리가 층 사이로 새고, 레이더는 클러터 억제로 정지물에 약하며, 초음파는 5m 이내다 — 그 사이(3~60m 조밀 형상)를 메운다. 출력=정규화 깊이 RT + `gridCols×gridRows` 최소거리 격자(관측 벡터용) + AEB 용 전방여유. ⚠**노면은 장애물이 아니다** — 아래로 기운 행은 항상 4~13m 앞 노면에 맞으므로 `obstacleMinHeight`(0.28m) 위로 솟은 것만 장애물로 셀 것(안 그러면 "전방 3m" 가짜 경고 상시).
  - **측위 융합**: 원시 GPS 를 그대로 쓰지 않는다. 예측=휠오도+IMU 요레이트 추측항법, 갱신=①GNSS(하늘 열림도로 R 가변) ②LGV2 차선 횡매칭 ③ZUPT. 출력=융합 포즈·위치 1σ·GNSS 음영·추측항법 표류·`TruthErrorMeters`(진단용). 하늘 열림도는 상방 레이캐스트로 자동 추정(도심 협곡·고가 밑에서 저하).
- **★재발 함정 4건(전부 이번에 물린 것)**
  1. **glTFast 임포트 머티리얼엔 `_BaseColor`/`_EmissionColor` 가 없다**(`Shader Graphs/glTF-pbrMetallicRoughness`). MaterialPropertyBlock 으로 색을 써도 무시돼 **신호등 램프가 한 번도 점등된 적이 없었다**. 램프는 URP/Lit 로 교체하거나 `emissiveFactor`/`_EmissiveFactor` 도 같이 쓰고 **`_EMISSION` 키워드를 켜야** 한다.
  2. **노면 도색 y 프로브**: 단일 `Raycast` 는 도로 리본이 아니라 그 아래 지형에 먼저 맞아 도색이 묻히고, 반대로 `RaycastAll` 최대 y 는 **머리 위 고가 리본**을 잡아 도색이 공중 16m 에 뜬다 → **지면 밴드(±3m) 내 최상단**을 취할 것.
  3. **`GroupBoundary` 그룹을 무신호 처리하면 강남역 사거리가 통째로 무신호**가 된다(구 버전 마스트 최근접 303m). 경계 그룹도 사이클을 주고, `GroupAxes==1` 그룹은 최소 2현시로 강제(안 그러면 상시녹색 84%).
  4. **`RightOf`/`LeftOf` 인접이 비어 있는 링크가 많다** → 지주가 1차로 한복판에 선다. 기하 프로브(옆으로 차선폭씩 밀며 동방향 차선 유무 확인)로 갓길까지 밀어낼 것. MAST_Signal 프리팹은 **헤드가 로컬 +z, 암이 로컬 −x** 라 램프를 마주 오는 차 쪽으로 돌리려면 `LookRotation(−tangent)` + **지주를 진행방향 좌측(중앙분리대측)** 에 세워야 암이 차도를 덮는다.
- **노변 장애물** `V2/StreetFurnitureV2.cs` = OSM 실측 + LGV2 도로변 절차생성 혼합. 실측은 `tools/v2_street_furniture.py`(OSM 태그 → `tools/nationwide_v2/streetfurniture/<region>.txt`, EPSG:5186; 점=가로등/가로수/소화전/볼라드/전신주/벤치/휴지통/쉘터/정류장/표지, 선=가드레일/펜스/담장/생울타리/연석). ⚠**한국 OSM 은 가로등 태깅이 사실상 없다**(강남 파일럿 0건) → 가로등·가로수는 최외곽 차로 가장자리+보도 오프셋에 절차 배치(28m/9m). 배치 가드는 `OnCarriageway()`(모든 차로 중심선 반폭+0.9m 점유 검사) — `IsOutermostRight` 만으로는 교차로 내부에 가로수가 심어진다.
- 조작: `W/S` 가감속(정지 중 S 유지=후진) · `A/D` 조향 · `Space` 핸드브레이크 · `Shift` 부스트 · `Q/E` 방향지시등 · `L` 전조등 · `R` 경로재설정 · `V`/`1~6` 카메라(운전석/보닛/추격/원거리/궤도/상공) · `F1~F4` HUD · `M` 미니맵 헤딩업.
- **★재발 함정 5건 추가 (2026-07-27 2차)**
  5. **비활성 MonoBehaviour 에도 `OnCollisionEnter` 가 전달된다.** 레거시 `CarPhysics` 를 `enabled=false` 로만 두었더니 그 구 핸들러가 보행자 충돌을 **먼저** 가로채 `Knock` 을 실행하고 레이어를 바꿔, 뒤에 도는 `RoadVehicleDynamics` 핸들러는 이미 Pedestrian 레이어가 아니어서 운동량 복원도 `LastPedHitTime`(RL 페널티 훅)도 유실됐다 → **`Destroy(_legacy)`** 로 물리 콜백 경로를 하나로 만들 것.
  6. **엔진브레이크를 음의 `motorTorque` 로 주면 안 된다.** WheelCollider 는 그대로 역방향 구동으로 해석해 타행 중인 차가 **뒤로 가속**한다(실측 50km/h 전진 → −61km/h). 반드시 `brakeTorque`(항상 회전 반대)로 변환해 구동륜 배분으로 걸 것.
  7. **차-보행자 충돌**: kinematic 보행자는 PhysX 가 **무한질량 벽**으로 풀어 1.5t 차를 즉시 세운다. 솔버 직전 속도를 캐시해 두었다가 완전비탄성 운동량 손실 `m_ped/(m_car+m_ped)≈4.3%` 만 적용해 되돌린다. `KnockablePedestrian.Knock` 은 레이어를 **Debris**(에고 무시)로 — Default(0)로 두면 래그돌이 다시 차를 막는다.
  8. **1인칭 카메라는 위치를 Lerp 하면 안 된다.** 가속 중 목표가 앞서가 카메라가 뒤로 처지고(운전자가 뒷좌석으로 밀리는 느낌) 감속 시 대시보드에 박힌다 → 실내 시점은 **위치 강체 고정**, 관성 변위는 cm 단위로 클램프. 가속도는 LateUpdate 가 아니라 **FixedUpdate 에서** 미분할 것(렌더 프레임이 잦으면 0/스파이크 교대로 떨림).
  9. **횡단보도 줄무늬 방향**: 바는 **차량 진행방향과 평행**하고 도로 폭 방향으로 반복한다. 구 코드처럼 crosswalk 장축(a→b) 전체 길이로 바를 그리면 **사다리 가로대**가 된다(사용자 신고 "가로여야 하는데 세로"). `AddStripePrism(center, across, along, halfDepth, barWidth/2, …)` 순서 주의.
- **전국화(2026-07-27 3차)**: `tools/v2_region_index.py` → `tools/nationwide_v2/region_index.json`(255 지역 EPSG:5186 bbox + 산출물 보유 플래그). Unity `V2/RegionCatalogV2.cs` 가 읽어 하드코딩 원점/경계를 대체 — 부트스트랩 `region` + `usePilotOrigin` 로 전환. ⚠sgg.json 의 `bbox` 는 **[minlat, minlon, maxlat, maxlon]**(위도 먼저) — lon/lat 순으로 읽으면 좌표변환이 inf 를 뱉는다. **차선그래프 249/255 · walk.bin 249/255 전국 완료**, 노변장애물은 `--all` 로 순차 수집(재개 가능). ⚠**v2 3D 타일(정사·건물)은 강남 9타일만 베이크** — 타 지역은 논리 계층만 동작(`RegionCatalogV2.HasBakedWorld`).
- **★재발 함정 4건 추가 (2026-07-27 3차)**
  10. **`Knock`/`Shove` 후 레이어를 Debris 로 보내면 에고가 무시**한다. 보행자(70kg)는 그게 맞지만 **차량(1.5t)은 두 번째 충돌이 통째로 무시**돼 통과해 버렸다 → 차량 전용 **`Wreck`(16)** 레이어 신설(도로·에고·다른 파손차와 충돌, 주행 NPC/보행자와는 무시). 지면 프로브 마스크에서도 제외할 것.
  11. **사고 잔해를 짧은 타이머로 되돌리면 눈앞에서 증발**한다(보행자 4초·차량 2.5초). 체류시간을 늘리고 **뷰어에서 멀거나 시야 뒤일 때만** 회수(`IsFarFromViewer`).
  12. **축을 맞바꾸면 삼각형 와인딩 부호가 뒤집힌다.** 횡단보도 줄무늬는 방향이 맞는데도 윗면 노멀이 −Y 가 되어 **백페이스 컬링으로 안 보였다** → 두 번째 축에 `-along` 을 넘겨 외적 부호를 +Y 로 유지.
  13. **IMGUI 회전 지도는 `BeginGroup` + 그룹로컬 피벗이면 창 밖으로 삐져나온다**(피벗이 스크린 좌표로 해석됨) → **`BeginClip`** 을 쓸 것. Blender→glTFast 임포트는 **X 축이 뒤집힌다** — 램프 배열 같은 좌우 순서는 반드시 Unity 에서 실측 확인(첫 빌드에서 적/녹이 반대로 나왔다).
- **자동변속기 인터록**: 기어 선택은 **운전자 의도**를 따른다. 구 코드는 `Gear<0 && speed>-0.6` 일 때만 D 로 복귀시켜, 드리프트로 뒤로 미끄러지는 동안 **R 에 갇혀 전진 페달이 후진 토크**가 됐다 → D 요청이면 속도 부호 무관 즉시 전진단, R 은 전진속도 `reverseInterlockSpeed`(2m/s) 미만에서만 체결.
- **방향지시등**: 차량 GLB 의 주황 발광 메시는 좌우가 한 덩어리라 **어느 쪽을 켜도 같이 켜져 비상등처럼** 보였다 → `VehicleLightRig` 가 차체 바운즈를 실측해 **전·후·사이드 리피터 6개 전용 램프**(주황 emissive + Point Light)를 생성해 좌/우 독립 구동. `H` = 비상등.
- **신호등 에셋**: Blender 제작 `KR_MastSignal`(국내 4구 가로형 — 렌즈 Ø300mm, 하우징 1.33×0.40×0.18m, **노란 테두리 프레임**, 등화별 차양, 좌회전 화살표 렌즈, 지주 Ø190·암 5.6m). 프리팹 규약 = **원점 지주 발밑 · 암 로컬 +X · 램프 로컬 +Z**, 배치는 `LookRotation(−tangent)` + **우측 갓길 지주**. 직좌 현시엔 국내 규약대로 **적 + 좌화살표** 동시 점등.
- **내비게이션** `V2/NavigationPlannerV2.cs`: `N` 키로 전체화면 OSM 지도 → 좌클릭으로 출발지·목적지 지정 → `Enter` 로 **출발지에 스폰 + 경로 설정**. 좌표는 화면→슬리피타일→WGS84→5186→씬월드 역변환 후 LGV2 최근접 차선 스냅.
- **★재발 함정 5건 추가 (2026-07-27 4차 — 전부 Play 실측으로 잡음)**
  14. **차체 BoxCollider 를 렌더러 bounds 로 만들면 바퀴까지 포함**돼 박스 바닥이 접지면까지 내려온다 → 노면·연석을 상시 긁고 건물에 한 번 물리면 못 나온다(실측: 박스가 `_Ortho`+`chunk0`(건물)에 동시 침투, 바퀴 슬립 0.77인데 속도 0). **바닥을 `bodyGroundClearance`(0.28m)만큼 올리고** 접지는 WheelCollider 에 맡길 것.
  15. **`TryRecoverCatastrophicState` 가 "마지막 안전 포즈"로만 복원하면 무한 루프**한다(그 포즈가 건물 안이면 디페네트레이션 폭주 → 폴트 → 같은 자리 복원, 실측 17연속). 반복 폴트는 **차선 스냅**(`PerformUnstick`)으로 뺄 것. 별도로 "구동 명령 있는데 안 움직임 + 바퀴 헛돎(또는 접지 0)"을 감지하는 anti-stuck 을 둔다(`T` 키 수동 구조).
  16. **`Shove`/`Knock` 후 잔해를 짧은 타이머로 되돌리면 눈앞에서 증발**, 반대로 Debris 로 보내면 **에고가 두 번째 충돌을 통째로 무시**한다. 차량은 `Wreck`(16, 에고와 충돌), 보행자는 `Debris`(에고 통과)로 갈라야 한다.
  17. **차량 GLB 의 방향지시등 메시는 좌우가 한 덩어리** — 가짜 램프를 붙이지 말고 **메시를 차량 로컬 X 부호로 분할**해 두 렌더러로 만들면 원래 등화 형상 그대로 좌/우 독립 점멸이 된다(`VehicleLightRig.SplitRendererByVehicleX`). 분할 시 **베이스 색은 건드리지 말고 발광만** 제어할 것(소등 시 원래 렌즈 색 복원 불가).
  18. **`TrafficManagerV2` 의 IDM 은 같은 링크/후속 링크 앞차만 본다** — 교차로에서 다른 접근로 차량과 횡단보도 보행자를 전혀 인식하지 못해 서로 통과했다. 공간격자 + 전방 원뿔(`ScanConflicts`)로 가상 리더를 만들 것.
- **버스전용차로**: 정밀도로지도 A2 `laneno` **특수코드 91/92** 가 전용차로다(실측 검증: laneno 91 은 좌측 이웃이 없고 우측 이웃이 1차로 = **중앙버스전용차로** 배치와 일치, 강남 1,402링크). `RoadMarkingsV2` 가 **청색 실선**으로 그리고 `TrafficManagerV2` 는 여기에 일반 NPC 를 스폰하지 않는다.
- **차로별 통행권**: `TrafficSignalDirectorV2.MovementOf(link)`(successor 방위로 직진/좌/우 판정·캐시) + `StateForLink` 가 국내 4구 규약을 반영 — 직좌 현시엔 좌회전 차로만 녹색·직진은 적색, 직진 녹색엔 좌회전 적색, 우회전은 상시 통행. ⚠HD맵 특성상 접근링크 대부분이 직진 successor 를 함께 가져 **전용 회전차로는 소수**(강남 9타일 기준 좌18·우10)다 — 회전은 주로 교차로 내부 연결링크에서 일어난다.
- **보행등 카운트다운**: `PedestrianSignalHeadRendererV2` 가 2자리 **7세그먼트**(쿼드 14장, 폰트 의존 없음)로 잔여시간을 표시. 점멸 구간(보행 녹색 말미)은 깜빡인다.
- **차량 물리 고도화(2026-07-27 2차)**: 스태빌라이저(좌우 서스펜션 압축차 → 반대 힘, 전 22kN/후 18kN), **마찰 원**(종·횡 그립 예산 공유 — 급가속 중 선회그립 ↓), **타이어 하중 민감도**(μ ∝ Fz^−0.15), 바퀴별 수직하중·차체 롤/피치·좌우 하중 불균형 텔레메트리. Pacejka 전 모델 대신 WheelCollider 마찰곡선 stiffness 를 바퀴별로 매 스텝 조정(저속 발산 문제 없음).
- **★3D 전환 후 잡은 결함 5건 (2026-07-30 2차, Play 실측)** — 전부 "평탄 씬 전제"의 잔재이거나 소스 z 특성:
  ①**절대 y 대입 잔재**: `LaneLineRenderer`(`a.y = 0.28`)·`RouteManager`(`a.y = ribbonY 0.22`) 가 오버레이를 절대고도에 그려 **F3 를 눌러도 지하 18m 에 묻혀 안 보였다** → `+=`(차선 z 위 상대높이).
  ②**노면표시가 링크당 상수 y**: `RoadMarkingsV2` 가 링크 중앙 y 하나로 도색해 종단구배 구간에서 도색이 뜨거나 묻혔다 → `AddQuad/AddTri` 를 **정점별 y**로.
  ③**이웃 차선 리본이 도색을 덮음**: 인접 차선 z 가 p90 12cm·p99 26cm 높아 경계 도색이 그 아래로 묻힘(실측 2.3%) → `LaneGraphV2.LaneTopY(p, ownY, radius 4.2m, band 1.2m)` = 같은 층 중 **최상층 z** 에 얹는다.
  ④**A2 z 노이즈**: 정점 간격 2~3m 라 ±5cm 오차가 곧 2~4% 구배(실측 p90 5.7%·max 35.7%) → 베이커·런타임 **동일 규칙**의 이동평균(창 ±5, 양끝 대칭축소로 이음새 보존).
  ⑤**단차·부양의 진짜 원인 3종**: (a)DSM 건물 덩어리가 카빙 하한(12m)에 걸려 안 파여 **노면 위 0.84m 지형 단차**(→하한 25m·절삭반경 11m) (b)성토 노반을 고가로 오판해 비카빙(→아래를 지나는 차선이 없으면 `EMBANK_MAX 12m` 까지 채움) (c)고가가 최근접일 때 카빙 포기 → 교각 주변 지형이 지상도로 위로 솟음(→**아래 지상차선 z 기준**으로 카빙). 결과: 6,600 표본에서 **0.3m↑ 단차 0 · 지형이 노면 위 0**, 노면 국부구배 p50 0.56%·p99 5.3%.
  ⑥**`TrafficManagerV2.ScanConflicts` 인덱스 예외**: 공간격자에 담긴 NPC 인덱스가 같은 프레임 회수로 노후화돼 `ArgumentOutOfRangeException` → 그 프레임 NPC 갱신 전체가 죽었다. `j < _npcs.Count`·`p < PedestrianCount` 가드 추가.
- **★타일 재베이크 파이프라인(CAR_test 는 자체 베이커가 없다)**: 베이커는 UAV_test 전용 `Assets/Editor/V2/TileBakerV2.cs` 이고 DEM 도 거기 있다. 절차 = ①UAV_test 에서 `BAKE_VER` bump → `StartBake("seoul_gangnamgu")`(110타일 ≈3.7s/타일) ②`scratchpad/copy_tiles_to_car.sh` 로 복사(.meta 동반 = GUID 유지) + **Building 레이어 remap `m_Layer: 8`→`11`**(UAV 8 / CAR 11) ③CAR_test 에서 `AssetDatabase.Refresh(ForceUpdate|ForceSynchronousImport)` — ⚠**에디터가 켜진 채 파일만 갈아끼우면 재임포트가 안 걸린다**(구 캐시가 그대로 ver13 을 반환했다). 구 타일은 삭제 말고 scratchpad 로 이동.
- **★함정 26 — 노면 리본 와인딩 반전은 도로를 조용히 지운다**: 타일 `_RoadSurf` 차선 리본을 `(0,2,1)/(1,2,3)` 으로 감았더니 노멀이 **전부 −Y** → 위에서 백페이스 컬링으로 안 보이고, **`Physics.queriesHitBackfaces=false` 라 레이캐스트도 통과**해 "도로가 없는" 것처럼 동작했다(에고가 지형에 착지). 올바른 순서는 `(0,1,2)/(1,3,2)`. 베이커에 **자가검증**(아래 향한 삼각형이 과반이면 `LogError`)을 넣어 재발을 막았다.
- **★함정 27 — 카빙 우선순위: 차선 회랑이 1순위, A3/보도 링은 보조**: `SurfSampler.CarvedGround` 가 링(A3 차도면·보도) 사면 블렌드를 먼저 돌리면 도로 옆 보도 링이 **노면과 DEM 의 중간값**을 반환하고 그대로 return 돼 차선 카빙이 통째로 무시된다(실측: 차선이 지형보다 0.91m 아래로 묻힘). 순서를 뒤집자 p50 −0.913 → **+0.066m**.
- **★함정 28 — 격자 해상도가 회랑보다 굵으면 카빙이 정점 사이로 샌다**: 지형 21.7m 격자(TERRAIN_SEG 46)에선 폭 5m 도로 회랑 안에 정점이 하나도 안 들어가 카빙이 무효였다 → **7.8m(SEG 128)** + **평탄 절삭 반경 9m**(격자보다 넓게) + 회랑 밖은 `clamp(DEM, z∓0.25·Δ)` 절토/성토 사면. 회랑 내 지형은 노면보다 **0.18m 아래**로 고정(격자 보간 잔차 ±0.14m 흡수).
- **★A3 차도면은 커버리지가 희박하다 — 노면 소스는 A2 차선**: 강남 전역 A3 = 364폴리곤뿐이고 강남역 타일엔 **1개**(반면 A2 주행링크는 그 타일에만 467개). A3 만으로 카빙/노면을 만들면 도심 대부분이 DSM 그대로 남는다. 베이커 v12+ 는 `hd_links`(A2, `laneno linktype roadrank fromnode tonode r_linkid l_linkid` + `lat lon z` 트리플릿)를 **2m 래스터 + 체임퍼 거리변환**(최근접 차선 z 전파)으로 인덱싱해 노면 리본·카빙·OSM 중복제거를 한다. A3 의 `roadtype`(1 지표 −2.6m / 2 터널 −36m / 3 고가 +7.6m / 4 지하차도 −6.8m, 실측 medΔ=z−DEM)은 카빙 여부 판정에 쓴다.
- **★도심 DSM 바이어스**: GLO-30 은 DSM 이라 스무딩 후에도 강남 도로가 DEM 보다 **중앙값 3.1m·p05 9.2m 아래**다. 높이 임계(±8m)만으로 터널/지표를 가르면 지표도로가 터널로 오판돼 묻힌다 — 카빙 하한을 12m(`LANE_CARVE_DOWN`)로 두고 속성(roadtype)을 우선한다.
- **★재발 함정 7건 추가 (2026-07-28 — 콘솔 전면 디버깅·NPC 신호·센서 확충 세션)**
  19. **kinematic Rigidbody 에 `MovePosition` 으로 텔레포트하면 PhysX 가 Δp/Δt 를 속도로 해석한다.** 보행자 리스폰(수백 m 점프)이 곧 **수만 m/s 의 벽**이 되어, 접촉한 에고를 10⁴~10⁸ m/s 로 발사했다(`[RoadPhysicsFault] linear-speed`의 진범). 텔레포트는 반드시 `rb.position` 직접 대입. 정상 이동분(1스텝 이동거리)을 넘는 점프는 자동으로 텔레포트 처리할 것.
  20. **`Awake` 에서 원본 레이어를 캡처하면 안 된다.** 프리팹에 `KnockablePedestrian`/`KnockableVehicle` 이 이미 붙어 있으면 Awake 가 **Instantiate 중**에 돌아, 매니저가 레이어를 씌우기 **전** 값(Default 0)을 저장한다 → 한 번 치인 보행자가 Default 로 복귀해 에고를 막는 kinematic 벽이 된다. 레이어는 **Knock/Shove 시점**에 읽을 것.
  21. **복구 로직이 깨진 좌표를 `safe pose` 로 캡처하면 차를 영영 잃는다.** 폭주 좌표(2.4e9)를 탐색 원점으로 쓰고 실패 시 제자리에 두고 그 값을 safe pose 로 저장 → 무한 낙하 + 카메라 NaN + `Look rotation viewing vector is zero`/`Screen position out of view frustum` 무한 출력. **좌표 타당성 검사 + 스폰 지점(home) 최후 보루**가 필수.
  22. **`Time.fixedDeltaTime` 의 소유자는 `SimulationClock`**(`Awake` 에서 덮어씀, 실행순서 −10000). 부트스트랩에서 `Time.fixedDeltaTime` 만 바꾸면 잠시 뒤 센서리그가 만드는 클록이 50Hz 로 되돌린다 → **클록을 먼저 만들어 `physicsRateHz` 를 설정**할 것. 그리고 **씬 직렬화 값이 스크립트 default 를 이긴다**(새 public 필드도 한 번 저장되면 씬 값이 이김).
  23. **프레임 병목은 물리 주기가 아니라 상시 살아 있는 kinematic 강체 수**였다. 실측(강남 836 NPC+140 보행자): LOD 전 50Hz 28.5fps·100Hz 8.4fps → **원거리 NPC/보행자 콜라이더 해제(물리 LOD)** 후 50Hz 43.3·70Hz 40.5·100Hz 32.6. 채택 = **70Hz + 물리 LOD**(`TrafficManagerV2.physicsRadiusMeters` 150m / 보행자 90m). URP 에서 `Camera.layerCullSpherical` 은 **경고만 뜨고 못 쓴다**(레이어별 거리 컬링 자체는 동작).
  24. **IMGUI 회전 지도는 `BeginGroup`·`BeginClip` 어느 쪽도 클립이 안 된다** — `GUI.matrix` 가 클립 확정 뒤에 적용돼 내용이 창 밖으로 샌다. **RenderTexture + `GL.LoadPixelMatrix`/`GL.MultMatrix` 로 RT 안에서 그리고 통째로 blit** 할 것(버퍼가 곧 경계라 어떤 각도에서도 안 샌다). `MinimapOSM` 이 이 방식.
  25. **EKF 분산 하한을 너무 낮게 잡으면 필터가 계통편차를 못 잡고 표류한다.** 차선매칭(횡 1축)으로 스칼라 분산을 깎았더니 GNSS 이득이 1% 로 떨어져 **진오차 42m 인데 σ 0.10m** 로 보고했다. ①1축 관측은 분산을 깎지 말 것 ②분산 하한 σ≈0.7m ③관측이 예측과 크게 벌어지면(>12m) **재초기화**. 수정 후 진오차 1~2.5m/σ0.71 로 정합.

### ★자율주행 학습은 UAV_test 로 이관 (2026-08-21, v4)

주행 스택의 **유지되는 사본은 이제 `UAV_test/Assets/CarDrive/`** 다(CAR_test 사본은 2026-08-11 에서 멈춰 있고 `GangnamDriveBootstrap` 만 1,352줄 갈렸다). 세계도 v2 타일이 아니라 **v1 시군구 씬**이다 — 씬 `Assets/Scenes/KDT_DriveTraining.unity`(Sun + `KoreaDriveBootstrap` 뿐) 이 시군구 씬을 additive 로 얹고 그 위에 주행 스택을 조립한다. 자율비행(`KoreaDigitalTwin`)과 같은 프로젝트·같은 세계다. 정본 `UAV_test/Documentation/ROAD_DRIVING_RL_DESIGN_V4.md`.

- **★액션 공간 = 계층형 [횡오프셋, 목표속도]**(구 [조향, 가감속] 폐기). 조향은 정책이 아니라 `RoadDrivingAgent.TrackPath` 의 **순수추종**(lookahead `clamp(0.9·v, 6, 22)m`)이 만들고, 경로 곡률로 `v ≤ √(a_lat_max/κ)` 감속까지 건다. 이유: **ML-Agents 연속액션은 가우시안+클리핑**이라 CaRL(CoRL 2025)의 Beta 분포 해법을 쓸 수 없고, σ≈1 조향을 10Hz 로 내보내면 채터링→"기어가기"가 국소최적이 된다(구 s4 실측 1 m/s 고착). 문헌의 다른 해법(계층형: 상위=설정점, 하위=기하추종기, atHRL·h-DDQN)을 택했다. 실측(규칙 기준선·역삼역): **차선 횡오차 평균 0.055m·p95 0.151m, 조향 반전 0.28회/s, 요레이트 RMS 0.092** — 사용자가 신고한 "좌우 흔들림"이 사라졌다. 오프셋 범위는 **±2.0m**(차선 안 한계는 ±0.73m 뿐이라 그 안에 묶으면 정차차량 회피가 원리적으로 불가능 → 능력은 열고 중앙 유지는 페널티가 맡는다). ⚠구 onnx 는 액션 의미·관측계약(BEV) 둘 다 달라 **재사용 금지**(`Models/RoadDriving_legacy_preBEV_s19.onnx` 로 내려 둠).
- **★v1 씬 ↔ CAR 런타임 중복 소유권**: v1 씬은 실측 B2 노면선(`_HdLaneMarks`)·OSM 신호(`_Features/Signals`)·OSM 횡단보도(`_Features/Crossings`)를 **이미 구워** 갖고 있는데, CAR 스택은 그걸 런타임에 또 만든다. v2 타일 경로는 `TileStreamerV2.hiddenGroups` 가 껐지만 **v1 씬엔 스트리머가 없어 아무도 안 껐다** → 사용자 신고 "중앙선 겹침·차선 이중"의 진범. 실측(`tools/v2_lane_paint_audit.py`, 역삼 ±1.5km): 합성 도색 표본의 **66.5%가 실측 B2 선 1m 안**(p50 0.69m), 중앙선은 대향 차로 두 링크가 같은 자리에 각각 황색 복선을 그려 **9.9% 중복**. 원리적으로 필연이다 — LGV2 의 `laneWidth` 자체가 B2 경계선에서 복원한 값이다. → 규칙: **도색 = v1 실측 B2**(`enableRoadMarkings=false`) / **신호·횡단보도 = CAR 런타임**(그쪽만 현시가 돈다). `KoreaDriveBootstrap.HideRegionDuplicates()` 가 OSM 그룹을 끈다. `RoadMarkingsV2` 쪽은 **링크 인덱스가 작은 쪽만 중앙선 소유**로 고쳤다(v2 타일 경로용).
- **★빌드 함정 — `Application.dataPath` 는 빌드에서 Assets 가 아니다**(`<build>/UAV_test_Data`). 런타임 코드가 `dataPath + "Scenes/Regions/sgg_index.json"`(시군구 경계)·`terrain/*.bin`(DEM)을 직접 읽고 있어 **빌드 플레이어에선 `SggNameAt` 이 null → "좌표가 시군구 범위 밖"으로 조용히 죽는다**. `RuntimeDataPath.Resolve/ResolveDir`(StreamingAssets 우선)로 통일하고 `RoadTrainingBuilder.PrepareStreamingAssets()` 가 학습 지역분만 복사한다. 차선그래프는 별도 정책 `AutonomyDataPaths`(`CAR_TEST_DATA_ROOT` 또는 StreamingAssets).
- **빌드/학습/검증 통로**: 빌드 `Tools/RoadDrive/Build RoadTrain Player`(외부에서는 `Temp/hd_rebuild.request` 에 `roadbuild=1`) → **시군구 씬은 학습 지역 하나만** 등록한다(한 개 479MB). 학습 `MLConfigs/run_road_drive.ps1`(`road_driving_ppo.yaml`, PPO 2e7, 관측 124+BEV 64³, 결정 10Hz). 무인 주행검증 `Temp/hd_rebuild.request` 에 `playtest=drive lat= lon= sec=` → `Temp/selftest_drive.json` 에 **주행 품질 수치**(조향 반전율·요레이트 RMS·차선 횡오차 p95)를 남긴다(`ModeSelfTest`).
- ⚠**씬 직렬화 값이 스크립트 기본값을 이긴다**(재확인): `KDT_DriveTraining.unity` 의 스폰이 **강남역(=행정구역상 서초구)** 으로 굳어 있어 강남구 차선그래프가 안 잡히는 상태였다 → 역삼역(37.5006, 127.0364)으로 정정.
- ⚠**스모크가 늘 FAIL 이면 신호가 없는 것과 같다**: `RoadDrivingAcceptanceProbe` 의 접지 판정이 8초 구간 **순간 최솟값**이라 리스폰 낙하 한 프레임으로도 0/4 가 찍혀 항상 FAIL 이었다(670m 무사고 주행도 FAIL). 게다가 그 낙하를 **프로브 자신의 재시도**가 만든다 → 합격 조건에서 빼고 비율만 보고한다.
- ⚠**정지선 색상 오류**: `tools/v2_hd_bucket.py` 의 `YELLOW_KIND` 에 530(정지선)이 들어 있어 정지선이 황색으로 구워졌다(국내 규약 백색). 스크립트는 정정했으나 **이미 구워진 버킷·씬은 재버킷 + HD 재임포트 전까지 황색**이다.
- 신호 체계 감사 결과 = **논리·데이터 모두 건전**(C1 차량등 4,086개·링크 부착 99.83%·정지선 복원 86%, 국내 4구 현시 + 차로별 통행권). 근사 2건만 알고 있을 것 — ①LGV2 가 현시축을 1개만 기록한 그룹이 546 중 379(69%)라 코드가 `max(2, axes)` 로 **가공 2현시**를 돌린다(안 그러면 상시녹색 84%) ②**우회전 차로는 상시 녹색**(2022년 개정법의 적색 시 일시정지 미구현).

## ★ML-Agents 학습 = Windows 빌드 → Linux 학습 (2026-09~)

Unity ML-Agents 학습도 이제 리눅스에서 돈다. 흐름은 **Windows 에서 플레이어를 굽고 → 서버로
보내고 → 산출물을 회수**다. 규약 넷:

- **산출물·플레이어는 `/home/ryu/roaddrive/` 안에서만** 다룬다. 학습박스의 리포 체크아웃
  `/home/ryu/MCI_UAV` 는 **다른 계정 세션이 쓰는 중**이라 건드리지 않는다.
- 두 박스가 공유하는 텍스트(학습 드라이버·커리큘럼 yaml·검증기)의 **정본은 Windows 의
  `tools/exp_drivers/`** 인데, 2026-09-09 부터 **git 추적 대상이 아니다**(원격에 Unity 코드를 안 남긴다).
  이동 통로는 `tools/exp_drivers/sync_roaddrive.sh` 하나뿐 — 서버 쪽 mtime 이 더 새로우면
  **덮지 않고 exit 3** 이다(서버에서 직접 핫픽스한 전례가 있다). 그때는 `--pull` 로 회수해
  로컬 파일에 반영한다(**커밋으로 남지 않으니 회수 자체를 빼먹지 말 것**).
- ⚠️**LF 는 이제 `.gitattributes` 가 지켜주지 않는다.** 미추적이라 git 이 손대지 않으므로
  `core.autocrlf` 변환도 없지만, 반대로 **Windows 편집기가 CRLF 로 저장하면 그대로 서버로 간다**.
  그러면 bash 가 `set -u` 을 못 읽고도 **exit 0 으로 계속 간다**(실측: `set: - : invalid option` 만
  찍히고 안전장치가 통째로 사라짐). 조용히 깨지는 종류다 — 새 `.sh` 는 LF 로 저장할 것.
- ⚠️**빌드 성공 판정은 로그가 아니라 DLL 내용으로.** 에디터가 열려 있으면 `-batchmode` 가
  `Temp/UnityLockfile` 때문에 죽는데(요청파일 `Temp/hd_rebuild.request` 경로는 열린 채로도 동작),
  구 바이너리가 그대로 전송돼도 학습은 멀쩡히 도는 것처럼 보인다. 실측 사고: 보상 7개를 고쳤는데
  세 번 연속 스테일 빌드가 나가 파인튠이 전부 무의미했다. 확인은
  `grep -ac <새필드> <build>/UAV_test_Data/Managed/Assembly-CSharp.dll` ≥ 1.

## tools/ data pipeline

- OSM via Overpass: `osm_roads2.py` (class/lanes/oneway/**struct** → detailed roads; line=`<class> <lanes> <oneway> <struct> <coords>`, struct 0=ground/1=bridge/2=tunnel, struct-less old files detected by 4th token being a decimal coord = backward-compat; **large bboxes return `geometry` with null points — filter `[p for p in geom if p and "lat" in p]` before `p["lat"]`**), `osm_features.py` (signals/crossings/bus stops/**streetlamp/hydrant**, `--trees` opt), `osm_areas.py` (park/green/water polygons), `osm_poi.py` (hospital/school/fire/police/fuel).
- **★`osm_water.py` — 수계는 이걸로 (2026-07-13)**: `osm_areas.py` 의 water 는 **쓰지 마라**. 그 쿼리는 `way[...]` 만 받는데 **한강·낙동강·안동호 등 국내 주요 수계는 전부 OSM multipolygon `relation`** 이라 통째로 누락돼 있었다(강남구 water = 0.10km² 연못뿐, 한강 없음). `osm_water.py` 가 relation 까지 받아 **구 경계로 클립**(대형 수계는 여러 구에 걸쳐 center-point 귀속이 불가) → 전국 255/255 구·총 **1,996 km²** 확보(0.10→2.31 강남, 0→5.12 영등포). 출력은 기존 `area/<name>.txt` 의 water 라인을 교체하는 **동일 포맷**(park/green 보존) → Unity 임포터 무수정. ⚠️**함정 3건**: ①Overpass `out tags geom` 은 relation 의 **members 를 안 준다**(진범) → **`out body geom`** 필수. ②링 1개 포맷엔 구멍이 없어 여의도·밤섬은 **키홀 절개**로 병합(검증: 여의도=물X, 한강본류=물O). ③**같은 area 파일에 두 프로세스가 동시 기록하면 줄이 붙고 park/green 이 통째 소실**된다 — Git-Bash 의 `ps -W | grep <스크립트명>` 은 **EXE 경로(`python.exe`)만 보여 스크립트명이 안 잡히므로** "죽었다"고 오판하기 쉽다(실제로 5개 구가 날아가 `osm_areas --force` 로 복구). 재개는 스탬프 `nationwide/.water2/<name>.ok`. All read `MCI_OVERPASS_URL`/`OVERPASS_URL` (local self-hosted Overpass, via `osm_overpass_endpoints.py`) else rotate 3 public mirrors to dodge 429, attribute features to a district by **center-point-in-polygon**, are resumable (skip existing output, `--force`), and write compact text to `tools/nationwide/<kind>/<name>.txt`. `osm_roads2.py footway` → `nationwide/footways/<name>.txt` (보행로/인도, roads2와 분리 레이어; `ImportFootwaysOne`이 `_Sidewalks` 콘크리트 리본 베이크, 주행망 제외).
- `vworld_fetch.py` pulls buildings/orthophoto tiles via the vWorld API. `nationwide_build.py`/`build_region_index.py` drive the citywide build + `region_index.json`. `scene_export.py`/`trace_export.py`/`run_sim_trace.py` bridge the Python sim output into Unity-loadable JSON. `scene_export.py` also emits per-route `states[]` (Kakao `traffic_state` 0–5, index-aligned to `pts`) for congestion-aware NPC density/speed.
- **★정밀도로지도 → 차선그래프/보행/교통 사이드카 바이너리 (2026-07, `tools/v2_lanegraph_build.py`·`v2_walk_extract.py`·`v2_stdlink_export.py`)**: 소스는 `hdmap_fetch.py` 가 받아둔 `tools/nationwide/hdmap/<layer>/<region>.geojson`(좌표 `[lon,lat,z]`), 산출은 `tools/nationwide_v2/lanegraph/` 밑 **리틀엔디언 bin**(Unity `CAR_test/Assets/Scripts/V2/LaneGraphV2.cs` 등이 `BinaryReader` 로 직독). 둘 다 gitignore(`tools/nationwide_v2/*`, 예외 `qa/` PNG) — **스크립트만 커밋**.
  - `<region>.bin` = **LGV2** 차선그래프: A2 주행링크=차선 1:1, `successor(L)={M | M.fromnodeid==L.tonodeid}`(직진성 오름차순 정렬, `[0]`=최직진), l/r 인접=차선변경, + C1 신호·B2 정지선(stopS)·차선폭·제한속도·교차로 그룹/현시축. 포맷·그래프 규약·합격기준은 **스크립트 docstring 이 정본**. `<region>.report.json` 의 `criteria`(wcc≥0.95·고립 dead-end<2%·신호부착≥99%·stopS≥85%·폭≥90%) 가 **PASS 여야 진행** — 미달은 원인 규명 먼저.
  - **전국 255구 전수 완료**: bin **249개** + report 255개. **정밀도로지도 미수록 6구**(`busan_{namgu,suyeonggu,yeongdogu}`·`gyeongbuk_ulleunggun`·`incheon_{ganghwagun,ongjingun}`)는 **bin 없이 report 만** 나오는 게 정상(d8c199e, vWorld WFS 실측 근거) — 빌드 실패로 오판하지 말 것.
  - `<region>.walk.bin`(**WLK2** 횡단보도/보행등/보도) 과 `<region>.stdlink.bin`(**STDL** 표준노드링크 MOCT_LINK 중점 → ITS 실시간 소통속도를 링크에 매칭)은 LGV2 를 건드리지 않는 **별도 사이드카**(기존 씬·`LaneRoadBuilderV2` 무영향). ⚠️**앵커 규약이 둘로 갈린다**: `walk.bin`은 **LGV2 와 같은 anchorE/N 을 그대로 써야** 월드 변환이 일치하고, `stdlink.bin`은 **자기 링크 중점 평균을 자체 앵커로 헤더에 실어 자기기술**한다 — 소비측이 LGV2 앵커를 가정하면 어긋난다. 좌표계는 셋 다 EPSG:5186.
  - walk/stdlink 는 **강남 파일럿만 생성됨**(LGV2 는 255구 전수). `v2_stdlink_export.py` 는 ITS 표준노드링크 배포본(`MOCT_LINK.shp`)이 로컬에 있어야 하고 기본값이 **박스별 Downloads 절대경로** — 다른 박스·다른 구는 `--shp/--bbox/--region` 으로 넘길 것(무인자 기본=강남 9타일).
- **★DEM 스무딩(v1 트윈 지형) + 건물 높이 인덱스 (2026-07-25)**: `tools/terrain_smooth.py` — GLO-30 은 **DSM** 이라 도심 건물 덩어리가 지면고도에 섞여 30m bilinear 가 도로/건물을 뚫는다(고질 아티팩트). 형태학적 **opening**(반경 2셀≈60m: 폭 120m 미만 양돌출만 제거 → 건물 O·산 X) + **총하강 상한 15m**(뾰족한 실제 봉우리 보호) + 가우시안 + **×3 업샘플**(≈10m 격자) 후 `Regions/terrain/<region>.{bin,json}` 을 **제자리 교체**(원본은 `Regions/terrain_raw/` 백업, 재실행 시 백업본을 입력으로 삼아 누적 방지). 소비자(런타임 `TerrainHeight`·임포터)는 json 격자를 읽으므로 무수정. Unity 쪽 반영은 `TerrainImporter.ApplyDemDelta`(**신−구 델타**를 정점·도로망 `y[]`·`RegionRoadGraph` nodeY/ptY·POI/신호 트랜스폼에 가산 — `RebaseGroundLayers` 가 비멱등이라 절대 재드레이프 불가) + `CarveRoadsIntoTerrain`(정사 정점을 도로면−0.45m 로, **상한 6m**=터널 보호) + 배치 메뉴 `Tools/MCI/Terrain/Smoothed DEM: {서울 25구 / 특별시·광역시 / 원본백업 있는 전 지역} 배치`. **2026-07-28 전국 255/255 스무딩+재베이크 완료**(`terrain_smooth.py --all --skip-done` → 배치 StartAll: delta 230=서울 밖 전량, carve 222). ⚠️업샘플은 셀 수를 scale² 배로 불려 대면적 군(옹진군 26M셀)이 런타임 수백 MB가 된다 — **`--max-cells`(기본 8M)로 scale 3→2→1 자동 강등**(132개 지역이 상한에 걸려 x1~x2, 스무딩 자체는 전부 적용). carve 미적용 8곳(홍천·안동·옹진·제주시·군산·신안·완도·여수)은 **기존 가드 `카빙 래스터 과대 — 스킵`**(정합엔 무해). 실측 지형>도로 비율: 강남 0.3%(2,091표본) / 부산중구 0.9%·대구중구 0.1%·광주서구 0.1%·대전서구 0.4%. `tools/bldg_height_index.py` — vWorld GIS건물통합(`tools/nationwide_v2/buildings/<region>/buildings.geojson`)에서 `lon lat height floors 이름` 인덱스를 뽑아 `tools/nationwide/bldgindex/<region>.txt` 로(런타임 `BuildingHeightIndex.cs` 가 100m 격자해시로 조회 → HUD 건물 라벨이 **실측 높이·층수·이름**). 원천 수집은 `tools/vworld_buildings_batch.py`(`--metro/--sido/--all`, 재개가능, 지역당 ~1분) — **2026-07-28 전국 255/255 완료**(geojson 31GB·gitignore, 인덱스 492MB/1,766만 행). 검증: 전국 최고가 롯데월드타워·해운대 엘시티 411.6m·서울국제금융센터 284m 로 실존 랜드마크와 일치, 400m 초과 4건뿐. ⚠️노이즈 방어 2단: ①height 가 층수와 모순이면(`floors×6m` 초과) `floors×3.5m` 로 대체 ②**세장비 게이트**(`height > 15×√발자국면적` 이면 그 링 제외) — height 결측 + `grnd_flr` 오기입(337층·128층)이 `×3.5` 로 부풀려져 **600m 팬텀 타워**가 생기던 것을 잡는다(탈락하는 건 3~26㎡ MultiPolygon 슬리버뿐, 실제 건물 본체는 통과).
- **★정사 해상도·해저·녹지 — 정합성의 실제 병목 (2026-07-30, 전국 255/255 반영)**: 건물이 산비탈에서 공중에 뜬 원인은 **건물이 아니라 정사 메시**였다(실측: 건물 밑면은 하이트맵에 ±0.3m 로 정확히 앉음 22,316점 / 정사 메시 대비는 평균 +1.889m·최대 207m). `TerrainImporter.Drape` 가 `Clamp(side/22f, 8, 160)` 이라 **상한 160 이 걸려** 8.1km 타일이 주석(22m)과 달리 **51m 간격**으로 구워졌고, 51m 평면 삼각형이 10m DEM 위를 덮으니 능선에서 현(chord)이 처져 **정사가 하이트맵보다 평균 −5.5m·\|Δ\|>1m 가 30%**. ⚠️`Clamp` 상한은 조용히 걸린다 — **의도 상수를 믿지 말고 파생값(정점수·실간격)을 실측**할 것.
  - **`TerrainImporter.ResubdivideOrtho`**(신설, 마커 `__ortho_res_v2`) — 기존 `ApplyToRegion` 은 `vertexCount != 4` 를 멱등 스킵해 구 저해상 타일을 못 고친다. 타일 XZ 경계로 격자를 다시 깔고 현재 DEM 재샘플(`ORTHO_TARGET_SPACING_M` 24m·`ORTHO_MAX_N` 400). 결과: 간격 51→24m, 메시표면−DEM \|Δ\|>1m 30.1%→**0.23%**, 건물−정사 평균 +1.889→**−0.031m**. 옛 작은 DEM 격자 밖 클램프로 **평면으로 굳은 타일**(거제 5장, 최대 −334m)도 같이 사라진다. 배치 kind = **`ortho-res`**(재세분+카빙 재적용+녹지 숨김 한 패스, 255개 ~15분, 씬 용량 +약 25%).
  - **★마커 의존 사슬(재발 1순위 함정)**: 어떤 패스가 서브트리를 **파괴하고 다시 지으면**, 그 서브트리에 결과를 넣었던 **후속 패스의 마커를 반드시 지워야** 한다. 안 지우면 그 패스가 `skipDone` 으로 건너뛰고 결과만 사라진다. 실제 사고 2건: ①`ImportAreasOne` 이 `_Areas` 재생성 → 물 머티리얼 소실, 그런데 `__water_applied` 는 vroot 에 있어 물 패스 skip → 전국 강·호수가 플랫 블루로 회귀 ②`ResubdivideOrtho` 가 정사 재생성 → 해수면 클립·도로 카빙 무효, 마커는 남아 skip. 현재 `ImportAreasOne` 은 `__water_applied`/`__water_v3` 를, `ResubdivideOrtho` 는 `__road_carve_v1`/`__sea_clipped` 를 각각 해제한다.
  - **해저 파기 `DeepenSeaOrtho`** (`ReclipSeaOne` 안에서 호출) — **GLO-30 은 수심이 없어 바다 밑 DEM 이 정확히 0.00m**(거제 3,366표본)이고 해수면은 y=0.2 라, 물 셰이더가 **온 바다를 수심 0.24m 여울로 보고 해안 포말로 칠했다**(그게 "회색 골판지 바다"의 진범). 정사 격자에서 바다 셀(현재 y<0.30)을 찾아 **육지로부터의 격자거리 램프**(`SEABED_RAMP_M` 260m)로 `SEABED_MAX_M` 18m 까지 낮춘다 → 외해는 Beer-Lambert 심해청, 포말은 진짜 해안선에만. 근거리 픽셀 0.365→**0.004**. ⚠️**절대 목표고도 대입**이라 멱등(초기 구현은 `-=` 라 재실행마다 파여 거제가 −36.5m 까지 갔다). 배치 kind = **`sea-reclip-force`**(마커 무시 강제판). 해안 판정 문턱은 `TerrainMinH < 0.5f` — 구 `-0.5f` 는 DEM 스무딩이 해안 최저점을 끌어올려 강릉(−0.038m)을 탈락시켰다.
  - **공원/녹지는 렌더러를 끈다**(`HidePaintedVegetation`) — 정사영상에 이미 실제 숲·잔디가 찍혀 있어 그 위에 불투명 단색을 덧칠하면 정보는 안 늘고 사실감만 깎인다. 지오메트리와 `SceneObjectMeta`(ML 시멘틱)는 보존. 물은 예외(반사·굴절은 정사만으로 안 나온다).
  - **물 셰이더 추가 수정**(`Shaders/SimpleWater.shader`): ①`fogFactor` 를 **프래그먼트에서** 계산 — `__sea` 는 47×46km 삼각형 2장이라 정점 보간 시 코앞 픽셀이 수십 km 밖 값을 물려받아 근거리에서도 회색으로 씻겼다(77m 에서 fog 18%) ②`WaveField` 기울기 **정규화**(`grad /= Σ amp·frq·lod`) — 옥타브마다 기울기 기여가 커져 합 4.5(경사 50°+)가 되던 것 ③옥타브 LOD 억제 3.2·주파수비 비조화(1.87)·저주파 도메인 워프로 모아레 제거 ④바닥이 안 보이면 `_DeepFallback`(40m) 수심 가정 ⑤정점 `uv1` = 하천 흐름벡터(임포터가 하천마다 굽는다).
  - **`Scripts/Sim/SeaLife.cs`**(신설, `UamCityLife.Create` 에서 기동) — 어선 14·부표 20 풀, 시야 1.9km 재배치, **파면 상하동+기울기 추종**(셰이더 파동장 CPU 저주파 근사 — 상수를 Water_Sea 프리셋과 맞춰야 따로 놀지 않는다), `TrailRenderer` 항적. 메시는 Blender 5.1 헤드리스 제작 → `Resources/SeaLife/{KR_FishingBoat,KR_NavBuoy}.prefab`(12.0m·4.1m), **프리팹 없으면 절차생성 폴백**이라 drop-in 교체 가능. ⚠️`URP/Unlit` 은 정점컬러를 안 먹어 TrailRenderer 그라디언트가 무시된다 → `URP/Particles/Unlit`.
  - ⚠️**렌더링 판정은 스크린샷이 아니라 픽셀 readback**. 게임뷰 감마/톤매핑이 선형 0.02 를 중간 회색으로 보여 이번 세션에 셰이더를 2회 헛수정했다. 선형 `RenderTexture`+`ReadPixels` + 셰이더 `_DebugMode`(수심/프레넬/body/sky/노멀) 항별 비교. 최종색이 항들의 합으로 설명 안 되면 아직 못 본 항(포말·안개)이 지배한다. ⚠️런타임 생성 카메라엔 `UniversalAdditionalCameraData` 가 없어 opaque/depth 텍스처가 안 만들어진다(콘텐츠 버그로 보이는 검증장비 버그).
## Working with Unity via MCP (hard-won gotchas)

Full cheat-sheet is in the auto-memory `reference_unity_mcp_osm_techniques.md`. The essentials:

- **Mass scene import**: never run a long single `execute_code` over many scenes — it hits the ~30s MCP receive timeout and the client **re-sends → duplicate-execution storm**. Use `VworldRegionImporter.StartBackgroundImport(kind)` (autonomous `EditorApplication.update` driver, idempotent via `skipDone`) and monitor progress by `grep`-ing scenes for `_Areas`/`_POI`/`_Features` on disk. Alternatively small time-boxed batches (≤12s budget) with `Resumable(start,max,budget,skipDone,onlyName)`.
- **Play mode frames freeze** under MCP when the Game view is unfocused + `runInBackground=false`: set `Application.runInBackground = true` after entering Play or coroutines/`Update` never tick (`Time.frameCount` stays 1).
- **New `.cs` files** may not be picked up by `refresh scope=scripts` (→ "type not found"); use `refresh scope=all force` before they compile.
- **`execute_code` runs as a method body**: no top-level `using` (fully-qualify `System.IO.File`/`Path`), and `Object` is ambiguous → use `UnityEngine.Object`.
- **Never write `mesh.vertices` on a shared built-in primitive** (`CreatePrimitive` Cube/Cylinder…): it corrupts the *global* shared mesh session-wide (not saved — scenes reference the built-in by GUID, so a Unity restart restores it). To reposition primitive-based objects (POI markers, traffic signals), move the **Transform**, not vertices.
- **Mass-scene batch pattern**: `EditorApplication.update` driver (1 scene/tick), idempotent via a **marker child GameObject** (e.g. `__terrain_rebased`) skipped on re-run; monitor with a `run_in_background` bash watcher polling scene `.unity` mtimes (the editor is too busy for reliable MCP polling).
- **Screenshot a scene**: temp `Camera`→`RenderTexture`→`ReadPixels`→`EncodeToPNG` to a path **outside `Assets/`**, then Read it. Frame on the *buildings* bounds — Sudogwon-frame geometry sits at huge world coords (~128000), so all-renderer bounds mis-frame to empty sky. Region scenes have **no Camera** → Play shows "No cameras rendering"; inspect via Scene view or Play the entry `SampleScene`.
- Entering Play after a script change briefly drops the MCP bridge ("No Unity Editor instances"); re-read `mcpforunity://instances` and retry.
- **Play/ML-Agents 학습 중엔 재컴파일 금지**: `.cs` 편집·`refresh(compile)`·play/stop = 도메인 리로드 = 실행 중 Play/학습 중단(+진행 중 `EditorApplication.update` 배치도 사망). 학습 중 안전작업은 `execute_code`(인메모리)·prefab/머티리얼/asset 조작뿐 — 스크립트 수정은 학습 종료 후로. (`editor_state`의 `play_mode.is_playing`으로 먼저 확인)
- **The Unity MCP bridge is often NOT connected to the Claude session** (only `claude.ai Figma/Notion/...` may be present) — confirm with `ListMcpResourcesTool` / `claude mcp list` **before** planning MCP-driven Unity work; `/unity-mcp-skill` is docs, not a connection. With no bridge you can still verify from disk: `.unity` scenes are YAML, so `grep -l '<marker>' Assets/Scenes/Regions/*.unity` confirms a pass applied across the 255 scenes, and a `StartBackgroundImport` pass saving scenes at all is **proof every edited runtime+editor script compiled clean** ("compile-proof" — Unity won't run editor code with any compile error).
- Flat/ribbon meshes need **CCW winding (viewed from above)** so `RecalculateNormals` faces +Y; CW → normals point down → backface-culled (invisible from above). Test visibility with a **URP** Unlit material (built-in `Unlit/Color` renders invisible under URP).
- **`VworldRegionImporter.StartBackgroundImport(kind)`** (realism/import 배치: `roads2/area/poi/feat/water/grass/despike/footway/furniture/infra/bldg-collider/sea-reclip/`**`sea-reclip-force`**`/redrape/`**`ortho-res`**)는 `EditorApplication.update` 1씬/틱 구동, 마커 자식(`__grass_applied`/`__ortho_despiked`/`__sidewalk_v1`/`__furnverN`/`__ortho_res_v2`…)으로 멱등. **Play 는 배치를 죽이지 않고 일시정지**시킨다(`isPlayingOrWillChangePlaymode` 가드) — 죽이는 건 **`.cs` 편집/재컴파일**(도메인 리로드로 static 델리게이트 소멸)이다. 완료감지는 `BackgroundImportStatus()` 폴링 또는 **`Editor.log` 완료줄 개수 증가**(⚠️완료줄은 세션 내내 누적되므로 존재 여부로 판정하면 이전 배치 줄에 오판한다 — 실제로 area v4 를 "완료"로 잘못 보고했다). ⚠️진행률을 `Assets/Scenes/Regions/*.unity` 전수 `grep` 으로 재지 말 것 — 총 77GB 라 폴링마다 수십 GB 를 읽어 배치와 디스크를 다툰다(60초 설정이 실제 3.3분 주기가 됐다). **강제 재실행=마커 버전 bump**(재컴파일 → 배치 사이에). 이 internal static들은 `execute_code` 리플렉션으로 호출.

##### 2026-09-09 (3차) — 노면선 형태 규칙 오류(도색 길이 36.9%) · 오진 2건 자백

사용자 지시로 "재베이크 전에 다른 결함 없는지 꼼꼼히" 검토한 결과. **내가 앞 절에서 보고한
"경계 중복 잔여 결함"은 존재하지 않았고**, 대신 훨씬 큰 것이 나왔다.

- ★★**B2 노면선의 실선/점선 규칙이 틀렸다 — 도색 길이의 36.9%**(표본 8지역 실측).
  `v2_hd_bucket.lane_style` 이 `type` **둘째자리**로 형태를 정했는데, 그 규칙이 두 종류를
  **정반대로** 뒤집고 있었다: `kind 503`(차선)이 전부 **실선**(강남 935km·송파 627km·
  미추홀 351km), `kind 515`(길가장자리)가 일부 지역서 **점선**(미추홀 120km).
  → 사용자 최초 신고 **"흰색 차선의 점선 및 실선이 제대로 배치되어 있지 않아"의 정체**다.
  종류를 **우리 데이터로 기하 검증**해 확정했다(추정 금지):
    · 503 = 보도까지 **10.3m**(차도 내부) · 도로와 평행(|cos|1.00) · 중앙 41m → **차선=점선**
    · 515 = 보도까지 **0.9m**(p90 4.9m) → **길가장자리=실선**
    · 530 = 길이 8.5m · **|cos| 0.02 = 도로와 수직** → **정지선=실선**
    · 504 = `type` 첫자리 3 의 **96%** → 버스전용=청색(구 매핑이 옳았다. 앞서 "청색은 버스
      전용 실데이터"라고 한 근거가 순환논리였는데 이번에 비순환으로 확인)
  수정 후 미추홀 흰선 점선 비중 **28.3% → 73.5%**(6차로면 차선5+가장자리2=71% 이므로 맞다).
- ⚠️**`type` 첫자리를 형태로 일반화하려다 새 결함을 만들 뻔했다.** ty[0]=1/2/3 이 실선/점선/
  버스로 두 지역에서 완벽히 일치해 그렇게 바꿨는데, 기하 검증에서 **530 정지선(211)이
  점선**이 되는 걸 잡았다(도로와 수직인 8.5m 선). → **기하로 증명한 503·515 만** 손대고
  505·506·525 는 구 규칙 유지. 205개 kind 조합의 의미를 다 모르는 상태의 일반화는 금물.
- ★**오진 1 — "경계 도색 이중 렌더"는 없었다.** 2장 감사의 ①89·⑦181.7% 를 중복이라 보고
  3시간 재베이크를 제안했는데, 원인은 **지표**였다: ⑦ 이 `regions[0]` 한 구의 기대치를 열린
  씬 **전부**의 렌더와 비교했다. 실증 — 강남 단독 99.1% · 서초 단독 98.7% · **기대 합
  11,518m vs 2장 렌더 11,396m = 98.9%**. ①의 89 는 소유권이 링크 **중점**으로 갈릴 때
  이웃 링크의 주인이 달라 생기는 **접합 이음매**(리본 끝 정점)다. 원자료도 확인했다 —
  `hd_links`·`hd_lanes`·`hd_marks` 가 두 구 버킷에서 **기하·ID 완전 동일**(296/580/250개)이고
  `OwnerAt` 은 (lat,lon) 순수 함수라 중복이 원리적으로 불가능하다. → ⑦ 은 전 지역 합으로,
  ① 은 **HD노면/골목 분리**로 고쳤다.
- ★**오진 2 — 도색 z 꼬리는 도색이 뜬 게 아니다.** ③ p95 +0.54~0.66m·최대 +1.29m 를 "도색을
  자기 z 로 클램프해서 뜬 것"이라 보고 임포터를 고쳤다가 **되돌렸다**: 꼬리 무변(p95
  0.541→0.558)에 본체 악화(묻힘 2.6%→**6.4%**). 전제가 틀렸다 — 원자료에서 **도색z−최근접
  링크z 는 중앙 0.030m · p95 0.120m**(미추홀 77,886 표본)로 이미 일치하고, 0.5m 초과 2.0%는
  거의 전부 **2m 초과 = 고가/지하**라 `PAINT_SNAP_MAX` 가 일부러 남긴 것이다. 꼬리는 **고가
  도색을 그 아래 지상 노면과 비교한 지표**의 산물 → ③ 을 ⑥/⑥b 처럼 **지상/고가로 분리**.
  ⚠️앞 절에서 "도색−아스팔트 중앙 +0.044 목표 달성"이라고 보고한 것은 **중앙값만 본 것**이고
  꼬리는 그때도 그대로였다. **분포는 중앙값 하나로 보고하지 말 것.**
- **골목(OSM) 소유권 필터 부재는 잠재 결함으로 남긴다.** `ReadOsmRoads` 루프에 `owns` 가
  없다. 다만 실측 겹침이 경계쌍 5곳 전부 **0.0~0.1%**(86~274m)로, OSM 페치가 이미 구별로
  잘려 있다. 필터를 붙이면 중점이 이웃 구에 떨어지는 경계 way 가 **양쪽에서 사라질** 위험이
  있어(각 구가 자기 클립본만 가짐) 이득 대비 위험이 크다 → 손대지 않음.
- ⑤ 단차 꼬리(1.9~11.9% >3cm·최대 0.82m)는 **램프 접속부**다 — `MAX_ADJ 0.40` 이 실제 높이차를
  일부러 보존한다. 최악 지점 좌표 로깅을 추가했다(좌표 없이 두 번 오진한 그 이유).
- ⑥b 골목 정사돌출(9.7~13.4%·최대 +1.41m)은 **경사지에서 4.5m 반경 안 정사가 원래 더 높은
  것**이다(골목은 카빙 대상이 아니다). 지표 반경의 산물이지 결함이 아니다.
