# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Two loosely-coupled codebases that share the same MCI (mass-casualty incident) scenario data:

1. **RL / simulation research (Python)** — learns AMB(ambulance)+UAV triage & transport decisions. Lives in `src/`, `tools/`, `scenarios/`, `results/`. This is what `origin` (`bbcc1017/MCI_UAV`) versions. See `README.md` for the full command walkthrough (scenario gen → heuristic sim → RL train → eval).
2. **Unity digital-twin visualization (C#)** — renders the disaster sim over a nationwide 3D Korea (255 시군구 scenes) with buildings, OSM roads, traffic, pedestrians. Lives in `external/ml-agents/UAV_test/`.

## Git / submodule structure (read this first)

- `external/ml-agents` is a **git submodule pointing at the upstream Unity-Technologies/ml-agents** repo (`ignore = dirty`). The entire Unity project `UAV_test/` sits **untracked inside that submodule's working tree**.
- Consequence: **all Unity C# / Assets changes are local-to-`C:` only — they are NOT tracked by `origin` and cannot be pushed** with the current layout. This is intentional ("Unity = local-C only"). Do not try to commit Unity changes to the parent repo; don't be alarmed that `git status` ignores them.
- Only `src/`, `tools/*.py`, docs config get committed to `origin`. Commit messages **and code comments** are written **in Korean** (keep the `Co-Authored-By` trailer in English); match the surrounding comment density.
- `.gitignore` excludes large/generated data: `scenarios/exp_*`, `results/`, `docs/`, `.codex/`, and `tools/nationwide/{roads2,feat,area,poi,blocks,...}` bulk data.

## Environment & commands

- Python 3.10 in conda env **`UAV`** (torch 2.8.0+cu128 for RTX 50; `requests`, `geopandas`, `shapely`, `pyproj` installed; SB3/MaskablePPO). `MCI`/`qgis_batch` envs also exist.
- **Running tools via `conda run` crashes on Korean stdout (cp949).** Always call the env python directly and force UTF-8:
  ```bash
  PYTHONIOENCODING=utf-8 /c/Users/User/anaconda3/envs/UAV/python.exe tools/<script>.py ...
  ```
- RL/sim entry points and exact flags are in `README.md` (`make_csv_yaml_dynamic.py` → `sim_src/main.py` → `rl_src/run_all_parallel.py` → `evaluate.py`/`hybrid_eval.py`/`cross_location_eval.py`). There is no test suite; "running a test" means running a scenario + sim/eval.
- **Generated scenario/trace data lives on `Y:/scenarios/...`**, not in the repo (the tracked `scenarios/` holds only gitignored seed inputs like the hospital xlsx). Unity's `TracePlayer` defaults to reading `scene.json`/`trace_flat.json` from `Y:`.

## RL/sim architecture (big picture)

Data flows: **scenario YAML → gym env → wrapper → trainer/evaluator.**

- `src/sce_src/make_csv_yaml_dynamic.py` builds a per-incident scenario (`scenarios/exp_*/(lat,lon)/config_*.yaml` + `scene.json`) from the hospital pool (`엑셀 결합 데이터.xlsx`) + AMB bases (`안전센터와 소방서.csv`). Distances come from **OSRM (default, `is_use_time=False`)** or **Kakao Mobility (`--is_use_time True --kakao_api_key`)**.
- `src/sim_src/` is the event-driven simulator core (treat as stable). `MCIEnvironment_gymnasium.py` exposes the gym env; AMB+UAV are both active when `amb_num>0`, giving `action_space = MultiDiscrete([3, H+1, 2])` (class, destination hospital, mode).
- `src/rl_src/env_wrapper.py` (`FlattenAndDiscreteWrapper`) is the keystone: dict→flat obs, MultiDiscrete→Discrete, **action masking**, and `encode_action`/`decode_action`. It **auto-adjusts dimensions** based on amb_num/uav_num. The hybrid evaluator (`hybrid_eval.py`, "2안") uses this encode/decode to let RL pick the UAV action while a heuristic rule overrides the AMB action.
- Heuristics (`sim_src/RuleManager.py`) enumerate 32 rule combos (START/ReSTART × RedOnly/YellowNearest × red/yellow modes); RL is compared against these.

### Multi-region / nationwide RL (Plan 1 + plan1nat)

Beyond single-coordinate training, there are two multi-region pipelines driven by **manifest JSONs in `scenarios/manifests/`** (`{region: config_path}`, with **absolute paths** — note training is also run on a Linux box, so paths there are `/home/...`):

- **Trainer branch on file extension**: `train_{ppo,dqn,reinforce}.py` check `config_path.endswith(".json")` — a `.json` manifest selects `rl_src/multi_region_env.py` (`MultiRegionEnv`, samples a region per `reset()`); a `.yaml` is the single-scenario path. All regions in a manifest **must share `fixed_hos_num`** so obs/action dims stay constant — regenerate scenarios with `gen_regions.py --fixed_hos_num` if not.
- **Plan 1 (per-region policies)**: `sce_src/gen_regions.py` builds the 17 광역시도 scenarios (coords from `cross_location_eval.LOCATIONS`, single source of truth) → `plan1_manifest.json`. `rl_src/run_grid_parallel.py` trains 17 regions × 3 algos as subprocesses (**CPU-forced via `CUDA_VISIBLE_DEVICES=""`** to avoid GPU contention). Diagonal eval: `run_grid_eval.py` fans out `eval_region.py` workers (each region's model vs its own heuristic best).
- **plan1nat (single national policy)**: trained on `national_train.json` → `plan1nat_manifest.json` via `MultiRegionEnv`. **Generalization eval** uses hold-out points: `sample_region_points.py` rejection-samples random WGS84 points inside `scenarios/ctprvn.shp` (통계청 시도 경계, EPSG:5179→4326) → `gen_eval_points.py` builds scenarios at those points (retries/re-samples on Kakao route failure or hospital-count mismatch).
- **sim_src debug-print spam**: the sim emits a `print` per event; trainers/workers therefore redirect **stdout → `/dev/null`** (monitor via TensorBoard) and capture only **stderr → `.err`** files. Don't "fix" this by editing `sim_src` — it's stable by decision.
- The many other `rl_src/*` scripts are research variants on the same wrapper (`enriched_env_wrapper`/`reward_redesign_wrapper`/`advantage_wrapper` obs-reward ablations, `train_ppo_bc.py`+`bc_dataset.py`+`distill_policy.py` for BC/distillation, `eval_*`/`aggregate_*`/`plot_*` for analysis). Read the module docstring — each states its reuse deps and purpose.

## Unity digital-twin architecture (big picture)

Project root: `external/ml-agents/UAV_test/`. Pipeline: **GIS/OSM fetch (tools) → Editor importers bake meshes into Region scenes → runtime additively loads needed scenes and plays the sim trace.**

- **Coordinate system**: `Assets/Scripts/Geo/RegionRegistry.cs` holds a per-시군구 EPSG:5186 "frame". `TryWorld(lat,lon)`/`TryWorldIn(frame,...)` convert WGS84→Unity world (meters). All importers and runtime spawning go through this. `tools/nationwide/sgg.json` is the source of the 255 districts (name/kor/frame/bbox/rings).
- **Scene structure**: `Assets/Scenes/SampleScene.unity` is the 3MB entry scene (only one in Build Settings) holding `MapVersionSelector`. `Assets/Scenes/Regions/<name>.unity` are 255 시군구 scenes (each ~50-80MB) with meshes embedded under a `Vworld_<name>` root: `_Ortho`, `_Buildings`, `_Roads`, `_Features`(traffic signals), `_Areas`(park/water), `_POI`(hospital/school/fire). There is no global terrain — districts are loaded **additively** on demand. (Legacy manual-build scenes were retired/deleted 2026-06; don't expect them.)
- **Import pipeline** (`Assets/Editor/VworldRegionImporter.cs`): reads `tools/nationwide/{roads2,feat,area,poi}/<name>.txt` and the vworld building/ortho data, then bakes ribbon/polygon/marker meshes into each Region scene under the `Vworld_<name>_*` roots. Menus under `Tools/MCI/...`. For mass (255-scene) imports use `StartBackgroundImport(kind)` (runs one scene per `EditorApplication.update` tick) — see Unity MCP notes below.
- **Runtime playback** (`Assets/Scripts/Sim/`): `MapVersionSelector` (IMGUI shown on Play) lets the user pick a scenario; `ScenarioSceneResolver` decides which Region scenes the trace actually traverses; selected scenes load additively; then `TracePlayer` animates `scene.json`+`trace_flat.json` (AMB/UAV dispatch, hospital `HospitalFacility`, camera modes) and spawns `TrafficManager` (NPC cars on `RegionRoadNetwork`, yield to emergency, stop at red `TrafficSignal`) and `PedestrianManager`. `FreePilotController` is the alternative free-drive/fly mode.
- **ML-Agents semantic layer**: `SceneObjectMeta` (category/width/lanes/oneway/speed/height) is attached to roads/areas/POI/signals so future ML-Agents observations can read each object's characteristics. `RegionRoadNetwork` stores drivable centerlines + oneway/width.

## tools/ data pipeline

- OSM via Overpass: `osm_roads2.py` (lanes/oneway/class → detailed roads), `osm_features.py` (signals/crossings/bus stops), `osm_areas.py` (park/green/water polygons), `osm_poi.py` (hospital/school/fire/police/fuel). All rotate 3 Overpass mirrors to dodge 429, attribute features to a district by **center-point-in-polygon**, are resumable (skip existing output, `--force`), and write compact text to `tools/nationwide/<kind>/<name>.txt`.
- `vworld_fetch.py` pulls buildings/orthophoto tiles via the vWorld API. `nationwide_build.py`/`build_region_index.py` drive the citywide build + `region_index.json`. `scene_export.py`/`trace_export.py`/`run_sim_trace.py` bridge the Python sim output into Unity-loadable JSON.

## Working with Unity via MCP (hard-won gotchas)

Full cheat-sheet is in the auto-memory `reference_unity_mcp_osm_techniques.md`. The essentials:

- **Mass scene import**: never run a long single `execute_code` over many scenes — it hits the ~30s MCP receive timeout and the client **re-sends → duplicate-execution storm**. Use `VworldRegionImporter.StartBackgroundImport(kind)` (autonomous `EditorApplication.update` driver, idempotent via `skipDone`) and monitor progress by `grep`-ing scenes for `_Areas`/`_POI`/`_Features` on disk. Alternatively small time-boxed batches (≤12s budget) with `Resumable(start,max,budget,skipDone,onlyName)`.
- **Play mode frames freeze** under MCP when the Game view is unfocused + `runInBackground=false`: set `Application.runInBackground = true` after entering Play or coroutines/`Update` never tick (`Time.frameCount` stays 1).
- **New `.cs` files** may not be picked up by `refresh scope=scripts` (→ "type not found"); use `refresh scope=all force` before they compile.
- Entering Play after a script change briefly drops the MCP bridge ("No Unity Editor instances"); re-read `mcpforunity://instances` and retry.
- Flat/ribbon meshes need **CCW winding (viewed from above)** so `RecalculateNormals` faces +Y; CW → normals point down → backface-culled (invisible from above). Test visibility with a **URP** Unlit material (built-in `Unlit/Color` renders invisible under URP).

## Long-running background work

Fetches/imports are launched detached (`nohup … &` in a `run_in_background` bash). Detached python isn't harness-tracked, so completion is detected with a **watcher** bash (also `run_in_background`) that polls a file count / completion marker / stall and exits — the harness then notifies on the watcher's exit.
