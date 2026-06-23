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
- **`origin` is pushed from multiple machines** (a Linux training box; **Codex** cloud branches `codex/*`) — a local push is often rejected ("fetch first"): `git fetch` → `git rebase origin/main` (changes are usually in non-overlapping files) → push. **Never force-push** (clobbers the other side's RL/sim commits). `AGENTS.md` is Codex's `CLAUDE.md` mirror.
- `.gitignore` excludes large/generated data: `scenarios/exp_*`, `results/`, `docs/`, `.codex/`, `osrm_data/`, `overpass_db/`, and `tools/nationwide/{roads2,feat,area,poi,blocks,infra,terrain,ortho_fill,...}` bulk data (the last holds **multi-GB** DEM/ortho — **stage tool scripts explicitly; never `git add -A`**).

## Environment & commands

- Python 3.10 in conda env **`UAV`** (torch 2.8.0+cu128 for RTX 50; `requests`, `geopandas`, `shapely`, `pyproj` installed; SB3/MaskablePPO). `MCI`/`qgis_batch` envs also exist.
- **Running tools via `conda run` crashes on Korean stdout (cp949).** Always call the env python directly and force UTF-8:
  ```bash
  PYTHONIOENCODING=utf-8 /c/Users/User/anaconda3/envs/UAV/python.exe tools/<script>.py ...
  ```
- **GDAL/rasterio are only in the `qgis_batch` env, not `UAV`** (UAV has numpy/PIL/requests but no raster libs). Use `qgis_batch` python for GeoTIFF/DEM — e.g. Copernicus GLO-30 via GDAL `/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/...` (windowed read, no key, no full download).
- **Bash loops over Windows-python `print` output need `| tr -d '\r'`** — CRLF leaves a trailing `\r` on each token → silent district-name mismatch (this failed all 255 fetches twice). Files shared between Windows-python and Git-Bash must use **project-relative paths** (Windows `/tmp` ≠ Git-Bash `/tmp`).
- RL/sim entry points and exact flags are in `README.md` (`make_csv_yaml_dynamic.py` → `sim_src/main.py` → `rl_src/run_all_parallel.py` → `evaluate.py`/`hybrid_eval.py`/`cross_location_eval.py`). There is no test suite; "running a test" means running a scenario + sim/eval.
- **Generated scenario/trace data lives on `Y:/scenarios/...`**, not in the repo (the tracked `scenarios/` holds only gitignored seed inputs like the hospital xlsx). Unity's `TracePlayer` defaults to reading `scene.json`/`trace_flat.json` from `Y:`.

## RL/sim architecture (big picture)

Data flows: **scenario YAML → gym env → wrapper → trainer/evaluator.**

- `src/sce_src/make_csv_yaml_dynamic.py` builds a per-incident scenario (`scenarios/exp_*/(lat,lon)/config_*.yaml` + `scene.json`) from the hospital pool (`엑셀 결합 데이터.xlsx`) + AMB bases (`안전센터와 소방서.csv`). Distances come from **OSRM (default, `is_use_time=False`)** or **Kakao Mobility (`--is_use_time True --kakao_api_key`)**.
- `src/sim_src/` is the event-driven simulator core (treat as stable). `MCIEnvironment_gymnasium.py` exposes the gym env; AMB+UAV are both active when `amb_num>0`, giving `action_space = MultiDiscrete([3, H+1, 2])` (class, destination hospital, mode).
- `src/rl_src/env_wrapper.py` (`FlattenAndDiscreteWrapper`) is the keystone: dict→flat obs, MultiDiscrete→Discrete, **action masking**, and `encode_action`/`decode_action`. It **auto-adjusts dimensions** based on amb_num/uav_num. The hybrid evaluator (`hybrid_eval.py`, "2안") uses this encode/decode to let RL pick the UAV action while a heuristic rule overrides the AMB action.
- Heuristics (`sim_src/RuleManager.py`) enumerate **64** rule combos (START/ReSTART × RedOnly/YellowNearest × **4 red modes × 4 yellow modes** = 2*2*4*4); RL is compared against these. (`config.yaml` `rule_info` lists the 4 modes per class; the `results_*_stat.txt` block is 64 rules × 5 metric groups = 320 rows.)

### Observation / action / reward encoding

- **Obs** (dict, flattened by the wrapper into one float32 vector): `p_states (incident_size,5)` = [class, rescued, move_start, moved, cared]; `h_states (H,3)` = [idle, queue, occupied]; `p_sent (H,)`; `amb_states`/`uav_states (n,3)` = [dest, time_remaining, severity]; `p_at_site (4,)` = [R/Y/G/B waiting]; `n_amb_at_site`, `n_uav_at_site`, `time`.
- **Action** `[class, dest, mode]`: class 0=Red/1=Yellow/2=Green; dest 0=stay on scene, 1..H=hospital; mode 0=AMB/1=UAV. Mode is auto-pinned when only one vehicle type exists (`amb_num=0`→UAV, `uav_num=0`→AMB). `encode_action`/`decode_action` map this to/from the flat Discrete index.
- **Reward** = patient survival probability at hospital-admit time (Red/Yellow decay with time, Green=1, Black=0). `reward_redesign_wrapper.py` reshapes it: `raw` | `woG` (drop Green; also exposed as `info['r_woG']`) | `rywt` (Red/Yellow weighted) — picked by `--reward_mode` on `train_ppo*.py` or the `MCI_REWARD_MODE` env var (works for any algo).
- **Masking is a hard constraint** via `action_masks()`, not a penalty: Red→Tier3-only, UAV→helipad hospitals only. Effective wrapper chain (outer→inner): `Monitor → ActionMasker → [HeuristicAdvantageWrapper] → FlattenAndDiscreteWrapper (or HybridAMBHeurWrapper) → [RewardRedesignWrapper] → base env` — keep core files unmodified; variants wrap.
- **`hybrid_eval.py --mode_split`**: `strict` = heuristic decides the full `[c,d,m]`; `loose` = RL decides class+dest, heuristic decides only mode.

### Multi-region / nationwide RL (Plan 1 + plan1nat)

Beyond single-coordinate training, there are two multi-region pipelines driven by **manifest JSONs in `scenarios/manifests/`** (`{region: config_path}`, with **absolute paths** — note training is also run on a Linux box, so paths there are `/home/...`):

- **Trainer branch on file extension**: `train_{ppo,dqn,reinforce}.py` check `config_path.endswith(".json")` — a `.json` manifest selects `rl_src/multi_region_env.py` (`MultiRegionEnv`, samples a region per `reset()`); a `.yaml` is the single-scenario path. All regions in a manifest **must share `fixed_hos_num`** so obs/action dims stay constant — regenerate scenarios with `gen_regions.py --fixed_hos_num` if not.
- **Plan 1 (per-region policies)**: `sce_src/gen_regions.py` builds the 17 광역시도 scenarios (coords from `cross_location_eval.LOCATIONS`, single source of truth) → `plan1_manifest.json`. `rl_src/run_grid_parallel.py` trains 17 regions × 3 algos as subprocesses (**CPU-forced via `CUDA_VISIBLE_DEVICES=""`** to avoid GPU contention). Diagonal eval: `run_grid_eval.py` fans out `eval_region.py` workers (each region's model vs its own heuristic best).
- **plan1nat (single national policy)**: trained on `national_train.json` → `plan1nat_manifest.json` via `MultiRegionEnv`. **Generalization eval** uses hold-out points: `sample_region_points.py` rejection-samples random WGS84 points inside `scenarios/ctprvn.shp` (통계청 시도 경계, EPSG:5179→4326) → `gen_eval_points.py` builds scenarios at those points (retries/re-samples on Kakao route failure or hospital-count mismatch).
- **sim_src debug-print spam**: the sim emits a `print` per event; trainers/workers therefore redirect **stdout → `/dev/null`** (monitor via TensorBoard) and capture only **stderr → `.err`** files. Don't "fix" this by editing `sim_src` — it's stable by decision.
- The many other `rl_src/*` scripts are research variants on the same wrapper (`enriched_env_wrapper`/`reward_redesign_wrapper`/`advantage_wrapper` obs-reward ablations, `train_ppo_bc.py`+`bc_dataset.py`+`distill_policy.py` for BC/distillation, `eval_*`/`aggregate_*`/`plot_*` for analysis). Read the module docstring — each states its reuse deps and purpose.

### Key env vars (RL/sim & scenario gen)

- **`MCI_REDUCED_OBS=1`** aggregates the obs to summary stats (smaller obs dim). **Must match between train and eval** or the model won't load; batch scripts (`run_seed_repro.py`, grid launchers) force it on.
- **`MCI_TIER_MASK=0`** disables tier-based action masking (backward compat). **`MCI_REWARD_MODE`** = `raw`|`woG`|`rywt`. **`MCI_OBS_VARIANT`** selects an obs-ablation variant (needs reduced obs).
- **`MCI_CAP_GATE`** = `occ` (default) | `psent` — which hospital-capacity signal gates the **RL action mask** (`MCIEnvironment_gymnasium.action_masks_joint`/`action_masks`) and the `cap_remain` feature in `HospitalFeatureWrapper`. `occ` = real-time occupancy (`h_states[:,-1] < max_send`, identical to the heuristic + the sim's actual admission gate `n_occupied < max_capa`); `psent` = cumulative dispatched (`p_sent < max_send`, never released → site-centric limited info). Run as two separate experiments: `occ` = hospital real-time comms, `psent` = on-site-only knowledge. (The heuristic `RuleManager` always gates on `occ`.)
- **`MCI_ADV_MODE` / `MCI_ADV_SUBTRACT_AT` / `MCI_ADV_CSV` / `MCI_ADV_REGION`** configure `advantage_wrapper.py` (baseline-relative reward shaping from a precomputed CSV).
- **Routing**: `MCI_OSRM_URL` (OSRM backend, default public router), `KAKAO_API_KEY` (Kakao mode). **Scenario-gen knobs** (fallbacks when the CLI flag is omitted): `MCI_UTIL_BY_TIER`, `MCI_BUFFER_RATIO` (default 1.5), `MCI_MAX_SEND_COEFF`.

## Unity digital-twin architecture (big picture)

Project root: `external/ml-agents/UAV_test/`. Pipeline: **GIS/OSM fetch (tools) → Editor importers bake meshes into Region scenes → runtime additively loads needed scenes and plays the sim trace.**

- **Coordinate system**: `Assets/Scripts/Geo/RegionRegistry.cs` holds a per-시군구 EPSG:5186 "frame". `TryWorld(lat,lon)`/`TryWorldIn(frame,...)` convert WGS84→Unity world (meters). All importers and runtime spawning go through this. `tools/nationwide/sgg.json` is the source of the 255 districts (name/kor/frame/bbox/rings). **`KoreaGeo.LatLonToWorld`/`TM5186ToWorld` are hardcoded to the legacy Sudogwon frame — don't use them per-region; use `TryWorldIn(frame,…)`.** `TryWorld` now adds terrain height (`TerrainHeight.Ground`) to y; `TryWorldIn` stays pure (importers depend on this). `KoreaGeo.TM5186ToLatLon` is the inverse projection.
- **Scene structure**: `Assets/Scenes/SampleScene.unity` is the 3MB entry scene (only one in Build Settings) holding `MapVersionSelector`. `Assets/Scenes/Regions/<name>.unity` are 255 시군구 scenes (each ~50-80MB) with meshes embedded under a `Vworld_<name>` root: `_Ortho`, `_Buildings`, `_Roads`, `_Features`(traffic signals), `_Areas`(park/water), `_POI`(hospital/school/fire). There is no global terrain — districts are loaded **additively** on demand. (Legacy manual-build scenes were retired/deleted 2026-06; don't expect them.)
- **Import pipeline** (`Assets/Editor/VworldRegionImporter.cs`): reads `tools/nationwide/{roads2,feat,area,poi}/<name>.txt` and the vworld building/ortho data, then bakes ribbon/polygon/marker meshes into each Region scene under the `Vworld_<name>_*` roots. Menus under `Tools/MCI/...`. For mass (255-scene) imports use `StartBackgroundImport(kind)` (runs one scene per `EditorApplication.update` tick) — see Unity MCP notes below. Realism passes (menus `Tools/MCI/Realism/...`): kinds `roads2` (re-drape roads to terrain), `bldg-collider` (MeshCollider + "Building" layer), `water` (SimpleWater material + sea plane), `sea-reclip` (clip sea to below-sea-level cells); each idempotent via a marker (`RegionRoadNetwork.terrainDraped`, `__bldg_collider`, `__water_applied`, `__sea_clipped`).
- **Runtime playback** (`Assets/Scripts/Sim/`): `MapVersionSelector` (IMGUI shown on Play) lets the user pick a scenario; `ScenarioSceneResolver` decides which Region scenes the trace actually traverses; selected scenes load additively; then `TracePlayer` animates `scene.json`+`trace_flat.json` (AMB/UAV dispatch, hospital `HospitalFacility`, camera modes) and spawns `TrafficManager` (NPC cars on `RegionRoadNetwork`, yield to emergency, stop at red `TrafficSignal`) and `PedestrianManager`. `FreePilotController` is the alternative free-drive/fly mode.
- **ML-Agents semantic layer**: `SceneObjectMeta` (category/width/lanes/oneway/speed/height) is attached to roads/areas/POI/signals so future ML-Agents observations can read each object's characteristics. `RegionRoadNetwork` stores drivable centerlines + oneway/width + per-point terrain-draped `y[]` (NPCs terrain-follow). Building chunks carry MeshColliders on the "Building" layer for camera-occlusion/NPC-stop raycasts; UAVs land on the real GIS roof via `TracePlayer.CityRoofAtCached`/`HospitalRoof`.

## tools/ data pipeline

- OSM via Overpass: `osm_roads2.py` (lanes/oneway/class → detailed roads), `osm_features.py` (signals/crossings/bus stops), `osm_areas.py` (park/green/water polygons), `osm_poi.py` (hospital/school/fire/police/fuel). All read `MCI_OVERPASS_URL`/`OVERPASS_URL` (local self-hosted Overpass, via `osm_overpass_endpoints.py`) else rotate 3 public mirrors to dodge 429, attribute features to a district by **center-point-in-polygon**, are resumable (skip existing output, `--force`), and write compact text to `tools/nationwide/<kind>/<name>.txt`.
- `vworld_fetch.py` pulls buildings/orthophoto tiles via the vWorld API. `nationwide_build.py`/`build_region_index.py` drive the citywide build + `region_index.json`. `scene_export.py`/`trace_export.py`/`run_sim_trace.py` bridge the Python sim output into Unity-loadable JSON. `scene_export.py` also emits per-route `states[]` (Kakao `traffic_state` 0–5, index-aligned to `pts`) for congestion-aware NPC density/speed.
- **Local self-hosted routing/OSM** (avoids Kakao cost + mirror 429; needs Docker Desktop/WSL2): `docker-compose.osrm.yml` (`MCI_OSRM_URL`; data via `tools/osrm_prepare_korea.ps1`→`osrm_start_local.ps1`) + `docker-compose.overpass.yml` (`MCI_OVERPASS_URL`); `tools/osm_fetch_local.ps1` runs the OSM scripts against local Overpass; `fill_h2h_road_osrm.py` backfills 0-valued hospital↔hospital road distances via OSRM table API. Guide: `docs/local_osm_osrm.md` (docs/ is gitignored).

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
- **The Unity MCP bridge is often NOT connected to the Claude session** (only `claude.ai Figma/Notion/...` may be present) — confirm with `ListMcpResourcesTool` / `claude mcp list` **before** planning MCP-driven Unity work; `/unity-mcp-skill` is docs, not a connection. With no bridge you can still verify from disk: `.unity` scenes are YAML, so `grep -l '<marker>' Assets/Scenes/Regions/*.unity` confirms a pass applied across the 255 scenes, and a `StartBackgroundImport` pass saving scenes at all is **proof every edited runtime+editor script compiled clean** ("compile-proof" — Unity won't run editor code with any compile error).
- Flat/ribbon meshes need **CCW winding (viewed from above)** so `RecalculateNormals` faces +Y; CW → normals point down → backface-culled (invisible from above). Test visibility with a **URP** Unlit material (built-in `Unlit/Color` renders invisible under URP).

## Long-running background work

Fetches/imports are launched detached (`nohup … &` in a `run_in_background` bash). Detached python isn't harness-tracked, so completion is detected with a **watcher** bash (also `run_in_background`) that polls a file count / completion marker / stall and exits — the harness then notifies on the watcher's exit.
