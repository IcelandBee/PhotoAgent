# Renderer interpolation ablation

This research runner compares three CPU point-cloud rasterizers while keeping
Depth Anything V2 Small, pseudo-depth mapping (`alpha=0.10`), source hFOV 60,
camera presets, and the immutable `PreparedPointCloud` fixed.

- `nearest_z`: explicit regression/debug baseline with the original radius-one policy.
- `bilinear`: production default, using four sub-pixel neighbors, a nearest-surface depth gate, and weighted RGB.
- `gaussian`: research-only radius-one 3x3 footprint (`sigma=0.75`).

Run the controlled three-source experiment from the repository root:

```powershell
.\.venv311\Scripts\python.exe experiments/renderer_interpolation_ablation/run_batch.py `
  --input-dir "D:\Data\videoagent\test_data\PhotoAgent\data\test_data\gpt_enhanced" `
  --output-dir "workdir\renderer_interpolation_ablation_20260921" `
  --depth-model "workdir\depth_models\depth-anything-v2-small" `
  --device cpu
```

The output directory must not already exist. Each formal timing uses one warm-up
and five measured calls. PNG writing, comparison generation, and metadata I/O
occur after timing. The production renderer is only called as an independent
oracle for the 12 exact `nearest_z` regression checks.

## 27-image nearest versus bilinear regression

The follow-up runner discovers all 27 images in the input directory and refuses
to run if the count differs. It uses exactly four presets and the original
bilinear math, then measures a separate cold, warm, and cached same-source
single-sketch path on video4, video5, and video23.

```powershell
.\.venv311\Scripts\python.exe experiments/renderer_interpolation_ablation/run_27_regression.py `
  --input-dir "D:\Data\videoagent\test_data\PhotoAgent\data\test_data\gpt_enhanced" `
  --output-dir "workdir\renderer_bilinear_regression_20260922" `
  --depth-model "workdir\depth_models\depth-anything-v2-small" `
  --device cpu
```

The runner writes a checkpointed summary and CSV after each source. E2E rows
account for image loading, depth stages, pseudo-Z mapping, intrinsics, geometry,
render, and PNG/mask/metadata save. A companion `calibrate_legacy_timing.py`
times production and research nearest render calls on identical warmed inputs.

```powershell
.\.venv311\Scripts\python.exe experiments/renderer_interpolation_ablation/calibrate_legacy_timing.py `
  --source "D:\Data\videoagent\test_data\PhotoAgent\data\test_data\gpt_enhanced\video4_final_01_gpt.png" `
  --depth "workdir\renderer_bilinear_regression_20260922\video4_final_01_gpt\depth\depth.npy" `
  --output "workdir\renderer_bilinear_regression_20260922\legacy_timing_calibration.json"
```
