# Inspect UI verification — 2026-11-03

The Inspect page now uses a large image viewer with input, heatmap-overlay and
pixel-mask tabs, fit/100%/zoom controls, a resizable controls pane, and collapsed
advanced smoothing controls. Train has a resizable preview pane and labels that
distinguish the highest image-AUROC result from a manual selection.

## Evidence

- `inspect-compact.png`: 1180 × 760 logical window size, 125% Windows scaling.
- `inspect-large.png`: 1440 × 820 logical window size, 125% Windows scaling.
- `inspect-advanced.png`: expanded advanced section; the controls pane scrolls.
- `train-selection.png`: reference comparison with the ranking explanation.

These are native Qt captures using bundled reference images and explicitly
labeled demonstration results. They are layout evidence, not a new model run.
All four captures were inspected. The production executable was also opened and
checked on Train and Inspect, including a missing remembered-model path.

## Validation

- MSVC 2022 / Qt 6.11.1 debug application build passed with camera SDK linkage.
- Native regression suite: 13 passed, 0 failed (including setup and cleanup).
- Viewer checks cover preserved aspect ratio, original-pixel zoom, zoom in,
  fit, compact resizing, clearing old images, advanced disclosure, and long paths.
- Selection checks cover the best image AUROC, manual selection, and no result.
- Existing inference-session, production-promotion, camera-selection and float
  mask threshold tests remain passing.
- No new real-model training or camera-acquisition run was performed.

Build and test logs are under `build/ui-improvements-app/build.log` and
`build/ui-improvements-tests/{build.log,results.txt}` (ignored build artifacts).
Set `QT_QPA_PLATFORM=windows` and `INSPECTION_QA_CAPTURE_DIR` to the desired output
directory when running `native_regression.exe` to reproduce the layout captures.
