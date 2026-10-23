# Resolution of the 17 review findings

The fixes address the review of commit `8f48bf3839d531cd266fc4640f7acaace5dad983`, dated 22 October 2026. The native UI, inference/export and ML training specialists implemented their respective changes; the coordinating reviewer implemented asset transactions, cross-checked integration behavior and ran combined verification. Changes remain in the working tree.

The original [review](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/REVIEW.md) and its probes describe the earlier behavior and are retained as historical evidence. The regression tests below exercise the corrected behavior.

| Finding | Resolution | Regression evidence |
| --- | --- | --- |
| 1 · P1 · Real model routed to demo | Inference is bound to an explicit loaded session mode. Global demo settings cannot reroute a real session. Simulated results are labelled. Fresh settings default to real mode. | Native `realLoadedModelIgnoresGlobalDemoToggle`, `demonstrationResultsAreLabelled` |
| 2 · P1 · Invalid production promotion | Promotion requires a completed real result, an existing checkpoint and its matching training-time SHA-256. The production manifest is written atomically and uses the result's immutable provenance. | Native `promotionRejectsInvalidArtifactsWithoutTouchingManifest`, `promotionUsesImmutableResultProvenance`; Python training provenance tests |
| 3 · P1 · Incorrect pixel thresholds | Calibrated maps retain their values. Both worker decisions and Qt threshold controls use float data; PNGs are display previews. A little-endian float32 sidecar preserves values across the worker/UI boundary. | Python calibrated-mask and preview tests; native `thresholdUsesCalibratedFloatValuesWithoutQuantization` |
| 4 · P1 · Export overwrite | Real exports are staged and published into unique model directories only after artifact and manifest creation succeeds. Previous exports survive later exports and failures. | Python `test_exports_are_unique_and_failure_preserves_first_delivery` |
| 5 · P2 · Incomparable data splits | Every model receives the same positive split seed, default 42. Dataset root, run ID and seed are frozen in the job snapshot and results. Seed zero, which would disable determinism in Anomalib, is rejected. | Actual Folder membership and split-disjointness assertions; persisted provenance assertions |
| 6 · P2 · ONNX map calibration lost | With no sigma override, adapters preserve graph outputs. Overrides reapply saved pixel calibration. Legacy graphs without calibration remain usable with their saved sigma. | CPU ONNX and OpenVINO graphs; enabled/disabled normalization, fallback thresholds and legacy output tests |
| 7 · P2 · Dark uint8 input scale | Input scaling follows the uint8 contract, including codes 0 and 1, in both ONNX adapters and the generic OpenVINO adapter. | Input tensor assertions for codes 1, 2 and 255 and already-normalized floats; installed OpenVINO preprocessing |
| 8 · P2 · Invalid prediction score | Missing, non-scalar and non-finite scores fail before artifacts or labels are produced. Non-finite maps also fail. Protocol serialization emits valid JSON. | Invalid score/map tests and strict JSON parsing |
| 9 · P2 · Invalid winner ranking | Only finite comparison metrics qualify. An unscored comparison selects no winner and disables promotion. | Stubbed training orchestration with NaN, Infinity and absent metrics; native empty-winner state |
| 10 · P2 · External ONNX weights omitted | Fingerprints include all referenced tensor files, with path and byte-range validation. Changed, missing or truncated data fails integrity checks. | Synthetic external-data ONNX package mutation and missing/truncated-file tests |
| 11 · P2 · Lost active job ownership | Occupancy checks run before job state or output preparation changes. Rejected starts preserve the original worker and Stop control. | Native `concurrentJobsPreserveOwnershipAndStopControl` |
| 12 · P2 · Capture shadows dataset aliases | Capture resolves existing normal/anomalous aliases before creating folders. Failed captures preserve existing membership. | Native `capturePreservesExistingClassAliasesEvenWhenCameraFails` |
| 13 · P2 · Camera index changes device | Selection stores a persistent camera ID; acquisition resolves that ID again. Reordering preserves selection; disappearance fails instead of substituting another camera. | Native `cameraSelectionSurvivesReorderingAndRejectsDisappearance` |
| 14 · P2 · Asset replacement loses prior files | HF imports/downloads validate in staging. Promotion retains the old directory until metadata commits and rolls back on failure. An OS file lock prevents concurrent asset writers. | Eight failure/success tests covering rename, metadata, corrupt staging, teacher archives and lock release |
| 15 · P2 · Invalid sigma/image combination | Training and inference validate `4 × sigma < min(width, height)`. UI controls enforce the same bound. One-shot overrides are applied after checkpoint restoration. | Real Gaussian component boundary tests; persistent, direct and one-shot backend checks; native constraints |
| 16 · P2 · PaDiM layer capacity | Explicit and automatic feature counts are checked against the selected backbone/layer channel sum. | Actual offline PaDiM constructor checks and native parameter validation |
| 17 · P2 · EfficientAD teacher width | The supplied teacher requires exactly 384 channels; worker validation and the UI enforce it. | Worker validation before construction and native parameter validation |

Regression sources:

- [Native Qt tests](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/tests/native/native_regression.cpp)
- [Inference and export tests](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/tests/test_inference_regressions.py)
- [Training tests](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/tests/test_training_regressions.py)
- [Asset transaction tests](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/tests/test_asset_regressions.py)
- [Existing worker suite](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/tests/test_worker.py)

## Verification

All checks below passed on 22 October 2026:

- **72 Python tests passed**, zero failures/errors/skips, in 91.930 seconds: the existing 37 tests plus 35 regressions (19 inference/export, 8 training, 8 assets). Command: `python -B -X utf8 -m unittest discover -s tests -v`, using `C:\Users\kivanc.ekici\Documents\KivancAnomaLibTest\.venv\Scripts\python.exe`. [Full log](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/verification/python-tests.log).
- **9 native Qt regression methods passed**, zero failures/skips, in 5.803 seconds. Qt reports 11 passes including initialization and cleanup. Built from `tests/native/native_regression.pro` with Qt 6.11.1 and MSVC 2022; executed with `QT_QPA_PLATFORM=offscreen`. [Test log](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/verification/native-tests.log), [test build log](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/verification/native-test-build.log).
- **Native application debug build passed** with Qt 6.11.1/MSVC 2022 and the Basler Pylon/OpenCV branch enabled. The final rebuild regenerated the embedded worker resource and linked successfully. [Build log](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/verification/app-build.log). Executable: [KivancAnomaLibTest.exe](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/build/fix-validation-app/debug/KivancAnomaLibTest.exe).
- `git diff --check` passed.

The Python log retains test output with terminal ANSI escapes and NUL padding removed for readability. Dependency deprecation notices and an offscreen font-directory warning did not fail the checks. Compiler execution required the existing sandbox escalation because MSVC's qmake preprocessing probe crashed within the sandbox; the approved build completed successfully.

Two existing fixtures were updated for the corrected contract: inspection predictions supply a valid scalar score, and the synthetic raw-map graph explicitly records identity calibration. The missing-assets preflight continues to report aggregated asset errors before accessing dataset configuration.

Older ONNX exports without saved pixel calibration remain usable at their saved sigma; re-export to enable overrides. External-data manifests that previously fingerprinted only the graph need a complete-package fingerprint or a fresh export.

Native tests isolate settings and data, use a JSON-line fixture worker, and simulate camera devices. Python tests exercise actual Folder splits, small Anomalib components, CPU ONNX/OpenVINO execution and artifact operations; model training/export orchestration uses stubs where stated. No full production training run, production accuracy benchmark, physical-camera acquisition or GPU benchmark was performed.
