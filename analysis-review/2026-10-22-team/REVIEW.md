Inspection Workbench review — 22 October 2026 (Europe/Istanbul)

Reviewed commit `8f48bf3839d531cd266fc4640f7acaace5dad983`. The working tree was clean before review. The review identified **17 actionable findings: four P1 and thirteen P2**. P1 means fix before relying on production decisions or artifact preservation; P2 means a reproducible correctness/reliability defect with a narrower trigger. These are findings against the current app, not necessarily newly introduced regressions.

Three specialist agents and the coordinating reviewer examined the app:

| Reviewer | Scope | Evidence |
| --- | --- | --- |
| Native UI expert | Qt state, process ownership, camera selection, operator flows | Source-path analysis; no live GUI or camera execution |
| ML training expert | Data splits, metrics, model configuration | Actual Folder setup and small component calls; stubbed engine ranking |
| Inference/export expert | Calibration, preprocessing, artifacts, runtime parity | Synthetic CPU ONNX graphs, capture runtimes, fake exporter with actual path logic |
| Coordinating reviewer | Cross-checks, full suite, integration failures | 37 worker tests, synthetic asset failure and invalid-score probes |

The existing suite passed **37/37 tests in 171.620 seconds**, exit 0, using the Python environment documented by this repository. Additional probes exposed behavior the suite does not currently reject. Earlier review notes were used as leads; the findings below were checked against current sources and fresh probes.

1. **[P1] A loaded real model can still produce demonstration predictions.** [UI:3558](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/mainwindow.cpp:3558)–3566.

   Demo mode defaults to enabled in Settings. Loading an artifact initializes the real backend and shows its device, but does not disable demo mode. Running inference then branches solely on that checkbox and invokes the synthetic worker. The result view ignores the demo flag and displays ordinary NORMAL/ANOMALY decisions. An operator can therefore believe the loaded model evaluated the image. Bind inference to an explicit session mode and label simulated results on the Inspect page. Validate by loading a real artifact with default settings and proving the prediction uses its backend/fingerprint. Source verification only.

2. **[P1] Production promotion accepts missing and synthetic checkpoints.** [UI:3064](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/mainwindow.cpp:3064)–3068, write at [UI:3096](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/mainwindow.cpp:3096).

   Startup always populates reference results containing nonexistent `demo/.../model.ckpt` paths. Promotion is available and neither validates the checkpoint/fingerprint nor requires a completed real run. Clicking it can replace an existing valid `production_model.json` with an unusable artifact and empty fingerprint while reporting success. Require real-run provenance, a readable valid artifact and a matching nonempty fingerprint before replacement. Acceptance: startup/demo/missing-checkpoint candidates are rejected and an existing production selection remains intact. Source verification only; no user's production manifest was changed.

3. **[P1] Per-image normalization corrupts pixel threshold decisions.** [worker:1876](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/worker/anomalib_worker.py:1876), threshold at [worker:1891](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/worker/anomalib_worker.py:1891), normalization at [worker:1810](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/worker/anomalib_worker.py:1810)–1815.

   The worker rescales each anomaly map to its own minimum/maximum before applying the operator threshold; constant maps become zero. At threshold 0.6, a calibrated map in 0.1–0.2 generated 50% anomalous pixels rather than zero. A constant 0.9 map generated zero rather than 100%. Preserve calibrated floating-point maps for decisions and apply display contrast separately. The Train threshold control also reads the normalized PNG and needs the same correction. Evidence: [inference results](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/inference/results.json) (`mask_threshold`).

4. **[P1] A later export overwrites the previous model and manifest.** [worker:2579](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/worker/anomalib_worker.py:2579)–2585; shared destination at [UI:3055](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/mainwindow.cpp:3055).

   All exports use the same root without a unique filename. Installed Anomalib defaults to `weights/<format>/model.*`. PatchCore followed by PaDiM exported to the same path; the second artifact and manifest replaced the first. Allocate an immutable run/model destination and stage the complete export before publishing its manifest. Acceptance: exporting two models preserves both hashes, and failure of the second leaves the first usable. Evidence: [export collision](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/inference/results.json). The exporter was mocked; orchestration and installed destination/default-name logic were real. No production export was overwritten.

5. **[P2] Compared models use different dataset splits.** [worker:2267](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/worker/anomalib_worker.py:2267)–2268; Folder arguments at [worker:2050](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/worker/anomalib_worker.py:2050)–2062.

   Each model creates a new Folder with its default seed of None, so test AUROCs used to pick a winner need not describe the same samples. Two fresh setups of 40 normal and 16 abnormal images each produced 32/12/12 train/validation/test counts, but shared only four of twelve test images. Seeded controls matched. Create and persist one split per experiment and reuse it across models. Evidence: [training probe results](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/ml-training/probe-results.json). This demonstrates differing membership, not observed leakage within one split or a measured change to a real production winner.

6. **[P2] ONNX replaces calibrated maps even when sigma override is off.** [worker:2967](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/worker/anomalib_worker.py:2967)–2971; repeated at [worker:3081](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/worker/anomalib_worker.py:3081).

   Override-capable graphs always replace their calibrated anomaly map with a blur of the raw map. A real synthetic Gaussian graph returned a calibrated mean of 0.062745; worker prediction with no override returned 0.250980. The image score stayed unchanged. Preserve the graph output when override is absent and reapply the saved calibration when overriding sigma. This is separate from finding 3; removing per-image normalization alone does not fix it. Evidence: [raw-map replacement](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/inference/results.json); CPU ONNX, not a production model benchmark.

7. **[P2] Almost-black uint8 images receive the wrong input scale.** [worker:2942](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/worker/anomalib_worker.py:2942)–2944; repeated at [worker:3055](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/worker/anomalib_worker.py:3055).

   Both ONNX adapters divide by 255 only when the maximum pixel value exceeds one. PIL supplies uint8 data, so an image containing only 0 and 1 reaches the model with code 1 mapped to 1.0, while code 2 maps to 0.007843. Normalize by dtype/input contract, independent of observed brightness. Acceptance: codes 0, 1, 2 and 255 always become 0, 1/255, 2/255 and 1. Evidence: [captured input tensors](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/inference/results.json); no real-model accuracy impact was measured.

8. **[P2] Invalid prediction scores produce fabricated decisions or invalid protocol data.** [worker:1899](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/worker/anomalib_worker.py:1899)–1906; scalar conversion at [worker:1774](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/worker/anomalib_worker.py:1774).

   A missing score falls back to the maximum of the already stretched map. A low 0.1–0.2 map therefore received score 1.0 and an anomalous label. NaN produced an internal normal label and nonstandard JSON; Infinity also escaped into JSON. Require a finite scalar score, or a documented calibrated fallback, before creating a decision. Evidence: [integration results](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/integration-results.json). This probe does not establish that Qt displays NORMAL for NaN: its JSON parser can reject that event instead.

9. **[P2] Invalid or absent metrics can still select a winner.** [worker:2325](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/worker/anomalib_worker.py:2325)–2329; conversion at [worker:1713](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/worker/anomalib_worker.py:1713)–1716.

   Ranking filters out None but accepts NaN/Infinity. The real training orchestration with stubbed engine results selected PatchCore=NaN over PaDiM=0.99 and emitted nonstandard JSON. When both metrics were absent, it still selected the first model. Normalize protocol values and require finite comparable metrics; report an incomplete comparison if none remain. Evidence: [ranking results](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/ml-training/metrics-probe-results.json). No real trained model was observed returning NaN during this review.

10. **[P2] ONNX fingerprints omit external weight files.** [worker:633](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/worker/anomalib_worker.py:633)–637.

    Fingerprinting includes the protobuf but not its referenced tensor files. Changing a synthetic external weight from 1 to 2 doubled real ONNX Runtime output while inspection returned the same fingerprint and accepted the original manifest. Imported external-data models are affected; the exporter also keeps external data above its 1.8-billion-byte packaging threshold. Hash and validate the complete referenced file set, including missing files. Evidence: [external-weight integrity](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/inference/results.json).

11. **[P2] Export during an active job clears ownership of the running worker.** [UI:2736](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/mainwindow.cpp:2736)–2743.

    Export controls remain usable during a job. After a model produces a checkpoint, selecting Export while another model trains changes `m_activeJob` before the process wrapper rejects the concurrent start. The resulting failure handler clears job state and hides Stop even though the original process continues. Check occupancy before mutating job state and disable conflicting actions. Acceptance: a rejected second job leaves the original job identity, progress and Stop control intact. Source-path verification; no live concurrent GUI run.

12. **[P2] Capture can hide existing samples in supported alias folders.** [UI:3921](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/mainwindow.cpp:3921)–3924; folder precedence at [UI:2476](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/mainwindow.cpp:2476).

    Capture always creates `normal` or `abnormal`. For a dataset using supported `good` or `anomalous` names, the newly created preferred folder shadows the old folder, so scanning/training omits previous samples. Directory creation occurs before acquisition; even a failed capture can leave an empty preferred folder. Resolve the existing class directory before creating a new one. Acceptance: successful and failed captures preserve the same sample membership except for a successfully added frame. Source-path verification.

13. **[P2] A stale camera index can capture a different device.** [cameraservice.cpp:70](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/cameraservice.cpp:70)–80.

    Discovery shows model/serial labels, but capture passes only the selection index into a freshly enumerated list. If A and B were discovered and selected A disconnects, index zero can now identify B while the UI still names A. Store a stable device identifier and fail if that specific device disappears. Acceptance: disconnecting the selected device never captures a replacement silently. This is a conditional source finding; no physical camera/hot-unplug test was performed.

14. **[P2] Asset replacement deletes the existing directory before promotion succeeds.** [worker:1012](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/worker/anomalib_worker.py:1012)–1016.

    During managed-asset repair/replacement, the old destination is recursively deleted before the staging directory is renamed. An injected rename PermissionError left the previous asset missing and only the staged candidate present. HF cache import also validates after this replacement. Validate staging first and use a backup/rollback or versioned-directory switch. Acceptance: rename/validation/lock-write failures preserve a usable previous asset where one exists. Evidence: [asset promotion failure](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/integration-results.json); only temporary synthetic directories were affected.

15. **[P2] Accepted PatchCore sigma can be incompatible with input dimensions.** [worker:319](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/worker/anomalib_worker.py:319)–321.

    A permitted 32×32 input with sigma 8 generates reflection padding of 32 and fails in the real AnomalyMapGenerator; sigma 7 works. The combination can fail during validation after feature extraction, and runtime overrides can hit it too. Validate the kernel/padding requirement against both image dimensions at training and override boundaries. Evidence: [PatchCore component probe](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/ml-training/probe-results.json); a small tensor/component invocation, not full training.

16. **[P2] PaDiM feature validation ignores selected layer capacity.** [worker:334](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/worker/anomalib_worker.py:334)–338.

    ResNet18 with layer1 only and automatic feature count passes app validation. Installed Anomalib chooses 100 features from a 64-channel layer and raises a ValueError. Explicit counts above the selected total also pass the app's 65,536 limit. Resolve automatic defaults and bounds against backbone/layer channel capacity before starting. Evidence: [PaDiM constructor probe](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/ml-training/probe-results.json); real offline constructor with pretrained weights disabled.

17. **[P2] EfficientAD accepts teacher widths incompatible with its supplied weights.** [worker:344](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/worker/anomalib_worker.py:344)–349.

    Teacher channels accept 1–65,536, but the app loads a fixed 384-channel teacher. Setting 383 passes validation and fails on the teacher's final convolution and bias shapes. Restrict channels to the supplied artifact's width unless compatible custom teachers are supported. Evidence: [teacher-shape probe](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/ml-training/probe-results.json); the managed teacher was inspected with weights-only loading on the meta device, without allocating its weight tensors.

The first fixes should prevent synthetic or invalid production decisions and preserve exported artifacts (findings 1–4). Then correct shared numeric contracts and comparison inputs (5–10), followed by state handling, asset rollback and configuration validation (11–17). Add outcome-based regression checks using the supplied counterexamples.

Validation artifacts are [the full unittest log](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/unittest.log), [the integration probe](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/integration_probe.py), [the inference/export probe](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/inference/probe_inference.py), [the training probe](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/ml-training/probe.py), and [the ranking probe](C:/Users/kivanc.ekici/Documents/GitHub/KivancAnomalib/analysis-review/2026-10-22-team/ml-training/metrics-probe.py). Run copied ML probes from the repository root; they write their fresh outputs under `.data/review-2026-10-22`. The other copied probes create only local synthetic fixtures alongside their scripts.

The app's source/configuration files were not modified. No native rebuild, live GUI walkthrough, physical camera test, production dataset/model inference, full training, GPU benchmark, download, installation, commit, or publication was performed. The findings identify code/component failures and do not quantify production accuracy or throughput. Review reports and small evidence files were added under `analysis-review/2026-10-22-team`.

