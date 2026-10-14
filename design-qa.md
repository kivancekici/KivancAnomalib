# Design QA — Experiment Bench

## Comparison Target

- Source visual truth:
  `C:\Users\kivanc.ekici\.codex\generated_images\019fb2cd-34b6-7692-bbe5-92e0913c4df6\call_S8RpRlLrs0rqPB9WGniPYXtV.png`
- Rendered implementation:
  `C:\Users\kivanc.ekici\Documents\GitHub\KivancAnomalib\qa\implementation-final.png`
- Full-view comparison:
  `C:\Users\kivanc.ekici\Documents\GitHub\KivancAnomalib\qa\comparison-source-implementation.png`
- Focused center-workspace comparison:
  `C:\Users\kivanc.ekici\Documents\GitHub\KivancAnomalib\qa\comparison-focused.png`
- State: Train page, completed PatchCore/PaDiM/EfficientAD reference experiment,
  PatchCore selected as winner.

## Viewport and Normalization

- Source image: 1487 × 1058 px at 96 DPI.
- Implementation capture: 1490 × 823 px at 120 DPI on a Windows display at
  125% scaling.
- Computer Use capture viewport: 1490 × 823.
- Normalized source: top 1487 × 823 crop, preserving source pixels at 1:1.
- Full comparison: normalized source and implementation stacked at equal
  pixel scale, with no resampling of the visible UI.
- The selected mock was designed for a taller desktop viewport. The
  implementation intentionally compacts inspection imagery and lets the center
  pane scroll on the available 823 px-tall window.

## Full-View Evidence

The final capture preserves the source hierarchy:

- 170 px navigation rail;
- dataset title, summary bar, and dataset-detail row;
- three-model selection row and one primary training action;
- aligned six-column comparison table with a green selected-winner row;
- dedicated inspection panel with input, mask, heatmap, threshold, predicted
  mask, promotion, and export controls.

The Qt implementation uses the same visual assets extracted from the selected
mock, rather than placeholders.

## Focused Evidence

The focused comparison covers the densest and most editable region: dataset
summary, model selection, training state, and results table. It confirms:

- matching content order and horizontal grid;
- matching metric labels and values;
- matching light neutral palette and blue/green semantic emphasis;
- legible Segoe UI hierarchy at the native Windows scale;
- consistent table alignment and row selection treatment.

## Required Fidelity Surfaces

- Fonts and typography: passed. Segoe UI is a close native match; headings,
  body copy, numeric metrics, weights, wrapping, and truncation remain legible.
- Spacing and layout rhythm: passed. Major tracks, margins, summary rhythm,
  model row, table density, and inspection hierarchy match. The shorter screen
  is handled as an intentional responsive constraint.
- Colors and visual tokens: passed. White/cool-gray surfaces, blue actions,
  green winner state, and warning colors match the source intent with adequate
  contrast.
- Image quality and asset fidelity: passed. Input, mask, heatmap, and predicted
  mask are exact crops from the selected generated design and remain sharp.
  Standard Qt icons are used for controls.
- Copy and content: passed. Dataset counts, model names, metrics, actions, and
  inspection labels preserve the chosen design's meaning.

## Comparison History

### Iteration 1

- [P2] A completed-state green progress rail added visual weight that was not
  present in the source.
- [P2] Model cards used a Unicode check character instead of a native icon.
- Fixes: the progress rail now appears only while a training job is active,
  and model cards use standard Qt icon assets.

### Iteration 2

- Post-fix evidence:
  `qa\implementation-final.png`,
  `qa\comparison-source-implementation.png`, and
  `qa\comparison-focused.png`.
- No actionable P0, P1, or P2 differences remain.

## Primary Interactions Tested

- left-navigation page changes;
- demonstration train/test flow for all three models;
- live progress, completion, results, and winner selection;
- settings environment verification;
- Basler camera discovery;
- real Basler single-frame acquisition and OpenCV PNG writing.

This is a native application, so browser console checks are not applicable.
The process launched without a startup error, the MSVC debug build passed, and
the worker stderr path remained clear during the tested flows.

## Follow-up Polish

- [P3] Native Windows/Qt icons differ slightly from the mock's monochrome
  outline icon family.
- [P3] Native window and menu chrome is taller than the generated mock.
- [P3] At 823 px height, right-panel images are deliberately cropped more
  tightly to keep promotion and export controls visible.

final result: passed
