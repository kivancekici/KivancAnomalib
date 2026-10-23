from pathlib import Path
import contextlib
import io
import json
import math
import sys
import types
from unittest import mock
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from worker import anomalib_worker as worker
worker.install_offline_network_guard()
OUT = ROOT / '.data/review-2026-10-22/ml-training'

class StubEngine:
    test_metrics = {}
    def __init__(self, **kwargs): self.best_model_path = ''
    def fit(self, **kwargs): pass
    def test(self, **kwargs): return [self.test_metrics[kwargs['model']]]

anomalib = types.ModuleType('anomalib')
anomalib.__version__ = '2.6.0'
engine = types.ModuleType('anomalib.engine')
engine.Engine = StubEngine
all_results = {}
for label, metrics in {'absent':{'PatchCore':{}, 'PaDiM':{}}, 'nan_first':{'PatchCore':{'image_AUROC':float('nan')}, 'PaDiM':{'image_AUROC':0.99}}, 'nan_second':{'PatchCore':{'image_AUROC':0.99}, 'PaDiM':{'image_AUROC':float('nan')}}}.items():
    StubEngine.test_metrics = metrics
    output = io.StringIO()
    params = {model:worker.normalize_model_parameters(model) for model in ('PatchCore','PaDiM')}
    with mock.patch.dict(sys.modules, {'anomalib':anomalib,'anomalib.engine':engine}), mock.patch.object(worker,'preflight_selected_models',return_value=params), mock.patch.object(worker,'configure_anomalib_asset_paths'), mock.patch.object(worker,'create_datamodule',return_value=None), mock.patch.object(worker,'create_model',side_effect=lambda name,parameters:name), mock.patch.object(worker,'create_inspection_artifacts',return_value={}), contextlib.redirect_stdout(output):
        worker.run_real_train({'models':['PatchCore','PaDiM'],'output_root':str(OUT / label),'asset_root':str(OUT / 'assets')})
    events=[json.loads(line) for line in output.getvalue().splitlines()]
    invalid=[]
    for index,line in enumerate(output.getvalue().splitlines()):
        try:
            json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        except ValueError as exc: invalid.append({'line':index+1,'invalid_constant':str(exc)})
    all_results[label]={'winner':events[-1]['winner'],'result_scores':{e['model']:e['image_auroc'] for e in events if e['event']=='result'},'invalid_json_lines':invalid}
print(json.dumps(all_results,indent=2))
(OUT / 'metrics-probe-results.json').write_text(json.dumps(all_results,indent=2),encoding='utf-8')
