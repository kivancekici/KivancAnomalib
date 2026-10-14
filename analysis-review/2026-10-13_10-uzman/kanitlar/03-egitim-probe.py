"""Read-only isolated API probes; no real model, dataset, or training run."""
from __future__ import annotations

import ast
import copy
import importlib.util
import io
import logging
from pathlib import Path
import pickletools
from types import SimpleNamespace
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[3]
SITE = ROOT / '.venv/Lib/site-packages'
spec = importlib.util.spec_from_file_location('review_worker', ROOT / 'worker/anomalib_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

def installed_ast(path):
    return ast.parse((SITE / path).read_text(encoding='utf-8'))

def class_node(path, name):
    return next(n for n in installed_ast(path).body if isinstance(n, ast.ClassDef) and n.name == name)

def run_node(node, namespace):
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<isolated installed source>', 'exec'), namespace)

print('Probe scope: AST-isolated installed API branches; no full Anomalib model construction, no weights loaded, no data training.')

# Actual worker normalization, followed by the installed PaDiM validation branch.
padim = worker.normalize_model_parameters('PaDiM', {'backbone': 'resnet18', 'layers': ['layer1'], 'pre_trained': False})
print('PaDiM worker accepted:', {k: padim[k] for k in ('backbone','layers','n_features')})
fn = copy.deepcopy(next(n for n in class_node('anomalib/models/image/padim/torch_model.py', 'PadimModel').body if isinstance(n, ast.FunctionDef) and n.name == '__init__'))
start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute) and n.targets[0].attr == 'n_features_original')
fn.body = fn.body[start:start+5]
ns = {'_N_FEATURES_DEFAULTS': {'resnet18': 100, 'wide_resnet50_2': 550}}
run_node(fn, ns)
try:
    ns['__init__'](SimpleNamespace(backbone='resnet18', feature_extractor=SimpleNamespace(out_dims=[64])), n_features=None)
except ValueError as exc:
    print('PaDiM installed validation branch:', type(exc).__name__, str(exc))
else:
    raise AssertionError('Expected invalid automatic dimensions')

# Read-only pickle opcode inspection. This does not call pickle.load or torch.load.
for filename, layer in [('pretrained_teacher_small.pth','conv4.weight'), ('pretrained_teacher_medium.pth','conv6.weight')]:
    path = ROOT / '.data/model-assets/anomalib/pre_trained/efficientad_pretrained_weights' / filename
    with zipfile.ZipFile(path) as archive:
        output = io.StringIO()
        pickletools.dis(archive.read(next(n for n in archive.namelist() if n.endswith('data.pkl'))), out=output)
        lines = output.getvalue().splitlines()
        index = next(i for i, line in enumerate(lines) if layer in line)
        print('Teacher metadata only:', filename, layer)
        print('\n'.join(lines[index:index+26]))
efficient = worker.normalize_model_parameters('EfficientAD', {'teacher_out_channels': 383})
print('EfficientAD worker accepted teacher_out_channels:', efficient['teacher_out_channels'])

# Execute installed cache update directly, without Trainer or hardware construction.
ns = {'logger': logging.getLogger('review')}
run_node(copy.deepcopy(class_node('anomalib/engine/engine.py', '_TrainerArgumentsCache')), ns)
cache = ns['_TrainerArgumentsCache'](max_epochs=5, devices=2)
cache.update(SimpleNamespace(trainer_arguments={'max_epochs': 1, 'devices': 1}))
print('Trainer cache requested max_epochs=5 devices=2 -> effective:', cache.args)

# Execute only coreset constructor. Tiny shape stub reaches the real early guard.
ratio = worker.normalize_model_parameters('PatchCore', {'coreset_sampling_ratio': 0.001})['coreset_sampling_ratio']
fn = copy.deepcopy(next(n for n in class_node('anomalib/models/components/sampling/k_center_greedy.py', 'KCenterGreedy').body if isinstance(n, ast.FunctionDef) and n.name == '__init__'))
ns = {}
run_node(fn, ns)
try:
    ns['__init__'](SimpleNamespace(), SimpleNamespace(shape=(128, 1536)), ratio)
except ValueError as exc:
    print('Coreset installed constructor guard:', type(exc).__name__, str(exc))
else:
    raise AssertionError('Expected zero coreset size')

# Tiny synthetic CPU parameters for the real installed scheduler setup only.
# No neural model is loaded and no optimizer/backward/training step is performed.
import torch
fn = copy.deepcopy(next(n for n in class_node('anomalib/models/image/efficient_ad/lightning_model.py', 'EfficientAd').body if isinstance(n, ast.FunctionDef) and n.name == 'configure_optimizers'))
ns = {'torch': torch}
run_node(fn, ns)
params = [torch.nn.Parameter(torch.tensor([0.0])), torch.nn.Parameter(torch.tensor([0.0]))]
fake = SimpleNamespace(
    model=SimpleNamespace(student=SimpleNamespace(parameters=lambda: [params[0]]), ae=SimpleNamespace(parameters=lambda: [params[1]])),
    trainer=SimpleNamespace(max_epochs=5, max_steps=-1, datamodule=SimpleNamespace(train_dataloader=lambda: range(100))),
    lr=0.0001, weight_decay=0.00001,
)
config = ns['configure_optimizers'](fake)
scheduler = config['lr_scheduler']
print('EfficientAD scheduler configured:', 'keys=', sorted(config), 'step_size=', scheduler.step_size)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for _ in range(5):
        scheduler.step()
print('Synthetic 5 epoch scheduler ticks: last_epoch=', scheduler.last_epoch, 'lr=', scheduler.get_last_lr())
types_tree = installed_ast('lightning/pytorch/utilities/types.py')
lr_class = next(n for n in types_tree.body if isinstance(n, ast.ClassDef) and n.name == 'LRSchedulerConfig')
interval = next(n.value.value for n in lr_class.body if isinstance(n, ast.AnnAssign) and n.target.id == 'interval')
assert interval == 'epoch'
print('Installed Lightning LRSchedulerConfig default interval:', interval)
print('All probes complete; real fit/test/export/inference not run.')
