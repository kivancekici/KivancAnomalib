from pathlib import Path
import json
import os
import sys
import traceback

ROOT = Path.cwd()
OUT = ROOT / '.data/review-2026-10-22/ml-training'
sys.path.insert(0, str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['WANDB_MODE'] = 'disabled'
from worker import anomalib_worker as worker
worker.install_offline_network_guard()

results = {}

def probe(name, fn):
    try:
        value = fn()
        results[name] = {'ok': True, 'value': value}
    except Exception as exc:
        results[name] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
    print(name, json.dumps(results[name]), flush=True)

import torch
import anomalib
from PIL import Image
from anomalib.data import Folder
from anomalib.models.image.padim.torch_model import PadimModel
from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator
from anomalib.models.image.efficient_ad.torch_model import SmallPatchDescriptionNetwork
results['versions'] = {'anomalib': anomalib.__version__, 'torch': torch.__version__}
torch.set_num_threads(2)

fixture = OUT / 'dataset'
for name, count, color in [('normal', 40, (60, 60, 60)), ('abnormal', 16, (220, 20, 20))]:
    target = fixture / name
    target.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new('RGB', (32, 32), color).save(target / f'{index:03}.png')
job = {'dataset_name':'review_fixture', 'dataset_root': str(fixture), 'normal_dir':'normal', 'abnormal_dir':'abnormal', 'num_workers':0}

def split_snapshot(dm):
    dm.setup()
    return {name: sorted(Path(p).relative_to(fixture).as_posix() for p in getattr(dm, name + '_data').samples.image_path) for name in ('train', 'val', 'test')}

def split_probe():
    a = split_snapshot(worker.create_datamodule(job))
    b = split_snapshot(worker.create_datamodule(job))
    seeded_args = dict(name='review_fixture', root=str(fixture), normal_dir='normal', abnormal_dir='abnormal', num_workers=0, seed=42)
    c = split_snapshot(Folder(**seeded_args))
    d = split_snapshot(Folder(**seeded_args))
    return {'a':a, 'b':b, 'equal':a==b, 'seeded_equal':c==d, 'counts':{key:len(value) for key,value in a.items()}, 'test_intersection_count':len(set(a['test']) & set(b['test']))}
probe('independent_splits', split_probe)

results['padim_accepted_parameters'] = worker.normalize_model_parameters('PaDiM', {'backbone':'resnet18','layers':['layer1'], 'pre_trained': False})
probe('padim_model_constructor', lambda: str(PadimModel(backbone='resnet18', layers=['layer1'], pre_trained=False, n_features=None)))

results['patchcore_accepted_parameters'] = worker.normalize_model_parameters('PatchCore', {'image_size':{'width':32,'height':32},'sigma':8})
probe('patchcore_sigma_8_size_32', lambda: list(AnomalyMapGenerator(sigma=8)(torch.zeros((1,1,4,4)), image_size=(32,32)).shape))
probe('patchcore_sigma_7_size_32', lambda: list(AnomalyMapGenerator(sigma=7)(torch.zeros((1,1,4,4)), image_size=(32,32)).shape))

results['efficientad_accepted_parameters'] = worker.normalize_model_parameters('EfficientAD', {'teacher_out_channels':383})
def teacher_shape_probe():
    path = ROOT / '.data/model-assets/anomalib/pre_trained/efficientad_pretrained_weights/pretrained_teacher_small.pth'
    state = torch.load(path, map_location='meta', weights_only=True)
    results['teacher_checkpoint_conv4_shape'] = list(state['conv4.weight'].shape)
    with torch.device('meta'):
        model = SmallPatchDescriptionNetwork(out_channels=383)
        model.load_state_dict(state)
    return 'loaded'
probe('efficientad_teacher_shape', teacher_shape_probe)

from anomalib.models import Patchcore, Padim
results['requested_training_parameters'] = worker.normalize_training_parameters({'max_epochs':5,'devices':2}, 'PatchCore')
results['patchcore_trainer_arguments'] = Patchcore.trainer_arguments.fget(None)
results['padim_trainer_arguments'] = Padim.trainer_arguments.fget(None)
(OUT / 'probe-results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print('Saved', OUT / 'probe-results.json')
