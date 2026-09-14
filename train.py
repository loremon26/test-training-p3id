"""Two-stage P&ID Relationformer training entry point (single GPU, gradient accumulation)."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import yaml

DEFAULT_CONFIG=Path(__file__).resolve().parent/'configs/road_2D.yaml'


def dict2obj(data):
    return json.loads(json.dumps(data),object_hook=lambda d:SimpleNamespace(**d))


def load_config(path, verbose=False):
    with open(path) as f:
        data=yaml.safe_load(f)
    return dict2obj(data)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--phase',choices=['pretrain','finetune'],default='pretrain')
    p.add_argument('--data-root',type=Path)
    p.add_argument('--output-dir',type=Path)
    loading=p.add_mutually_exclusive_group()
    loading.add_argument('--pretrained',type=Path,help='Transfer weights only; reset optimizer/early stopping')
    loading.add_argument('--resume',type=Path,help='Restore complete run state from a trusted local checkpoint')
    p.add_argument('--device',default='cuda')
    p.add_argument('--cuda_visible_device',type=int,nargs='+',default=[0])
    p.add_argument('--epochs',type=int)
    p.add_argument('--batch-size',type=int)
    p.add_argument('--effective-batch-size',type=int)
    p.add_argument('--no-imagenet',action='store_true')
    p.add_argument('--max-steps',type=int,help='Smoke run: stop/save after this many optimizer steps')
    p.add_argument('--real-drawings',type=int,default=60)
    p.add_argument('--finetune-synthetic-drawings',type=int,default=500)
    p.add_argument('--synthetic-source',default='Dataset PID',
                   help='Patched source used as synthetic data (default: Dataset PID; use Generated for the local generator)')
    a=p.parse_args()
    if len(a.cuda_visible_device)!=1:
        p.error('This pipeline uses one GPU; use gradient accumulation for effective batch 20')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.cuda_visible_device[0])
    data=yaml.safe_load(a.config.read_text())
    data['TRAIN']['PHASE']=a.phase
    if a.data_root:
        data['DATA']['DATA_PATH']=str(a.data_root)
        os.environ['PID2GRAPH_DATA_PATH']=str(a.data_root)
    if a.output_dir:
        data['TRAIN']['SAVE_PATH']=str(a.output_dir)
    for value,key in ((a.epochs,'EPOCHS'),(a.effective_batch_size,'EFFECTIVE_BATCH_SIZE')):
        if value is not None:
            data['TRAIN'][key]=value
    if a.batch_size is not None:
        data['DATA']['BATCH_SIZE']=a.batch_size
    if a.no_imagenet or a.resume or a.pretrained:
        data['MODEL']['ENCODER']['PRETRAINED']=False
    if a.phase=='finetune':
        data['DATA']['TRAIN_SOURCES']=[a.synthetic_source,'Real']
        data['DATA']['SOURCE_LIMITS']={a.synthetic_source:a.finetune_synthetic_drawings,'Real':a.real_drawings}
        data['TRAIN']['REAL_DRAWINGS']=a.real_drawings
    else:
        data['DATA']['TRAIN_SOURCES']=[a.synthetic_source]
        data['DATA']['SOURCE_LIMITS']={}
    data['DATA']['TEST_SOURCES']=[]
    from trainer import run_training
    run_training(dict2obj(data),a,data)


if __name__=='__main__':
    main()
