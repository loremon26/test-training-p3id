"""Native PyTorch training loop, resumable at optimizer-step boundaries."""
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_road_network import build_road_network_data, image_graph_collate_road_network
from models import build_model
from models.matcher import build_matcher
from losses import SetCriterion
from evaluator import evaluate


def batch_target(batch,device):
    names=('nodes','edges','node_classes','edge_classes','boxes')
    return {name:[x.to(device) for x in batch[i]] for name,i in zip(names,(1,2,4,5,6))}


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def save_checkpoint(path,state):
    temporary=path.with_suffix('.tmp')
    torch.save(state,temporary)
    os.replace(temporary,path)


def run_training(config,args,config_dict):
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: enable Kaggle GPU or explicitly select --device cpu')
    seed=config.DATA.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(seed)
        names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        print(f'GPUs: {names}, free/total bytes: {torch.cuda.mem_get_info(device)}',flush=True)
    batch_size=config.DATA.BATCH_SIZE
    effective=config.TRAIN.EFFECTIVE_BATCH_SIZE
    if batch_size<1 or effective<batch_size or effective%batch_size:
        raise ValueError('Effective batch size must be a positive multiple of microbatch size')
    output=Path(config.TRAIN.SAVE_PATH)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'last.pt').exists() and not args.resume:
        raise FileExistsError(f'Existing run in {output}; use --resume or a fresh directory')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(output/'train.log')],force=True)
    train_ds,val_ds=build_road_network_data(config)
    benchmarks={'Dataset PID','PID2Graph Synthetic','PID2Graph OPEN100'}
    for ds in (train_ds,val_ds):
        for image,graph,identifier in ds.samples:
            if benchmarks.intersection(image.resolve().parts):
                raise ValueError(f'Benchmark data cannot enter training/validation: {image}')
    all_ids=train_ds.ids+val_ds.ids
    drawings={tuple(identifier.split('/')[:2]) for identifier in all_ids}
    if config.TRAIN.PHASE=='finetune':
        real_count=sum(source=='Real' for source,_ in drawings)
        if real_count!=config.TRAIN.REAL_DRAWINGS:
            raise ValueError(f'Expected {config.TRAIN.REAL_DRAWINGS} real drawings, found {real_count}')
    signature=hashlib.sha256(json.dumps([(sid,im.stat().st_size,hashlib.sha256(g.read_bytes()).hexdigest())
                      for ds in (train_ds,val_ds) for im,g,sid in ds.samples]).encode()).hexdigest()
    split_manifest=dict(train=train_ds.ids,validation=val_ds.ids,drawings=sorted(drawings),
                        data_signature=signature,real_weight=config.TRAIN.REAL_WEIGHT)
    model=build_model(config).to(device)
    # DataParallel keeps checkpoints in the unwrapped format and gathers outputs
    # on the primary GPU, where the variable-size graph loss is evaluated.
    forward_model=torch.nn.DataParallel(model) if device.type=='cuda' and torch.cuda.device_count()>1 else model
    if forward_model is not model:
        print(f'Using {torch.cuda.device_count()} GPUs via DataParallel',flush=True)
    criterion=SetCriterion(config,build_matcher(config),model).to(device)
    optimizer=torch.optim.AdamW([
        {'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.0.')],
         'lr':float(config.TRAIN.LR)},
        {'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('encoder.0.')],
         'lr':float(config.TRAIN.LR_BACKBONE)}],weight_decay=float(config.TRAIN.WEIGHT_DECAY))
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=config.TRAIN.LR_DROP)
    amp=bool(config.TRAIN.AMP) and device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    state=dict(epoch=0,cursor=0,step=0,best=float('inf'),bad_epochs=0,train_sum={},train_count=0)
    if args.pretrained:
        checkpoint=torch.load(args.pretrained,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['net'],strict=True)
    if args.resume:
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=False)
        if checkpoint['data_signature']!=signature:
            raise ValueError('Resume dataset differs from checkpoint; restore original data or start a new run')
        if checkpoint['config']['TRAIN']['PHASE']!=config.TRAIN.PHASE:
            raise ValueError('Use --pretrained, not --resume, to change training phase')
        if checkpoint['config']['DATA']['BATCH_SIZE']!=batch_size or checkpoint['config']['TRAIN']['EFFECTIVE_BATCH_SIZE']!=effective:
            raise ValueError('Resume requires the saved batch configuration')
        for section, keys in [('DATA', ['SEED','IMG_SIZE','AUGMENT']),
                              ('TRAIN',['REAL_WEIGHT','LR','LR_BACKBONE','WEIGHT_DECAY','LR_DROP',
                                        'LOSSES','W_BBOX','W_CLASS','W_CARD','W_NODE','W_EDGE',
                                        'RANDOMIZE_EDGE_DIRECTIONS','MAX_POS_EDGES','NEG_EDGE_RATIO','MIN_NEG_EDGES'])]:
            if any(checkpoint['config'][section].get(k)!=config_dict[section].get(k) for k in keys):
                raise ValueError(f'Resume requires unchanged {section} sampling/loss settings')
        model.load_state_dict(checkpoint['net'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        state=checkpoint['state']
        restore_rng(checkpoint['rng'])
    (output/'split.json').write_text(json.dumps(split_manifest,indent=2))
    (output/'config.json').write_text(json.dumps(config_dict,indent=2))
    import yaml
    (output/'config.yaml').write_text(yaml.safe_dump(config_dict,sort_keys=False))
    def checkpoint(name):
        save_checkpoint(output/name,dict(net=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),state=state,
            rng=rng_state(),config=config_dict,data_signature=signature))
    start=time.monotonic()
    real_weight=int(config.TRAIN.REAL_WEIGHT) if config.TRAIN.PHASE=='finetune' else 1
    if real_weight<1:
        raise ValueError('REAL_WEIGHT must be a positive integer')
    order_base=[(i,repeat) for i,sid in enumerate(train_ds.ids)
                for repeat in range(real_weight if sid.split('/')[0]=='Real' else 1)]
    print(f'Train patches={len(train_ds)}, sampled per epoch={len(order_base)}, validation={len(val_ds)}, effective batch={effective}',flush=True)
    patience=config.TRAIN.EARLY_STOPPING_PATIENCE if config.TRAIN.PHASE=='finetune' else 0
    while state['epoch']<config.TRAIN.EPOCHS:
        train_ds.epoch=state['epoch']
        order=order_base.copy()
        random.Random(seed+state['epoch']).shuffle(order)
        loader=DataLoader(train_ds,batch_size=batch_size,sampler=order[state['cursor']:],
            num_workers=config.DATA.NUM_WORKERS,collate_fn=image_graph_collate_road_network,
            pin_memory=device.type=='cuda',generator=torch.Generator().manual_seed(seed+state['epoch']))
        model.train()
        criterion.train()
        optimizer.zero_grad(set_to_none=True)
        pending=0
        for batch in loader:
            images=batch[0].to(device)
            with torch.autocast(device.type,dtype=torch.float16,enabled=amp):
                h,out=forward_model(images)
                losses=criterion(h,out,batch_target(batch,device))
            if not torch.isfinite(losses['total']):
                raise FloatingPointError(f'Non-finite loss at epoch {state["epoch"]}, cursor {state["cursor"]}')
            count=len(images)
            scaler.scale(losses['total']*count).backward()
            pending+=count
            state['cursor']+=count
            state['train_count']+=count
            for key,value in losses.items():
                state['train_sum'][key]=state['train_sum'].get(key,0.)+float(value.detach())*count
            if pending>=effective or state['cursor']==len(order):
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(pending)
                torch.nn.utils.clip_grad_norm_(model.parameters(),config.TRAIN.CLIP_MAX_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending=0
                state['step']+=1
                if state['step']%config.TRAIN.LOG_INTERVAL==0:
                    logging.info(f'epoch {state["epoch"]+1}, step {state["step"]}, patch {state["cursor"]}/{len(order)}, loss {float(losses["total"].detach()):.4f}')
                stop=(args.max_steps is not None and state['step']>=args.max_steps) or (
                    config.TRAIN.MAX_HOURS and time.monotonic()-start>=config.TRAIN.MAX_HOURS*3600)
                if state['step']%config.TRAIN.CHECKPOINT_STEPS==0 or stop:
                    checkpoint('last.pt')
                if stop:
                    print(f'Saved resumable checkpoint: {output/"last.pt"}',flush=True)
                    return
        val_loader=DataLoader(val_ds,batch_size=batch_size,num_workers=config.DATA.NUM_WORKERS,
            collate_fn=image_graph_collate_road_network,generator=torch.Generator().manual_seed(seed))
        validation=evaluate(forward_model,criterion,val_loader,config,device,
                            output/'validation'/f'epoch_{state["epoch"]+1:03d}',amp)
        current=validation['loss']['total']
        improved=current < state['best']-config.TRAIN.EARLY_STOPPING_MIN_DELTA
        if improved:
            state['best']=current
            state['bad_epochs']=0
        else:
            state['bad_epochs']+=1
        record=dict(epoch=state['epoch']+1,step=state['step'],
            train_loss={k:v/state['train_count'] for k,v in state['train_sum'].items()},
            validation=validation,lr=scheduler.get_last_lr())
        with (output/'history.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        scheduler.step()
        state.update(epoch=state['epoch']+1,cursor=0,train_sum={},train_count=0)
        checkpoint('last.pt')
        if improved:
            checkpoint('best.pt')
        if patience and state['bad_epochs']>=patience:
            print(f'Early stopping on validation loss after {state["epoch"]} epochs',flush=True)
            break
