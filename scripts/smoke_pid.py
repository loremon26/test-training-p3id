"""Runnable end-to-end checks. Uses public templates, a full-size model and CPU by default."""
import argparse
import copy
import json
import random
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from scripts.generate_synthetic_pid import generate
from scripts.merge_pid_graph import merge_graphs,fuse
from pid_graph import patch_plan,crop_graph,point_node,parse_graph,write_graph,save_graph_json,overlay
from dataset_road_network import load_pid_graph,PatchedPIDDataset,augment_pid_sample
from train import load_config
from models import build_model
from models.matcher import build_matcher
from losses import SetCriterion
from inference import relation_infer,prediction_graph
from metric_pid import PIDMetrics,ranked_ap


def check(templates,output,device,amp=False):
    torch.set_num_threads(4)
    torch.manual_seed(11)
    output.mkdir(parents=True,exist_ok=True)
    library=json.loads((templates/'templates.json').read_text())
    generated=[]
    for index in range(2):
        for seed in range(10+index*1000,110+index*1000):
            try:
                image,nodes,edges=generate(seed,templates,library['templates'])
                break
            except ValueError as error:
                if str(error)!='route intersects equipment':
                    raise
        generated.append((nodes,edges))
        stem=output/'Complete'/f'drawing{index}'
        stem.parent.mkdir(exist_ok=True)
        image.save(stem.with_suffix('.png'))
        write_graph(stem.with_suffix('.graphml'),nodes,edges)
        manifest=patch_plan(stem.with_suffix('.png'),stem.with_suffix('.graphml'),output/'Patched'/'Generated'/stem.name)
        for patch in manifest['patches']:
            parse_graph(output/'Patched'/'Generated'/stem.name/patch['graph'])
    image2,ns2,es2=generate(seed,templates,library['templates'])
    assert image.tobytes()==image2.tobytes() and nodes==ns2 and edges==es2,'Generator must be deterministic'
    kwargs=dict(root_path=output/'Patched',cache_dir=output/'cache',train_sources=['Generated'],augment=True)
    train=PatchedPIDDataset(split='train',**kwargs)
    valid=PatchedPIDDataset(split='valid',**kwargs)
    assert {s.split('/')[1] for s in train.ids}.isdisjoint({s.split('/')[1] for s in valid.ids})
    gpath=train.samples[0][1]
    graph=load_pid_graph(gpath,1500,1500)
    sample=train[0]
    assert sample[0].shape==(1,1,512,512)
    assert (sample[6][:,2:]>0).all()
    # Through-going line: both endpoints are outside the central patch.
    ns=[point_node('left',[100,500],'general',20),point_node('right',[2900,500],'valve',20)]
    es=[dict(source='left',target='right',label='non_solid',score=1.)]
    clipped,cut_edges=crop_graph(ns,es,[750,0,2250,1500])
    assert len(clipped)==2 and all(n['label']=='border' for n in clipped) and len(cut_edges)==1
    source=output/'line.png'
    Image.new('L',(3000,1500),255).save(source)
    write_graph(source.with_suffix('.graphml'),ns,es)
    manifest=patch_plan(source,source.with_suffix('.graphml'),output/'line_patches',resize=(3000,1500))
    truth_patches=[]
    for p in manifest['patches']:
        pn,pe=parse_graph(output/'line_patches'/p['graph'])
        truth_patches.append(dict(nodes=pn,edges=pe))
    mn,me=merge_graphs(manifest,truth_patches)
    assert len(mn)==2 and len(me)==1 and me[0]['label']=='non_solid',(mn,me)
    save_graph_json(output/'line_merged.json',mn,me)
    assert len(fuse(ns,1.,False)[0])==2,'NMS threshold 1 must terminate'
    metrics=PIDMetrics()
    metrics.add(dict(nodes=ns,edges=es),dict(nodes=ns,edges=es))
    result=metrics.compute()
    assert all(abs(result[k]-1)<1e-6 for k in ('symbol_mAP50','node_AP50','edge_mAP')),result
    missing=PIDMetrics()
    missing.add(dict(nodes=ns,edges=es),dict(nodes=ns,edges=[]))
    assert missing.compute()['edge_mAP']==0
    wrong=PIDMetrics()
    wrong.add(dict(nodes=ns,edges=es),dict(nodes=ns,edges=[dict(es[0],label='solid')]))
    assert wrong.compute()['edge_mAP']==0
    assert abs(ranked_ap([(.9,1),(.8,0),(.7,1)],3)-5/9)<1e-9
    # Empty and isolated-node annotations are legitimate detection examples.
    for name,targets in [('empty',[]),('isolated',ns)]:
        path=output/f'{name}.graphml'
        write_graph(path,targets,[])
        parsed=load_pid_graph(path,3000,1500)
        assert parsed[1].shape==(0,2) and len(parsed[0])==len(targets)
    config=load_config('configs/road_2D.yaml')
    config.MODEL.ENCODER.PRETRAINED=False
    model=build_model(config).to(device)
    criterion=SetCriterion(config,build_matcher(config),model).to(device)
    sample=train[0]
    x=sample[0].to(device)
    target={k:[sample[i].to(device)] for k,i in [('nodes',1),('edges',2),('node_classes',4),('edge_classes',5),('boxes',6)]}
    with torch.autocast(device.type,dtype=torch.float16,enabled=amp):
        h,out=model(x)
        losses=criterion(h,out,target)
    assert h.shape==(1,401,256) and out['pred_logits'].shape==(1,400,11)
    assert all(torch.isfinite(v) for v in losses.values())
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    for attempt in range(12):
        if attempt:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.float16,enabled=amp):
                h,out=model(x)
                losses=criterion(h,out,target)
        scaler.scale(losses['total']).backward()
        assert model.relation_embed.layers[-1].weight.grad is not None
        scaler.unscale_(optimizer)
        finite=all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(),.1)
        # GradScaler may legitimately skip initial updates while lowering its scale.
        scaler.step(optimizer)
        scaler.update()
        if finite:
            break
    assert finite,'AMP failed to find a finite gradient scale'
    for path in (output/'empty.graphml',output/'isolated.graphml'):
        g=load_pid_graph(path,3000,1500)
        t={k:[g[i].to(device)] for k,i in [('nodes',0),('edges',1),('node_classes',2),('edge_classes',3),('boxes',4)]}
        with torch.no_grad():
            assert all(torch.isfinite(v) for v in criterion(h,out,t).values())
    model.eval()
    criterion.eval()
    with torch.no_grad():
        h,out=model(x)
        first=criterion(h,out,target)
        second=criterion(h,out,target)
    assert all(torch.equal(first[k],second[k]) for k in first),'Validation must not randomly resample edges'
    decoded=relation_infer(h,out,model,400,1,map_=True,node_threshold=.05,edge_threshold=.5)
    pred=prediction_graph(decoded,0,(1500,1500))
    save_graph_json(output/'patch_prediction.json',pred['nodes'],pred['edges'])
    single=dict(patches=[dict(id='0',offset=[0,0],size=[1500,1500])],scale=[1,1])
    merged_nodes,merged_edges=merge_graphs(single,[pred])
    write_graph(output/'predicted_merged.graphml',merged_nodes,merged_edges)
    parse_graph(output/'predicted_merged.graphml')
    overlay(train.samples[0][0],pred['nodes'],pred['edges'],output/'prediction.png')
    report=dict(status='passed',device=str(device),amp=amp,model_tokens=list(h.shape),
                loss={k:float(v.detach()) for k,v in losses.items()},perfect_graph_metrics=result,
                synthetic_counts=[dict(nodes=len(n),edges=len(e)) for n,e in generated],
                template_count=len(library['templates']))
    (output/'test_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--templates',type=Path,default=Path('data/templates'))
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--amp',action='store_true',help='CUDA mixed precision plus scaler optimizer step')
    a=p.parse_args()
    output=a.output_dir or Path(tempfile.mkdtemp(prefix='pid-smoke-'))
    if a.amp and not a.device.startswith('cuda'):
        p.error('--amp requires a CUDA device')
    check(a.templates,output,torch.device(a.device),a.amp)
