"""Infer an entire P&ID through overlapping patches, or a single patch."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from train import DEFAULT_CONFIG, dict2obj, load_config
from models import build_model
from inference import relation_infer, prediction_graph
from pid_graph import patch_plan, save_graph_json, write_graph, overlay
from scripts.merge_pid_graph import merge_graphs


def load_predictor(checkpoint_path, config_path=None, device='cuda'):
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; select --device cpu or enable a GPU')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    config=load_config(config_path) if config_path else (
        dict2obj(checkpoint['config']) if 'config' in checkpoint else load_config(DEFAULT_CONFIG))
    config.MODEL.ENCODER.PRETRAINED=False
    model=build_model(config).to(device)
    model.load_state_dict(checkpoint['net'],strict=True)
    model.eval()
    return model,config,device


@torch.no_grad()
def predict_patches(model, config, paths, device, node_threshold=None, edge_threshold=None):
    tensors,sizes=[],[]
    for path in paths:
        with Image.open(path) as image:
            sizes.append(image.size)
            array=np.array(image.convert('L').resize(tuple(config.DATA.IMG_SIZE),Image.Resampling.BILINEAR),dtype=np.float32)/255
        tensors.append(torch.from_numpy(array)[None])
    with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda' and config.TRAIN.AMP):
        h,out=model(torch.stack(tensors).to(device))
    decoded=relation_infer(h,out,model,config.MODEL.DECODER.OBJ_TOKEN,config.MODEL.DECODER.RLN_TOKEN,
            nms=config.INFERENCE.NMS,map_=True,
            node_threshold=config.INFERENCE.NODE_THRESHOLD if node_threshold is None else node_threshold,
            edge_threshold=config.INFERENCE.EDGE_THRESHOLD if edge_threshold is None else edge_threshold)
    return [prediction_graph(decoded,i,size) for i,size in enumerate(sizes)]


def predict_plan(model, config, image, output_dir, device, patch_size=1500, stride=750,
                 resize=(7000,4500), batch_size=2, node_threshold=None, edge_threshold=None,
                 merge_threshold=.15, nms_iou=.8, wbf_iou=.4, border_tolerance=8):
    output_dir=Path(output_dir)
    manifest=patch_plan(image,None,output_dir/'patches',patch_size,stride,resize)
    predictions=[]
    for start in range(0,len(manifest['patches']),batch_size):
        group=manifest['patches'][start:start+batch_size]
        graphs=predict_patches(model,config,[output_dir/'patches'/p['image'] for p in group],device,
                              node_threshold,edge_threshold)
        predictions.extend(graphs)
        for patch,graph in zip(group,graphs):
            save_graph_json(output_dir/'predictions'/f"{patch['id']}.json",graph['nodes'],graph['edges'])
    nodes,edges=merge_graphs(manifest,predictions,merge_threshold,nms_iou,wbf_iou,border_tolerance)
    save_graph_json(output_dir/'graph.json',nodes,edges,image=str(Path(image).resolve()),manifest='patches/manifest.json')
    write_graph(output_dir/'graph.graphml',nodes,edges)
    overlay(image,nodes,edges,output_dir/'prediction.png',confidence=True)
    return dict(nodes=nodes,edges=edges)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('image',type=Path)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--single-patch',action='store_true')
    p.add_argument('--patch-size',type=int,default=1500)
    p.add_argument('--stride',type=int,default=750)
    p.add_argument('--resize',nargs=2,type=int,default=[7000,4500],metavar=('WIDTH','HEIGHT'))
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--node-threshold',type=float)
    p.add_argument('--edge-threshold',type=float)
    p.add_argument('--merge-threshold',type=float,default=.15)
    p.add_argument('--nms-iou',type=float,default=.8)
    p.add_argument('--wbf-iou',type=float,default=.4)
    p.add_argument('--border-tolerance',type=float,default=8)
    a=p.parse_args()
    if a.batch_size<1:
        p.error('--batch-size must be positive')
    model,config,device=load_predictor(a.checkpoint,a.config,a.device)
    if a.single_patch:
        graph=predict_patches(model,config,[a.image],device,a.node_threshold,a.edge_threshold)[0]
        save_graph_json(a.output_dir/'graph.json',graph['nodes'],graph['edges'],image=str(a.image.resolve()))
        write_graph(a.output_dir/'graph.graphml',graph['nodes'],graph['edges'])
        overlay(a.image,graph['nodes'],graph['edges'],a.output_dir/'prediction.png',True)
    else:
        graph=predict_plan(model,config,a.image,a.output_dir,device,a.patch_size,a.stride,a.resize,a.batch_size,
                           a.node_threshold,a.edge_threshold,a.merge_threshold,a.nms_iou,a.wbf_iou,a.border_tolerance)
    print(f"{len(graph['nodes'])} nodes, {len(graph['edges'])} edges -> {a.output_dir}")


if __name__=='__main__':
    main()
