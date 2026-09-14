"""Validation loss, P&ID metrics and qualitative overlays."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from inference import relation_infer, prediction_graph
from metric_pid import PIDMetrics
from pid_graph import overlay


def target_graph(batch,index,size):
    from prepare_pid2graph_paper_dataset import PID_NODE_CLASS_TO_ID,PID_EDGE_CLASS_TO_ID
    names={v:k for k,v in PID_NODE_CLASS_TO_ID.items()}
    enames={v:k for k,v in PID_EDGE_CLASS_TO_ID.items()}
    width,height=size
    nodes=[]
    for i,(b,c) in enumerate(zip(batch[6][index].tolist(),batch[4][index].tolist())):
        x,y,w,h=b
        nodes.append(dict(id=str(i),label=names[c],score=1.,xmin=(x-w/2)*width,
                          ymin=(y-h/2)*height,xmax=(x+w/2)*width,ymax=(y+h/2)*height))
    edges=[dict(source=str(u),target=str(v),label=enames[c],score=1.)
           for (u,v),c in zip(batch[2][index].tolist(),batch[5][index].tolist())]
    return dict(nodes=nodes,edges=edges)


@torch.no_grad()
def evaluate(model,criterion,loader,config,device,preview_dir=None,amp=False):
    from trainer import batch_target
    model.eval()
    criterion.eval()
    metrics=PIDMetrics()
    sums={}
    count=0
    for batch in loader:
        images=batch[0].to(device)
        with torch.autocast(device.type,dtype=torch.float16,enabled=amp):
            h,out=model(images)
            losses=criterion(h,out,batch_target(batch,device))
        if not torch.isfinite(losses['total']):
            raise FloatingPointError('Non-finite validation loss')
        for key,value in losses.items():
            sums[key]=sums.get(key,0.)+float(value)*len(images)
        relation_model=getattr(model,'module',model)
        decoded=relation_infer(h,out,relation_model,config.MODEL.DECODER.OBJ_TOKEN,config.MODEL.DECODER.RLN_TOKEN,
                map_=True,node_threshold=config.INFERENCE.METRIC_NODE_THRESHOLD,
                edge_threshold=config.INFERENCE.METRIC_EDGE_THRESHOLD)
        size=(images.shape[-1],images.shape[-2])
        for i in range(len(images)):
            prediction=prediction_graph(decoded,i,size)
            truth=target_graph(batch,i,size)
            metrics.add(truth,prediction)
            if count==0 and i==0 and preview_dir is not None:
                preview_dir=Path(preview_dir)
                image=Image.fromarray((batch[0][i,0].numpy()*255).astype(np.uint8))
                overlay(image,truth['nodes'],truth['edges'],preview_dir/'truth.png')
                # Render only display-confidence predictions, keep low scores for AP.
                shown=[n for n in prediction['nodes'] if n['score']>=config.INFERENCE.NODE_THRESHOLD]
                ids={n['id'] for n in shown}
                es=[e for e in prediction['edges'] if e['source'] in ids and e['target'] in ids and e['score']>=config.INFERENCE.EDGE_THRESHOLD]
                overlay(image,shown,es,preview_dir/'prediction.png',True)
        count+=len(images)
    return dict(loss={k:v/count for k,v in sums.items()},metrics=metrics.compute())
