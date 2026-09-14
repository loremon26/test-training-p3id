"""Relationformer decoding with bounded-memory pair scoring."""
import torch
from torchvision.ops import batched_nms
from box_ops_2D import box_cxcywh_to_xyxy
from prepare_pid2graph_paper_dataset import PID_NODE_CLASS_TO_ID, PID_EDGE_CLASS_TO_ID


@torch.no_grad()
def relation_infer(h, out, model, obj_token, rln_token, nms=False, map_=False,
                   node_threshold=.5, edge_threshold=.5):
    object_token=h[..., :obj_token, :]
    probs=out["pred_logits"].softmax(-1)
    scores,classes=probs[...,1:].max(-1)
    classes=classes+1
    values=[[] for _ in range(7)]
    for batch in range(h.shape[0]):
        ids=torch.where((scores[batch]>=node_threshold) & (probs[batch].argmax(-1)!=0))[0]
        if nms and len(ids):
            keep=batched_nms(box_cxcywh_to_xyxy(out["pred_nodes"][batch,ids]),
                             scores[batch,ids],classes[batch,ids],.9)
            ids=ids[keep]
        boxes=out["pred_nodes"][batch,ids].detach()
        pairs=torch.triu_indices(len(ids),len(ids),offset=1,device=h.device).T
        kept,edge_scores,edge_classes=[],[],[]
        for chunk in pairs.split(2048):
            if not len(chunk):
                continue
            a=object_token[batch,ids[chunk[:,0]]]
            b=object_token[batch,ids[chunk[:,1]]]
            forward,reverse=[a,b],[b,a]
            if rln_token:
                relation=h[batch,obj_token:obj_token+rln_token].expand(len(chunk),-1)
                forward.append(relation)
                reverse.append(relation)
            logits=(model.relation_embed(torch.cat(forward,1))+
                    model.relation_embed(torch.cat(reverse,1)))/2
            edge_probs=logits.softmax(-1)
            confidence,label=edge_probs[:,1:].max(-1)
            valid=(confidence>=edge_threshold) & (edge_probs.argmax(-1)!=0)
            kept.append(chunk[valid])
            edge_scores.append(confidence[valid])
            edge_classes.append(label[valid]+1)
        es=torch.cat(kept) if kept else torch.empty((0,2),dtype=torch.long,device=h.device)
        ec=torch.cat(edge_classes) if edge_classes else torch.empty(0,dtype=torch.long,device=h.device)
        conf=torch.cat(edge_scores) if edge_scores else torch.empty(0,device=h.device)
        values[0].append(boxes[:,:2])
        for bucket,value in zip(values[1:],(es,boxes,scores[batch,ids],classes[batch,ids],conf,ec)):
            bucket.append(value.cpu().numpy())
    return tuple(values) if map_ else tuple(values[:2])


def prediction_graph(decoded, index, size):
    """Convert normalized cxcywh model outputs to canonical patch-pixel graphs."""
    width,height=size
    boxes,scores,classes=decoded[2][index],decoded[3][index],decoded[4][index]
    node_names={v:k for k,v in PID_NODE_CLASS_TO_ID.items()}
    edge_names={v:k for k,v in PID_EDGE_CLASS_TO_ID.items()}
    nodes=[]
    for i,(b,score,cls) in enumerate(zip(boxes,scores,classes)):
        x,y,w,h=map(float,b)
        nodes.append(dict(id=str(i),label=node_names[int(cls)],score=float(score),
                          xmin=(x-w/2)*width,ymin=(y-h/2)*height,
                          xmax=(x+w/2)*width,ymax=(y+h/2)*height))
    edges=[dict(source=str(int(e[0])),target=str(int(e[1])),label=edge_names[int(c)],score=float(s))
           for e,s,c in zip(decoded[1][index],decoded[5][index],decoded[6][index])]
    return dict(schema_version=1,coordinate_space="pixels",nodes=nodes,edges=edges)
