"""Merge pixel-coordinate patch predictions using III-B confidence/NMS/WBF."""
import argparse
import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from metric_map import box_iou_2d_np
from pid_graph import box, center, set_box, overlay, save_graph_json, write_graph, transform_graph
from prepare_pid2graph_paper_dataset import PID_NODE_CLASS_TO_ID, PID_EDGE_CLASS_TO_ID


def fuse(nodes, threshold, weighted):
    """Class-aware clustering; suppressed node IDs still map to the survivor."""
    remaining = sorted(range(len(nodes)), key=lambda i: -nodes[i]['score'])
    fused, mapping = [], {}
    all_boxes = np.array([box(n) for n in nodes]).reshape(-1,4)
    # ponytail: quadratic per drawing; add a spatial index if dense plans dominate runtime.
    while remaining:
        i = remaining[0]
        overlaps = box_iou_2d_np(all_boxes[i:i+1],all_boxes[remaining])[0]
        group = [j for j,v in zip(remaining,overlaps) if j==i or (nodes[j]['label']==nodes[i]['label'] and v>=threshold)]
        chosen = set(group)
        remaining = [j for j in remaining if j not in chosen]
        n = dict(nodes[i], id=str(len(fused)))
        if weighted:
            weights = [max(nodes[j]['score'],1e-6) for j in group]
            n = set_box(n, np.average(all_boxes[group],axis=0,weights=weights))
        for j in group:
            mapping[nodes[j]['id']] = n['id']
        fused.append(n)
    return fused, mapping


def remap_edges(edges, mapping):
    result = {}
    for e in edges:
        a, b = mapping.get(e['source']), mapping.get(e['target'])
        if a is None or b is None or a == b:
            continue
        key = tuple(sorted((a,b)))
        if key not in result or e['score'] > result[key]['score']:
            result[key] = dict(e,source=key[0],target=key[1])
    return list(result.values())


def reconnect_borders(nodes, edges, tolerance=8):
    """Split overlapping edge segments at border points, then contract degree-two cuts.

    Geometry/thresholds are an explicit approximation: the paper does not specify
    border reconnection. Unresolved borders remain in the exported graph.
    """
    lookup = {n['id']: n for n in nodes}
    identity = {k:k for k in lookup}
    borders = [n for n in nodes if n['label']=='border']
    for n in borders:
        candidates = [m for m in nodes if m['id'] != n['id']
                      and (m['label']!='border' or int(m['id']) < int(n['id']))
                      and np.linalg.norm(center(m)-center(n)) <= tolerance]
        if candidates:
            m = min(candidates,key=lambda m: np.linalg.norm(center(m)-center(n)))
            identity[n['id']] = identity[m['id']]
    edges = remap_edges(edges,identity)
    nodes = [n for n in nodes if identity[n['id']]==n['id']]
    lookup = {n['id']: n for n in nodes}
    borders = [n for n in nodes if n['label']=='border']
    incident = {n['id']:set() for n in borders}
    for e in edges:
        for k in (e['source'],e['target']):
            if k in incident:
                incident[k].add(e['label'])
    split = []
    for e in edges:
        a, b = center(lookup[e['source']]), center(lookup[e['target']])
        delta = b-a
        length2 = float(delta@delta)
        cuts = [(0.,e['source']),(1.,e['target'])]
        if length2 > 1e-9:
            for n in borders:
                if n['id'] in (e['source'],e['target']) or e['label'] not in incident[n['id']]:
                    continue
                xy = center(n)
                t = float((xy-a)@delta/length2)
                if 1e-6 < t < 1-1e-6 and np.linalg.norm(a+t*delta-xy)<=tolerance:
                    cuts.append((t,n['id']))
        cuts.sort()
        split.extend(dict(e,source=u[1],target=v[1]) for u,v in zip(cuts,cuts[1:]))
    edges = remap_edges(split,{k:k for k in lookup})
    for n in borders:
        adjacent = [e for e in edges if n['id'] in (e['source'],e['target'])]
        if len(adjacent)==2 and adjacent[0]['label']==adjacent[1]['label']:
            ends = [e['target'] if e['source']==n['id'] else e['source'] for e in adjacent]
            edges = [e for e in edges if e not in adjacent]
            edges.append(dict(source=ends[0],target=ends[1],label=adjacent[0]['label'],
                              score=min(e['score'] for e in adjacent)))
            lookup.pop(n['id'])
    edges = remap_edges(edges,{k:k for k in lookup})
    active = {e[k] for e in edges for k in ('source','target')}
    return [n for n in lookup.values() if n['id'] in active], edges


def merge_graphs(manifest, predictions, threshold=.15, nms_iou=.8, wbf_iou=.4, border_tolerance=8):
    if not 0 < wbf_iou <= nms_iou <= 1 or not 0 <= threshold <= 1:
        raise ValueError('Require 0 < WBF IoU <= NMS IoU <= 1 and confidence in [0,1]')
    if len(predictions) != len(manifest['patches']):
        raise ValueError('Missing patch predictions')
    scale=np.asarray(manifest['scale'],float)
    if scale.shape!=(2,) or not np.isfinite(scale).all() or (scale<=0).any() or border_tolerance<0:
        raise ValueError('Invalid scale or border tolerance')
    if len({p['id'] for p in manifest['patches']})!=len(manifest['patches']):
        raise ValueError('Duplicate patch IDs')
    nodes, edges = [], []
    for patch, prediction in zip(manifest['patches'],predictions):
        if prediction.get('coordinate_space','pixels') != 'pixels':
            raise ValueError('Merge requires patch-pixel boxes, not normalized model boxes')
        offset = np.tile(patch['offset'],2)
        width,height = patch['size']
        if min(width,height)<=0 or width!=height or not np.isfinite(offset).all():
            raise ValueError('Expected positive square patches and finite offsets')
        ids = set()
        for n in prediction['nodes']:
            if n['id'] in ids:
                raise ValueError('Duplicate predicted node ID')
            ids.add(n['id'])
            confidence=float(n.get('score',1.))
            if not math.isfinite(confidence) or not 0<=confidence<=1 or n['label'] not in PID_NODE_CLASS_TO_ID:
                raise ValueError('Invalid predicted node class/confidence')
            bounds = box(n)
            if not np.isfinite(bounds).all() or np.any(bounds[2:]<=bounds[:2]):
                raise ValueError('Invalid predicted bounding box')
            d = max(0.,min(bounds[0],bounds[1],width-bounds[2],height-bounds[3]))
            score = confidence - .4*math.exp(-3*abs(2*d/width))
            if score >= threshold:
                nodes.append(set_box(dict(n,id=f"{patch['id']}:{n['id']}",score=score),bounds+offset))
        for e in prediction['edges']:
            confidence=float(e.get('score',1.))
            if not math.isfinite(confidence) or not 0<=confidence<=1 or e['label'] not in PID_EDGE_CLASS_TO_ID:
                raise ValueError('Invalid predicted edge class/confidence')
            if e['source'] not in ids or e['target'] not in ids:
                raise ValueError('Dangling predicted edge')
            edges.append(dict(e,source=f"{patch['id']}:{e['source']}",
                              target=f"{patch['id']}:{e['target']}",score=float(e.get('score',1.))))
    for iou,weighted in ((nms_iou,False),(wbf_iou,True)):
        nodes,mapping = fuse(nodes,iou,weighted)
        edges = remap_edges(edges,mapping)
    nodes,edges = reconnect_borders(nodes,edges,border_tolerance)
    sx,sy = manifest['scale']
    nodes,edges = transform_graph(nodes,edges,[[1/sx,0,0],[0,1/sy,0]])
    return nodes,edges


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='Output stem, without extension')
    p.add_argument('--threshold',type=float,default=.15)
    p.add_argument('--nms-iou',type=float,default=.8)
    p.add_argument('--wbf-iou',type=float,default=.4)
    p.add_argument('--border-tolerance',type=float,default=8)
    p.add_argument('--image',type=Path,help='Override original image location after moving data')
    a=p.parse_args()
    manifest=json.loads(a.manifest.read_text())
    predictions=[json.loads((a.predictions/f"{p['id']}.json").read_text()) for p in manifest['patches']]
    nodes,edges=merge_graphs(manifest,predictions,a.threshold,a.nms_iou,a.wbf_iou,a.border_tolerance)
    save_graph_json(a.output.with_suffix('.json'),nodes,edges,manifest=str(a.manifest))
    write_graph(a.output.with_suffix('.graphml'),nodes,edges)
    overlay(a.image or manifest['image'],nodes,edges,a.output.with_suffix('.png'),confidence=True)
    print(f'{len(nodes)} nodes, {len(edges)} edges -> {a.output}')


if __name__=='__main__':
    main()
