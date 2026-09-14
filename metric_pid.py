"""III-E metrics. Reuse existing COCO-style node AP; A1 edges use gIoU assignment."""
from collections import Counter
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from metric_map import BBoxEvaluator
from box_ops_2D import generalized_box_iou
from pid_graph import box
from prepare_pid2graph_paper_dataset import PID_NODE_CLASS_TO_ID


def ranked_ap(records, positives):
    """Non-interpolated AP with missing positives in the recall denominator.

    This is sklearn's threshold-grouped precision/recall integration on the
    observed predictions, multiplied by achievable recall TP/(TP+FN).
    Missing edges contribute no invented prediction confidence.
    """
    if not positives:
        return None
    if not records:
        return 0.
    rows=np.array(sorted(records,key=lambda r:-r[0]),float)
    tp=np.cumsum(rows[:,1])
    end=np.r_[np.where(np.diff(rows[:,0])!=0)[0],len(rows)-1]
    recalls=tp[end]/positives
    precision=tp[end]/(end+1)
    return float(np.sum(np.diff(np.r_[0.,recalls])*precision))


class PIDMetrics:
    def __init__(self, include_borders=False, edge_class_mismatch_fn=False):
        names=[k for k,v in PID_NODE_CLASS_TO_ID.items() if v<=7]
        self.symbols=BBoxEvaluator(names,max_detections=100000,iou_threshold=.5)
        self.nodes=BBoxEvaluator(['node'],max_detections=100000,iou_threshold=.5)
        self.include_borders=include_borders
        self.edge_class_mismatch_fn=edge_class_mismatch_fn
        self.edge_records={'solid':[],'non_solid':[]}
        self.edge_positives=Counter()
        self.edge_gt=Counter()
        self.count=0

    def add(self, truth, prediction):
        tn,te=truth['nodes'],truth['edges']
        pn,pe=prediction['nodes'],prediction['edges']
        for evaluator,symbol_only in ((self.symbols,True),(self.nodes,False)):
            def keep(n):
                return PID_NODE_CLASS_TO_ID[n['label']]<=7 if symbol_only else (self.include_borders or n['label']!='border')
            gt=[n for n in tn if keep(n)]
            pred=[n for n in pn if keep(n)]
            evaluator.add([np.array([box(n) for n in pred]).reshape(-1,4)],
                [np.array([PID_NODE_CLASS_TO_ID[n['label']] if symbol_only else 1 for n in pred],int)],
                [np.array([n.get('score',1.) for n in pred])],
                [np.array([box(n) for n in gt]).reshape(-1,4)],
                [np.array([PID_NODE_CLASS_TO_ID[n['label']] if symbol_only else 1 for n in gt],int)])
        mapping={}
        if tn and pn:
            cost=-generalized_box_iou(torch.tensor(np.array([box(n) for n in tn])),
                                      torch.tensor(np.array([box(n) for n in pn]))).numpy()
            gt_idx,pred_idx=linear_sum_assignment(cost)
            mapping={pn[p]['id']:tn[t]['id'] for t,p in zip(gt_idx,pred_idx)}
        target={tuple(sorted((e['source'],e['target']))):e['label'] for e in te}
        seen,matched=set(),set()
        for e in sorted(pe,key=lambda e:-e.get('score',1.)):
            a,b=mapping.get(e['source']),mapping.get(e['target'])
            pair=tuple(sorted((a,b))) if a is not None and b is not None else None
            correct=pair in target and target[pair]==e['label'] and pair not in matched
            self.edge_records[e['label']].append((float(e.get('score',1.)),int(correct)))
            if pair in target:
                seen.add(pair)
            if correct:
                matched.add(pair)
                self.edge_positives[e['label']]+=1
        for pair,label in target.items():
            self.edge_gt[label]+=1
            # Literal A1 lines 14-17: wrong-type existing pairs are FP, not also FN.
            if pair not in (matched if self.edge_class_mismatch_fn else seen):
                self.edge_positives[label]+=1
        self.count+=1

    def compute(self):
        if not self.count:
            raise ValueError('No drawings evaluated')
        symbol_scores=self.symbols.eval()
        node_scores=self.nodes.eval()
        per_edge={c:ranked_ap(self.edge_records[c],self.edge_positives[c])
                  if self.edge_positives[c] else (0. if self.edge_gt[c] else None) for c in self.edge_records}
        present=[v for v in per_edge.values() if v is not None]
        symbol_ap=float(symbol_scores['AP_IoU_0.50_MaxDet_100000'])
        node_ap=float(node_scores['AP_IoU_0.50_MaxDet_100000'])
        return dict(symbol_mAP50=symbol_ap if symbol_ap>=0 else None,
                    node_AP50=node_ap if node_ap>=0 else None,
                    edge_mAP=float(np.mean(present)) if present else None,
                    edge_AP=per_edge,images=self.count,
                    include_borders=self.include_borders,edge_class_mismatch_fn=self.edge_class_mismatch_fn)
