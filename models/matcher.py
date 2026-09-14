# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from box_ops_2D import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, config):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_nodes = config.MODEL.MATCHER.C_NODE
        self.cost_class = config.MODEL.MATCHER.C_CLASS
        self.cost_box = getattr(config.MODEL.MATCHER, 'C_BOX', 2)

    @torch.no_grad()
    def forward(self, outputs, targets):
        """[summary]

        Args:
            outputs ([type]): [description]
            targets ([type]): [description]

        Returns:
            [type]: [description]
        """
        bs, num_queries = outputs['pred_nodes'].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_nodes = outputs['pred_nodes'][...,:2].flatten(0, 1)  # [batch_size * num_queries, 2]

        # Also concat the target labels and boxes
        tgt_nodes = torch.cat([v for v in targets['nodes']])

        # Compute the L1 cost between nodes
        cost_nodes = torch.cdist(out_nodes, tgt_nodes, p=1)

        # Compute the cls cost
        tgt_ids = torch.cat([v.to(out_nodes.device) for v in targets["node_classes"]])
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        cost_class = -out_prob[:, tgt_ids]

        # Final cost matrix
        C = self.cost_nodes * cost_nodes + self.cost_class * cost_class
        if 'boxes' in targets:
            predicted_boxes = outputs['pred_nodes'].flatten(0, 1)
            target_boxes = torch.cat(targets['boxes'])
            C += self.cost_box * (torch.cdist(predicted_boxes, target_boxes, p=1)
                 - generalized_box_iou(box_cxcywh_to_xyxy(predicted_boxes),
                                       box_cxcywh_to_xyxy(target_boxes)))
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v) for v in targets['nodes']]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(config):
    return HungarianMatcher(config)
