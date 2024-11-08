import numpy as np
import torch

from mmdet.registry import TASK_UTILS
from .base_bbox_coder import BaseBBoxCoder


@TASK_UTILS.register_module()
class DeltaXYOffsetCoder(BaseBBoxCoder):
    def __init__(self,
                 target_means=(0., 0.),
                 target_stds=(0.5, 0.5)):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_offsets):
        assert bboxes.size(0) == gt_offsets.size(0)
        assert gt_offsets.size(-1) == 2
        encoded_offsets = self.offset2delta(bboxes, gt_offsets)
        return encoded_offsets

    def decode(self,
               bboxes,
               pred_offsets,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        assert pred_offsets.size(0) == bboxes.size(0)
        decoded_offsets = self.delta2offset(bboxes, pred_offsets, max_shape)

        return decoded_offsets

    def offset2delta(self, proposals, gt):
        assert proposals.size()[0] == gt.size()[0]

        proposals = proposals.float()
        gt = gt.float()
        pw = proposals[..., 2] - proposals[..., 0]
        ph = proposals[..., 3] - proposals[..., 1]

        gx = gt[..., 0]
        gy = gt[..., 1]

        dx = gx / pw
        dy = gy / ph
        deltas = torch.stack([dx, dy], dim=-1)

        means = deltas.new_tensor(self.means).unsqueeze(0)
        stds = deltas.new_tensor(self.stds).unsqueeze(0)
        deltas = deltas.sub_(means).div_(stds)

        return deltas

    def delta2offset(self, rois, deltas, max_shape=None):
        means = deltas.new_tensor(self.means).repeat(1, deltas.size(1) // 2)
        stds = deltas.new_tensor(self.stds).repeat(1, deltas.size(1) // 2)
        denorm_deltas = deltas * stds + means
        dx = denorm_deltas[:, 0::2]
        dy = denorm_deltas[:, 1::2]
        # Compute width/height of each roi
        pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dx)
        ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dy)
        # Use network energy to shift the center of each roi
        gx = pw * dx
        gy = ph * dy
        if max_shape is not None:
            gx = gx.clamp(min=-max_shape[1], max=max_shape[1])
            gy = gy.clamp(min=-max_shape[0], max=max_shape[0])
        bboxes = torch.stack([gx, gy], dim=-1).view_as(deltas)
        return bboxes
