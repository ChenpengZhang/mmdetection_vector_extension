from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np

from mmengine.config import ConfigDict
from mmengine.model import BaseModule, ModuleList
from torch import Tensor
from torch.nn.modules.utils import _pair
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from mmdet.registry import MODELS
from mmdet.structures.mask import mask_target

from mmcv.ops import Conv2d


@MODELS.register_module()
class OffsetHead(BaseModule):

    def __init__(self,
                 roi_feat_size=7,
                 in_channels=256,
                 num_convs=4,
                 num_fcs=2,
                 reg_num=2,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 offset_coordinate='rectangle',
                 reg_decoded_offset=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_offset: ConfigType = dict(type='MSELoss', loss_weight=1.0)
                 ):
        super(OffsetHead, self).__init__()
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.offset_coordinate = offset_coordinate
        self.reg_decoded_offset = reg_decoded_offset
        self.reg_num = reg_num
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # Build offset coder and offset loss calculator
        # self.offset_coder = MODELS.build(offset_coder)
        self.loss_offset = MODELS.build(loss_offset)

        # Build convolution layers
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            in_channels = (self.in_channels if i == 0 else self.conv_out_channels)
            self.convs.append(
                Conv2d(
                    in_channels,
                    self.conv_out_channels,
                    3,
                    padding=1))

        roi_feat_size = _pair(roi_feat_size)
        roi_feat_area = roi_feat_size[0] * roi_feat_size[1]
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            in_channels = (
                self.conv_out_channels *
                roi_feat_area if i == 0 else self.fc_out_channels)
            self.fcs.append(nn.Linear(in_channels, self.fc_out_channels))
        self.fc_offset = nn.Linear(self.fc_out_channels, self.reg_num)
        self.relu = nn.ReLU()

    def forward(self, x):

        if x.size(0) == 0:
            return x.new_empty(x.size(0), 2)
        for conv in self.convs:
            x = self.relu(conv(x))

        self.vis_featuremap = x.clone()

        x = x.view(x.size(0), -1)

        for fc in self.fcs:
            x = self.relu(fc(x))
        offset = self.fc_offset(x)

        return offset

    def _offset_target_single(self,
                              pos_proposals: Tensor,
                              pos_assigned_gt_inds: Tensor,
                              gt_offsets: List[Tensor],
                              cfg: ConfigDict) -> Tensor:
        device = pos_proposals.device
        num_pos = pos_proposals.size(0)

        pos_gt_offsets = []

        if num_pos > 0:
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
            for i in range(num_pos):
                gt_offset = gt_offsets[pos_assigned_gt_inds[i]]
                pos_gt_offsets.append(gt_offset.tolist())

            pos_gt_offsets = np.array(pos_gt_offsets)
            pos_gt_offsets = torch.from_numpy(np.stack(pos_gt_offsets)).float().to(device)

            if not self.reg_decoded_offset:
                offset_targets = self.offset_coder.encode(pos_proposals, pos_gt_offsets)
            else:
                offset_targets = pos_gt_offsets
        else:
            offset_targets = pos_proposals.new_zeros((0, 2))

        return offset_targets

    def get_targets(self, sampling_results,
                    batch_gt_instances,
                    rcnn_train_cfg,
                    concat=True):
        """
        这个函数的主要目的是为了将正样本和真值一一匹配起来,例如正样本可能有100个,
        但图中的物体只有10个,那么就需要做个匹配,将100个正样本对应到10个物体上,便于拿取.
        :param sampling_results:
        :param batch_gt_instances:
        :param rcnn_train_cfg:
        :param concat:
        :return:
        """
        # 提取出所有的正样本包围框
        pos_proposals = [res.pos_priors for res in sampling_results]
        # 每个正样本都有一个实际的图片中的标记物体与其关联,取出这个关联的ind(号码)
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        # 提取出所有gt_offset
        gt_offsets = [res.offsets for res in batch_gt_instances]
        # 将设置做成一个列表以便于使用map函数
        rcnn_train_cfgs = [rcnn_train_cfg for _ in range(len(pos_proposals))]
        offset_targets = map(
            self._offset_target_single,
            pos_proposals,
            pos_assigned_gt_inds,
            gt_offsets,
            rcnn_train_cfgs)
        # 如果指定要拼接起来的话
        if concat:
            offset_targets = torch.cat(list(offset_targets), dim=0)
        if self.reg_num == 2:
            return offset_targets
        elif self.reg_num == 3:
            length = offset_targets[:, 0]
            angle = offset_targets[:, 1]
            angle_cos = torch.cos(angle)
            angle_sin = torch.sin(angle)
            offset_targets = torch.stack([length, angle_cos, angle_sin], dim=-1)
            return offset_targets
        else:
            raise (RuntimeError("error reg_num value: ", self.reg_num))

    def loss_and_target(self, offset_preds: Tensor,
                        sampling_results: List[SamplingResult],
                        batch_gt_instances: InstanceList,
                        rcnn_train_cfg: ConfigDict) -> dict:
        """

        :param offset_preds: offset的预测结果
        :param sampling_results: 采样的完整结果,包含了正样本,负样本,以及一些额外的指示来指示这个正样本是gt还是rpn生成的.
        :param batch_gt_instances: 选出的比较好的gt,包含了gt_mask, gt_label等.
        :param rcnn_train_cfg: 训练参数字典.
        :return:
        """
        offset_targets = self.get_targets(
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        loss = dict()
        if offset_preds.size(0) == 0:
            # 如果程序认为这个地方没有框
            loss_offset = offset_preds.sum() * 0
        else:
            # 调用loss_offset的前向函数来计算loss
            loss_offset = self.loss_offset(offset_preds, offset_targets)
        loss['loss_offset'] = loss_offset
        return dict(loss_offset=loss, offset_targets=offset_targets)

