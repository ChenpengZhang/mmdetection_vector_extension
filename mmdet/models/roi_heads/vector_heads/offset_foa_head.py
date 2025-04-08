from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np
import math

import torch.nn.functional as F
import torch.nn.init as init

from mmengine.config import ConfigDict
from mmengine.model import BaseModule, ModuleList
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn.modules.utils import _pair
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.mask import mask_target
from mmcv.ops import Conv2d


@MODELS.register_module()
class OffsetFOAHead(BaseModule):
    def __init__(self,
                 roi_feat_size=7,
                 in_channels=256,
                 num_convs=4,
                 num_fcs=2,
                 reg_num=2,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 expand_feature_num=4,
                 share_expand_fc=False,
                 rotations=[0, 90, 180, 270],
                 offset_coordinate='rectangle',
                 offset_coder=dict(
                    type='DeltaXYOffsetCoder',
                    target_means=[0.0, 0.0],
                    target_stds=[0.5, 0.5]),
                 reg_decoded_offset=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_offset=dict(type='MSELoss', loss_weight=1.0)):
        super(OffsetFOAHead, self).__init__()
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.offset_coordinate = offset_coordinate
        self.reg_decoded_offset = reg_decoded_offset
        self.reg_num = reg_num
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        # expand_feature_num is the branch numbers
        self.expand_feature_num = expand_feature_num
        self.share_expand_fc = share_expand_fc

        # the rotation angle of feature transformation
        self.rotations = rotations
        self.flips = ['h', 'v']

        # define the conv and fc operations
        self.expand_convs = nn.ModuleList()
        for _ in range(self.expand_feature_num):
            convs = nn.ModuleList()
            for i in range(num_convs):
                in_channels = self.in_channels if i == 0 else self.conv_out_channels

                conv = Conv2d(in_channels, self.conv_out_channels, 3, padding=1)

                # 初始化 Conv2d 权重
                init.xavier_uniform_(conv.weight)

                # 如果有偏置，初始化为零
                if conv.bias is not None:
                    init.zeros_(conv.bias)

                convs.append(conv)

            self.expand_convs.append(convs)

        roi_feat_size = _pair(roi_feat_size)
        roi_feat_area = roi_feat_size[0] * roi_feat_size[1]
        if not self.share_expand_fc:
            self.expand_fcs = nn.ModuleList()
            for _ in range(self.expand_feature_num):
                fcs = nn.ModuleList()
                for i in range(num_fcs):
                    in_channels = (
                        self.conv_out_channels *
                        roi_feat_area if i == 0 else self.fc_out_channels)
                    fcs.append(nn.Linear(in_channels, self.fc_out_channels))
                self.expand_fcs.append(fcs)
            self.expand_fc_offsets = nn.ModuleList()
            for _ in range(self.expand_feature_num):
                fc_offset = nn.Linear(self.fc_out_channels, self.reg_num)
                self.expand_fc_offsets.append(fc_offset)
        else:
            self.fcs = nn.ModuleList()
            for i in range(num_fcs):
                in_channels = (
                    self.conv_out_channels *
                    roi_feat_area if i == 0 else self.fc_out_channels)
                self.fcs.append(nn.Linear(in_channels, self.fc_out_channels))

            self.fc_offset = nn.Linear(self.fc_out_channels, self.reg_num)

        self.relu = nn.ReLU()
        self.offset_coder = TASK_UTILS.build(offset_coder)
        self.loss_offset = MODELS.build(loss_offset)

    def forward(self, x):
        if x.size(0) == 0:
            return x.new_empty(x.size(0), 2 * self.expand_feature_num)
        input_feature = x.clone()
        offsets = []
        for idx in range(self.expand_feature_num):
            x = self.expand_feature(input_feature, idx)
            convs = self.expand_convs[idx]
            for conv in convs:
                x = self.relu(conv(x))

            x = x.view(x.size(0), -1)
            # share the fully connected parameters
            if not self.share_expand_fc:
                fcs = self.expand_fcs[idx]
                for fc in fcs:
                    x = self.relu(fc(x))
                fc_offset = self.expand_fc_offsets[idx]
                offset = fc_offset(x)
            else:
                for fc in self.fcs:
                    x = self.relu(fc(x))
                offset = self.fc_offset(x)

            offsets.append(offset)

        offsets = torch.cat(offsets, 0)
        return offsets

    def expand_feature(self, feature, operation_idx):
        """rotate the feature by operation index

        Args:
            feature (torch.Tensor): input feature map
            operation_idx (int): operation index -> rotation angle

        Returns:
            torch.Tensor: rotated feature
        """
        if operation_idx < 4:
            # rotate feature map
            rotate_angle = self.rotations[operation_idx]
            theta = torch.zeros((feature.size()[0], 2, 3), requires_grad=False, device=feature.device)

            with torch.no_grad():
                # counterclockwise
                angle = rotate_angle * math.pi / 180.0

                theta[:, 0, 0] = torch.tensor(math.cos(angle), requires_grad=False, device=feature.device)
                theta[:, 0, 1] = torch.tensor(math.sin(-angle), requires_grad=False, device=feature.device)
                theta[:, 1, 0] = torch.tensor(math.sin(angle), requires_grad=False, device=feature.device)
                theta[:, 1, 1] = torch.tensor(math.cos(angle), requires_grad=False, device=feature.device)

            grid = F.affine_grid(theta, feature.size())
            transformed_feature = F.grid_sample(feature, grid).to(feature.device)

        elif operation_idx >= 4 and operation_idx < 8:
            # rotate and flip feature map
            raise NotImplementedError
        else:
            raise NotImplementedError

        return transformed_feature

    def offset_coordinate_transform(self, offset, transform_flag='xy2la'):
        """transform the coordinate of offsets

        Args:
            offset (list): list of offset
            transform_flag (str, optional): flag of transform. Defaults to 'xy2la'.

        Returns:
            list: transformed offsets
        """
        if transform_flag == 'xy2la':
            offset_x, offset_y = offset
            length = math.sqrt(offset_x ** 2 + offset_y ** 2)
            angle = math.atan2(offset_y, offset_x)
            offset = [length, angle]
        elif transform_flag == 'la2xy':
            length, angle = offset
            offset_x = length * math.cos(angle)
            offset_y = length * math.sin(angle)
            offset = [offset_x, offset_y]
        else:
            raise NotImplementedError

        return offset

    def offset_rotate(self, offset, rotate_angle):
        """rotate the offset

        Args:
            offset (np.array): input offset
            rotate_angle (int): rotation angle

        Returns:
            np.array: rotated offset
        """
        offset = self.offset_coordinate_transform(offset, transform_flag='xy2la')
        # counterclockwise
        offset = [offset[0], offset[1] - rotate_angle * math.pi / 180.0]
        offset = self.offset_coordinate_transform(offset, transform_flag='la2xy')

        return offset

    def expand_gt_offset(self, gt_offset, operation_idx):
        """rotate the ground truth of offset

        Args:
            gt_offset (np.array): offset ground truth
            operation_idx (int): operation index

        Returns:
            np.array: rotated offset
        """
        if operation_idx < 4:
            # rotate feature map
            rotate_angle = self.rotations[operation_idx]
            transformed_offset = self.offset_rotate(gt_offset, rotate_angle)
        elif operation_idx >= 4 and operation_idx < 8:
            # rotate and flip feature map
            raise NotImplementedError
        else:
            raise NotImplementedError

        return transformed_offset

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

    def get_targets(self,
                    sampling_results,
                    batch_gt_instances,
                    rcnn_train_cfg,
                    concat=True):
        """
        这个函数的主要目的是为了将正样本和真值一一匹配起来,例如正样本可能有100个,
        但图中的物体只有10个,那么就需要做个匹配,将100个正样本对应到10个物体上,便于拿取.
        在foa模块中，还额外在此步骤加入了正样本的旋转.
        :param sampling_results:
        :param batch_gt_instances:
        :param rcnn_train_cfg:
        :param concat:
        :return:
        """
        pos_proposals = [res.pos_priors for res in sampling_results]
        # 每个正样本都有一个实际的图片中的标记物体与其关联,取出这个关联的ind(号码)
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        # 提取出所有gt_offset
        gt_offsets = [res.offsets for res in batch_gt_instances]
        # 将设置做成一个列表以便于使用map函数
        rcnn_train_cfgs = [rcnn_train_cfg for _ in range(len(pos_proposals))]
        expand_offset_targets = []
        for idx in range(self.expand_feature_num):
            idxs = [idx for _ in range(len(pos_proposals))]
            offset_targets = map(
                self._offset_target_single,
                pos_proposals,
                pos_assigned_gt_inds,
                gt_offsets,
                rcnn_train_cfgs,
                idxs)
            if concat:
                offset_targets = torch.cat(list(offset_targets), dim=0)
            expand_offset_targets.append(offset_targets)
        expand_offset_targets = torch.cat(expand_offset_targets, 0)
        return expand_offset_targets

    def _offset_target_single(self,
                              pos_proposals: Tensor,
                              pos_assigned_gt_inds: Tensor,
                              gt_offsets: List[Tensor],
                              cfg: ConfigDict,
                              operation_idx: int) -> Tensor:
        device = pos_proposals.device
        num_pos = pos_proposals.size(0)

        pos_gt_offsets = []

        if num_pos > 0:
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
            for i in range(num_pos):
                gt_offset = gt_offsets[pos_assigned_gt_inds[i]].tolist()
                gt_offset = self.expand_gt_offset(gt_offset, operation_idx=operation_idx)
                pos_gt_offsets.append(gt_offset)

            pos_gt_offsets = np.array(pos_gt_offsets)
            pos_gt_offsets = torch.from_numpy(np.stack(pos_gt_offsets)).float().to(device)

            if not self.reg_decoded_offset:
                if self.rotations[operation_idx] == 90 or self.rotations[operation_idx] == 270:
                    # if rotation angle is 90 or 270, the position of x and y need to be exchange
                    offset_targets = self.offset_coder.encode(pos_proposals, pos_gt_offsets[:, [1, 0]])
                    offset_targets = offset_targets[:, [1, 0]]
                else:
                    offset_targets = self.offset_coder.encode(pos_proposals, pos_gt_offsets)
            else:
                offset_targets = pos_gt_offsets
        else:
            offset_targets = pos_proposals.new_zeros((0, 2))

        return offset_targets

    def offset_fusion(self, offset_pred, model='max'):
        """Fuse the predicted offsets in inference stage

        Args:
            offset_pred (torch.Tensor): predicted offsets
            model (str, optional): fusion model. Defaults to 'max'. Max -> keep the max of offsets, Mean -> keep the mean value of offsets.

        Returns:
            np.array: fused offsets
        """
        split_offsets = offset_pred.split(int(offset_pred.shape[0] / self.expand_feature_num), dim=0)
        main_offsets = split_offsets[0]
        if model == 'mean':
            # Mean model for offset fusion
            offset_values = 0
            for idx in range(self.expand_feature_num):
                # 1. processing the rotation, rotation angle in (90, 270) -> switch the position of (x, y)
                if self.rotations[idx] == 90 or self.rotations[idx] == 270:
                    current_offsets = split_offsets[idx][:, [1, 0]]
                elif self.rotations[idx] == 0 or self.rotations[idx] == 180:
                    current_offsets = split_offsets[idx]
                else:
                    raise NotImplementedError(
                        f"rotation angle: {self.rotations[idx]} (self.rotations = {self.rotations})")

                offset_values += torch.abs(current_offsets)
            offset_values /= 1
        elif model == 'max':
            # Max model for offset fusion
            if self.expand_feature_num == 2 and self.rotations == [0, 180]:
                offset_value_x = torch.cat([split_offsets[0][:, 0].unsqueeze(dim=1),
                                            split_offsets[1][:, 0].unsqueeze(dim=1)], dim=1)
                offset_value_y = torch.cat([split_offsets[0][:, 1].unsqueeze(dim=1),
                                            split_offsets[1][:, 1].unsqueeze(dim=1)], dim=1)
            elif self.expand_feature_num == 2 and self.rotations == [0, 90]:
                offset_value_x = torch.cat([split_offsets[0][:, 0].unsqueeze(dim=1),
                                            split_offsets[1][:, 1].unsqueeze(dim=1)], dim=1)
                offset_value_y = torch.cat([split_offsets[0][:, 1].unsqueeze(dim=1),
                                            split_offsets[1][:, 0].unsqueeze(dim=1)], dim=1)
            elif self.expand_feature_num == 3 and self.rotations == [0, 90, 180]:
                offset_value_x = torch.cat([split_offsets[0][:, 0].unsqueeze(dim=1),
                                            split_offsets[1][:, 1].unsqueeze(dim=1),
                                            split_offsets[2][:, 0].unsqueeze(dim=1)], dim=1)
                offset_value_y = torch.cat([split_offsets[0][:, 1].unsqueeze(dim=1),
                                            split_offsets[1][:, 0].unsqueeze(dim=1),
                                            split_offsets[2][:, 1].unsqueeze(dim=1)], dim=1)
            elif self.expand_feature_num == 4:
                offset_value_x = torch.cat([split_offsets[0][:, 0].unsqueeze(dim=1),
                                            split_offsets[1][:, 1].unsqueeze(dim=1),
                                            split_offsets[2][:, 0].unsqueeze(dim=1),
                                            split_offsets[3][:, 1].unsqueeze(dim=1)], dim=1)
                offset_value_y = torch.cat([split_offsets[0][:, 1].unsqueeze(dim=1),
                                            split_offsets[1][:, 0].unsqueeze(dim=1),
                                            split_offsets[2][:, 1].unsqueeze(dim=1),
                                            split_offsets[3][:, 0].unsqueeze(dim=1)], dim=1)
            else:
                raise NotImplementedError

            offset_values = torch.cat([torch.max(torch.abs(offset_value_x), dim=1)[0].unsqueeze(dim=1),
                                       torch.max(torch.abs(offset_value_y), dim=1)[0].unsqueeze(dim=1)], dim=1)
        else:
            raise NotImplementedError

        offset_polarity = torch.zeros(main_offsets.size(), device=offset_pred.device)
        offset_polarity[main_offsets > 0] = 1
        offset_polarity[main_offsets <= 0] = -1

        fused_offsets = offset_values * offset_polarity

        return fused_offsets

    def predict_by_feat(self,
                        offset_preds: Tuple[Tensor],
                        results_list: List[InstanceData],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg: ConfigDict,
                        rescale: bool = False,
                        activate_map: bool = False) -> InstanceList:
        assert len(offset_preds) == len(results_list) == len(batch_img_metas)

        for img_id in range(len(batch_img_metas)):
            results = results_list[img_id]
            offset_pred = offset_preds[img_id]
            offset_pred_fused = self.offset_fusion(offset_pred)
            offset_pred_decoded = self.offset_coder.decode(results['bboxes'], offset_pred_fused)

            if self.offset_coordinate == 'rectangle':
                pass
            elif self.offset_coordinate == 'polar':
                length, angle = offset_pred_decoded[:, 0], offset_pred_decoded[:, 1]
                offset_x = length * np.cos(angle)
                offset_y = length * np.sin(angle)
                offset_pred_decoded = np.stack([offset_x, offset_y], axis=-1)

            results.offsets = offset_pred_decoded
        return results_list
