import warnings
from typing import List, Optional, Tuple

import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.utils.misc import floordiv
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, InstanceList, MultiConfig, OptConfigType
from ..layers import mask_matrix_nms
from ..utils import center_of_mass, generate_coordinate, multi_apply
from .solov2_head import SOLOV2Head


@MODELS.register_module()
class SOLOVecHead(SOLOV2Head):

    def __init__(self, loss_offset=dict(type='SmoothL1Loss', loss_weight=2.0), **kwargs):
        super().__init__(**kwargs)
        self.loss_vec = MODELS.build(loss_offset)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.vec_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        conv_cfg = None
        for i in range(self.stacked_convs):
            if self.with_dcn:
                if self.dcn_apply_to_all_conv:
                    conv_cfg = self.dcn_cfg
                elif i == self.stacked_convs - 1:
                    # light head
                    conv_cfg = self.dcn_cfg

            chn = self.in_channels + 2 if i == 0 else self.feat_channels
            self.kernel_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

            chn = self.in_channels + 2 if i == 0 else self.feat_channels
            self.vec_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)

        self.conv_kernel = nn.Conv2d(
            self.feat_channels, self.kernel_out_channels, 3, padding=1)

        self.conv_vec = nn.Conv2d(
            self.feat_channels, 2, 3, padding=1)

    def forward(self, x):
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores, mask prediction,
            and mask features.

                - mlvl_kernel_preds (list[Tensor]): Multi-level dynamic kernel
                  prediction. The kernel is used to generate instance
                  segmentation masks by dynamic convolution. Each element in
                  the list has shape
                  (batch_size, kernel_out_channels, num_grids, num_grids).
                - mlvl_cls_preds (list[Tensor]): Multi-level scores. Each
                  element in the list has shape
                  (batch_size, num_classes, num_grids, num_grids).
                - mlvl_vec_preds (list[Tensor]): Multi-level vector. Each
                  element in the list has shape
                  (batch_size, 2, num_grids, num_grids).
                - mask_feats (Tensor): Unified mask feature map used to
                  generate instance segmentation masks by dynamic convolution.
                  Has shape (batch_size, mask_out_channels, h, w).
        """
        assert len(x) == self.num_levels
        mask_feats = self.mask_feature_head(x)
        ins_kernel_feats = self.resize_feats(x)
        mlvl_kernel_preds = []
        mlvl_cls_preds = []
        mlvl_vec_preds = []
        for i in range(self.num_levels):
            ins_kernel_feat = ins_kernel_feats[i]
            # ins branch
            # concat coord
            coord_feat = generate_coordinate(ins_kernel_feat.size(),
                                             ins_kernel_feat.device)
            ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

            # kernel branch
            kernel_feat = ins_kernel_feat
            kernel_feat = F.interpolate(
                kernel_feat,
                size=self.num_grids[i],
                mode='bilinear',
                align_corners=False)

            cate_feat = kernel_feat[:, :-2, :, :]
            vec_feat = kernel_feat

            kernel_feat = kernel_feat.contiguous()
            for i, kernel_conv in enumerate(self.kernel_convs):
                kernel_feat = kernel_conv(kernel_feat)
            kernel_pred = self.conv_kernel(kernel_feat)

            # cate branch
            cate_feat = cate_feat.contiguous()
            for i, cls_conv in enumerate(self.cls_convs):
                cate_feat = cls_conv(cate_feat)
            cate_pred = self.conv_cls(cate_feat)

            # vec branch
            vec_feat = vec_feat.contiguous()
            for i, vec_conv in enumerate(self.vec_convs):
                vec_feat = vec_conv(vec_feat)
            vec_pred = self.conv_vec(vec_feat)

            mlvl_kernel_preds.append(kernel_pred)
            mlvl_cls_preds.append(cate_pred)
            mlvl_vec_preds.append(vec_pred)

        return mlvl_kernel_preds, mlvl_cls_preds, mlvl_vec_preds, mask_feats

    def _get_targets_single(self,
                            gt_instances: InstanceData,
                            featmap_sizes: Optional[list] = None) -> tuple:
        """Compute targets for predictions of single image.

        Args:
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes``, ``labels``,
                and ``masks`` attributes.
            featmap_sizes (list[:obj:`torch.size`]): Size of each
                feature map from feature pyramid, each element
                means (feat_h, feat_w). Defaults to None.

        Returns:
            Tuple: Usually returns a tuple containing targets for predictions.

                - mlvl_pos_mask_targets (list[Tensor]): Each element represent
                  the binary mask targets for positive points in this
                  level, has shape (num_pos, out_h, out_w).
                - mlvl_labels (list[Tensor]): Each element is
                  classification labels for all
                  points in this level, has shape
                  (num_grid, num_grid).
                - mlvl_pos_masks  (list[Tensor]): Each element is
                  a `BoolTensor` to represent whether the
                  corresponding point in single level
                  is positive, has shape (num_grid **2).
                - mlvl_pos_indexes  (list[list]): Each element
                  in the list contains the positive index in
                  corresponding level, has shape (num_pos).
        """
        gt_labels = gt_instances.labels
        device = gt_labels.device

        gt_bboxes = gt_instances.bboxes
        gt_areas = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0]) *
                              (gt_bboxes[:, 3] - gt_bboxes[:, 1]))
        gt_masks = gt_instances.masks.to_tensor(
            dtype=torch.bool, device=device)

        gt_offsets = gt_instances.offsets

        mlvl_pos_mask_targets = []
        mlvl_pos_indexes = []
        mlvl_labels = []
        mlvl_pos_masks = []
        mlvl_offsets = []
        for (lower_bound, upper_bound), num_grid \
                in zip(self.scale_ranges, self.num_grids):
            mask_target = []
            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            pos_index = []
            labels = torch.zeros([num_grid, num_grid],
                                 dtype=torch.int64,
                                 device=device) + self.num_classes
            # Set default background offset to 0, indicating that there is no offset
            offsets = torch.zeros([num_grid, num_grid, 2],
                                  dtype=torch.float32,
                                  device=device)
            pos_mask = torch.zeros([num_grid ** 2],
                                   dtype=torch.bool,
                                   device=device)

            gt_inds = ((gt_areas >= lower_bound) &
                       (gt_areas <= upper_bound)).nonzero().flatten()
            if len(gt_inds) == 0:
                mlvl_pos_mask_targets.append(
                    torch.zeros([0, featmap_sizes[0], featmap_sizes[1]],
                                dtype=torch.uint8,
                                device=device))
                mlvl_labels.append(labels)
                mlvl_offsets.append(offsets)
                mlvl_pos_masks.append(pos_mask)
                mlvl_pos_indexes.append([])
                continue
            hit_gt_bboxes = gt_bboxes[gt_inds]
            hit_gt_labels = gt_labels[gt_inds]
            hit_gt_offsets = gt_offsets[gt_inds]
            hit_gt_masks = gt_masks[gt_inds, ...]

            pos_w_ranges = 0.5 * (hit_gt_bboxes[:, 2] -
                                  hit_gt_bboxes[:, 0]) * self.pos_scale
            pos_h_ranges = 0.5 * (hit_gt_bboxes[:, 3] -
                                  hit_gt_bboxes[:, 1]) * self.pos_scale
            # Make sure hit_gt_masks has a value
            valid_mask_flags = hit_gt_masks.sum(dim=-1).sum(dim=-1) > 0

            for gt_mask, gt_label, gt_offset, pos_h_range, pos_w_range, \
                    valid_mask_flag in \
                    zip(hit_gt_masks, hit_gt_labels, hit_gt_offsets, pos_h_ranges,
                        pos_w_ranges, valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (featmap_sizes[0] * self.mask_stride,
                                  featmap_sizes[1] * self.mask_stride)
                center_h, center_w = center_of_mass(gt_mask)

                coord_w = int(
                    floordiv((center_w / upsampled_size[1]), (1. / num_grid),
                             rounding_mode='trunc'))
                coord_h = int(
                    floordiv((center_h / upsampled_size[0]), (1. / num_grid),
                             rounding_mode='trunc'))

                # left, top, right, down
                top_box = max(
                    0,
                    int(
                        floordiv(
                            (center_h - pos_h_range) / upsampled_size[0],
                            (1. / num_grid),
                            rounding_mode='trunc')))
                down_box = min(
                    num_grid - 1,
                    int(
                        floordiv(
                            (center_h + pos_h_range) / upsampled_size[0],
                            (1. / num_grid),
                            rounding_mode='trunc')))
                left_box = max(
                    0,
                    int(
                        floordiv(
                            (center_w - pos_w_range) / upsampled_size[1],
                            (1. / num_grid),
                            rounding_mode='trunc')))
                right_box = min(
                    num_grid - 1,
                    int(
                        floordiv(
                            (center_w + pos_w_range) / upsampled_size[1],
                            (1. / num_grid),
                            rounding_mode='trunc')))

                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)

                labels[top:(down + 1), left:(right + 1)] = gt_label
                offsets[top:(down + 1), left:(right + 1)] = gt_offset
                # ins
                gt_mask = np.uint8(gt_mask.cpu().numpy())
                # Follow the original implementation, F.interpolate is
                # different from cv2 and opencv
                gt_mask = mmcv.imrescale(gt_mask, scale=1. / self.mask_stride)
                gt_mask = torch.from_numpy(gt_mask).to(device=device)

                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        index = int(i * num_grid + j)
                        this_mask_target = torch.zeros(
                            [featmap_sizes[0], featmap_sizes[1]],
                            dtype=torch.uint8,
                            device=device)
                        this_mask_target[:gt_mask.shape[0], :gt_mask.
                        shape[1]] = gt_mask
                        mask_target.append(this_mask_target)
                        pos_mask[index] = True
                        pos_index.append(index)
            if len(mask_target) == 0:
                mask_target = torch.zeros(
                    [0, featmap_sizes[0], featmap_sizes[1]],
                    dtype=torch.uint8,
                    device=device)
            else:
                mask_target = torch.stack(mask_target, 0)
            mlvl_pos_mask_targets.append(mask_target)
            mlvl_labels.append(labels)
            mlvl_offsets.append(offsets)
            mlvl_pos_masks.append(pos_mask)
            mlvl_pos_indexes.append(pos_index)
        return (mlvl_pos_mask_targets, mlvl_labels, mlvl_offsets, mlvl_pos_masks,
                mlvl_pos_indexes)

    def loss_by_feat(self, mlvl_kernel_preds: List[Tensor],
                     mlvl_cls_preds: List[Tensor],
                     mlvl_offset_preds: List[Tensor], mask_feats: Tensor,
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict], **kwargs) -> dict:
        """Calculate the loss based on the features extracted by the mask head.

        Args:
            mlvl_kernel_preds (list[Tensor]): Multi-level dynamic kernel
                prediction. The kernel is used to generate instance
                segmentation masks by dynamic convolution. Each element in the
                list has shape
                (batch_size, kernel_out_channels, num_grids, num_grids).
            mlvl_cls_preds (list[Tensor]): Multi-level scores. Each element
                in the list has shape
                (batch_size, num_classes, num_grids, num_grids).
            mask_feats (Tensor): Unified mask feature map used to generate
                instance segmentation masks by dynamic convolution. Has shape
                (batch_size, mask_out_channels, h, w).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``masks``,
                and ``labels`` attributes.
            batch_img_metas (list[dict]): Meta information of multiple images.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = mask_feats.size()[-2:]

        pos_mask_targets, labels, offsets, pos_masks, pos_indexes = multi_apply(
            self._get_targets_single,
            batch_gt_instances,
            featmap_sizes=featmap_sizes)

        mlvl_mask_targets = [
            torch.cat(lvl_mask_targets, 0)
            for lvl_mask_targets in zip(*pos_mask_targets)
        ]

        mlvl_pos_kernel_preds = []
        for lvl_kernel_preds, lvl_pos_indexes in zip(mlvl_kernel_preds,
                                                     zip(*pos_indexes)):
            lvl_pos_kernel_preds = []
            for img_lvl_kernel_preds, img_lvl_pos_indexes in zip(
                    lvl_kernel_preds, lvl_pos_indexes):
                img_lvl_pos_kernel_preds = img_lvl_kernel_preds.view(
                    img_lvl_kernel_preds.shape[0], -1)[:, img_lvl_pos_indexes]
                lvl_pos_kernel_preds.append(img_lvl_pos_kernel_preds)
            mlvl_pos_kernel_preds.append(lvl_pos_kernel_preds)

        # make multilevel mlvl_mask_pred
        mlvl_mask_preds = []
        for lvl_pos_kernel_preds in mlvl_pos_kernel_preds:
            lvl_mask_preds = []
            for img_id, img_lvl_pos_kernel_pred in enumerate(
                    lvl_pos_kernel_preds):
                if img_lvl_pos_kernel_pred.size()[-1] == 0:
                    continue
                img_mask_feats = mask_feats[[img_id]]
                h, w = img_mask_feats.shape[-2:]
                num_kernel = img_lvl_pos_kernel_pred.shape[1]
                img_lvl_mask_pred = F.conv2d(
                    img_mask_feats,
                    img_lvl_pos_kernel_pred.permute(1, 0).view(
                        num_kernel, -1, self.dynamic_conv_size,
                        self.dynamic_conv_size),
                    stride=1).view(-1, h, w)
                lvl_mask_preds.append(img_lvl_mask_pred)
            if len(lvl_mask_preds) == 0:
                lvl_mask_preds = None
            else:
                lvl_mask_preds = torch.cat(lvl_mask_preds, 0)
            mlvl_mask_preds.append(lvl_mask_preds)
        # dice loss
        num_pos = 0
        for img_pos_masks in pos_masks:
            for lvl_img_pos_masks in img_pos_masks:
                # Fix `Tensor` object has no attribute `count_nonzero()`
                # in PyTorch 1.6, the type of `lvl_img_pos_masks`
                # should be `torch.bool`.
                num_pos += lvl_img_pos_masks.nonzero().numel()
        loss_mask = []
        for lvl_mask_preds, lvl_mask_targets in zip(mlvl_mask_preds,
                                                    mlvl_mask_targets):
            if lvl_mask_preds is None:
                continue
            loss_mask.append(
                self.loss_mask(
                    lvl_mask_preds,
                    lvl_mask_targets,
                    reduction_override='none'))
        if num_pos > 0:
            loss_mask = torch.cat(loss_mask).sum() / num_pos
        else:
            loss_mask = mask_feats.sum() * 0

        # cate
        flatten_labels = [
            torch.cat(
                [img_lvl_labels.flatten() for img_lvl_labels in lvl_labels])
            for lvl_labels in zip(*labels)
        ]
        flatten_labels = torch.cat(flatten_labels)

        flatten_cls_preds = [
            lvl_cls_preds.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for lvl_cls_preds in mlvl_cls_preds
        ]
        flatten_cls_preds = torch.cat(flatten_cls_preds)

        loss_cls = self.loss_cls(
            flatten_cls_preds, flatten_labels, avg_factor=num_pos + 1)
        
        # vec
        flatten_offsets = [
            torch.cat([
                img_lvl_offsets.reshape(-1, 2) for img_lvl_offsets in lvl_offsets])
            for lvl_offsets in zip(*offsets)
        ]
        flatten_offsets = torch.cat(flatten_offsets)

        flatten_vec_preds = [
            lvl_vec_preds.permute(0, 2, 3, 1).reshape(-1, 2)
            for lvl_vec_preds in mlvl_offset_preds
        ]
        flatten_vec_preds = torch.cat(flatten_vec_preds)

        loss_vec = self.loss_vec(flatten_vec_preds, flatten_offsets)
        
        return dict(loss_mask=loss_mask, loss_cls=loss_cls, loss_offset=loss_vec)

    def predict_by_feat(self, mlvl_kernel_preds: List[Tensor],
                        mlvl_cls_scores: List[Tensor],
                        mlvl_offset_preds: List[Tensor], mask_feats: Tensor,
                        batch_img_metas: List[dict], **kwargs) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        mask results.

        Args:
            mlvl_kernel_preds (list[Tensor]): Multi-level dynamic kernel
                prediction. The kernel is used to generate instance
                segmentation masks by dynamic convolution. Each element in the
                list has shape
                (batch_size, kernel_out_channels, num_grids, num_grids).
            mlvl_cls_scores (list[Tensor]): Multi-level scores. Each element
                in the list has shape
                (batch_size, num_classes, num_grids, num_grids).
            mask_feats (Tensor): Unified mask feature map used to generate
                instance segmentation masks by dynamic convolution. Has shape
                (batch_size, mask_out_channels, h, w).
            batch_img_metas (list[dict]): Meta information of all images.

        Returns:
            list[:obj:`InstanceData`]: Processed results of multiple
            images.Each :obj:`InstanceData` usually contains
            following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,).
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).
        """
        num_levels = len(mlvl_cls_scores)
        assert len(mlvl_kernel_preds) == len(mlvl_cls_scores) == len(mlvl_offset_preds)

        for lvl in range(num_levels):
            cls_scores = mlvl_cls_scores[lvl]
            offsets = mlvl_offset_preds[lvl]
            cls_scores = cls_scores.sigmoid()
            local_max = F.max_pool2d(cls_scores, 2, stride=1, padding=1)
            keep_mask = local_max[:, :, :-1, :-1] == cls_scores
            cls_scores = cls_scores * keep_mask
            offsets = offsets * keep_mask
            mlvl_cls_scores[lvl] = cls_scores.permute(0, 2, 3, 1)
            mlvl_offset_preds[lvl] = offsets.permute(0, 2, 3, 1)

        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_cls_pred = [
                mlvl_cls_scores[lvl][img_id].view(-1, self.cls_out_channels)
                for lvl in range(num_levels)
            ]
            img_offset_pred = [
                mlvl_offset_preds[lvl][img_id].view(-1, 2)
                for lvl in range(num_levels)
            ]
            img_mask_feats = mask_feats[[img_id]]
            img_kernel_pred = [
                mlvl_kernel_preds[lvl][img_id].permute(1, 2, 0).view(
                    -1, self.kernel_out_channels) for lvl in range(num_levels)
            ]
            img_cls_pred = torch.cat(img_cls_pred, dim=0)
            img_offset_pred = torch.cat(img_offset_pred, dim=0)
            img_kernel_pred = torch.cat(img_kernel_pred, dim=0)
            result = self._predict_by_feat_single(
                img_kernel_pred,
                img_cls_pred,
                img_offset_pred,
                img_mask_feats,
                img_meta=batch_img_metas[img_id])
            result_list.append(result)
        return result_list

    def _predict_by_feat_single(self,
                                kernel_preds: Tensor,
                                cls_scores: Tensor,
                                offset_preds: Tensor,
                                mask_feats: Tensor,
                                img_meta: dict,
                                cfg: OptConfigType = None) -> InstanceData:
        """Transform a single image's features extracted from the head into
        mask results.

        Args:
            kernel_preds (Tensor): Dynamic kernel prediction of all points
                in single image, has shape
                (num_points, kernel_out_channels).
            cls_scores (Tensor): Classification score of all points
                in single image, has shape (num_points, num_classes).
            mask_feats (Tensor): Mask prediction of all points in
                single image, has shape (num_points, feat_h, feat_w).
            img_meta (dict): Meta information of corresponding image.
            cfg (dict, optional): Config used in test phase.
                Defaults to None.

        Returns:
            :obj:`InstanceData`: Processed results of single image.
             it usually contains following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,).
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).
        """

        def empty_results(cls_scores, ori_shape):
            """Generate a empty results."""
            results = InstanceData()
            results.scores = cls_scores.new_ones(0)
            results.masks = cls_scores.new_zeros(0, *ori_shape)
            results.labels = cls_scores.new_ones(0)
            results.bboxes = cls_scores.new_zeros(0, 4)
            return results

        cfg = self.test_cfg if cfg is None else cfg
        assert len(kernel_preds) == len(cls_scores) == len(offset_preds)

        featmap_size = mask_feats.size()[-2:]

        # overall info
        h, w = img_meta['img_shape'][:2]
        upsampled_size = (featmap_size[0] * self.mask_stride,
                          featmap_size[1] * self.mask_stride)

        # process.
        score_mask = (cls_scores > cfg.score_thr)
        cls_scores = cls_scores[score_mask]
        offset_mask = score_mask.squeeze(1)
        offset_preds = offset_preds[offset_mask]
        if len(cls_scores) == 0:
            return empty_results(cls_scores, img_meta['ori_shape'][:2])

        # cate_labels & kernel_preds
        inds = score_mask.nonzero()
        cls_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        lvl_interval = cls_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(lvl_interval[-1])

        strides[:lvl_interval[0]] *= self.strides[0]
        for lvl in range(1, self.num_levels):
            strides[lvl_interval[lvl -
                                 1]:lvl_interval[lvl]] *= self.strides[lvl]
        strides = strides[inds[:, 0]]

        # mask encoding.
        kernel_preds = kernel_preds.view(
            kernel_preds.size(0), -1, self.dynamic_conv_size,
            self.dynamic_conv_size)
        mask_preds = F.conv2d(
            mask_feats, kernel_preds, stride=1).squeeze(0).sigmoid()
        # mask.
        masks = mask_preds > cfg.mask_thr
        sum_masks = masks.sum((1, 2)).float()
        keep = sum_masks > strides
        if keep.sum() == 0:
            return empty_results(cls_scores, img_meta['ori_shape'][:2])
        masks = masks[keep]
        mask_preds = mask_preds[keep]
        sum_masks = sum_masks[keep]
        cls_scores = cls_scores[keep]
        cls_labels = cls_labels[keep]
        offset_preds = offset_preds[keep]

        # maskness.
        mask_scores = (mask_preds * masks).sum((1, 2)) / sum_masks
        cls_scores *= mask_scores

        scores, labels, _, keep_inds = mask_matrix_nms(
            masks,
            cls_labels,
            cls_scores,
            mask_area=sum_masks,
            nms_pre=cfg.nms_pre,
            max_num=cfg.max_per_img,
            kernel=cfg.kernel,
            sigma=cfg.sigma,
            filter_thr=cfg.filter_thr)
        if len(keep_inds) == 0:
            return empty_results(cls_scores, img_meta['ori_shape'][:2])
        mask_preds = mask_preds[keep_inds]
        offset_preds = offset_preds[keep_inds]
        mask_preds = F.interpolate(
            mask_preds.unsqueeze(0),
            size=upsampled_size,
            mode='bilinear',
            align_corners=False)[:, :, :h, :w]
        mask_preds = F.interpolate(
            mask_preds,
            size=img_meta['ori_shape'][:2],
            mode='bilinear',
            align_corners=False).squeeze(0)
        masks = mask_preds > cfg.mask_thr

        results = InstanceData()
        results.masks = masks
        results.labels = labels
        results.scores = scores
        results.offsets = offset_preds
        # create an empty bbox in InstanceData to avoid bugs when
        # calculating metrics.
        results.bboxes = results.scores.new_zeros(len(scores), 4)

        return results
