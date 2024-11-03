# -*- encoding: utf-8 -*-
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from mmdet.utils import ConfigType, InstanceList
from mmdet.structures import DetDataSample, SampleList

from mmdet.structures.bbox.transforms import bbox2roi, bbox2result, roi2bbox
from mmdet.registry import MODELS
from .standard_roi_head import StandardRoIHead
from ..task_modules.samplers import SamplingResult
from ..utils import empty_instances, unpack_gt_instances


@MODELS.register_module()
class LoftRoIHead(StandardRoIHead):

    def __init__(self,
                offset_roi_extractor,
                offset_head,
                **kwargs):
        
        super(LoftRoIHead, self).__init__(**kwargs)

        # Initialize offset head, same logic in ./base_roi_head.py
        if offset_head is not None:
            self.init_offset_head(offset_roi_extractor, offset_head)

    # Customize the property of having offset head
    @property
    def with_offset(self):
        return hasattr(self, 'offset_head') and self.offset_head is not None

    # Customize the property of having offset head
    def init_offset_head(self, offset_roi_extractor, offset_head):
        self.offset_roi_extractor = MODELS.build(offset_roi_extractor)
        self.offset_head = MODELS.build(offset_head)

    # Override from class BaseModule
    def init_weights(self):
        super(LoftRoIHead, self).init_weights()
        self.offset_head.init_weights()

    # 前向函数,很有可能没什么用,先放在这里
    def forward(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList = None) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            x (List[Tensor]): Multi-level features that may have different
                resolutions.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
            the meta information of each image and corresponding
            annotations.

        Returns
            tuple: A tuple of features from ``bbox_head`` and ``mask_head``
            forward.
        """
        results = ()
        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        rois = bbox2roi(proposals)
        # bbox head
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            results = results + (bbox_results['cls_score'],
                                 bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            results = results + (mask_results['mask_preds'], )

        # offset head
        if self.with_offset:
            offset_results = self._offset_forward(x, rois)
            results = results + (offset_results['cls_score'],
                                 offset_results['bbox_pred'])
        return results
    
    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.
        主要起计算作用的函数，计算前向传播的过程也在其中，实际上并没有调用forward。

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        # 先选出我们想要的ground truth，存储在batch_gt_instances[i]里。
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            # 在下面的代码中，bboxes和priors的意义是相同的
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')
            # 将每一个候选框都安排一个最近的（简单理解起见）gt框，框号的数组称为gt_inds
            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            # 根据候选框和它们参考的重叠度（以及更多信息）选出正负样本
            # 正样本可以包含gt，在pos_is_gt中指明了每个框是不是本来就是gt
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        losses = dict()
        # 计算候选框的预测量和loss
        if self.with_bbox:
            # 获得每个候选框的分类得分,预测结果,本身的内含属性张量,和loss
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results['loss_bbox'])

        # 根据正候选框来计算mask的预测量和loss
        if self.with_mask:
            # 用sample(采样)获得的所有positive(正)候选框来计算mask的前向过程和loss
            mask_results = self.mask_loss(x, sampling_results,
                                          bbox_results['bbox_feats'],
                                          batch_gt_instances)
            losses.update(mask_results['loss_mask'])

        # 计算本模型中独有的offset预测量和对应loss
        if self.with_offset:
            # 获得offset预测结果,特征张量,和loss
            offset_results = self.offset_loss(x, sampling_results,
                                              bbox_results['bbox_feats'],
                                              batch_gt_instances)
            losses.update(offset_results['loss_offset'])

        return losses

    def offset_loss(self, x: Tuple[Tensor],
                     sampling_results: List[SamplingResult],
                     bbox_feats: Tensor,
                     batch_gt_instances: InstanceList) -> dict:
        """

        :param x:
        :param sampling_results:
        :param bbox_feats:
        :param batch_gt_instances:
        :return:
        """
        # 先选出所有的正样本,本模型中只会在正样本的方框(bbox/prior)中预测offset
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
        # 获得这些正样本里的offset预测结果和特征张量
        offset_results = self._offset_forward(x, pos_rois)
        # 用预测结果来计算loss. target理论上会被返回,但实际上没有用
        offset_loss_and_target = self.offset_head.loss_and_target(
            offset_preds=offset_results['offset_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg)

        offset_results.update(loss_offset=offset_loss_and_target['loss_offset'])
        return offset_results

    def _offset_forward(self, x,
                        rois=None,
                        pos_inds=None,
                        bbox_feats=None):
        """
        调用offset_head前向函数的前序函数,根据正样本的位置获取offset的预测结果和特征张量
        :param x: 多级feature图
        :param rois: 可选参数,正样本位置
        :param pos_inds: 可选参数,正样本的ind
        :param bbox_feats: 可选参数,正样本所对应的特征张量的列表
        :return: offset的预测结果和特征张量
        """
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        # 如果我们知道正样本的位置:
        if rois is not None:
            # 没有具体深入看这一部分,总之我们获得了这些正样本的特征张量
            offset_feats = self.offset_roi_extractor(
                x[:self.offset_roi_extractor.num_inputs], rois)
        else:
            # 但是如果我们已经有了相关的特征张量存起来了,那就不用再算了直接取出即可
            assert bbox_feats is not None
            offset_feats = bbox_feats[pos_inds]
        # 调用offset_head的forward前向函数,输出所有的offset预测结果
        offset_preds = self.offset_head(offset_feats)
        # 返回预测结果和offset特征张量
        offset_results = dict(offset_preds=offset_preds, offset_feats=offset_feats)
        return offset_results

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        # 先调用父类方法计算好所有的bbox和mask预测量
        results_list = super().predict(x, rpn_results_list, batch_data_samples, rescale)
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        if self.with_offset:
            results_list = self.predict_offset(x, batch_img_metas, results_list, rescale)
        return results_list

    def predict_offset(self,
                       x: Tuple[Tensor],
                       batch_img_metas: List[dict],
                       results_list: InstanceList,
                       rescale: bool = False) -> InstanceList:
        bboxes = [res.bboxes for res in results_list]
        offset_rois = bbox2roi(bboxes)

        if offset_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                offset_rois.device,
                task_type='mask',
                instance_results=results_list)
            return results_list

        offset_results = self._offset_forward(x, offset_rois)
        # 计算预测结果和gt的iou
        offset_preds = offset_results['offset_preds']
        num_offset_rois_per_img = [len(res) for res in results_list]
        offset_preds = offset_preds.split(num_offset_rois_per_img, 0)

        results_list = self.offset_head.predict_by_feat(
            offset_preds=offset_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)

        return results_list

