import numpy as np

from mmdet.registry import TRANSFORMS
from mmcv.transforms import RandomFlip
from .transforms import Resize
from .formatting import PackDetInputs


@TRANSFORMS.register_module()
class ResizeLoft(Resize):
    """
    由于mmdetection并没有针对图像提取结果的向量进行缩放的代码，
    我们需要重载部分方法实现向量缩放。
    """
    def transform(self, results: dict) -> dict:
        # 先调用父类方法完成大部分的变换
        results = super().transform(results)
        # 再将所有的offset缩放即可
        self._resize_offsets(results)
        return results

    def _resize_offsets(self, results: dict):
        if results.get('gt_offsets', None) is not None:
            if self.keep_ratio:
                results['gt_offsets'] = results['gt_offsets'] * results['scale_factor']
            else:
                # x方向的缩放倍数
                x_factor = results['img_shape'][0] / results['ori_shape'][0]
                # y方向额缩放倍数
                y_factor = results['img_shape'][0] / results['ori_shape'][0]
                # 构建一个线性变换矩阵并相乘
                multipliers = np.array([[x_factor, 0], [0, y_factor]])
                results['gt_masks'] = results['gt_masks'] @ multipliers


@TRANSFORMS.register_module()
class FlipLoft(RandomFlip):
    """
    由于mmcv并没有针对图像提取结果的向量进行翻转的代码，
    我们需要重载部分方法实现向量翻转。
    """
    def _flip(self, results: dict) -> None:
        if results.get('gt_offsets', None) is not None:
            results['gt_offsets'] = self._flip_offsets(results['gt_offsets'], results['flip_direction'])
        super()._flip(results)


    def _flip_offsets(self, offsets: np.ndarray, direction: str) -> np.ndarray:
        if direction == 'horizontal':
            offsets[:, 0] *= -1
        elif direction == 'vertical':
            offsets[:, 1] *= -1
        elif direction == 'diagonal':
            offsets[:, 0] *= -1
            offsets[:, 1] *= -1
        return offsets


@TRANSFORMS.register_module()
class PackLoft(PackDetInputs):
    """
    由于mmdetection并没有针对图像提取结果的向量进行打包的代码，
    我们需要重载部分方法实现向量打包。
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
        'gt_offsets': 'offsets',
    }
