from mmdet.datasets.transforms import LoadAnnotations
from mmdet.registry import TRANSFORMS
import numpy as np


@TRANSFORMS.register_module()
class LoadLoft(LoadAnnotations):
    def __init__(self, with_offset: bool = True, **kwargs):
        super(LoadLoft, self).__init__(**kwargs)
        self.with_offset = with_offset

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        if self.with_offset:
            self._load_offsets(results)
        return results
    
    def _load_offsets(self, results: dict) -> None:
        gt_bboxes_offsets = []
        for instance in results.get('instances', []):
            gt_bboxes_offsets.append(instance['offset'])
        if gt_bboxes_offsets:
            results['gt_offsets'] = np.array(gt_bboxes_offsets, dtype=np.int64)
        else:
            results['gt_offsets'] = np.zeros((0, 2), dtype=np.int64)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_offset={self.with_offset}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str