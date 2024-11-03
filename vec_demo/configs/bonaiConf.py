# 新配置继承了基本配置，并做了必要的修改
_base_ = '../mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py'

# 模型配置
model = dict(
    type='LOFT',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        # 图像归一化参数
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        # 图像 padding 参数
        pad_mask=True,  # 在实例分割中，需要将 mask 也进行 padding
        pad_size_divisor=32),   # 图像 padding 到 32 的倍数
    backbone=dict(
        # 由于batch size太小了，只能用GN
        norm_cfg=dict(requires_grad=True, num_groups=32, type='GN'),
        style='pytorch'
    ),
    roi_head=dict(
        type='LoftRoIHead',
        bbox_head=dict(num_classes=1), 
        mask_head=dict(num_classes=1),
        offset_head=dict(
            type='OffsetHead',
            reg_decoded_offset=False,
            loss_offset=dict(type='SmoothL1Loss', loss_weight=2.0),
            offset_coder=dict(type='DeltaXYOffsetCoder')
        ),
        offset_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        )
    )
)

# 修改数据集相关配置
data_root = 'data/ard/coco/'
dataset_type = 'BonaiDataset'
metainfo = {
    'classes': ('building', ),
    'palette': [
        (220, 20, 60),
    ]
}
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(poly2mask=True, type='LoadLoft', with_bbox=True, with_mask=True, with_offset=True, box_type=None),
    dict(keep_ratio=True,
         scales=[(1024, 1024)],
         resize_type='ResizeLoft',  # 这里需要更改resize的方式以实现缩放offset
         type='RandomChoiceResize'),
    dict(prob=0.5, type='FlipLoft'),
    dict(type='PackLoft')
]
train_dataloader = dict(
    batch_size=8, # 实测最大了...
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/train.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline,
        type='BonaiDataset')
)
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadLoft', with_bbox=True, with_mask=True, with_offset=True),
    dict(keep_ratio=True, scale=(1024, 1024), type='ResizeLoft'),
    dict(type='PackLoft', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                                     'scale_factor'))
]
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/val.json',
        data_prefix=dict(img='val/'),
        pipeline=test_pipeline,
        test_mode=True,
        type='BonaiDataset'))
test_dataloader = val_dataloader

# 修改评价指标相关配置
val_evaluator = dict(
    type='VectorMetric',
    metric=[
        'bbox',
        'segm',
        'vector'
    ],
    outfile_prefix='D:/zcp/heightStereo/work_dirs/bonaiConf/results/',  # 保存结果文件的路径
    ann_file=data_root + 'val/val.json')  # 验证集的标注文件
test_evaluator = val_evaluator

# 修改训练相关配置
train_cfg = dict(max_epochs=300, type='EpochBasedTrainLoop', val_interval=1)

# 修改优化器相关配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,  # 把学习率设置高一些看看有没有用
        momentum=0.9,
        weight_decay=0.0001),
    clip_grad=dict(
        max_norm=35,
        norm_type=2
    )
)

# 修改学习率相关配置
param_scheduler = [
    dict(
        type='LinearLR',  # 使用线性学习率预热
        start_factor=0.001,  # 学习率预热的系数
        by_epoch=False,  # 按 iteration 更新预热学习率
        begin=0,  # 从第一个 iteration 开始
        end=300)  # 到第 500 个 iteration 结束
    # 暂时不需要学习率衰减的设置
    # dict(
    #    type='MultiStepLR',  # 在训练过程中使用 multi step 学习率策略
    #    by_epoch=True,  # 按 epoch 更新学习率
    #    milestones=[90, 96],  # 在哪几个 epoch 进行学习率衰减
    #    gamma=0.1)  # 学习率衰减系数
]

# 修改测试相关配置
evaluation = dict(
    interval=2,
    metric=['bbox', 'segm', 'offset'])

# 修改可视化设置
visualizer = dict(
    name='visualizer',
    type='BonaiVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ]
)
