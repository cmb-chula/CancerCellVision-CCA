# dataset settings (brightfield + B channel)
dataset_type = 'CTCDataset'
data_root = 'data/dataset/CTC/'
img_norm_cfg = dict(
    mean=[0,0, 0 ], std=[1, 1, 1], to_rgb=False)
train_pipeline = [
    dict(type='LoadCTCImageFromFile', to_float32 = True, blue_input = True, fl_input = False, normalize = True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(540, 540), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadCTCImageFromFile', to_float32 = True, blue_input = True, fl_input = False, normalize = True),
    dict(
        type='MultiScaleFlipAug',
        img_scale= (540, 540),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=4,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'train.txt',
            ],
            img_prefix=[data_root],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val.txt',
        img_prefix=data_root ,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.txt',
        img_prefix=data_root,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='mAP')
