# dataset settings (brightfield + Fl channel)
dataset_type = 'CTCDataset'
data_root = 'data/dataset/CTC/'
img_norm_cfg = dict(
    mean=[0, 0, 0, 0, 0, 0], std=[1, 1, 1, 1, 1, 1], to_rgb=False)
train_pipeline = [
    dict(type='LoadCTCImageFromFile', to_float32 = True, blue_input = False, fl_input = True, normalize = True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale_factor = 1.0, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=1),
    dict(
        type='PackDetInputs',
        meta_keys=('img', 'gt_bboxes', 'gt_bboxes_labels','img_shape', 'pad_shape'))
]
test_pipeline = [

    dict(type='LoadCTCImageFromFile', to_float32 = True, blue_input = False, fl_input = True, normalize = True),
    dict(type='Resize', scale_factor = 1.0, keep_ratio=True),
    # avoid bboxes being resized
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor')
        )
]

train_dataloader = dict(
    batch_size=4,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True, ),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'train.txt',
            data_prefix=dict(sub_data_root=data_root), 
            pipeline=train_pipeline
            ))
            )


val_dataloader = dict(
    batch_size=4,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'val.txt',
            data_prefix=dict(sub_data_root=data_root), 
            pipeline=test_pipeline
            ))
)


test_dataloader = dict(
    batch_size=4,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'test.txt',
            data_prefix=dict(sub_data_root=data_root), 
            pipeline=test_pipeline
            ))
)

 
evaluation = dict(type='VOCMetric', metric='mAP')
val_evaluator = evaluation
test_evaluator = evaluation
