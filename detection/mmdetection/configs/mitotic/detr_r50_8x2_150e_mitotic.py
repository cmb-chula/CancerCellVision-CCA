_base_ = [
    '../_base_/datasets/mitotic_CMC_base.py', '../_base_/default_runtime.py'
]
model = dict(
    type='DETR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    bbox_head=dict(
        type='DETRHead',
        num_classes=1,
        in_channels=2048,
        transformer=dict(
            type='Transformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetrTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1),
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=1.),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=100))
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.

train_pipeline = [
    dict(type='LoadMitoticSlide'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 480), (448, 448), (412, 412), (512, 512),
                           (544, 544), (576, 576), (608, 608)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(480, 480), (512, 512), (544, 544)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(448, 448),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(480, 480), (448, 448), (412, 412), (512, 512),
                           (544, 544), (576, 576), (608, 608)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadMitoticSlide'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

dataset_type = 'MitoticDataset'
data_root = 'data/dataset/CMC/'

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=16,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='MitoticInferenceDataset',
            ann_file=[
                data_root + 'train.txt',
            ],
            img_prefix=[data_root],
            pipeline=train_pipeline)),
    val=dict(
        type='MitoticInferenceDataset',
        ann_file=data_root + 'val.txt',
        img_prefix=data_root ,
        pipeline=test_pipeline),
    test=dict(
        type='MitoticInferenceDataset',
        ann_file=data_root + 'test.txt', #inference_train, test_tiling
        # ann_file=data_root + 'inference_train.txt', #inference_train, test_tiling

        img_prefix=data_root,
        pipeline=test_pipeline))


# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(pipeline=train_pipeline),
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[16])
runner = dict(type='EpochBasedRunner', max_epochs=25)
load_from = 'detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'
