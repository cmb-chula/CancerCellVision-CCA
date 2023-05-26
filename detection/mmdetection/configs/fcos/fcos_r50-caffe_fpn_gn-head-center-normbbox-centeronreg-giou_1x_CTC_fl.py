_base_ = 'fcos_r50-caffe_fpn_gn-head_1x_coco_CTC_fl.py'

# model setting
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        #move this to dataset config instead
        # mean=[103.530, 116.280, 123.675, 0, 0, 0],
        # std=[1.0, 1.0, 1.0, 1, 1, 1],
        bgr_to_rgb=False,
        pad_size_divisor=1),
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    bbox_head=dict(
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),
    # training and testing settings
    test_cfg=dict(nms=dict(type='nms', iou_threshold=0.6)))
