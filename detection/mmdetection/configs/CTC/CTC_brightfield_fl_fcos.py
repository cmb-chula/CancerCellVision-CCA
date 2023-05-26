_base_ = [
    '../fcos/fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_CTC_fl.py',
    '../_base_/datasets/CTC/CTC_brightfield_fl_fcos.py',
    '../_base_/schedules/schedule_half.py', 
    '../_base_/default_runtime.py'
]
