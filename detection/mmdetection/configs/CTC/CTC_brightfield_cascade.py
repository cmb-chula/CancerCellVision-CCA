_base_ = [
    '../_base_/models/cascade_rcnn_r101_fpn_CTC.py',
    '../_base_/datasets/CTC/CTC_brightfield.py',
    '../_base_/schedules/schedule_half.py', 
    '../_base_/default_runtime.py'
]
