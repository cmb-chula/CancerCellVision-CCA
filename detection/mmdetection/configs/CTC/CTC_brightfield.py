_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_CTC.py',
    '../_base_/datasets/CTC/CTC_brightfield.py',
    '../_base_/schedules/schedule_half.py', 
    '../_base_/default_runtime.py'
]
