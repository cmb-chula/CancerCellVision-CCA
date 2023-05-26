_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_CTC_fl.py',
    '../_base_/datasets/CTC/CTC_brightfield_fl.py',
    '../_base_/schedules/schedule_half.py', 
    '../_base_/default_runtime.py'
]
