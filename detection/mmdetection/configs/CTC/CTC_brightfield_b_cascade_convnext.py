_base_ = [
    '../_base_/models/cascade_convnextb_fpn_CTC.py',
    '../_base_/datasets/CTC/CTC_brightfield_b_cascade.py',
    '../_base_/schedules/schedule_half.py', 
    '../_base_/default_runtime.py'
]
