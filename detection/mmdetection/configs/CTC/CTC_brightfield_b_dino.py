_base_ = [
    '../dino/dino-4scale_r50_CTC.py',
    '../_base_/datasets/CTC/CTC_brightfield_b_dino.py',
    '../_base_/schedules/schedule_dino.py', 
    '../_base_/default_runtime.py'
]

load_from = './checkpoints/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'