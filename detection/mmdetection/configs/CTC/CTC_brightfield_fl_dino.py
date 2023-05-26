_base_ = [
    '../dino/dino-4scale_r50_CTC_fl.py',
    '../_base_/datasets/CTC/CTC_brightfield_fl_dino.py',
    '../_base_/schedules/schedule_dino.py', 
    '../_base_/default_runtime.py'
]
# download it from https://github.com/open-mmlab/mmdetection/tree/main/configs/dino
load_from = './checkpoints/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'