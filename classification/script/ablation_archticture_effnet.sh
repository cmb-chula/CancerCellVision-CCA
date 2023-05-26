python3 tools/train/train_CTC_torch.py -i config/effnet_b1/CTC_brightfield_b_RGU.py -o CTC/effnet_b1/BrU_3cls_0_effnet_b1
python3 tools/train/train_CTC_torch.py -i config/effnet_b1/CTC_brightfield_b_RGU.py -o CTC/effnet_b1/BrU_3cls_1_effnet_b1
python3 tools/train/train_CTC_torch.py -i config/effnet_b1/CTC_brightfield_b_RGU.py -o CTC/effnet_b1/BrU_3cls_2_effnet_b1

python3 tools/train/train_CTC_torch.py -i config/effnet_b1/CTC_brightfield_RGU.py -o CTC/effnet_b1/Br_3cls_0_effnet_b1
python3 tools/train/train_CTC_torch.py -i config/effnet_b1/CTC_brightfield_RGU.py -o CTC/effnet_b1/Br_3cls_1_effnet_b1
python3 tools/train/train_CTC_torch.py -i config/effnet_b1/CTC_brightfield_RGU.py -o CTC/effnet_b1/Br_3cls_2_effnet_b1

python3 tools/train/train_CTC_torch.py -i config/effnet_b1/CTC_brightfield_fl_RGU.py -o CTC/effnet_b1/BrFl_3cls_0_effnet_b1
python3 tools/train/train_CTC_torch.py -i config/effnet_b1/CTC_brightfield_fl_RGU.py -o CTC/effnet_b1/BrFl_3cls_1_effnet_b1
python3 tools/train/train_CTC_torch.py -i config/effnet_b1/CTC_brightfield_fl_RGU.py -o CTC/effnet_b1/BrFl_3cls_2_effnet_b1


python3 tools/train/train_CTC_torch.py -i config/effnet_b7/CTC_brightfield_b_RGU.py -o CTC/effnet_b7/BrU_3cls_0_effnet_b7
python3 tools/train/train_CTC_torch.py -i config/effnet_b7/CTC_brightfield_b_RGU.py -o CTC/effnet_b7/BrU_3cls_1_effnet_b7
python3 tools/train/train_CTC_torch.py -i config/effnet_b7/CTC_brightfield_b_RGU.py -o CTC/effnet_b7/BrU_3cls_2_effnet_b7

python3 tools/train/train_CTC_torch.py -i config/effnet_b7/CTC_brightfield_RGU.py -o CTC/effnet_b7/Br_3cls_0_effnet_b7
python3 tools/train/train_CTC_torch.py -i config/effnet_b7/CTC_brightfield_RGU.py -o CTC/effnet_b7/Br_3cls_1_effnet_b7
python3 tools/train/train_CTC_torch.py -i config/effnet_b7/CTC_brightfield_RGU.py -o CTC/effnet_b7/Br_3cls_2_effnet_b7

python3 tools/train/train_CTC_torch.py -i config/effnet_b7/CTC_brightfield_fl_RGU.py -o CTC/effnet_b7/BrFl_3cls_0_effnet_b7
python3 tools/train/train_CTC_torch.py -i config/effnet_b7/CTC_brightfield_fl_RGU.py -o CTC/effnet_b7/BrFl_3cls_1_effnet_b7
python3 tools/train/train_CTC_torch.py -i config/effnet_b7/CTC_brightfield_fl_RGU.py -o CTC/effnet_b7/BrFl_3cls_2_effnet_b7


python3 tools/test/eval.py -i config/effnet_b1/CTC_brightfield_fl_RGU.py -o CTC/effnet_b1/BrFl_3cls_0_effnet_b1