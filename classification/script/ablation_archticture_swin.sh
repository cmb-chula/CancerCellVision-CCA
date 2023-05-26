
python3 tools/train/train_CTC_torch.py -i config/Swin_B/CTC_brightfield_b_RGU.py -o CTC/Swin_B/BrU_3cls_0_Swin_B_2x
python3 tools/train/train_CTC_torch.py -i config/Swin_B/CTC_brightfield_b_RGU.py -o CTC/Swin_B/BrU_3cls_1_Swin_B_2x
python3 tools/train/train_CTC_torch.py -i config/Swin_B/CTC_brightfield_b_RGU.py -o CTC/Swin_B/BrU_3cls_2_Swin_B_2x

python3 tools/train/train_CTC_torch.py -i config/Swin_B/CTC_brightfield_RGU.py -o CTC/Swin_B/Br_3cls_0_Swin_B_2x
python3 tools/train/train_CTC_torch.py -i config/Swin_B/CTC_brightfield_RGU.py -o CTC/Swin_B/Br_3cls_1_Swin_B_2x
python3 tools/train/train_CTC_torch.py -i config/Swin_B/CTC_brightfield_RGU.py -o CTC/Swin_B/Br_3cls_2_Swin_B_2x

python3 tools/train/train_CTC_torch.py -i config/Swin_B/CTC_brightfield_fl_RGU.py -o CTC/Swin_B/BrFl_3cls_0_Swin_B_2x
python3 tools/train/train_CTC_torch.py -i config/Swin_B/CTC_brightfield_fl_RGU.py -o CTC/Swin_B/BrFl_3cls_1_Swin_B_2x
python3 tools/train/train_CTC_torch.py -i config/Swin_B/CTC_brightfield_fl_RGU.py -o CTC/Swin_B/BrFl_3cls_2_Swin_B_2x


python3 tools/test/eval.py -i config/Swin_B/CTC_brightfield_b_RGU.py -o CTC/Swin_B/BrU_3cls_0_Swin_B_2x


python3 tools/train/train_CTC_torch.py -i config/Swin_S/CTC_brightfield_b_RGU.py -o CTC/Swin_S/BrU_3cls_0_Swin_S_2x
python3 tools/train/train_CTC_torch.py -i config/Swin_S/CTC_brightfield_b_RGU.py -o CTC/Swin_S/BrU_3cls_1_Swin_S_2x
python3 tools/train/train_CTC_torch.py -i config/Swin_S/CTC_brightfield_b_RGU.py -o CTC/Swin_S/BrU_3cls_2_Swin_S_2x

python3 tools/train/train_CTC_torch.py -i config/Swin_S/CTC_brightfield_RGU.py -o CTC/Swin_S/Br_3cls_0_Swin_S_2x
python3 tools/train/train_CTC_torch.py -i config/Swin_S/CTC_brightfield_RGU.py -o CTC/Swin_S/Br_3cls_1_Swin_S_2x
python3 tools/train/train_CTC_torch.py -i config/Swin_S/CTC_brightfield_RGU.py -o CTC/Swin_S/Br_3cls_2_Swin_S_2x

python3 tools/train/train_CTC_torch.py -i config/Swin_S/CTC_brightfield_fl_RGU.py -o CTC/Swin_S/BrFl_3cls_0_Swin_S_2x
python3 tools/train/train_CTC_torch.py -i config/Swin_S/CTC_brightfield_fl_RGU.py -o CTC/Swin_S/BrFl_3cls_1_Swin_S_2x
python3 tools/train/train_CTC_torch.py -i config/Swin_S/CTC_brightfield_fl_RGU.py -o CTC/Swin_S/BrFl_3cls_2_Swin_S_2x


python3 tools/test/eval.py -i config/Swin_S/CTC_brightfield_fl_RGU.py -o CTC/Swin_S/BrFl_3cls_0_Swin_S_2x

