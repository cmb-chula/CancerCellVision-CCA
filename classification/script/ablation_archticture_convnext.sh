python3 tools/train/train_CTC_torch.py -i config/convnext_s/CTC_brightfield_b_RGU.py -o     
python3 tools/train/train_CTC_torch.py -i config/convnext_s/CTC_brightfield_b_RGU.py -o CTC/convnext_s/BrU_3cls_1_convnext_s_2x
python3 tools/train/train_CTC_torch.py -i config/convnext_s/CTC_brightfield_b_RGU.py -o CTC/convnext_s/BrU_3cls_2_convnext_s_2x

python3 tools/train/train_CTC_torch.py -i config/convnext_s/CTC_brightfield_RGU.py -o CTC/convnext_s/Br_3cls_0_convnext_s_2x
python3 tools/train/train_CTC_torch.py -i config/convnext_s/CTC_brightfield_RGU.py -o CTC/convnext_s/Br_3cls_1_convnext_s_2x
python3 tools/train/train_CTC_torch.py -i config/convnext_s/CTC_brightfield_RGU.py -o CTC/convnext_s/Br_3cls_2_convnext_s_2x

python3 tools/train/train_CTC_torch.py -i config/convnext_s/CTC_brightfield_fl_RGU.py -o CTC/convnext_s/BrFl_3cls_0_convnext_s_2x
python3 tools/train/train_CTC_torch.py -i config/convnext_s/CTC_brightfield_fl_RGU.py -o CTC/convnext_s/BrFl_3cls_1_convnext_s_2x
python3 tools/train/train_CTC_torch.py -i config/convnext_s/CTC_brightfield_fl_RGU.py -o CTC/convnext_s/BrFl_3cls_2_convnext_s_2x


python3 tools/test/eval.py -i config/Swin_B/CTC_brightfield_b_RGU.py -o CTC/Swin_B/BrU_3cls_0_Swin_B_2x

python3 tools/train/train_CTC_torch.py -i config/convnext_l/CTC_brightfield_b_RGU.py -o CTC/convnext_l/BrU_3cls_1_convnext_l
python3 tools/train/train_CTC_torch.py -i config/convnext_l/CTC_brightfield_b_RGU.py -o CTC/convnext_l/BrU_3cls_2_convnext_l

python3 tools/train/train_CTC_torch.py -i config/convnext_l/CTC_brightfield_RGU.py -o CTC/convnext_l/Br_3cls_0_convnext_l
python3 tools/train/train_CTC_torch.py -i config/convnext_l/CTC_brightfield_RGU.py -o CTC/convnext_l/Br_3cls_1_convnext_l
python3 tools/train/train_CTC_torch.py -i config/convnext_l/CTC_brightfield_RGU.py -o CTC/convnext_l/Br_3cls_2_convnext_l
python3 tools/train/train_CTC_torch.py -i config/convnext_l/CTC_brightfield_b_RGU.py -o CTC/convnext_l/BrU_3cls_0_convnext_l

python3 tools/train/train_CTC_torch.py -i config/convnext_l/CTC_brightfield_fl_RGU.py -o CTC/convnext_l/BrFl_3cls_0_convnext_l
python3 tools/train/train_CTC_torch.py -i config/convnext_l/CTC_brightfield_fl_RGU.py -o CTC/convnext_l/BrFl_3cls_1_convnext_l
python3 tools/train/train_CTC_torch.py -i config/convnext_l/CTC_brightfield_fl_RGU.py -o CTC/convnext_l/BrFl_3cls_2_convnext_l


python3 tools/train/train_CTC_torch.py -i config/effnet_b4/CTC_brightfield_b_RGU.py -o CTC/test

