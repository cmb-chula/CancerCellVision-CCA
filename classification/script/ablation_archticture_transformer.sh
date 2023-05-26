
python3 tools/train/train_CTC_torch.py -i config/VIT_B/CTC_brightfield_b_RGU.py -o CTC/VIT_B/BrU_3cls_0_VIT_B_2x
python3 tools/train/train_CTC_torch.py -i config/VIT_B/CTC_brightfield_b_RGU.py -o CTC/VIT_B/BrU_3cls_1_VIT_B_2x
python3 tools/train/train_CTC_torch.py -i config/VIT_B/CTC_brightfield_b_RGU.py -o CTC/VIT_B/BrU_3cls_2_VIT_B_2x

python3 tools/train/train_CTC_torch.py -i config/VIT_B/CTC_brightfield_RGU.py -o CTC/VIT_B/Br_3cls_0_VIT_B_2x
python3 tools/train/train_CTC_torch.py -i config/VIT_B/CTC_brightfield_RGU.py -o CTC/VIT_B/Br_3cls_1_VIT_B_2x
python3 tools/train/train_CTC_torch.py -i config/VIT_B/CTC_brightfield_RGU.py -o CTC/VIT_B/Br_3cls_2_VIT_B_2x

python3 tools/train/train_CTC_torch.py -i config/VIT_B/CTC_brightfield_fl_RGU.py -o CTC/VIT_B/BrFl_3cls_0_VIT_B_2x
python3 tools/train/train_CTC_torch.py -i config/VIT_B/CTC_brightfield_fl_RGU.py -o CTC/VIT_B/BrFl_3cls_1_VIT_B_2x
python3 tools/train/train_CTC_torch.py -i config/VIT_B/CTC_brightfield_fl_RGU.py -o CTC/VIT_B/BrFl_3cls_2_VIT_B_2x

python3 tools/test/eval.py -i config/VIT_B/CTC_brightfield_fl_RGU.py -o CTC/VIT_B/BrFl_3cls_0_VIT_B_2x


python3 tools/train/train_CTC_torch.py -i config/VIT_L/CTC_brightfield_b_RGU.py -o CTC/VIT_L/BrU_3cls_0_VIT_L_2x
python3 tools/train/train_CTC_torch.py -i config/VIT_L/CTC_brightfield_b_RGU.py -o CTC/VIT_L/BrU_3cls_1_VIT_L_2x
python3 tools/train/train_CTC_torch.py -i config/VIT_L/CTC_brightfield_b_RGU.py -o CTC/VIT_L/BrU_3cls_2_VIT_L_2x

python3 tools/train/train_CTC_torch.py -i config/VIT_L/CTC_brightfield_RGU.py -o CTC/VIT_L/Br_3cls_0_VIT_L_2x
python3 tools/train/train_CTC_torch.py -i config/VIT_L/CTC_brightfield_RGU.py -o CTC/VIT_L/Br_3cls_1_VIT_L_2x
python3 tools/train/train_CTC_torch.py -i config/VIT_L/CTC_brightfield_RGU.py -o CTC/VIT_L/Br_3cls_2_VIT_L_2x

python3 tools/train/train_CTC_torch.py -i config/VIT_L/CTC_brightfield_fl_RGU.py -o CTC/VIT_L/BrFl_3cls_0_VIT_L_2x
python3 tools/train/train_CTC_torch.py -i config/VIT_L/CTC_brightfield_fl_RGU.py -o CTC/VIT_L/BrFl_3cls_1_VIT_L_2x
python3 tools/train/train_CTC_torch.py -i config/VIT_L/CTC_brightfield_fl_RGU.py -o CTC/VIT_L/BrFl_3cls_2_VIT_L_2x

python3 tools/test/eval.py -i config/VIT_L/CTC_brightfield_fl_RGU.py -o CTC/VIT_L/BrFl_3cls_0_VIT_L_2x



python3 tools/train/train_CTC_torch.py -i config/VIT_B_224/CTC_brightfield_b_RGU.py -o CTC/VIT_B_224/BrU_3cls_0_VIT_B_224_2x
python3 tools/train/train_CTC_torch.py -i config/VIT_B_224/CTC_brightfield_b_RGU.py -o CTC/VIT_B_224/BrU_3cls_1_VIT_B_224_2x
python3 tools/train/train_CTC_torch.py -i config/VIT_B_224/CTC_brightfield_b_RGU.py -o CTC/VIT_B_224/BrU_3cls_2_VIT_B_224_2x

python3 tools/train/train_CTC_torch.py -i config/VIT_B_224/CTC_brightfield_RGU.py -o CTC/VIT_B_224/Br_3cls_0_VIT_B_224_2x
python3 tools/train/train_CTC_torch.py -i config/VIT_B_224/CTC_brightfield_RGU.py -o CTC/VIT_B_224/Br_3cls_1_VIT_B_224_2x
python3 tools/train/train_CTC_torch.py -i config/VIT_B_224/CTC_brightfield_RGU.py -o CTC/VIT_B_224/Br_3cls_2_VIT_B_224_2x

python3 tools/train/train_CTC_torch.py -i config/VIT_B_224/CTC_brightfield_fl_RGU.py -o CTC/VIT_B_224/BrFl_3cls_0_VIT_B_224_2x
python3 tools/train/train_CTC_torch.py -i config/VIT_B_224/CTC_brightfield_fl_RGU.py -o CTC/VIT_B_224/BrFl_3cls_1_VIT_B_224_2x
python3 tools/train/train_CTC_torch.py -i config/VIT_B_224/CTC_brightfield_fl_RGU.py -o CTC/VIT_B_224/BrFl_3cls_2_VIT_B_224_2x

python3 tools/test/eval.py -i config/VIT_B/CTC_brightfield_fl_RGU.py -o CTC/VIT_B/BrFl_3cls_0_VIT_B_2x

