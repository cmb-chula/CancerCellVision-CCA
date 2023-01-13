
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU_effnet.py -o CTC/effent_publish/BrFl_3cls_0_eff
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU_effnet.py -o CTC/effent_publish/BrFl_3cls_1_eff
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU_effnet.py -o CTC/effent_publish/BrFl_3cls_2_eff

python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU_effnet.py -o CTC/effent_publish/BrU_3cls_0_eff
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU_effnet.py -o CTC/effent_publish/BrU_3cls_1_eff
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU_effnet.py -o CTC/effent_publish/BrU_3cls_2_eff

python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU_effnet.py -o CTC/effent_publish/Br_3cls_0_eff
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU_effnet.py -o CTC/effent_publish/Br_3cls_1_eff
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU_effnet.py -o CTC/effent_publish/Br_3cls_2_eff

# python3 tools/test/eval.py -i config/CTC_brightfield_RGU_effnet.py -o CTC/effent_publish/Br_3cls_0_eff

# python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU_effnet.py -o CTC/effnet_test/test

# python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU_effnet.py -o CTC/effent_test/BrU_3cls_0_eff
# python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU_effnet.py -o CTC/effent_test/BrU_3cls_1_eff
# python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU_effnet.py -o CTC/effent_test/BrU_3cls_2_eff

