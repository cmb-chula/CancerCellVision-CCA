
python3 tools/train/train_CTC_torch.py -i config/resnet/CTC_brightfield_b_RGU.py -o CTC/resnet_publish_fix/BrU_3cls_0_res
python3 tools/train/train_CTC_torch.py -i config/resnet/CTC_brightfield_b_RGU.py -o CTC/resnet_publish_fix/BrU_3cls_1_res
python3 tools/train/train_CTC_torch.py -i config/resnet/CTC_brightfield_b_RGU.py -o CTC/resnet_publish_fix/BrU_3cls_2_res

python3 tools/train/train_CTC_torch.py -i config/resnet/CTC_brightfield_RGU.py -o CTC/resnet_publish_fix/Br_3cls_0_res
python3 tools/train/train_CTC_torch.py -i config/resnet/CTC_brightfield_RGU.py -o CTC/resnet_publish_fix/Br_3cls_1_res
python3 tools/train/train_CTC_torch.py -i config/resnet/CTC_brightfield_RGU.py -o CTC/resnet_publish_fix/Br_3cls_2_res

python3 tools/train/train_CTC_torch.py -i config/resnet/CTC_brightfield_fl_RGU.py -o CTC/resnet_publish_fix/BrFl_3cls_0_res
python3 tools/train/train_CTC_torch.py -i config/resnet/CTC_brightfield_fl_RGU.py -o CTC/resnet_publish_fix/BrFl_3cls_1_res
python3 tools/train/train_CTC_torch.py -i config/resnet/CTC_brightfield_fl_RGU.py -o CTC/resnet_publish_fix/BrFl_3cls_2_res


python3 tools/test/eval.py -i config/resnet/CTC_brightfield_b_RGU.py -o CTC/resnet_publish_fix/BrU_3cls_0_res
python3 tools/test/eval.py -i config/resnet/CTC_brightfield_b_RGU.py -o CTC/resnet_publish_fix/BrU_3cls_1_res
python3 tools/test/eval.py -i config/resnet/CTC_brightfield_b_RGU.py -o CTC/resnet_publish_fix/BrU_3cls_2_res

python3 tools/test/eval.py -i config/resnet/CTC_brightfield_RGU.py -o CTC/resnet_publish_fix/Br_3cls_0_res
python3 tools/test/eval.py -i config/resnet/CTC_brightfield_RGU.py -o CTC/resnet_publish_fix/Br_3cls_1_res
python3 tools/test/eval.py -i config/resnet/CTC_brightfield_RGU.py -o CTC/resnet_publish_fix/Br_3cls_2_res

python3 tools/test/eval.py -i config/resnet/CTC_brightfield_fl_RGU.py -o CTC/resnet_publish_fix/BrFl_3cls_0_res
python3 tools/test/eval.py -i config/resnet/CTC_brightfield_fl_RGU.py -o CTC/resnet_publish_fix/BrFl_3cls_1_res
python3 tools/test/eval.py -i config/resnet/CTC_brightfield_fl_RGU.py -o CTC/resnet_publish_fix/BrFl_3cls_2_res


python3 tools/train/train_CTC_torch.py -i config/densenet/CTC_brightfield_b_RGU.py -o CTC/densenet_publish_fix/BrU_3cls_0_den
python3 tools/train/train_CTC_torch.py -i config/densenet/CTC_brightfield_b_RGU.py -o CTC/densenet_publish_fix/BrU_3cls_1_den
python3 tools/train/train_CTC_torch.py -i config/densenet/CTC_brightfield_b_RGU.py -o CTC/densenet_publish_fix/BrU_3cls_2_den

python3 tools/train/train_CTC_torch.py -i config/densenet/CTC_brightfield_RGU.py -o CTC/densenet_publish_fix/Br_3cls_0_den
python3 tools/train/train_CTC_torch.py -i config/densenet/CTC_brightfield_RGU.py -o CTC/densenet_publish_fix/Br_3cls_1_den
python3 tools/train/train_CTC_torch.py -i config/densenet/CTC_brightfield_RGU.py -o CTC/densenet_publish_fix/Br_3cls_2_den

python3 tools/train/train_CTC_torch.py -i config/densenet/CTC_brightfield_fl_RGU.py -o CTC/densenet_publish_fix/BrFl_3cls_0_den
python3 tools/train/train_CTC_torch.py -i config/densenet/CTC_brightfield_fl_RGU.py -o CTC/densenet_publish_fix/BrFl_3cls_1_den
python3 tools/train/train_CTC_torch.py -i config/densenet/CTC_brightfield_fl_RGU.py -o CTC/densenet_publish_fix/BrFl_3cls_2_den


python3 tools/test/eval.py -i config/densenet/CTC_brightfield_fl_RGU.py -o CTC/densenet_publish_fix/BrFl_3cls_0_den
python3 tools/test/eval.py -i config/densenet/CTC_brightfield_fl_RGU.py -o CTC/densenet_publish_fix/BrFl_3cls_1_den
python3 tools/test/eval.py -i config/densenet/CTC_brightfield_fl_RGU.py -o CTC/densenet_publish_fix/BrFl_3cls_2_den

python3 tools/test/eval.py -i config/densenet/CTC_brightfield_b_RGU.py -o CTC/densenet_publish_fix/BrU_3cls_0_den
python3 tools/test/eval.py -i config/densenet/CTC_brightfield_b_RGU.py -o CTC/densenet_publish_fix/BrU_3cls_1_den
python3 tools/test/eval.py -i config/densenet/CTC_brightfield_b_RGU.py -o CTC/densenet_publish_fix/BrU_3cls_2_den

python3 tools/test/eval.py -i config/densenet/CTC_brightfield_RGU.py -o CTC/densenet_publish_fix/Br_3cls_0_den
python3 tools/test/eval.py -i config/densenet/CTC_brightfield_RGU.py -o CTC/densenet_publish_fix/Br_3cls_1_den
python3 tools/test/eval.py -i config/densenet/CTC_brightfield_RGU.py -o CTC/densenet_publish_fix/Br_3cls_2_den


python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU_effnet.py -o CTC/effent_publish_fix/BrFl_3cls_0_eff
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU_effnet.py -o CTC/effent_publish_fix/BrFl_3cls_1_eff
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU_effnet.py -o CTC/effent_publish_fix/BrFl_3cls_2_eff

python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU_effnet.py -o CTC/effent_publish_fix/BrU_3cls_0_eff
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU_effnet.py -o CTC/effent_publish_fix/BrU_3cls_1_eff
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU_effnet.py -o CTC/effent_publish_fix/BrU_3cls_2_eff

python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU_effnet.py -o CTC/effent_publish_fix/Br_3cls_0_eff
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU_effnet.py -o CTC/effent_publish_fix/Br_3cls_1_eff
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU_effnet.py -o CTC/effent_publish_fix/Br_3cls_2_eff



python3 tools/test/eval.py -i config/CTC_brightfield_fl_RGU_effnet.py -o CTC/effent_publish_fix/BrFl_3cls_0_eff
python3 tools/test/eval.py -i config/CTC_brightfield_fl_RGU_effnet.py -o CTC/effent_publish_fix/BrFl_3cls_1_eff
python3 tools/test/eval.py -i config/CTC_brightfield_fl_RGU_effnet.py -o CTC/effent_publish_fix/BrFl_3cls_2_eff

python3 tools/test/eval.py -i config/CTC_brightfield_b_RGU_effnet.py -o CTC/effent_publish_fix/BrU_3cls_0_eff
python3 tools/test/eval.py -i config/CTC_brightfield_b_RGU_effnet.py -o CTC/effent_publish_fix/BrU_3cls_1_eff
python3 tools/test/eval.py -i config/CTC_brightfield_b_RGU_effnet.py -o CTC/effent_publish_fix/BrU_3cls_2_eff

python3 tools/test/eval.py -i config/CTC_brightfield_RGU_effnet.py -o CTC/effent_publish_fix/Br_3cls_0_eff
python3 tools/test/eval.py -i config/CTC_brightfield_RGU_effnet.py -o CTC/effent_publish_fix/Br_3cls_1_eff
python3 tools/test/eval.py -i config/CTC_brightfield_RGU_effnet.py -o CTC/effent_publish_fix/Br_3cls_2_eff