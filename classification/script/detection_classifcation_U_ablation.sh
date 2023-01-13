#brightfield + hoescht eval
python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_b_0_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_b_brightfield_0.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RG.py -o CTC/abalation_U/BrU_2cls_0 \
-ip CTC_result/CTC_b_brightfield_0.pkl -op CTC_result/CTC_b_brightfield_0_2cls_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_0_2cls_clsweight=70p.pkl


python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_b_1_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_b_brightfield_1.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RG.py -o CTC/abalation_U/BrU_2cls_1 \
-ip CTC_result/CTC_b_brightfield_1.pkl -op CTC_result/CTC_b_brightfield_1_2cls_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_1_2cls_clsweight=70p.pkl


python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_b_2_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_b_brightfield_2.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RG.py -o CTC/abalation_U/BrU_2cls_2 \
-ip CTC_result/CTC_b_brightfield_2.pkl -op CTC_result/CTC_b_brightfield_2_2cls_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_2_2cls_clsweight=70p.pkl



#brightfield + hoescht ablation1
python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_b_0_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_b_brightfield_0.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RG_ablation1.py -o CTC/abalation_U/BrU_2cls_ablation1_0 \
-ip CTC_result/CTC_b_brightfield_0.pkl -op CTC_result/CTC_b_brightfield_0_2cls_ablation1_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_0_2cls_ablation1_clsweight=70p.pkl

python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_b_1_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_b_brightfield_1.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RG_ablation1.py -o CTC/abalation_U/BrU_2cls_ablation1_1 \
-ip CTC_result/CTC_b_brightfield_1.pkl -op CTC_result/CTC_b_brightfield_1_2cls_ablation1_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_1_2cls_ablation1_clsweight=70p.pkl

python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_b_2_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_b_brightfield_2.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RG_ablation1.py -o CTC/abalation_U/BrU_2cls_ablation1_2 \
-ip CTC_result/CTC_b_brightfield_2.pkl -op CTC_result/CTC_b_brightfield_2_2cls_ablation1_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_2_2cls_ablation1_clsweight=70p.pkl



python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_b_0_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_b_brightfield_0.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RG_ablation2.py -o CTC/abalation_U/BrU_2cls_ablation2_0 \
-ip CTC_result/CTC_b_brightfield_0.pkl -op CTC_result/CTC_b_brightfield_0_2cls_ablation2_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_0_2cls_ablation2_clsweight=70p.pkl

python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_b_1_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_b_brightfield_1.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RG_ablation2.py -o CTC/abalation_U/BrU_2cls_ablation2_1 \
-ip CTC_result/CTC_b_brightfield_1.pkl -op CTC_result/CTC_b_brightfield_1_2cls_ablation2_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_1_2cls_ablation2_clsweight=70p.pkl

python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_b_2_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_b_brightfield_2.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RG_ablation2.py -o CTC/abalation_U/BrU_2cls_ablation2_2 \
-ip CTC_result/CTC_b_brightfield_2.pkl -op CTC_result/CTC_b_brightfield_2_2cls_ablation2_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_2_2cls_ablation2_clsweight=70p.pkl


