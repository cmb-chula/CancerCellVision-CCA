#brightfield eval
python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_0_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_brightfield_0.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fix_main/Br_3cls_0 \
-ip CTC_result/CTC_brightfield_0.pkl -op CTC_result/CTC_brightfield_0_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_0_clsweight=70p.pkl


python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_1_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_brightfield_1.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fix_main/Br_3cls_1 \
-ip CTC_result/CTC_brightfield_1.pkl -op CTC_result/CTC_brightfield_1_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_1_clsweight=70p.pkl


python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_2_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_brightfield_2.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fix_main/Br_3cls_2 \
-ip CTC_result/CTC_brightfield_2.pkl -op CTC_result/CTC_brightfield_2_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_2_clsweight=70p.pkl



#brightfield + hoescht eval
python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_b_0_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_b_brightfield_0.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fix_main/BrU_3cls_0 \
-ip CTC_result/CTC_b_brightfield_0.pkl -op CTC_result/CTC_b_brightfield_0_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_0_clsweight=70p.pkl

python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_b_1_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_b_brightfield_1.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fix_main/BrU_3cls_1 \
-ip CTC_result/CTC_b_brightfield_1.pkl -op CTC_result/CTC_b_brightfield_1_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_1_clsweight=70p.pkl

python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_b_2_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_b_brightfield_2.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fix_main/BrU_3cls_2 \
-ip CTC_result/CTC_b_brightfield_2.pkl -op CTC_result/CTC_b_brightfield_2_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_2_clsweight=70p.pkl


#brightfield + fluorescence eval
python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_fl_0_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_fl_brightfield_0.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fix_main/BrFl_3cls_0 \
-ip CTC_result/CTC_fl_brightfield_0.pkl -op CTC_result/CTC_fl_brightfield_0_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_fl_brightfield_0_clsweight=70p.pkl

python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_fl_1_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_fl_brightfield_1.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fix_main/BrFl_3cls_1 \
-ip CTC_result/CTC_fl_brightfield_1.pkl -op CTC_result/CTC_fl_brightfield_1_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_fl_brightfield_1_clsweight=70p.pkl

python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_fl_2_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_fl_brightfield_2.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fix_main/BrFl_3cls_2 \
-ip CTC_result/CTC_fl_brightfield_2.pkl -op CTC_result/CTC_fl_brightfield_2_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_fl_brightfield_2_clsweight=70p.pkl




