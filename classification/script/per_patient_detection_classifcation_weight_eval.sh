#train on all patient
python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_b_0_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_b_brightfield_0.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fix_main/BrU_3cls_0 \
-ip CTC_result/CTC_b_brightfield_0.pkl -op CTC_result/CTC_b_brightfield_0_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_0_clsweight=70p.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_0_clsweight=70p.pkl -pid r06
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_0_clsweight=70p.pkl -pid r07
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_0_clsweight=70p.pkl -pid r08


python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_b_1_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_b_brightfield_1.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fix_main/BrU_3cls_1 \
-ip CTC_result/CTC_b_brightfield_1.pkl -op CTC_result/CTC_b_brightfield_1_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_1_clsweight=70p.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_1_clsweight=70p.pkl -pid r06
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_1_clsweight=70p.pkl -pid r07
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_1_clsweight=70p.pkl -pid r08


python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_b_2_wnorm_test_1080_fix.pkl \
-op CTC_result/CTC_b_brightfield_2.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fix_main/BrU_3cls_2 \
-ip CTC_result/CTC_b_brightfield_2.pkl -op CTC_result/CTC_b_brightfield_2_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_2_clsweight=70p.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_2_clsweight=70p.pkl -pid r06
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_2_clsweight=70p.pkl -pid r07
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_2_clsweight=70p.pkl -pid r08




#train on p06
python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/patient_test/CTC_brightfield_b_r06_0.pkl \
-op CTC_result/CTC_brightfield_b_r06_0.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_patient/BrU_3cls_0_r06 \
-ip CTC_result/CTC_brightfield_b_r06_0.pkl -op CTC_result/CTC_brightfield_b_r06_0_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r06_0_clsweight=70p.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r06_0_clsweight=70p.pkl -pid r06
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r06_0_clsweight=70p.pkl -pid r07
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r06_0_clsweight=70p.pkl -pid r08


python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/patient_test/CTC_brightfield_b_r06_1.pkl \
-op CTC_result/CTC_brightfield_b_r06_1.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_patient/BrU_3cls_1_r06 \
-ip CTC_result/CTC_brightfield_b_r06_1.pkl -op CTC_result/CTC_brightfield_b_r06_1_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r06_1_clsweight=70p.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r06_1_clsweight=70p.pkl -pid r06
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r06_1_clsweight=70p.pkl -pid r07
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r06_1_clsweight=70p.pkl -pid r08


python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/patient_test/CTC_brightfield_b_r06_2.pkl \
-op CTC_result/CTC_brightfield_b_r06_2.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_patient/BrU_3cls_2_r06 \
-ip CTC_result/CTC_brightfield_b_r06_2.pkl -op CTC_result/CTC_brightfield_b_r06_2_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r06_2_clsweight=70p.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r06_2_clsweight=70p.pkl -pid r06
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r06_2_clsweight=70p.pkl -pid r07
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r06_2_clsweight=70p.pkl -pid r08





#train on p07
python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/patient_test/CTC_brightfield_b_r07_0.pkl \
-op CTC_result/CTC_brightfield_b_r07_0.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_patient/BrU_3cls_0_r07 \
-ip CTC_result/CTC_brightfield_b_r07_0.pkl -op CTC_result/CTC_brightfield_b_r07_0_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r07_0_clsweight=70p.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r07_0_clsweight=70p.pkl -pid r06
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r07_0_clsweight=70p.pkl -pid r07
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r07_0_clsweight=70p.pkl -pid r08


python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/patient_test/CTC_brightfield_b_r07_1.pkl \
-op CTC_result/CTC_brightfield_b_r07_1.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_patient/BrU_3cls_1_r07 \
-ip CTC_result/CTC_brightfield_b_r07_1.pkl -op CTC_result/CTC_brightfield_b_r07_1_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r07_1_clsweight=70p.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r07_1_clsweight=70p.pkl -pid r06
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r07_1_clsweight=70p.pkl -pid r07
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r07_1_clsweight=70p.pkl -pid r08


python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/patient_test/CTC_brightfield_b_r07_2.pkl \
-op CTC_result/CTC_brightfield_b_r07_2.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_patient/BrU_3cls_2_r07 \
-ip CTC_result/CTC_brightfield_b_r07_2.pkl -op CTC_result/CTC_brightfield_b_r07_2_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r07_2_clsweight=70p.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r07_2_clsweight=70p.pkl -pid r06
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r07_2_clsweight=70p.pkl -pid r07
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r07_2_clsweight=70p.pkl -pid r08



#train on p08
python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/patient_test/CTC_brightfield_b_r08_0.pkl \
-op CTC_result/CTC_brightfield_b_r08_0.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_patient/BrU_3cls_0_r08 \
-ip CTC_result/CTC_brightfield_b_r08_0.pkl -op CTC_result/CTC_brightfield_b_r08_0_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r08_0_clsweight=70p.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r08_0_clsweight=70p.pkl -pid r06
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r08_0_clsweight=70p.pkl -pid r07
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r08_0_clsweight=70p.pkl -pid r08


python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/patient_test/CTC_brightfield_b_r08_1.pkl \
-op CTC_result/CTC_brightfield_b_r08_1.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_patient/BrU_3cls_1_r08 \
-ip CTC_result/CTC_brightfield_b_r08_1.pkl -op CTC_result/CTC_brightfield_b_r08_1_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r08_1_clsweight=70p.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r08_1_clsweight=70p.pkl -pid r06
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r08_1_clsweight=70p.pkl -pid r07
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r08_1_clsweight=70p.pkl -pid r08


python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/patient_test/CTC_brightfield_b_r08_2.pkl \
-op CTC_result/CTC_brightfield_b_r08_2.pkl --generate_image
python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_patient/BrU_3cls_2_r08 \
-ip CTC_result/CTC_brightfield_b_r08_2.pkl -op CTC_result/CTC_brightfield_b_r08_2_clsweight=70p.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r08_2_clsweight=70p.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r08_2_clsweight=70p.pkl -pid r06
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r08_2_clsweight=70p.pkl -pid r07
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_b_r08_2_clsweight=70p.pkl -pid r08

