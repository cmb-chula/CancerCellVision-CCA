python3 tools/utils/format_output.py -ip ../detection2/mmdetection/CTC_result/CTC_brightfield_b_dino_0_final.pkl \
-op CTC_result/test.pkl --generate_image

python3 tools/test/eval_image_level.py -ip CTC_result/test.pkl

python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fix_main/BrU_3cls_0 \
-ip CTC_result/test.pkl -op CTC_result/test2.pkl -c 1
python3 tools/test/eval_image_level.py -ip CTC_result/test2.pkl

python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fix_main/BrU_3cls_0 \
-ip CTC_result/test.pkl -op CTC_result/test2.pkl -c 0.7
python3 tools/test/eval_image_level.py -ip CTC_result/test2.pkl

