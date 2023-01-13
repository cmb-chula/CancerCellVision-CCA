python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_b_0_wnorm_test_1080_fix.pkl -op CTC_result/CTC_b_brightfield_0.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_0.pkl

python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_b_1_wnorm_test_1080_fix.pkl -op CTC_result/CTC_b_brightfield_1.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_1.pkl

python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_b_2_wnorm_test_1080_fix.pkl -op CTC_result/CTC_b_brightfield_2.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_b_brightfield_2.pkl

python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_0_wnorm_test_1080_fix.pkl -op CTC_result/CTC_brightfield_0.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_0.pkl

python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_1_wnorm_test_1080_fix.pkl -op CTC_result/CTC_brightfield_1.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_1.pkl

python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_2_wnorm_test_1080_fix.pkl -op CTC_result/CTC_brightfield_2.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_2.pkl

python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_fl_0_wnorm_test_1080_fix.pkl -op CTC_result/CTC_fl_brightfield_0.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_fl_brightfield_0.pkl

python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_fl_1_wnorm_test_1080_fix.pkl -op CTC_result/CTC_fl_brightfield_1.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_fl_brightfield_1.pkl

python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield_fl_2_wnorm_test_1080_fix.pkl -op CTC_result/CTC_fl_brightfield_2.pkl
python3 tools/test/eval_image_level.py -ip CTC_result/CTC_fl_brightfield_2.pkl

