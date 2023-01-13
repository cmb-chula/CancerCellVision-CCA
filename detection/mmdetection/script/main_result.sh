python tools/train.py  configs/CTC/CTC_brightfield.py --seed 40 --work-dir checkpoints/CTC_brightfield_0_wnorm_test_1080_fix
python tools/train.py  configs/CTC/CTC_brightfield.py --seed 41 --work-dir checkpoints/CTC_brightfield_1_wnorm_test_1080_fix
python tools/train.py  configs/CTC/CTC_brightfield.py --seed 42 --work-dir checkpoints/CTC_brightfield_2_wnorm_test_1080_fix

python tools/test.py  configs/CTC/CTC_brightfield.py checkpoints/CTC_brightfield_0_wnorm_test_1080_fix/epoch_8.pth --out CTC_result/CTC_brightfield_0_wnorm_test_1080_fix.pkl
python tools/test.py  configs/CTC/CTC_brightfield.py checkpoints/CTC_brightfield_1_wnorm_test_1080_fix/epoch_8.pth --out CTC_result/CTC_brightfield_1_wnorm_test_1080_fix.pkl
python tools/test.py  configs/CTC/CTC_brightfield.py checkpoints/CTC_brightfield_2_wnorm_test_1080_fix/epoch_8.pth --out CTC_result/CTC_brightfield_2_wnorm_test_1080_fix.pkl


python tools/train.py  configs/CTC/CTC_brightfield_b.py --seed 40 --work-dir checkpoints/CTC_brightfield_b_0_wnorm_test_1080_fix
python tools/train.py  configs/CTC/CTC_brightfield_b.py --seed 41 --work-dir checkpoints/CTC_brightfield_b_1_wnorm_test_1080_fix
python tools/train.py  configs/CTC/CTC_brightfield_b.py --seed 42 --work-dir checkpoints/CTC_brightfield_b_2_wnorm_test_1080_fix

python tools/test.py  configs/CTC/CTC_brightfield_b.py checkpoints/CTC_brightfield_b_0_wnorm_test_1080_fix/epoch_8.pth --out CTC_result/CTC_brightfield_b_0_wnorm_test_1080_fix.pkl
python tools/test.py  configs/CTC/CTC_brightfield_b.py checkpoints/CTC_brightfield_b_1_wnorm_test_1080_fix/epoch_8.pth --out CTC_result/CTC_brightfield_b_1_wnorm_test_1080_fix.pkl
python tools/test.py  configs/CTC/CTC_brightfield_b.py checkpoints/CTC_brightfield_b_2_wnorm_test_1080_fix/epoch_8.pth --out CTC_result/CTC_brightfield_b_2_wnorm_test_1080_fix.pkl


python tools/train.py  configs/CTC/CTC_brightfield_fl.py --seed 40 --work-dir checkpoints/CTC_brightfield_fl_0_wnorm_test_1080_fix
python tools/train.py  configs/CTC/CTC_brightfield_fl.py --seed 41 --work-dir checkpoints/CTC_brightfield_fl_1_wnorm_test_1080_fix
python tools/train.py  configs/CTC/CTC_brightfield_fl.py --seed 42 --work-dir checkpoints/CTC_brightfield_fl_2_wnorm_test_1080_fix

python tools/test.py  configs/CTC/CTC_brightfield_fl.py checkpoints/CTC_brightfield_fl_0_wnorm_test_1080_fix/epoch_8.pth --out CTC_result/CTC_brightfield_fl_0_wnorm_test_1080_fix.pkl
python tools/test.py  configs/CTC/CTC_brightfield_fl.py checkpoints/CTC_brightfield_fl_1_wnorm_test_1080_fix/epoch_8.pth --out CTC_result/CTC_brightfield_fl_1_wnorm_test_1080_fix.pkl
python tools/test.py  configs/CTC/CTC_brightfield_fl.py checkpoints/CTC_brightfield_fl_2_wnorm_test_1080_fix/epoch_8.pth --out CTC_result/CTC_brightfield_fl_2_wnorm_test_1080_fix.pkl