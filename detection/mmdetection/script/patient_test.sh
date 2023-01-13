python tools/train.py  configs/CTC/CTC_brightfield_b_r06.py --seed 40 --work-dir checkpoints/CTC_brightfield_b_r06_0
python tools/train.py  configs/CTC/CTC_brightfield_b_r06.py --seed 41 --work-dir checkpoints/CTC_brightfield_b_r06_1
python tools/train.py  configs/CTC/CTC_brightfield_b_r06.py --seed 42 --work-dir checkpoints/CTC_brightfield_b_r06_2

python tools/test.py  configs/CTC/CTC_brightfield_b_r06.py checkpoints/CTC_brightfield_b_r06_0/epoch_8.pth --out CTC_result/CTC_brightfield_b_r06_0.pkl
python tools/test.py  configs/CTC/CTC_brightfield_b_r06.py checkpoints/CTC_brightfield_b_r06_1/epoch_8.pth --out CTC_result/CTC_brightfield_b_r06_1.pkl
python tools/test.py  configs/CTC/CTC_brightfield_b_r06.py checkpoints/CTC_brightfield_b_r06_2/epoch_8.pth --out CTC_result/CTC_brightfield_b_r06_2.pkl

python tools/train.py  configs/CTC/CTC_brightfield_b_r07.py --seed 40 --work-dir checkpoints/CTC_brightfield_b_r07_0
python tools/train.py  configs/CTC/CTC_brightfield_b_r07.py --seed 41 --work-dir checkpoints/CTC_brightfield_b_r07_1
python tools/train.py  configs/CTC/CTC_brightfield_b_r07.py --seed 42 --work-dir checkpoints/CTC_brightfield_b_r07_2

python tools/test.py  configs/CTC/CTC_brightfield_b_r07.py checkpoints/CTC_brightfield_b_r07_0/epoch_8.pth --out CTC_result/CTC_brightfield_b_r07_0.pkl
python tools/test.py  configs/CTC/CTC_brightfield_b_r07.py checkpoints/CTC_brightfield_b_r07_1/epoch_8.pth --out CTC_result/CTC_brightfield_b_r07_1.pkl
python tools/test.py  configs/CTC/CTC_brightfield_b_r07.py checkpoints/CTC_brightfield_b_r07_2/epoch_8.pth --out CTC_result/CTC_brightfield_b_r07_2.pkl

python tools/train.py  configs/CTC/CTC_brightfield_b_r08.py --seed 40 --work-dir checkpoints/CTC_brightfield_b_r08_0
python tools/train.py  configs/CTC/CTC_brightfield_b_r08.py --seed 41 --work-dir checkpoints/CTC_brightfield_b_r08_1
python tools/train.py  configs/CTC/CTC_brightfield_b_r08.py --seed 42 --work-dir checkpoints/CTC_brightfield_b_r08_2

python tools/test.py  configs/CTC/CTC_brightfield_b_r08.py checkpoints/CTC_brightfield_b_r08_0/epoch_8.pth --out CTC_result/CTC_brightfield_b_r08_0.pkl
python tools/test.py  configs/CTC/CTC_brightfield_b_r08.py checkpoints/CTC_brightfield_b_r08_1/epoch_8.pth --out CTC_result/CTC_brightfield_b_r08_1.pkl
python tools/test.py  configs/CTC/CTC_brightfield_b_r08.py checkpoints/CTC_brightfield_b_r08_2/epoch_8.pth --out CTC_result/CTC_brightfield_b_r08_2.pkl
