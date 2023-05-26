python tools/train.py configs/CTC/CTC_brightfield_b_dino.py --work-dir checkpoints/CTC_brightfield_b_dino_0
python tools/train.py configs/CTC/CTC_brightfield_b_dino.py --work-dir checkpoints/CTC_brightfield_b_dino_1
python tools/train.py configs/CTC/CTC_brightfield_b_dino.py --work-dir checkpoints/CTC_brightfield_b_dino_2

python tools/train.py configs/CTC/CTC_brightfield_dino.py --work-dir checkpoints/CTC_brightfield_dino_1
python tools/train.py configs/CTC/CTC_brightfield_dino.py --work-dir checkpoints/CTC_brightfield_dino_2

python tools/train.py configs/CTC/CTC_brightfield_b_dino.py --work-dir checkpoints/CTC_brightfield_b_dino_test

python tools/test.py configs/CTC/CTC_brightfield_b_dino.py checkpoints/CTC_brightfield_b_dino_0/epoch_8.pth  --out CTC_result/test.pkl

python tools/train.py configs/CTC/CTC_brightfield_fl_dino.py --work-dir checkpoints/CTC_brightfield_fl_dino_test

python tools/test.py configs/CTC/CTC_brightfield_fl_dino.py checkpoints/CTC_brightfield_fl_dino_test/epoch_32.pth  --out CTC_result/test.pkl

python tools/train.py configs/CTC/CTC_brightfield_fl.py --work-dir checkpoints/CTC_brightfield_fl_test
python tools/test.py configs/CTC/CTC_brightfield_fl.py checkpoints/CTC_brightfield_fl_test/epoch_32.pth  --out CTC_result/test.pkl



sleep 4h;
python tools/train.py configs/CTC/CTC_brightfield_dino.py --work-dir checkpoints/CTC_brightfield_dino_0_n
python tools/train.py configs/CTC/CTC_brightfield_dino.py --work-dir checkpoints/CTC_brightfield_dino_1_n
python tools/train.py configs/CTC/CTC_brightfield_dino.py --work-dir checkpoints/CTC_brightfield_dino_2_n


python tools/train.py configs/CTC/CTC_brightfield_dino.py --work-dir checkpoints/CTC_brightfield_dino_0_nt
python tools/train.py configs/CTC/CTC_brightfield_dino.py --work-dir checkpoints/CTC_brightfield_dino_1_nt

python tools/train.py configs/CTC/CTC_brightfield_b_dino.py --work-dir checkpoints/CTC_brightfield_b_dino_0_nt
python tools/train.py configs/CTC/CTC_brightfield_b_dino.py --work-dir checkpoints/CTC_brightfield_b_dino_1_nt

python tools/train.py configs/CTC/CTC_brightfield_fcos.py --work-dir checkpoints/CTC_brightfield_fcos_0
python tools/train.py configs/CTC/CTC_brightfield_fcos.py --work-dir checkpoints/CTC_brightfield_fcos_1
python tools/train.py configs/CTC/CTC_brightfield_fcos.py --work-dir checkpoints/CTC_brightfield_fcos_2

python tools/train.py configs/CTC/CTC_brightfield_b_fcos.py --work-dir checkpoints/CTC_brightfield_b_fcos_0
python tools/train.py configs/CTC/CTC_brightfield_b_fcos.py --work-dir checkpoints/CTC_brightfield_b_fcos_1
python tools/train.py configs/CTC/CTC_brightfield_b_fcos.py --work-dir checkpoints/CTC_brightfield_b_fcos_2


python tools/train.py configs/CTC/CTC_brightfield_fl_fcos.py --work-dir checkpoints/CTC_brightfield_fl_fcos_0
python tools/train.py configs/CTC/CTC_brightfield_fl_fcos.py --work-dir checkpoints/CTC_brightfield_fl_fcos_1
python tools/train.py configs/CTC/CTC_brightfield_fl_fcos.py --work-dir checkpoints/CTC_brightfield_fl_fcos_2


python tools/train.py configs/CTC/CTC_brightfield_fl_dino.py --work-dir checkpoints/CTC_brightfield_fl_dino_0_n
python tools/train.py configs/CTC/CTC_brightfield_fl_dino.py --work-dir checkpoints/CTC_brightfield_fl_dino_1_n
python tools/train.py configs/CTC/CTC_brightfield_fl_dino.py --work-dir checkpoints/CTC_brightfield_fl_dino_2_n

python tools/test.py configs/CTC/CTC_brightfield_fl_fcos.py checkpoints/CTC_brightfield_fl_fcos_0/epoch_32.pth  --out CTC_result/CTC_brightfield_fl_fcos_0.pkl
python tools/test.py configs/CTC/CTC_brightfield_fl_fcos.py checkpoints/CTC_brightfield_fl_fcos_1/epoch_32.pth  --out CTC_result/CTC_brightfield_fl_fcos_1.pkl
python tools/test.py configs/CTC/CTC_brightfield_fl_fcos.py checkpoints/CTC_brightfield_fl_fcos_2/epoch_32.pth  --out CTC_result/CTC_brightfield_fl_fcos_2.pkl

python tools/test.py configs/CTC/CTC_brightfield_b_fcos.py checkpoints/CTC_brightfield_b_fcos_0/epoch_32.pth  --out CTC_result/CTC_brightfield_b_fcos_0.pkl
python tools/test.py configs/CTC/CTC_brightfield_b_fcos.py checkpoints/CTC_brightfield_b_fcos_1/epoch_32.pth  --out CTC_result/CTC_brightfield_b_fcos_1.pkl
python tools/test.py configs/CTC/CTC_brightfield_b_fcos.py checkpoints/CTC_brightfield_b_fcos_2/epoch_32.pth  --out CTC_result/CTC_brightfield_b_fcos_2.pkl

python tools/test.py configs/CTC/CTC_brightfield_fcos.py checkpoints/CTC_brightfield_fcos_0/epoch_32.pth  --out CTC_result/CTC_brightfield_fcos_0.pkl
python tools/test.py configs/CTC/CTC_brightfield_fcos.py checkpoints/CTC_brightfield_fcos_1/epoch_32.pth  --out CTC_result/CTC_brightfield_fcos_1.pkl
python tools/test.py configs/CTC/CTC_brightfield_fcos.py checkpoints/CTC_brightfield_fcos_2/epoch_32.pth  --out CTC_result/CTC_brightfield_fcos_2.pkl


python tools/train.py configs/CTC/CTC_brightfield_b_dino.py --work-dir checkpoints/CTC_brightfield_b_dino_0_nn


python tools/train.py configs/CTC/CTC_brightfield_b_dino.py --work-dir checkpoints/CTC_brightfield_b_dino_0_norm
python tools/train.py configs/CTC/CTC_brightfield_b_dino.py --work-dir checkpoints/CTC_brightfield_b_dino_1_norm
python tools/train.py configs/CTC/CTC_brightfield_b_dino.py --work-dir checkpoints/CTC_brightfield_b_dino_2_norm

python tools/train.py configs/CTC/CTC_brightfield_b_dino_2x.py --work-dir checkpoints/CTC_brightfield_b_dino_0_norm_2x
python tools/train.py configs/CTC/CTC_brightfield_b_dino_2x.py --work-dir checkpoints/CTC_brightfield_b_dino_1_norm_2x
python tools/train.py configs/CTC/CTC_brightfield_b_dino_2x.py --work-dir checkpoints/CTC_brightfield_b_dino_2_norm_2x

python tools/test.py configs/CTC/CTC_brightfield_fl_dino.py checkpoints/CTC_brightfield_fl_dino_0/epoch_32.pth  --out CTC_result/CTC_brightfield_fl_dino_0.pkl
python tools/test.py configs/CTC/CTC_brightfield_dino.py checkpoints/CTC_brightfield_dino_0/epoch_32.pth  --out CTC_result/CTC_brightfield_dino_0.pkl



python tools/train.py configs/CTC/CTC_brightfield_fl_dino.py --work-dir checkpoints/CTC_brightfield_fl_dino_0_final
python tools/train.py configs/CTC/CTC_brightfield_fl_dino.py --work-dir checkpoints/CTC_brightfield_fl_dino_1_final
python tools/train.py configs/CTC/CTC_brightfield_fl_dino.py --work-dir checkpoints/CTC_brightfield_fl_dino_2_final

python tools/train.py configs/CTC/CTC_brightfield_b_dino.py --work-dir checkpoints/CTC_brightfield_b_dino_0_final
python tools/train.py configs/CTC/CTC_brightfield_b_dino.py --work-dir checkpoints/CTC_brightfield_b_dino_1_final
python tools/train.py configs/CTC/CTC_brightfield_b_dino.py --work-dir checkpoints/CTC_brightfield_b_dino_2_final

python tools/train.py configs/CTC/CTC_brightfield_dino.py --work-dir checkpoints/CTC_brightfield_dino_0_final
python tools/train.py configs/CTC/CTC_brightfield_dino.py --work-dir checkpoints/CTC_brightfield_dino_1_final
python tools/train.py configs/CTC/CTC_brightfield_dino.py --work-dir checkpoints/CTC_brightfield_dino_2_final

python tools/test.py configs/CTC/CTC_brightfield_dino.py checkpoints/CTC_brightfield_dino_0_final/epoch_32.pth  --out CTC_result/CTC_brightfield_dino_0_final.pkl
python tools/test.py configs/CTC/CTC_brightfield_dino.py checkpoints/CTC_brightfield_dino_1_final/epoch_32.pth  --out CTC_result/CTC_brightfield_dino_1_final.pkl
python tools/test.py configs/CTC/CTC_brightfield_dino.py checkpoints/CTC_brightfield_dino_2_final/epoch_32.pth  --out CTC_result/CTC_brightfield_dino_2_final.pkl

python tools/test.py configs/CTC/CTC_brightfield_b_dino.py checkpoints/CTC_brightfield_b_dino_0_final/epoch_32.pth  --out CTC_result/CTC_brightfield_b_dino_0_final.pkl
python tools/test.py configs/CTC/CTC_brightfield_b_dino.py checkpoints/CTC_brightfield_b_dino_1_final/epoch_32.pth  --out CTC_result/CTC_brightfield_b_dino_1_final.pkl
python tools/test.py configs/CTC/CTC_brightfield_b_dino.py checkpoints/CTC_brightfield_b_dino_2_final/epoch_32.pth  --out CTC_result/CTC_brightfield_b_dino_2_final.pkl

python tools/test.py configs/CTC/CTC_brightfield_fl_dino.py checkpoints/CTC_brightfield_fl_dino_0_final/epoch_32.pth  --out CTC_result/CTC_brightfield_fl_dino_0_final.pkl
python tools/test.py configs/CTC/CTC_brightfield_fl_dino.py checkpoints/CTC_brightfield_fl_dino_1_final/epoch_32.pth  --out CTC_result/CTC_brightfield_fl_dino_1_final.pkl
python tools/test.py configs/CTC/CTC_brightfield_fl_dino.py checkpoints/CTC_brightfield_fl_dino_2_final/epoch_32.pth  --out CTC_result/CTC_brightfield_fl_dino_2_final.pkl

python tools/train.py configs/CTC/CTC_brightfield_fl_dino.py --work-dir checkpoints/CTC_brightfield_fl_dino_3_final

python tools/test.py configs/CTC/CTC_brightfield_fl_dino.py checkpoints/CTC_brightfield_fl_dino_1_final/epoch_32.pth  --out CTC_result/CTC_brightfield_fl_dino_1_final.pkl
python tools/test.py configs/CTC/CTC_brightfield_fl_dino.py checkpoints/CTC_brightfield_fl_dino_3_final/epoch_32.pth  --out CTC_result/CTC_brightfield_fl_dino_3_final.pkl

python tools/train.py configs/CTC/CTC_brightfield_fl_dino.py --work-dir checkpoints/CTC_brightfield_fl_dino_1_RGU
python tools/test.py configs/CTC/CTC_brightfield_fl_dino.py checkpoints/CTC_brightfield_fl_dino_0_RGU/epoch_12.pth  --out CTC_result/CTC_brightfield_fl_dino_0_RGU.pkl
