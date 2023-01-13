python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_0_f50 -f 0.5
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_1_f50 -f 0.5
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_2_f50 -f 0.5

python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_0_f20 -f 0.2
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_1_f20 -f 0.2
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_2_f20 -f 0.2

python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_0_f10 -f 0.1
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_1_f10 -f 0.1
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_2_f10 -f 0.1

python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_0_f5 -f 0.05
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_1_f5 -f 0.05
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_2_f5 -f 0.05

python3 tools/test/eval.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_0_f50 
python3 tools/test/eval.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_1_f50 
python3 tools/test/eval.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_2_f50 

python3 tools/test/eval.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_0_f20 
python3 tools/test/eval.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_1_f20 
python3 tools/test/eval.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_2_f20

python3 tools/test/eval.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_0_f10
python3 tools/test/eval.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_1_f10
python3 tools/test/eval.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_2_f10 

python3 tools/test/eval.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_0_f5 
python3 tools/test/eval.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_1_f5 
python3 tools/test/eval.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish_fraction/BrU_3cls_2_f5 



python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_0_f50 -f 0.5
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_1_f50 -f 0.5
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_2_f50 -f 0.5

python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_0_f20 -f 0.2
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_1_f20 -f 0.2
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_2_f20 -f 0.2

python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_0_f10 -f 0.1
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_1_f10 -f 0.1
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_2_f10 -f 0.1

python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_0_f5 -f 0.05
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_1_f5 -f 0.05
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_2_f5 -f 0.05


python3 tools/test/eval.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_0_f50 
python3 tools/test/eval.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_1_f50 
python3 tools/test/eval.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_2_f50 

python3 tools/test/eval.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_0_f20 
python3 tools/test/eval.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_1_f20 
python3 tools/test/eval.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_2_f20

python3 tools/test/eval.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_0_f10
python3 tools/test/eval.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_1_f10
python3 tools/test/eval.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_2_f10 

python3 tools/test/eval.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_0_f5 
python3 tools/test/eval.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_1_f5 
python3 tools/test/eval.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish_fraction/Brfl_3cls_2_f5 


python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_0_f50 -f 0.5
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_1_f50 -f 0.5
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_2_f50 -f 0.5

python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_0_f20 -f 0.2
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_1_f20 -f 0.2
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_2_f20 -f 0.2

python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_0_f10 -f 0.1
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_1_f10 -f 0.1
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_2_f10 -f 0.1

python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_1_f5 -f 0.05
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_2_f5 -f 0.05
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_0_f5 -f 0.05


python3 tools/test/eval.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_0_f50
python3 tools/test/eval.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_1_f50
python3 tools/test/eval.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_2_f50

python3 tools/test/eval.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_0_f20 
python3 tools/test/eval.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_1_f20
python3 tools/test/eval.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_2_f20

python3 tools/test/eval.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_0_f10
python3 tools/test/eval.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_1_f10
python3 tools/test/eval.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_2_f10

python3 tools/test/eval.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_1_f5 
python3 tools/test/eval.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_2_f5 
python3 tools/test/eval.py -i config/CTC_brightfield_RGU.py -o CTC/publish_fraction/Br_3cls_0_f5
