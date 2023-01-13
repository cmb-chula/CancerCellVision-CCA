
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU.py -o CTC/publish/Br_3cls_0
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU.py -o CTC/publish/Br_3cls_1
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU.py -o CTC/publish/Br_3cls_2

python3 tools/test/eval.py -i config/CTC_brightfield_RGU.py -o CTC/publish/Br_3cls_0
python3 tools/test/eval.py -i config/CTC_brightfield_RGU.py -o CTC/publish/Br_3cls_1
python3 tools/test/eval.py -i config/CTC_brightfield_RGU.py -o CTC/publish/Br_3cls_2



python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish/BrU_3cls_0
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish/BrU_3cls_1
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish/BrU_3cls_2

python3 tools/test/eval.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish/BrU_3cls_0 
python3 tools/test/eval.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish/BrU_3cls_1
python3 tools/test/eval.py -i config/CTC_brightfield_b_RGU.py -o CTC/publish/BrU_3cls_2



python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish/Brfl_3cls_0
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish/Brfl_3cls_1
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish/Brfl_3cls_2

python3 tools/test/eval.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish/Brfl_3cls_0
python3 tools/test/eval.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish/Brfl_3cls_1
python3 tools/test/eval.py -i config/CTC_brightfield_fl_RGU.py -o CTC/publish/Brfl_3cls_2

