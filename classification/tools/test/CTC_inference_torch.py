import sys
sys.path.append('./')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import argparse
import glob
from engine.trainer_torch import do_train
from data.CTC_loader_torch import CTCLoader
import pickle

from model import build_model_torch
from sklearn.model_selection import train_test_split
import numpy as np
from utils.utils import load_config
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# ---------------------------path configuration------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--ckpt_path", required=True, help="Checkpoint directory")
parser.add_argument("-i", "--config_path", required=True, help="Path to config file")
parser.add_argument("-c", "--cls_weight", required=False, default=1, help="confidence weight of classification model (1 - omega)")
parser.add_argument("-ip", "--input_path", required=True, help="Input pickle file path")
parser.add_argument("-op", "--output_path", required=True, help="Output pickle file path")

args = vars(parser.parse_args())

name = args["ckpt_path"]
cfg = load_config(args["config_path"])
base_pkl = pickle.load(open(args["input_path"], "rb"))
cls_weight = float(args["cls_weight"])
assert 0 <= cls_weight <= 1 

print(cfg.data_cfg.visualize)


ckpt_path = 'log/{}/{}'.format(name, 'model')
# log_path = 'log/{}/model_final.h5'.format(name)
log_path = 'log/{}/model.pth'.format(name)

cfg.data_cfg.return_index = True
cfg.data_cfg.quality = False

dst = 'data/dataset/CTC_image_for_classification_inference'
data_path = np.array( glob.glob('{}/brightfield/R/*'.format(dst)) )
print(len(data_path))

test_loader = torch.utils.data.DataLoader(CTCLoader(data_path, cfg.data_cfg, flag = 'test'), batch_size=cfg.data_cfg.BATCH_SIZE )
print("Number of proposed cells : ", len(base_pkl['raw'].keys()))


model = build_model_torch(cfg.data_cfg).cuda()
model = torch.load(log_path)

if(cfg.data_cfg.visualize):
    model.remove_top()

from tqdm import tqdm
pbar = tqdm()

cls_prediction_dict = {}
for test_data in test_loader:
    X_test, Y_test, idx = test_data
    X_test = X_test.cuda()
    # X_test = np.transpose(X_test, (0, 3, 1, 2))
    # X_test = torch.tensor(X_test, dtype = torch.float32).cuda()

    import torch.nn.functional as F
    res = model(X_test)
    if(not cfg.data_cfg.visualize):
        res = F.softmax(res)
    res = res.cpu().detach().numpy()

    if(cfg.data_cfg.visualize):
        y_pred = np.array(res, dtype = np.float32)
        file_list = [i.split('/')[-1].split('.')[0] for i in idx]
        pred = list(y_pred)

        for i, j in zip(file_list, pred):
            cls_prediction_dict[i] = j
    else:
        if(cfg.data_cfg.train_mode == 'CTC'):
            y_pred = np.array(res[0] , dtype = np.float32)
        else:
            y_pred = np.array(res, dtype = np.float32)
        file_list = [i.split('/')[-1].split('.')[0] for i in idx]

        pred = list(y_pred[:, 0])

        for i, j in zip(file_list, pred):
            cls_prediction_dict[i] = j

    pbar.update(1)
pbar.close()


#override detection confidence with classfier's confidence
for cell_name in base_pkl['raw'].keys():
    base_pkl['raw'][cell_name][-1] = (1 - cls_weight) * base_pkl['raw'][cell_name][-1] + (cls_weight) * cls_prediction_dict[cell_name]

#reformat prediction metadata
for key in base_pkl.keys():
    if(key == 'raw'): continue
    base_pkl[key]['pred'] = []

for cell_id in base_pkl['raw'].keys():
    image_id = cell_id.split('_')[0]
    base_pkl[image_id]['pred'].append(base_pkl['raw'][cell_id])

#create pickle for newly processed data
pickle.dump(base_pkl, open(args["output_path"], 'wb'))

