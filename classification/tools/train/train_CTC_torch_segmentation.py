import sys
sys.path.append('./')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import argparse
import glob
from engine.evaluate import evaluate
from engine.trainer_segmentation_torch import do_train
from data.CTC_segmentaion_loader_torch import CTCLoader
from model import build_model_torch
from sklearn.model_selection import train_test_split
import numpy as np
from utils.utils import load_config
import pickle
from torch.utils.tensorboard import SummaryWriter

# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# ---------------------------path configuration------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--ckpt_path", required=True, help="Checkpoint directory")
parser.add_argument("-i", "--config_path", required=True, help="Path to config file")
parser.add_argument("-w", "--seg_weight", required=False, help="Seg weight")

args = vars(parser.parse_args())

name = args["ckpt_path"]
cfg = load_config(args["config_path"])
seg_weight = float(args["seg_weight"])

cfg.data_cfg.visualize = False

target_dir = 'log/' + name + '/'
if not os.path.exists(target_dir): 
    os.makedirs(target_dir)

ckpt_path = 'log/{}/{}'.format(name, 'model')
log_path = 'log/{}'.format(name)

# ---------------------------fetch data------------------------
train_data = np.array(glob.glob('{}/*/*'.format(cfg.data_cfg.train_path)))
val_data = np.array(glob.glob('{}/*/*'.format(cfg.data_cfg.val_path)))
print("N train = ", len(train_data), "N val = ",  len(val_data))

import torch
from torch.utils.data import BatchSampler, SequentialSampler
train_loader = torch.utils.data.DataLoader(CTCLoader(train_data, cfg.data_cfg, flag = 'train'), 
                    batch_size=cfg.data_cfg.BATCH_SIZE, shuffle = True, pin_memory = True)
val_loader = torch.utils.data.DataLoader(CTCLoader(val_data, cfg.data_cfg, flag = 'test'), batch_size=cfg.data_cfg.BATCH_SIZE )

model = build_model_torch(cfg.data_cfg).cuda()

writer = SummaryWriter(log_path)

do_train(writer, cfg.data_cfg, model, train_loader, val_loader, scheduler=cfg.scheduler_cfg, ckpt_path=ckpt_path, criterion = None, seg_weight = seg_weight)
