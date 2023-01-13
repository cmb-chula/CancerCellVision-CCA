import sys
sys.path.append('./')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import argparse
import glob
from engine.evaluate import evaluate
from engine.trainer_torch import do_train
from data.CTC_loader_torch import CTCLoader
from model import build_model_torch
from sklearn.model_selection import train_test_split
import numpy as np
from utils.utils import load_config
import pickle
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter

# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# ---------------------------path configuration------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--ckpt_path", required=True, help="Checkpoint directory")
parser.add_argument("-i", "--config_path", required=True, help="Path to config file")
parser.add_argument("-f", "--fraction", required=False, default=1, help="Training fraction")


args = vars(parser.parse_args())

name = args["ckpt_path"]
cfg = load_config(args["config_path"])
training_fraction = float(args["fraction"])
assert training_fraction > 0

cfg.data_cfg.visualize = False

target_dir = 'log/' + name + '/'
if not os.path.exists(target_dir): 
    os.makedirs(target_dir)

ckpt_path = 'log/{}/{}'.format(name, 'model')
log_path = 'log/{}'.format(name)

# ---------------------------fetch data------------------------
if(type(cfg.data_cfg.train_path) is not list):
    train_data = glob.glob('{}/*/*'.format(cfg.data_cfg.train_path))
else:
    train_data = []
    for i in cfg.data_cfg.train_path:
        train_data += glob.glob('{}/*/*'.format(i))


if(training_fraction != 1):
    train_data, _ = train_test_split(train_data, train_size = training_fraction)
val_data = np.array(glob.glob('{}/*/*'.format(cfg.data_cfg.val_path)))
print("N train = ", len(train_data), "N val = ",  len(val_data))

import torch
from torch.utils.data import BatchSampler, SequentialSampler
train_loader = torch.utils.data.DataLoader(CTCLoader(train_data, cfg.data_cfg, flag = 'train'), 
                    batch_size=cfg.data_cfg.BATCH_SIZE, shuffle = True, pin_memory = True)
val_loader = torch.utils.data.DataLoader(CTCLoader(val_data, cfg.data_cfg, flag = 'test'), batch_size=cfg.data_cfg.BATCH_SIZE )

model = build_model_torch(cfg.data_cfg).cuda()
writer = SummaryWriter(log_path)

do_train(writer, cfg.data_cfg, model, train_loader, val_loader, scheduler=cfg.scheduler_cfg, ckpt_path=ckpt_path, criterion = None)
