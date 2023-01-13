import sys
sys.path.append('./')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import argparse
import glob
from engine.evaluate import evaluate
from engine.trainer_CTC import do_train
from data.build import make_CTC_semantic_loader
from model import build_model
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from utils.utils import load_config
tf.get_logger().setLevel('ERROR')
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20000)])

# ---------------------------path configuration------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--ckpt_path", required=True, help="Checkpoint directory")
parser.add_argument("-i", "--config_path", required=True, help="Path to config file")
parser.add_argument("-a", "--lreg", required=False, default=1, help="lamda_reg")
parser.add_argument("-b", "--lcls", required=False, default=1, help="lamda_cls")

args = vars(parser.parse_args())

name = args["ckpt_path"]
cfg = load_config(args["config_path"])
alpha = float(args["lreg"])
beta = float(args["lcls"])

target_dir = 'log/' + name + '/'
if not os.path.exists(target_dir): 
    os.makedirs(target_dir)

ckpt_path = 'log/{}/{}'.format(name, 'model')
log_path = 'log/{}'.format(name)

# ---------------------------fetch data------------------------
data_path = glob.glob('{}/*/*'.format(cfg.data_cfg.dataset_path))
print(len(data_path))
data_path = np.array(['/'.join(i.split('/')[-2:]) for i in data_path])
train_data, val_data = train_test_split(
    data_path, train_size=0.95, random_state=42)
print(len(train_data))

train_loader = make_CTC_semantic_loader(cfg.data_cfg, train_data, is_train=True)
val_loader = make_CTC_semantic_loader(cfg.data_cfg, val_data, is_train=False, fetch_into_mem = False)

model = build_model(cfg.data_cfg)
train_summary_writer = tf.summary.create_file_writer(log_path)

import tensorflow_addons as tfa
import tensorflow as tf

with train_summary_writer.as_default():
    do_train(cfg.data_cfg, model, train_loader, val_loader,
            scheduler=cfg.scheduler_cfg, ckpt_path=ckpt_path, loss_fn = [tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE), tf.keras.losses.MAE],
            reg_w = alpha, reg_c = beta )