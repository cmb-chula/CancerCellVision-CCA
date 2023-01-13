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

from model import build_model_torch
from sklearn.model_selection import train_test_split
import numpy as np
from utils.utils import load_config
import pickle
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_curve, auc, accuracy_score, roc_auc_score

# ---------------------------path configuration------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--ckpt_path", required=True, help="Checkpoint directory")
parser.add_argument("-i", "--config_path", required=True, help="Path to config file")
parser.add_argument('--detail', action='store_true', help='generate detailed prediction and embedding')

args = vars(parser.parse_args())

name = args["ckpt_path"]
cfg = load_config(args["config_path"])
get_full_detail = args["detail"]
ckpt_path = 'log/{}/{}'.format(name, 'model')
log_path = 'log/{}/model.pth'.format(name)

cfg.data_cfg.return_index = True
cfg.data_cfg.quality = False

data_path = np.array( glob.glob('{}/*/*'.format(cfg.data_cfg.test_path)) )
print(len(data_path))
test_loader = torch.utils.data.DataLoader(CTCLoader(data_path, cfg.data_cfg, flag = 'test'), batch_size=cfg.data_cfg.BATCH_SIZE )

model = build_model_torch(cfg.data_cfg).cuda()
model = torch.load(log_path)

pbar = tqdm()

predictions = {}
for test_data in test_loader:
    X_test, Y_test, idx = test_data
    X_test = X_test.cuda()

    import torch.nn.functional as F
    if(get_full_detail):
        pred_conf, embeddings = model(X_test, return_embedding = True)
        embeddings = embeddings.cpu().detach().numpy()
    else:
        pred_conf = model(X_test)

    if(not cfg.data_cfg.visualize):
        pred_conf = F.softmax(pred_conf, dim = 1)
    pred_conf = pred_conf.cpu().detach().numpy()

    for i in range(len(X_test)):
        filename = idx[i].split('/')[-1].split('.')[0]
        predictions[filename] = {}
        predictions[filename]['pred_conf'] = pred_conf[i]
        predictions[filename]['true_class'] = Y_test[i]
        predictions[filename]['real_class'] = idx[i].split('/')[-2]
        if(get_full_detail):
            predictions[filename]['embedding'] = embeddings[i]

    pbar.update(1)
pbar.close()

y_pred_raw = np.array([predictions[i]['pred_conf'] for i in predictions], dtype = np.float32)
y_true = np.array([predictions[i]['true_class'] for i in predictions], dtype = np.float32)
y_true_binary = y_true.copy()
y_true_binary[y_true_binary > 1] = 1


max_thresh = -1
max_f1, max_rec, max_acc = -1, -1, -1
comparator = lambda x, thresh : 1 if x < thresh else 0

for threshold in np.arange(0.05, 1, 0.01):
    y_pred = np.array([ comparator(i[0], threshold) for i in y_pred_raw.copy()])
    prec, rec, f1=  precision_score(y_true_binary, y_pred, pos_label = 0), recall_score(y_true_binary, y_pred, pos_label = 0), f1_score(y_true_binary, y_pred, pos_label = 0)
    print("thresh = {:.2f}, prec = {:.2f}, rec = {:.2f}, F1 = {:.2f}".format(threshold, 100*prec, 100*rec, 100*f1))
    
    if(f1 < 0.001): 
        print("TERMINATED DUE TO TOO LOW F1")
        break

    if(f1 > max_f1):
        max_f1, max_rec, max_prec = f1, rec, prec
        max_thresh = threshold


print("Best thresh = {:.2f}, prec = {:.2f}, rec = {:.2f}, F1 = {:.2f}".format(max_thresh, 100*max_prec, 100*max_rec, 100*max_f1))
# print("AUC = {:.2f}".format(100 * roc_auc_score(y_true, np.array([predictions[i]['pred_conf'] for i in predictions], dtype = np.float32), multi_class = 'ovr')))
print("AUC = {:.2f}".format(100 * roc_auc_score(y_true_binary, 1 - y_pred_raw[:, 0])))


y_pred_argmax =  np.argmax(y_pred_raw, axis = 1)
print("accuracy = {:.2f}".format(100 * accuracy_score(y_true, y_pred_argmax)))

conf_matrix= confusion_matrix(y_true, y_pred_argmax)
print(conf_matrix)
print(np.around(100 * conf_matrix  / conf_matrix.sum(axis = 1).reshape(-1, 1), decimals = 2))
predictions['_metadata'] = {'max_thresh' : max_thresh, 'confusion_matrix' : conf_matrix}


y_pred = np.array([ comparator(i[0], max_thresh) for i in y_pred_raw.copy()])
conf_matrix= confusion_matrix(y_true_binary, y_pred)
print(conf_matrix)
print(np.around(100 * conf_matrix / conf_matrix.sum(axis = 1).reshape(-1, 1), decimals = 2))
print(classification_report(y_true_binary, y_pred))

if(get_full_detail):
    pickle.dump(predictions, open('metadata/{}.pkl'.format(name.split('/')[-1]), 'wb'))