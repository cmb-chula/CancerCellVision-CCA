from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.utils import  plot_confusion_matrix
import pickle
import torch.optim as optim
import torch
import torch.nn.functional as F

def accuracy(predictions, trues):
    # print(trues)
    # labels = torch.argmax(trues, dim=1)
    classes = torch.argmax(predictions, dim=1)
    return torch.mean((classes == trues).float())

def do_train(writer, cfg, model, train_loader, val_loader, optimizer=None, scheduler=None, criterion=None, ckpt_path=None):
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.1)
    scaler = torch.cuda.amp.GradScaler()

    train_stat = {}
    max_val_acc = 0
    for step in tqdm(range(scheduler.NUM_ITERATION)):
        for g in optimizer.param_groups:
            g['lr'] = scheduler.schedule(step)
        w = None
        X_train, Y_train, indices = next(iter(train_loader))
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()

        # optimizer.zero_grad()
        # y_pred = model(X_train)
        # train_loss = criterion(y_pred, Y_train)
        # train_loss.backward()
        # optimizer.step()


        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            y_pred = model(X_train)
            train_loss = criterion(y_pred, Y_train)
        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if (step + 1)% 25 == 0:
            train_acc = accuracy(y_pred, Y_train)
            writer.add_scalar('loss/train_loss', train_loss, step)
            writer.add_scalar('lr', scheduler.schedule(step), step)
            writer.add_scalar('accuracy/train_acc', train_acc, step)

        if (step + 1)% scheduler.val_freq == 0:
            with torch.no_grad():
                model.eval()
                y_true, y_pred = [], []
                for val_data in val_loader:
                    X_val, Y_val, indices = val_data

                    X_val = X_val.cuda()
                    Y_val = Y_val.cuda()
                    # print(X_val.shape, indices[-1])
                    p = model(X_val)
                    y_true.append(Y_val)
                    y_pred.append(p)

                y_true = torch.cat( y_true, axis = 0)
                y_pred = torch.cat( y_pred, axis = 0)
                # print(y_true.shape, y_pred.shape)
                val_loss = criterion(y_pred, y_true)
                val_accuracy =  accuracy(y_pred, y_true)

                print("val_acc", val_accuracy)
                metric = val_accuracy
                if(metric > max_val_acc and step > 1000):
                    print("-->", metric)
                    max_val_acc = metric
                    torch.save(model, '{}.pth'.format(ckpt_path))

                y_true = y_true.cpu().detach().numpy()
                y_pred = y_pred.cpu().detach().numpy()
                y_pred = np.argmax(y_pred, axis=1)
                writer.add_figure("Confusion Matrix", plot_confusion_matrix(
                    confusion_matrix(y_true, y_pred), cfg.test_class_mapper.keys()), step)
                writer.add_scalar('loss/val_loss', val_loss, step)
                writer.add_scalar('accuracy/val_acc', val_accuracy, step)
            model.train()

    # pickle.dump(train_stat, open('train_stat_step2.pkl', 'wb'))
    torch.save(model, '{}_final.pth'.format(ckpt_path))
    del train_loader