import numpy as np


def evaluate(model, val_loader):
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    import collections

    X_test, Y_test = val_loader
    y_true = np.argmax(Y_test, axis=1)
    y_pred = model.predict(X_test, verbose=1)
    print(y_pred.shape)
    y_pred_argmax = np.argmax(y_pred, axis=1)

    y_pred_argmax[y_pred_argmax > 1] = 1
    y_true[y_true > 1] = 1
    counter = collections.Counter(y_pred_argmax)

    # print(counter)
    print(classification_report(y_true, y_pred_argmax, digits=4))
    print(confusion_matrix(y_true, y_pred_argmax))
    return y_true, y_pred
