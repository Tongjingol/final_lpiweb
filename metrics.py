import numpy as np
import torch
import argparse
from sklearn.preprocessing import minmax_scale, scale
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
import math
import streamlit as st


def scaley(ymat):
    return (ymat - ymat.min()) / ymat.max()


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def neighborhood(feat, k):
    featprod = np.dot(feat.T, feat)
    smat = np.tile(np.diag(featprod), (feat.shape[1], 1))
    dmat = smat + smat.T - 2 * featprod
    dsort = np.argsort(dmat)[:, 1:k + 1]
    C = np.zeros((feat.shape[1], feat.shape[1]))
    m = 0
    for i in range(feat.shape[1]):
        m = m + 1
        k = 0
        for j in dsort[i]:
            C[i, j] = 1.0
            k = k + 1
    return C


def normalized(wmat):
    deg = np.diag(np.sum(wmat, axis=0))
    degpow = np.power(deg, -0.5)
    degpow[np.isinf(degpow)] = 0
    W = np.dot(np.dot(degpow, wmat), degpow)
    return W


def norm_adj(feat):
    C = neighborhood(feat.T, k=5)
    norm_adj = normalized(C.T * C + np.eye(C.shape[0]))
    g = torch.from_numpy(norm_adj).float()
    return g


def show_auc(ymat, i):
    lpi = np.loadtxt('./dataset1/interaction.csv', delimiter=',')
    y_true = lpi.flatten()
    ymat = ymat.flatten()
    fpr, tpr, rocth = roc_curve(y_true, ymat)
    auroc = auc(fpr, tpr)
    precision, recall, prth = precision_recall_curve(y_true, ymat)

    aupr = auc(recall, precision)
    # st.info('AUROC= %.4f | AUPR= %.4f' % (auroc, aupr))
    # np.savetxt('fusion' + str(i) + '_roc.txt', np.vstack((fpr, tpr)), fmt='%10.5f', delimiter=',')
    # np.savetxt('fusion'+str(i)+'_pr.txt',np.vstack((recall,precision)),fmt='%10.5f',delimiter=',')
    y_true = np.reshape(y_true, [-1])
    y_pred = np.reshape(ymat, [-1])
    y_pred = np.rint(y_pred)

    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred)
    st.info('AUROC= %.4f | AUPR= %.4f| Precision= %.4f | Recall= %.4f | F1score= %.4f' % (auroc, aupr, p, r, f1score))
    return auroc, aupr, p, r, f1score
