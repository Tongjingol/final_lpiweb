import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.preprocessing import minmax_scale
import time
from gcn import GCN
from metrics import *
from omic_learn_2 import *
from utils.ui_helper import objdict


def s5_main(state):
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=12, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.015,
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-7,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=250, help='Dimension of representations')
    parser.add_argument('--alpha', type=float, default=0.62,
                        help='Weight between lncRNA space and protein space')
    parser.add_argument('--beta', type=float, default=1.5,
                        help='Hyperparameter beta')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.seed = state.random_seed
    set_seed(args.seed, args.cuda)
    args.epochs = state.Epochs
    args.weight_decay = state.weight_decay
    args.lr = state.learning_rate
    args.hidden = state.hidden
    args.alpha = state.alpha
    args.beta = state.beta

    def scaley(ymat):
        return (ymat - ymat.min()) / ymat.max()

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

    t = time.time()
    lpi = np.loadtxt('./dataset1/interaction.csv', delimiter=',')
    lpi = torch.from_numpy(lpi).float()

    rnafeat = np.loadtxt('./dataset1/' + state.sample_file + '.csv', delimiter=',')
    protfeat = np.loadtxt('./dataset1/' + state.sample_file_protein + '.csv', delimiter=',')

    rnafeat = minmax_scale(rnafeat, axis=0)
    rnafeatorch = torch.from_numpy(rnafeat).float()

    protfeat = minmax_scale(protfeat, axis=0)
    protfeatorch = torch.from_numpy(protfeat).float()

    gl = norm_adj(rnafeat)
    gp = norm_adj(protfeat)

    if args.cuda:
        lpi = lpi.cuda()
        gm = gl.cuda()
        gd = gp.cuda()
        rnafeatorch = rnafeatorch.cuda()
        protfeatorch = protfeatorch.cuda()

    class GCNs(nn.Module):
        def __init__(self):
            super(GCNs, self).__init__()
            self.gcnl = GCN(args.hidden, lpi.shape[1])
            self.gcnp = GCN(args.hidden, lpi.shape[0])

        def forward(self, y0):
            yl = self.gcnl(gm, y0)
            yp = self.gcnp(gd, y0.t())
            return yl, yp

    def train(gcn, y0, epoch, alpha):
        beta = args.beta
        if state.optimizer == "Adam":
            optp = torch.optim.Adam(gcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif state.optimizer == "SGD":
            optp = torch.optim.SGD(gcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif state.optimizer == "Adagrad":
            optp = torch.optim.Adagrad(gcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif state.optimizer == "RMSprop":
            optp = torch.optim.RMSprop(gcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif state.optimizer == "AdamW":
            optp = torch.optim.AdamW(gcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for e in range(epoch):
            gcn.train()
            yl, yp = gcn(y0)
            lossl = F.binary_cross_entropy(yl, y0)
            lossp = F.binary_cross_entropy(yp, y0.t())
            loss = beta * (alpha * lossl + (1 - alpha) * lossp)

            optp.zero_grad()
            loss.backward()
            optp.step()
            gcn.eval()
            with torch.no_grad():
                yl, yp = gcn(y0)

            if e % 20 == 0 and e != 0:
                st.info('Epoch %d | Lossp: %.4f' % (e, loss.item()))

        return alpha * yl + (1 - alpha) * yp.t()

    def trainres(A0):
        gcn = GCNs()
        if args.cuda:
            gcn = gcn.cuda()

        train(gcn, A0, args.epochs, args.alpha)
        gcn.eval()
        yli, ypi = gcn(A0)
        resi = args.alpha * yli + (1 - args.alpha) * ypi.t()
        return resi

    def fivefoldcv_sign(A):
        N = A.shape[0]
        idx = np.arange(N)
        np.random.shuffle(idx)
        aurocl = np.zeros(state.cv_splits)
        auprl = np.zeros(state.cv_splits)
        pl = np.zeros(state.cv_splits)
        rl = np.zeros(state.cv_splits)
        f1l = np.zeros(state.cv_splits)

        for i in range(state.cv_splits):
            st.info("Fold {}".format(i + 1))
            A0 = A.clone()
            for j in range(i * N // state.cv_splits, (i + 1) * N // state.cv_splits):
                A0[idx[j], :] = torch.zeros(A.shape[1])
            resi = trainres(A0)
            if args.cuda:
                resi = resi.cpu().detach().numpy()
            else:
                resi = resi.detach().numpy()

            auroc, aupr, p, r, f1 = show_auc(resi, i)
            aurocl[i] = auroc
            auprl[i] = aupr
            pl[i] = p
            rl[i] = r
            f1l[i] = f1

        st.info("===Final result===")
        st.info('AUROC= %.4f \u00B1 %.4f | AUPR= %.4f \u00B1 %.4f | Recall= %.4f \u00B1 %.4f | Precision= %.4f \u00B1 '
                '%.4f| F1score= %.4f\u00B1 %.4f'
                % (aurocl.mean(), aurocl.std(), auprl.mean(), auprl.std(), rl.mean(), rl.std(), pl.mean(), pl.std(),
                   f1l.mean(), f1l.std()))
        state["auc"] = aurocl.mean()
        state["aupr"] = aurocl.mean()
        state["recall"] = rl.mean()
        state["precision"] = pl.mean()
        state["f1"] = f1l.mean()
    fivefoldcv_sign(lpi)
    return state
