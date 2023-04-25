import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.preprocessing import minmax_scale
import time
from omic_learn_2 import *
from gcn import GCN
from metrics import *
import streamlit as st


def loocv(state):
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
    args.lr = state.learning_rate
    args.weight_decay = state.weight_decay
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

    lpi = np.loadtxt('./dataset1/interaction.csv', delimiter=',')
    lpi = torch.from_numpy(lpi).float()

    ep = np.loadtxt('./dataset1/ep.csv', delimiter=',')
    ep = torch.from_numpy(ep).float()
    ep = norm_adj(ep)
    ct = np.loadtxt('./dataset1/ct.csv', delimiter=',')
    ct = torch.from_numpy(ct).float()
    ct = norm_adj(ct)
    swl = np.loadtxt('./dataset1/swl.csv', delimiter=',')
    swl = torch.from_numpy(swl).float()
    swl = norm_adj(swl)

    go = np.loadtxt('./dataset1/go.csv', delimiter=',')
    go = torch.from_numpy(go).float()
    go = norm_adj(go)
    swp = np.loadtxt('./dataset1/swp.csv', delimiter=',')
    swp = torch.from_numpy(swp).float()
    swp = norm_adj(swp)
    ps = np.loadtxt('./dataset1/ps.csv', delimiter=',')
    ps = torch.from_numpy(ps).float()
    ps = norm_adj(ps)

    if args.cuda:
        lpi = lpi.cuda()
        ep = ep.cuda()
        ct = ct.cuda()
        swl = swl.cuda()

        go = go.cuda()
        swp = swp.cuda()
        ps = ps.cuda()

    class GCNs(nn.Module):
        def __init__(self):
            super(GCNs, self).__init__()
            self.gcnl = GCN(args.hidden, lpi.shape[1])
            self.gcnp = GCN(args.hidden, lpi.shape[0])

        def forward(self, y0):
            lep = self.gcnl(ep, y0)
            lct = self.gcnl(ct, y0)
            lswl = self.gcnl(swl, y0)

            pgo = self.gcnp(go, y0.t())
            pps = self.gcnp(ps, y0.t())
            pswp = self.gcnp(swp, y0.t())
            return lep, lct, lswl, pgo, pps, pswp

    def train(gcn, y0, epoch, alpha):
        # global f1, f2, f3, f4, f5, f6, f7, f8, f9
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
            lep, lct, lswl, pgo, pps, pswp = gcn(y0)

            lossl1 = F.binary_cross_entropy(lep, y0)
            lossl2 = F.binary_cross_entropy(lct, y0)
            lossl3 = F.binary_cross_entropy(lswl, y0)
            lossp1 = F.binary_cross_entropy(pgo, y0.t())
            lossp2 = F.binary_cross_entropy(pps, y0.t())
            lossp3 = F.binary_cross_entropy(pswp, y0.t())

            loss1 = beta * (alpha * lossl1 + (1 - alpha) * lossp1)
            loss2 = beta * (alpha * lossl1 + (1 - alpha) * lossp2)
            loss3 = beta * (alpha * lossl1 + (1 - alpha) * lossp3)
            loss4 = beta * (alpha * lossl2 + (1 - alpha) * lossp1)
            loss5 = beta * (alpha * lossl2 + (1 - alpha) * lossp2)
            loss6 = beta * (alpha * lossl2 + (1 - alpha) * lossp3)
            loss7 = beta * (alpha * lossl3 + (1 - alpha) * lossp1)
            loss8 = beta * (alpha * lossl3 + (1 - alpha) * lossp2)
            loss9 = beta * (alpha * lossl3 + (1 - alpha) * lossp3)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9) / 9
            # loss = loss1
            optp.zero_grad()
            loss.backward()
            optp.step()
            gcn.eval()
            with torch.no_grad():
                lep, lct, lswl, pgo, pps, pswp = gcn(y0)

            # if e % 20 == 0 and e != 0:
                # st.info('Epoch %d | Lossp: %.4f' % (e, loss.item()))
            f1 = alpha * lep + (1 - alpha) * pgo.t()
            f2 = alpha * lep + (1 - alpha) * pps.t()
            f3 = alpha * lep + (1 - alpha) * pswp.t()
            f4 = alpha * lct + (1 - alpha) * pgo.t()
            f5 = alpha * lct + (1 - alpha) * pps.t()
            f6 = alpha * lct + (1 - alpha) * pswp.t()
            f7 = alpha * lswl + (1 - alpha) * pgo.t()
            f8 = alpha * lswl + (1 - alpha) * pps.t()
            f9 = alpha * lswl + (1 - alpha) * pswp.t()
        return (f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9) / 9
        # return f1

    def trainres(A0):
        gcn = GCNs()
        if args.cuda:
            gcn = gcn.cuda()

        # train(gcn, A0, args.epochs, args.alpha)
        # gcn.eval()
        # lep, lct, lswl, pgo, pps, pswp = gcn(A0)
        # resi = args.alpha * yli + (1 - args.alpha) * ypi.t()
        # f1 = args.alpha * lep + (1 - args.alpha) * pgo.t()
        # f2 = args.alpha * lep + (1 - args.alpha) * pps.t()
        # f3 = args.alpha * lep + (1 - args.alpha) * pswp.t()
        # f4 = args.alpha * lct + (1 - args.alpha) * pgo.t()
        # f5 = args.alpha * lct + (1 - args.alpha) * pps.t()
        # f6 = args.alpha * lct + (1 - args.alpha) * pswp.t()
        # f7 = args.alpha * lswl + (1 - args.alpha) * pgo.t()
        # f8 = args.alpha * lswl + (1 - args.alpha) * pps.t()
        # f9 = args.alpha * lswl + (1 - args.alpha) * pswp.t()
        return train(gcn, A0, args.epochs, args.alpha)

    def fivefoldcv(A):
        A0 = A.clone()
        if state.ID=="protein":
            A0[:state.proteinid - 1] = 0
            resi = trainres(A0)
            if args.cuda:
                resi = resi.cpu().detach().numpy()
            else:
                resi = resi.detach().numpy()

            np.savetxt('./casestudy/pro'+str(state.proteinid)+'.csv', resi, fmt='%10.5f', delimiter=',')
        else:
            A0[state.lncRNAid-1:] = 0
            resi = trainres(A0)
            if args.cuda:
                resi = resi.cpu().detach().numpy()
            else:
                resi = resi.detach().numpy()

            np.savetxt('./casestudy/lnc'+str(state.lncRNAid)+'.csv', resi, fmt='%10.5f', delimiter=',')
    fivefoldcv(lpi)
    return state
