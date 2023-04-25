import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, drop=0.5, bias=False, activation=None):
        super(GraphConv, self).__init__()
        self.dropout = nn.Dropout(drop)
        self.activation = activation
        self.w = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.w.weight)
        self.bias = bias
        if self.bias:
            nn.init.zeros_(self.w.bias)

    def forward(self, adj, x):
        x = self.dropout(x)
        x = adj.mm(x)
        x = self.w(x)
        if self.activation:
            return self.activation(x)
        else:
            return x


class GCN(nn.Module):
    def __init__(self, hid_dim, out_dim, bias=False):
        super(GCN, self).__init__()
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.res1 = GraphConv(out_dim, hid_dim, bias=bias, activation=F.relu)
        self.res2 = GraphConv(hid_dim, hid_dim, bias=bias, activation=torch.tanh)
        self.res3 = GraphConv(hid_dim, hid_dim, bias=bias, activation=F.relu)
        self.res4 = GraphConv(hid_dim, out_dim, bias=bias, activation=torch.sigmoid)

    def forward(self, g, z):
        z = self.res1(g, z)
        res = self.res4(g, z)

        return res
