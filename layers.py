import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class hyperedge_encoder(nn.Module):
    def __init__(self, num_in_edge, num_hidden, dropout, act=F.tanh):  # 1408, 32
        super(hyperedge_encoder, self).__init__()
        self.num_in_edge = num_in_edge
        self.num_hidden = num_hidden
        # self.num_out_features = num_out_features
        self.dropout = dropout
        self.act = act

        self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_edge, self.num_hidden), dtype=torch.float))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.float))
        # self.W1 = nn.Linear(self.num_in_edge, self.num_hidden, bias=True)
        # nn.init.xavier_normal_(self.W1.weight, gain=1.414)

    def forward(self, H_T):
        z1 = self.act(torch.mm(H_T, self.W1) + self.b1)
        return z1

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_in_edge) + ' -> ' + str(self.num_hidden)


class node_encoder(nn.Module):
    def __init__(self, num_in_node, num_hidden, dropout, act=F.tanh):  # 1482, 32
        super(node_encoder, self).__init__()
        self.num_in_node = num_in_node
        self.num_hidden = num_hidden
        # self.num_out_features = num_out_features
        self.dropout = dropout
        self.act = act
        self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_node, self.num_hidden), dtype=torch.float))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.float))
        # self.W1 = nn.Linear(self.num_in_node, self.num_hidden, bias=True)
        # nn.init.xavier_normal_(self.W1.weight, gain=1.414)

    def forward(self, H):
        z1 = self.act(H.mm(self.W1) + self.b1)
        return z1

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_in_node) + ' -> ' + str(self.num_hidden)


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):  #
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):  # x: torch.Tensor, G: torch.Tensor
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class HGNN1(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN1, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)

    def forward(self, x, G):
        x = F.dropout(x, self.dropout)
        x = F.tanh(self.hgc1(x, G))
        return x


class HGNN2(nn.Module):
    def __init__(self, in_ch, n_hid, n_class, dropout=0.5):
        super(HGNN2, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = F.dropout(x, self.dropout)
        x = F.tanh(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x


class sample_latents(nn.Module):  # 图卷积
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(sample_latents, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):  #（参数初始化）
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)  # uniform_(from=0, to=1) → Tensor 将tensor用从均匀分布中抽样得到的值填充。
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, z):  # D_V1, z
        # support = torch.mm(input, self.weight)
        # output = torch.mm(adj, support)
        support = torch.mm(z, self.weight)
        output = torch.mm(input, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self): #内置的方法，使对象具体化
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class decoder1(nn.Module):
    # def __init__(self, test_idx, num_re=16, num_in=16, num_out_node=1004, num_out_edge=387, dropout=0.5):  # 32, 32, 1482, 1408
    def __init__(self, in_features, out_features, dropout=0.5, bias=True):  # 824, 16
        super(decoder1, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):  # （参数初始化）
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)  # uniform_(from=0, to=1) → Tensor 将tensor用从均匀分布中抽样得到的值填充。
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, H_DT_T, z):
        z_T = self.dropout(z).t()
        sim_output = torch.sigmoid((F.tanh(H_DT_T.mm(self.weight) + self.bias)).mm(z_T))

        return sim_output


class decoder2(nn.Module):
    def __init__(self, dropout=0.5, act=torch.sigmoid):  # 32, 32, 1482, 1408
        super(decoder2, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, z_node, z_hyperedge):  # 1408*16 1482*16
        z_node_ = self.dropout(z_node)
        z_hyperedge_ = self.dropout(z_hyperedge)
        z = self.act(z_node_.mm(z_hyperedge_.t()))
        return z


class hgnn(nn.Module):
    def __init__(self, num_in, num_hidden, act=F.tanh, bias=True):  # 1482, 32
        super(hgnn, self).__init__()
        self.num_in = num_in
        self.num_hidden = num_hidden
        # self.num_out_features = num_out_features
        self.act = act
        bias = bias
        self.weight = nn.Parameter(torch.zeros(size=(self.num_in, self.num_hidden), dtype=torch.float))
        # nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.num_hidden))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # self.W1 = nn.Linear(self.num_in_node, self.num_hidden, bias=True)
        # nn.init.xavier_normal_(self.W1.weight, gain=1.414)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, H):
        z1 = self.act(H.mm(self.weight) + self.bias)
        return z1

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_in) + ' -> ' + str(self.num_hidden)


