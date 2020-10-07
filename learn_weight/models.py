#!/usr/bin/python
# -*- encoding:utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers import GraphConvolution	  # 简单的GCN层


class GCN(nn.Module):	# nn.Module类的单继承
    def __init__(self, nfeat, nhid, nclass, dropout, meta_size):
        """
        GCN由两个GraphConvolution层构成,输出为输出层做log_softmax变换的结果
        :param nfeat: 底层节点的参数，feature的个数
        :param nhid: 隐层节点个数
        :param nclass: 最终的分类数
        :param dropout: dropout参数
        :param meta_size: 元路径数量
        """
        super(GCN, self).__init__()
        self.meta_size = meta_size
        self.gc1_outdim = nhid
        self.gc2_outdim = nclass
        # 用Variable引入权重
        self.W = Parameter(torch.FloatTensor(self.meta_size, 1))
        nn.init.xavier_uniform_(self.W.data)
        self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc1代表GraphConvolution()，gc1输入尺寸nfeat，输出尺寸nhid
        self.gc2 = GraphConvolution(nhid, nclass)
        # self.gc2代表GraphConvolution()，gc2输入尺寸nhid，输出尺寸ncalss
        self.dropout = dropout
        # dropout参数

    def forward(self, x, adj):
        """
        :param x: 输入特征
        :param adj: 邻接矩阵
        :return:
        """
        # 每条meta-path分别传入GCN
        gcn_out = []
        shape = x.shape[0]

        for i in range(self.meta_size):
            gcn_out.append(F.relu(self.gc1(x, adj[i])))
            gcn_out[i] = F.relu(self.gc2(gcn_out[i], adj[i]))
            gcn_out[i] = gcn_out[i].view(1, shape * self.gc2_outdim)
        x = gcn_out[0]
        for i in range(1, self.meta_size):
            x = torch.cat((x, gcn_out[i]), 0)
        x = torch.t(x)
        # print(self.W)
        x = F.relu(torch.mm(x, self.W))
        x = x.view(shape, self.gc2_outdim)
        # training=self.training表示将模型整体的training状态参数传入dropout函数，没有此参数无法进行dropout
        x = F.dropout(x, self.dropout, training=self.training)
        # 输出为输出层做log_softmax变换的结果，dim表示log_softmax将计算的维度
        return F.log_softmax(x, dim=1)
