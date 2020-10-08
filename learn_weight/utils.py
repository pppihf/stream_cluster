#!/usr/bin/python
# -*- encoding:utf-8 -*-
import numpy as np
import scipy.sparse as sp
import torch
import os


def encode_onehot(labels):
    """
    将标签转换为one-hot编码形式
    :param labels: 原始数据标签
    :return: 
    """
    classes = set(labels)
    # 这一句主要功能就是进行转化成dict字典数据类型，且键为元素，值为one-hot编码
    # enumerate()将可遍历对象组合成一个含数据下标和数据的索引序列
    # for i,c in XXX 将XXX序列进行循环遍历赋给(i,c)，这里是i得数据下标，c得数据
    # len()返回元素的个数，np.identity()函数创建对角矩阵，返回主对角线元素为1，其余元素为0
    # 矩阵[i,:]是仅保留第一维度的下标i的元素和第二维度所有元素，直白来看就是提取了矩阵的第i行
    # {}生成了字典，c:xxx 是字典的形式，c作为键，xxx作为值，在for in循环下进行组成字典
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    # map(function, iterable)是对指定序列iterable中的每一个元素调用function函数
    # 根据提供的函数对指定序列做映射，返回包含每次function函数返回值的新列表
    # 将输入一一对应one - hot编码进行输出
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="../data/GCNdata/"):
    """
    加载数据
    :param path: 传入GCN的邻接矩阵和特征矩阵的路径
    :return: 表示图的邻接矩阵, 样本特征张量，样本标签，训练集索引列表，验证集索引列表，测试集索引列表
    """
    features = np.load(os.path.join(path, 'features.npy'))
    # 将其转换为csr矩阵（压缩稀疏行矩阵）
    print(features.shape)
    features = sp.csr_matrix(features, dtype=np.float32)

    # 提取样本的类别标签，并将其转换为one-hot编码形式
    idx_labels = np.genfromtxt("{}instance2label.txt".format(path), dtype=np.dtype(str))
    labels = encode_onehot(idx_labels[:, 1])
    # 构建图
    adj = []
    data = np.load(os.path.join(path, "22_adj_data.npy"))
    for d in data:
        d = normalize(d, diag_lambda=2)
        # d = sparse_mx_to_torch_sparse_tensor(d)
        adj.append(d)
    adj = torch.FloatTensor(np.array(adj))
    # print(adj)
    # exit()
    # features是样本特征的压缩稀疏矩阵，行规范化稀疏矩阵
    # features = normalize(features, diag_lambda=0)
    # 分割为train，val，test三个集，最终数据加载为torch的格式并且分成三个数据集

    idx_train = range(9000)  # 训练集索引列表
    idx_val = range(9000, 10000)  # 验证集索引列表
    idx_test = range(10000)	 # 测试集索引列表
    # 将特征矩阵转化为张量形式, .todense()与.csr_matrix()对应，将压缩的稀疏矩阵进行还原
    features = torch.FloatTensor(np.array(features.todense()))
    # np.where(condition)，输出满足条件condition(非0)的元素的坐标，np.where()[1]则表示返回列的索引、下标值
    # 说白了就是将每个标签one-hot向量中非0元素位置输出成标签
    # one-hot向量label转常规label：0,1,2,3,……
    labels = torch.LongTensor(np.where(labels)[1])
    # 将scipy稀疏矩阵转换为torch稀疏张量，具体函数下面有定义
    idx_train = torch.LongTensor(idx_train)	 # 训练集索引列表
    idx_val = torch.LongTensor(idx_val)  # 验证集索引列表
    idx_test = torch.LongTensor(idx_test)  # 测试集索引列表
    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx, diag_lambda):
    """
    行规范化稀疏矩阵, 这里是计算D^-1A，而不是计算论文中的D^-1/2AD^-1/2
    这个函数思路就是在邻接矩阵基础上转化出度矩阵，并求D^-1A随机游走归一化拉普拉斯算子
    函数实现的规范化方法是将输入左乘一个D^-1算子，就是将矩阵每行进行归一化
    :param mx:
    :return: 
    """
    rowsum = np.array(mx.sum(1))  # .sum(1)矩阵的第1维度求和，这里是将二维矩阵的每一行元素求和
    r_inv = np.power(rowsum, -1).flatten() # rowsum数组元素求-1次方，flatten()返回一个折叠成一维的数组（默认按行的方向降维）
    # 求倒数
    r_inv[np.isinf(r_inv)] = 0.
    # isinf()测试元素是否为正无穷或负无穷,若是则返回真，否则是假，最后返回一个与输入形状相同的布尔数组
    # 如果某一行全为0，则倒数r_inv算出来会等于无穷大，将这些行的r_inv置为0
    # 这句就是将数组中无穷大的元素置0处理
    r_mat_inv = sp.diags(r_inv)	 # 构建对角元素为r_inv的对角矩阵
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘
    mx = r_mat_inv.dot(mx)
    mx = mx + diag_lambda * sp.diags(mx.diagonal())  # 对角增强
    return mx  # D^-1A


def accuracy(output, labels):
    """
    计算准确率
    :param output: 模型输出结果
    :param labels: 原始标签
    :return: 
    """
    # max(1)返回每一行最大值组成的一维数组和索引,output.max(1)[1]表示最大值所在的索引indice
    preds = output.max(1)[1].type_as(labels)  # type_as()将张量转化为labels类型
    correct = preds.eq(labels).double()  # eq是判断preds与labels是否相等，相等的话对应元素置1，不等置0
    correct = correct.sum()  # 对其求和，即求出相等(置1)的个数
    return correct / len(labels)	# 计算准确率


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    将scipy稀疏矩阵转换为torch稀疏张量
    :param sparse_mx: 稀疏矩阵
    :return:
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)  # 将矩阵转换为Coo格式，再转换数组的数据类型
    # vstack()将两个数组按垂直方向堆叠成一个新数组
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)) # Coo的索引
    values = torch.from_numpy(sparse_mx.data) # Coo的值
    shape = torch.Size(sparse_mx.shape)  # Coo的形状大小
    return torch.sparse.FloatTensor(indices, values, shape)	 # 构造稀疏张量



