import os
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import random
import scipy.io
from sklearn.decomposition import non_negative_factorization

random.seed(1)

def pre_processed_DTnet_1():
    dataset_dir = os.path.sep.join(['DTnet2'])
    i_m = np.genfromtxt(os.path.sep.join([dataset_dir, 'drugProtein.txt']), dtype=np.int32)
    print(len(i_m), len(i_m[0]))

    edge = []
    for i in range(len(i_m)):
        for j in range(len(i_m[0])):
            if i_m[i][j] == 1:
                edge.append([i, j])
    print(len(edge))

    # with open(os.path.sep.join([dataset_dir, "drug_target_interaction.txt"]), "w") as f0:
    #     for i in range(len(edge)):
    #         s = str(edge[i]).replace('[', ' ').replace(']', ' ')
    #         s = s.replace("'", ' ').replace(',', '') + '\n'
    #         f0.write(s)

# pre_processed_DTnet_1()


def load_data_deepDTnet(dataset_train="DTnet_drug_cold_start_train_0", dataset_test="DTnet_drug_cold_start_test_0",
                        dataset_val="DTnet_drug_cold_start_test_0"):
    dataset_dir = os.path.sep.join(['deepDTnet'])

    # build incidence matrix
    edge_train = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_train)]), dtype=np.int32)
    edge_all = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format("deepDTnet_all")]), dtype=np.int32)
    # edge_train_pro = []
    # for i in edge_all:
    #     edge_train_pro.append([i[0], i[1] + 732])
    # with open(os.path.sep.join([dataset_dir, "edge_train_pro.txt"]), "w") as f0:
    #     for i in range(len(edge_train_pro)):
    #         s = str(edge_train_pro[i]).replace('[', ' ').replace(']', ' ')
    #         s = s.replace("'", ' ').replace(',', '') + '\n'
    #         f0.write(s)
    edge_test = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_test)]), dtype=np.int32)
    # edge_test_pro = []
    # for i in edge_test:
    #     edge_test_pro.append([i[0], i[1] + 732])
    # with open(os.path.sep.join([dataset_dir, "edge_test_pro.txt"]), "w") as f0:
    #     for i in range(len(edge_test_pro)):
    #         s = str(edge_test_pro[i]).replace('[', ' ').replace(']', ' ')
    #         s = s.replace("'", ' ').replace(',', '') + '\n'
    #         f0.write(s)
    # print('edge_test', len(edge_test) / 2)
    edge_val = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_val)]), dtype=np.int32)
    # print(edge_train)

    i_m = np.genfromtxt(os.path.sep.join([dataset_dir, 'drugProtein.txt']), dtype=np.int32)

    H_T = np.zeros((len(i_m), len(i_m[0])), dtype=np.int32)
    H_T_all = np.zeros((len(i_m), len(i_m[0])), dtype=np.int32)

    for i in edge_train:
        H_T[i[0]][i[1]] = 1

    # for i in edge_all:
    #     H_T_all[i[0]][i[1]] = 1

    # val = np.zeros(len(edge_val))
    test = np.zeros(len(edge_test))
    for i in range(len(test)):
        if i <= len(edge_test) // 2:
            test[i] = 1
            # val[i] = 1

    np.set_printoptions(threshold=np.inf)
    # print(H_T[0])
    # print(H_T_all[0])

    H_T = torch.Tensor(H_T)
    H = H_T.t()
    H_T_all = torch.Tensor(H_T_all)
    H_all = H_T_all.t()
    print("deepDTnet", H.size())  # 1915, 732
    # drug_feat = torch.Tensor(scipy.io.loadmat(os.path.sep.join([dataset_dir, 'drug_feat.mat']))['drug_feat'])
    # drug_feat = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'drugsim6network.txt']), dtype=np.float))
    drug_feat = torch.eye(732)
    # prot_feat = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'proteinsim4network.txt']), dtype=np.float))
    prot_feat = torch.eye(1915)
    # prot_feat = torch.Tensor(scipy.io.loadmat(os.path.sep.join([dataset_dir, 'prot_feat.mat']))['prot_feat'])
    # print(drug_feat.size())  # 732, 732
    # print(prot_feat.size())  # 1915, 1915
    drugDisease = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'drugDisease.txt']), dtype=np.int32))
    proteinDisease = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'proteinDisease.txt']), dtype=np.int32))
    drugdrug = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'drugdrug.txt']), dtype=np.int32))
    proteinprotein = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'proteinprotein.txt']), dtype=np.int32))
    # print(drugDisease.size())  # 732, 440
    # print(proteinDisease.size())  # 1915, 440
    # print(drugdrug.size())  # 732, 732
    # print(proteinprotein.size())  # 1915, 1915

    return drugDisease, proteinDisease, drug_feat, prot_feat, H, H_T, edge_test, test


# load_data_DTnet1()


"""
# generate txt file from mat file
dataset_dir = os.path.sep.join(['deepDTnet'])

a = scipy.io.loadmat(os.path.sep.join([dataset_dir, 'drug_feat.mat']))['drug_feat']
print(a)
print(len(a[0]))
print(a.shape)
with open(os.path.sep.join([dataset_dir, "drug_feat.txt"]), "w") as f:
    for i in range(len(a)):
        s = str(a[i]).replace('[', ' ').replace(']', ' ')
        s = s.replace("'", ' ').replace(',', '') + '\n'
        f.write(s)
drug_feat = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'drug_feat.txt']), dtype=np.float))
print(drug_feat)
print(drug_feat.size())
c = scipy.io.loadmat(os.path.sep.join([dataset_dir, 'prot_feat.mat']))['prot_feat']
print(c.shape)
with open(os.path.sep.join([dataset_dir, "prot_feat.txt"]), "w") as f:
    for i in range(len(c)):
        s = str(c[i]).replace('[', ' ').replace(']', ' ')
        s = s.replace("'", ' ').replace(',', '') + '\n'
        f.write(s)

edge = np.genfromtxt(os.path.sep.join([dataset_dir, 'drugProtein.txt']), dtype=np.int32)  # dtype='U75'
# print(edge.shape)
"""


def generate_data_2(dataset_str="drug_target_interaction"):
    # 将数据集分为训练集，测试集
    dataset_dir = os.path.sep.join(['deepDTnet'])
    # edge = np.genfromtxt("edges.txt", dtype=np.int32)
    edge = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_str)]), dtype=np.int32)  # dtype='U75'
    # print(edge)

    data = torch.utils.data.DataLoader(edge, shuffle=True)
    edge_shuffled = []
    for i in data:
        edge_shuffled.append(i[0].tolist())
    # print(edge_shuffled)

    drugs = []
    targets = []
    for i in edge:
        if i[0] not in drugs:
            drugs.append(i[0])
        if i[1] not in targets:
            targets.append(i[1])

    test_ration = [0.8]
    for d in test_ration:
        for a in (range(1)):
            edge_test = edge_shuffled[a * int(len(edge_shuffled) * d): (a + 1) * int(len(edge_shuffled) * d)]
            edge_train = edge_shuffled[: a * int(len(edge_shuffled) * d)] + edge_shuffled[(a + 1) * int(len(edge_shuffled) * d):]

            test_zeros = []
            while len(test_zeros) != len(edge_test):
                x1 = random.sample(range(0, 732), 1)[0]
                y1 = random.sample(range(0, 1915), 1)[0]
                if [x1, y1] not in edge and [x1, y1] not in test_zeros and len(test_zeros) != len(edge_test):
                    test_zeros.append([x1, y1])

            edge_test = edge_test + test_zeros

            with open(os.path.sep.join([dataset_dir, "DTnet_train_{ratio}_{fold}.txt".format(ratio=d, fold=a)]), "w") as f0:
                for i in range(len(edge_train)):
                    s = str(edge_train[i]).replace('[', ' ').replace(']', ' ')
                    s = s.replace("'", ' ').replace(',', '') + '\n'
                    f0.write(s)

            with open(os.path.sep.join([dataset_dir, "DTnet_test_{ratio}_{fold}.txt".format(ratio=d, fold=a)]), "w") as f1:
                for i in range(len(edge_test)):
                    s = str(edge_test[i]).replace('[', ' ').replace(']', ' ')
                    s = s.replace("'", ' ').replace(',', '') + '\n'
                    f1.write(s)

    with open(os.path.sep.join([dataset_dir,  "DTnet_all.txt"]), "w") as f3:
        for i in range(len(edge)):
            s = str(edge[i]).replace('[', ' ').replace(']', ' ')
            s = s.replace("'", ' ').replace(',', '') + '\n'
            f3.write(s)


# load_data_2()


def generate_data_3(object='target', dataset_str="drug_target_interaction"):
    # generate cold-start train dataset and test dataset
    dataset_dir = os.path.sep.join(['deepDTnet'])
    # edge = np.genfromtxt("edges.txt", dtype=np.int32)
    edge = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_str)]), dtype=np.int32)  # dtype='U75'
    data = torch.utils.data.DataLoader(edge, shuffle=True)
    edge_shuffled = []
    for i in data:
        edge_shuffled.append(i[0].tolist())
    # print(edge_shuffled)
    edge_ = []
    for i in edge_shuffled:
        edge_.append(list(i))
    edge = edge_
    print(len(edge))  # 4978

    drugs = []
    targets = []
    for i in edge:
        if i[0] not in drugs:
            drugs.append(i[0])
        if i[1] not in targets:
            targets.append(i[1])
    # print(len(drugs))  # 538
    # print(len(targets))  # 697

    test_ration = 0.1
    test_number = int(len(drugs) * test_ration)
    # print(test_number)  # 53
    for a in range(10):
        # test = random.sample(drugs, test_number)
        edge_test = []
        edge_train = []
        if object == 'drug':
            test = drugs[a * int(len(drugs) * test_ration): (a + 1) * int(len(drugs) * test_ration)]
            for i in edge:
                if i[0] in test:
                    edge_test.append(i)
                else:
                    edge_train.append(i)
            # print(edge_train)
            # print(edge_test)
        else:
            test = targets[a * int(len(targets) * test_ration): (a + 1) * int(len(targets) * test_ration)]
            for i in edge:
                if i[1] in test:
                    edge_test.append(i)
                else:
                    edge_train.append(i)
        test_zeros = []
        while len(test_zeros) != len(edge_test):
            x1 = random.sample(range(0, 732), 1)[0]
            y1 = random.sample(range(0, 1915), 1)[0]
            if [x1, y1] not in edge and [x1, y1] not in test_zeros and len(test_zeros) != len(edge_test):
                test_zeros.append([x1, y1])
        edge_test = edge_test + test_zeros

        with open(os.path.sep.join([dataset_dir, "DTnet_{object}_cold_start_train_{fold}.txt".format(object=object, fold=a)]), "w") as f0:
            for i in range(len(edge_train)):
                s = str(edge_train[i]).replace('[', ' ').replace(']', ' ')
                s = s.replace("'", ' ').replace(',', '') + '\n'
                f0.write(s)

        with open(os.path.sep.join([dataset_dir, "DTnet_{object}_cold_start_test_{fold}.txt".format(object=object, fold=a)]), "w") as f1:
            for i in range(len(edge_test)):
                s = str(edge_test[i]).replace('[', ' ').replace(']', ' ')
                s = s.replace("'", ' ').replace(',', '') + '\n'
                f1.write(s)


# load_data_3(object='target')


