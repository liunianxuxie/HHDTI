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


def load_data_deepDTnet(dataset_train="DTnet_test_0.2_0_more_v", dataset_test="DTnet_test_0.2_0_more_v"):
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
    # print(edge_train)

    i_m = np.genfromtxt(os.path.sep.join([dataset_dir, 'drugProtein.txt']), dtype=np.int32)

    H_T = np.zeros((len(i_m), len(i_m[0])), dtype=np.int32)
    H_T_all = np.zeros((len(i_m), len(i_m[0])), dtype=np.int32)

    for i in edge_train:
        H_T[i[0]][i[1]] = 1

    for i in edge_all:
        H_T_all[i[0]][i[1]] = 1

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
    drugDisease = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'drugDisease.txt']), dtype=np.int32))  # 732, 440
    proteinDisease = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'proteinDisease.txt']), dtype=np.int32))  # 1915, 440
    drugsideeffect = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'drugsideEffect.txt']), dtype=np.int32))  # 732 12904
    # H = torch.cat((H, proteinDisease), 1)
    # H_T = torch.cat((H_T, drugDisease), 1)
    # print(drugsideeffect.size())  # 732, 440
    # print(proteinDisease.size())  # 1915, 440
    # print(drugdrug.size())  # 732, 732
    # print(proteinprotein.size())  # 1915, 1915
    # print(drugsideeffect.size())
    # return drugsideeffect, drugDisease, proteinDisease, drug_feat, prot_feat, H, H_T, edge_test, test
    return drugDisease, proteinDisease, drug_feat, prot_feat, H, H_T, edge_test, test


# load_data_deepDTnet()


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

    # drugs = []
    # targets = []
    # for i in edge:
    #     if i[0] not in drugs:
    #         drugs.append(i[0])
    #     if i[1] not in targets:
    #         targets.append(i[1])

    test_ration = [0.99]
    for d in test_ration:
        for a in (range(1)):
            edge_test = edge_shuffled[a * int(len(edge_shuffled) * d): (a + 1) * int(len(edge_shuffled) * d)]
            edge_train = edge_shuffled[: a * int(len(edge_shuffled) * d)] + edge_shuffled[(a + 1) * int(len(edge_shuffled) * d):]

            test_zeros = []
            while len(test_zeros) != len(edge_test):
                x1 = random.sample(range(0, 732), 1)[0]
                y1 = random.sample(range(0, 1915), 1)[0]
                if [x1, y1] not in edge.tolist() and [x1, y1] not in test_zeros and len(test_zeros) != len(edge_test):
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


# generate_data_2()


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
    print(len(edge))  # 4978

    drugs = []
    targets = []
    for i in edge_shuffled:
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
            for i in edge_shuffled:
                if i[0] in test:
                    edge_test.append(i)
                else:
                    edge_train.append(i)
            # print(edge_train)
            # print(edge_test)
        else:
            test = targets[a * int(len(targets) * test_ration): (a + 1) * int(len(targets) * test_ration)]
            for i in edge_shuffled:
                if i[1] in test:
                    edge_test.append(i)
                else:
                    edge_train.append(i)
        test_zeros = []
        while len(test_zeros) != len(edge_test):
            x1 = random.sample(range(0, 732), 1)[0]
            y1 = random.sample(range(0, 1915), 1)[0]
            if [x1, y1] not in edge_shuffled and [x1, y1] not in test_zeros and len(test_zeros) != len(edge_test):
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


# generate_data_3(object='target')


def Degree():
    dataset_dir = os.path.sep.join(['deepDTnet'])
    i_m = np.genfromtxt(os.path.sep.join([dataset_dir, 'drugProtein.txt']), dtype=np.int32)
    print(len(i_m), len(i_m[0]))  # 732 1915
    H_T_all = i_m

    H_all = H_T_all.T

    av_De = H_all.sum(axis=0).sum() / 732  # 9.25  538      732   6.80
    e_Degree_List = H_all.sum(axis=0)
    av_Dv = H_all.sum(axis=1).sum() / 1915  # 7.14  697     1915   2.59
    v_Degree_List= H_all.sum(axis=1)
    print(av_De, av_Dv)
    e_Degree_List.sort()  # 1, 229
    v_Degree_List.sort()  # 1, 110
    print(e_Degree_List[462], v_Degree_List[1566])  # 4, 3

    e_div = 7
    v_div = 3

    sum1 = 0
    n1 = 0
    while sum1 < 2448:
        sum1 += e_Degree_List[n1]
        n1 += 1
    print(n1)  # 674
    print(e_Degree_List[674])  # 21

    sum2 = 0
    n2 = 0
    while sum2 < 2448:
        sum2 += v_Degree_List[n2]
        n2 += 1
    print(n2)  # 1869
    print(v_Degree_List[1869])  # 21

    edge_more = []
    edge_less = []
    for i in range(len(H_all)):
        if H_all[i].sum() >= e_div:
            for j in range(len(H_all[i])):
                if H_all[i][j] == 1:
                    edge_more.append([j, i])
        else:
            for j in range(len(H_all[i])):
                if H_all[i][j] == 1:
                    edge_less.append([j, i])

    with open(os.path.sep.join([dataset_dir, "DTnet_more_e.txt"]), "w") as f1:
        for i in range(len(edge_more)):
            s = str(edge_more[i]).replace('[', ' ').replace(']', ' ')
            s = s.replace("'", ' ').replace(',', '') + '\n'
            f1.write(s)

    with open(os.path.sep.join([dataset_dir, "DTnet_less_e.txt"]), "w") as f1:
        for i in range(len(edge_less)):
            s = str(edge_less[i]).replace('[', ' ').replace(']', ' ')
            s = s.replace("'", ' ').replace(',', '') + '\n'
            f1.write(s)

    node_more = []
    node_less = []
    for i in range(len(H_T_all)):
        if H_T_all[i].sum() >= v_div:
            for j in range(len(H_T_all[i])):
                if H_T_all[i][j] == 1:
                    node_more.append([i, j])
        else:
            for j in range(len(H_T_all[i])):
                if H_T_all[i][j] == 1:
                    node_less.append([i, j])

    with open(os.path.sep.join([dataset_dir, "DTnet_more_v.txt"]), "w") as f1:
        for i in range(len(node_more)):
            s = str(node_more[i]).replace('[', ' ').replace(']', ' ')
            s = s.replace("'", ' ').replace(',', '') + '\n'
            f1.write(s)

    with open(os.path.sep.join([dataset_dir, "DTnet_less_v.txt"]), "w") as f1:
        for i in range(len(node_less)):
            s = str(node_less[i]).replace('[', ' ').replace(']', ' ')
            s = s.replace("'", ' ').replace(',', '') + '\n'
            f1.write(s)

# Degree()


def generate_data_4(dataset_str="DTnet_less_v"):
    # 将数据集分为训练集，测试集
    dataset_dir = os.path.sep.join(['deepDTnet'])

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

    test_ration = [0.2]
    for d in test_ration:
        for a in (range(5)):
            edge_test = edge_shuffled[a * int(len(edge_shuffled) * d): (a + 1) * int(len(edge_shuffled) * d)]
            edge_train = edge_shuffled[: a * int(len(edge_shuffled) * d)] + edge_shuffled[(a + 1) * int(len(edge_shuffled) * d):]

            test_zeros = []
            while len(test_zeros) != len(edge_test):
                x1 = random.sample(range(0, 732), 1)[0]
                y1 = random.sample(range(0, 1915), 1)[0]
                if [x1, y1] not in edge_shuffled and [x1, y1] not in test_zeros and len(test_zeros) != len(edge_test):
                    test_zeros.append([x1, y1])

            edge_test = edge_test + test_zeros

            with open(os.path.sep.join([dataset_dir, "DTnet_train_{ratio}_{fold}_less_v.txt".format(ratio=d, fold=a)]), "w") as f0:
                for i in range(len(edge_train)):
                    s = str(edge_train[i]).replace('[', ' ').replace(']', ' ')
                    s = s.replace("'", ' ').replace(',', '') + '\n'
                    f0.write(s)

            with open(os.path.sep.join([dataset_dir, "DTnet_test_{ratio}_{fold}_less_v.txt".format(ratio=d, fold=a)]), "w") as f1:
                for i in range(len(edge_test)):
                    s = str(edge_test[i]).replace('[', ' ').replace(']', ' ')
                    s = s.replace("'", ' ').replace(',', '') + '\n'
                    f1.write(s)


# generate_data_4()


# edge = np.array([[1,2], [4,6], [4,7]])
# edge = np.array([[1,2], [4,6], [4,7], [43,1]])
# print([1, 42] in edge)  # True
# print(id(edge))
# data = torch.utils.data.DataLoader(edge, shuffle=True)
# # edge_shuffled = []
# # for i in data:
# #     edge_shuffled.append(i[0].tolist())
# print(id(edge))
# print([1, 42] in edge)  # True
