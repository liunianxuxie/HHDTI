import os
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import random
from hypergraph_utils import _generate_G_from_H, generate_G_from_H

random.seed(1)
torch.manual_seed(1)

def pre_processed_DTInet(dataset_train="drug_bank_train_0.8_0", dataset_test="drug_bank_test_0.8_0", dataset_val="drug_bank_val_0.8_0"):

    dataset_dir = os.path.sep.join(['DTInet'])
    i_m = np.genfromtxt(os.path.sep.join([dataset_dir, 'mat_drug_protein.txt']), dtype=np.int32)
    # print(len(i_m), len(i_m[0]))
    a = np.genfromtxt(os.path.sep.join([dataset_dir, 'drug_vector_d100.txt']), dtype=np.float)
    b = np.genfromtxt(os.path.sep.join([dataset_dir, 'protein_vector_d400.txt']), dtype=np.float)
    print(a)
    print(len(a), len(a[0]))  # 708, 100
    print(len(b), len(b[0]))  #  1512, 400
    edge = []
    for i in range(len(i_m)):
        for j in range(len(i_m[0])):
            if i_m[i][j] == 1:
                edge.append([i, j])
    # print(edge)

    with open(os.path.sep.join([dataset_dir, "drug_target_interaction.txt"]), "w") as f0:
        for i in range(len(edge)):
            s = str(edge[i]).replace('[', ' ').replace(']', ' ')
            s = s.replace("'", ' ').replace(',', '') + '\n'
            f0.write(s)


# pre_processed_DTInet()


def load_data_DTInet(dataset_train="DTInet_train_0.1_0", dataset_test="DTInet_test_0.1_0",
                        dataset_val="DTInet_test_0.1_0"):  # 更改测试集
    dataset_dir = os.path.sep.join(['DTInet'])

    # build incidence matrix
    edge_train = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_train)]), dtype=np.int32)
    edge_all = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format("DTInet_all")]), dtype=np.int32)
    edge_test = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_test)]), dtype=np.int32)
    edge_val = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_val)]), dtype=np.int32)
    # print(edge_train)
    i_m = np.genfromtxt(os.path.sep.join([dataset_dir, 'mat_drug_protein.txt']), dtype=np.int32)

    H_T = np.zeros((len(i_m), len(i_m[0])), dtype=np.int32)
    H_T_all = np.zeros((len(i_m), len(i_m[0])), dtype=np.int32)

    for i in edge_train:
        H_T[i[0]][i[1]] = 1

    for i in edge_all:
        H_T_all[i[0]][i[1]] = 1

    val = np.zeros(len(edge_val))
    test = np.zeros(len(edge_test))
    for i in range(len(test)):
        if i <= len(edge_test) // 2:
            test[i] = 1
            val[i] = 1
    # edge_val = edge_test
    # val = test
    np.set_printoptions(threshold=np.inf)
    # print(H_T[0])
    # print(H_T_all[0])

    H_T = torch.Tensor(H_T)
    H = H_T.t()
    H_T_all = torch.Tensor(H_T_all)
    H_all = H_T_all.t()
    drug_feat = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'Similarity_Matrix_Drugs.txt']), dtype=np.float))
    protein_feat = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'Similarity_Matrix_Proteins.txt']), dtype=np.float) / 100)
    drug_feat = torch.eye(708)
    protein_feat = torch.eye(1512)
    # print(drug_feat.size())  # 708, 708
    # print(protein_feat.size())  # 1512, 1512

    drugDisease = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'mat_drug_disease.txt']), dtype=np.int32))
    proteinDisease = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'mat_protein_disease.txt']), dtype=np.int32))
    drugdrug = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'mat_drug_drug.txt']), dtype=np.int32))
    proteinprotein = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'mat_protein_protein.txt']), dtype=np.int32))
    # print(drugDisease.size())  # 708, 5608
    # print(proteinDisease.size())  # 1512, 5608
    # print(drugdrug.size())  # 708, 708
    # print(proteinprotein.size())  # 1512, 1512
    print("DTInet", H.size())  # 1512, 708
    # print(H)
    """
    a = []
    for i in sim_H:
        if max(i) != 0:
            i = i / max(i)
        a.append(i.tolist())
    sim_H_ = torch.Tensor(a)

    b = []
    for i in sim_H_T:
        if max(i) != 0:
            i = i / max(i)
        b.append(i.tolist())
    sim_H_T_ = torch.Tensor(b)
    # print(H)
    """

    return drugDisease, proteinDisease, drug_feat, protein_feat, H, H_T, edge_test, test


# load_data_DTInet()


def generate_data_2(dataset_str="drug_target_interaction"):
    # 将数据集分为训练集，测试集
    dataset_dir = os.path.sep.join(['DTInet'])
    # edge = np.genfromtxt("edges.txt", dtype=np.int32)
    edge = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_str)]), dtype=np.int32)  # dtype='U75'
    print(edge)

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

    test_ration = [0.1]
    for d in test_ration:
        for a in (range(10)):
            edge_test = edge_shuffled[a * int(len(edge_shuffled) * d): (a + 1) * int(len(edge_shuffled) * d)]
            edge_train = edge_shuffled[: a * int(len(edge_shuffled) * d)] + edge_shuffled[(a + 1) * int(len(edge_shuffled) * d):]

            test_zeros = []
            while len(test_zeros) != len(edge_test):
                x1 = random.sample(range(0, 708), 1)[0]
                y1 = random.sample(range(0, 1512), 1)[0]
                if [x1, y1] not in edge and [x1, y1] not in test_zeros and len(test_zeros) != len(edge_test):
                    test_zeros.append([x1, y1])

            edge_test = edge_test + test_zeros

            with open(os.path.sep.join([dataset_dir, "DTInet_train_{ratio}_{fold}.txt".format(ratio=d, fold=a)]), "w") as f0:
                for i in range(len(edge_train)):
                    s = str(edge_train[i]).replace('[', ' ').replace(']', ' ')
                    s = s.replace("'", ' ').replace(',', '') + '\n'
                    f0.write(s)

            with open(os.path.sep.join([dataset_dir, "DTInet_test_{ratio}_{fold}.txt".format(ratio=d, fold=a)]), "w") as f1:
                for i in range(len(edge_test)):
                    s = str(edge_test[i]).replace('[', ' ').replace(']', ' ')
                    s = s.replace("'", ' ').replace(',', '') + '\n'
                    f1.write(s)

    with open(os.path.sep.join([dataset_dir,  "DTInet_all.txt"]), "w") as f3:
        for i in range(len(edge)):
            s = str(edge[i]).replace('[', ' ').replace(']', ' ')
            s = s.replace("'", ' ').replace(',', '') + '\n'
            f3.write(s)


# load_data_2()


def generate_data_3(object='target', dataset_str="drug_target_interaction"):
    # 生成冷启动训练集，测试集
    dataset_dir = os.path.sep.join(['DTInet'])
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
    # print(len(edge))  # 1923

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
        edge_test = []
        edge_train = []
        # test = random.sample(drugs, test_number)
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
            x1 = random.sample(range(0, 708), 1)[0]
            y1 = random.sample(range(0, 1512), 1)[0]
            if [x1, y1] not in edge and [x1, y1] not in test_zeros and len(test_zeros) != len(edge_test):
                test_zeros.append([x1, y1])
        edge_test = edge_test + test_zeros

        with open(os.path.sep.join([dataset_dir, "DTInet_{object}_cold_start_train_{fold}.txt".format(object=object, fold=a)]), "w") as f0:
            for i in range(len(edge_train)):
                s = str(edge_train[i]).replace('[', ' ').replace(']', ' ')
                s = s.replace("'", ' ').replace(',', '') + '\n'
                f0.write(s)

        with open(os.path.sep.join([dataset_dir, "DTInet_{object}_cold_start_test_{fold}.txt".format(object=object, fold=a)]), "w") as f1:
            for i in range(len(edge_test)):
                s = str(edge_test[i]).replace('[', ' ').replace(']', ' ')
                s = s.replace("'", ' ').replace(',', '') + '\n'
                f1.write(s)


# load_data_3(object='drug')


