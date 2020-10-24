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

def pre_processed_kegg_kg():
    dataset_dir = os.path.sep.join(['kegg'])
    dti = pd.read_table(os.path.sep.join([dataset_dir, 'dt_kegg_med.txt']), header=None)
    # print(dti)
    drug_list = set(dti[0])
    # print(len(drug_list))  # 4284
    target_list = set(dti[2])
    # print(len(target_list))  # 945
    kg = pd.read_table(os.path.sep.join([dataset_dir, 'kegg_kg.txt']), header=None)
    kg_drug = kg[kg[1].str.contains('PATHWAY_DRUG')]
    # print(kg_drug)
    drug_pathway = []
    for row in kg_drug.itertuples():
        if getattr(row, '_3') in drug_list:
            drug_pathway.append([getattr(row, '_1'), getattr(row, '_3')])
    # print("drug_pathway", drug_pathway)

    kg_gene = kg[kg[1].str.contains('PATHWAY_GENE')]
    gene_pathway = []
    for row in kg_gene.itertuples():
        if getattr(row, '_3') in target_list:
            gene_pathway.append([getattr(row, '_1'), getattr(row, '_3')])
    # print("gene_pathway", gene_pathway)

    drug_dict = {}
    for index, i in enumerate(drug_list):
        drug_dict[i] = index
    print("drug_dict", drug_dict)

    target_dict = {}
    for index, i in enumerate(target_list):
        target_dict[i] = index
    # print("target_dict", target_dict)

    pathway1 = np.array(drug_pathway)[:, 0]
    pathway2 = np.array(gene_pathway)[:, 0]
    pathway = set(np.concatenate((pathway1, pathway2), axis=0))
    # print(len(pathway))  # 105

    pathway_dict = {}
    for index, i in enumerate(pathway):
        pathway_dict[i] = index
    # print("pathway_dict", pathway_dict)

    drug_pathway_processed = []
    for i in drug_pathway:
        if i[0] in pathway_dict:
            drug_pathway_processed.append([pathway_dict[i[0]], drug_dict[i[1]]])
    # print("drug_pathway_processed", drug_pathway_processed)
    # print(len(drug_pathway_processed))  # 7087

    target_pathway_processed = []
    for i in gene_pathway:
        if i[0] in pathway_dict:
            target_pathway_processed.append([pathway_dict[i[0]], target_dict[i[1]]])
    # print("target_pathway_processed", target_pathway_processed)
    # print(len(target_pathway_processed))  # 3390

    drug_target_processed = []
    for row in dti.itertuples():
        drug_target_processed.append([drug_dict[getattr(row, '_1')], target_dict[getattr(row, '_3')]])
    # print("drug_target_processed", drug_target_processed)
    # print(len(drug_target_processed))  # 12112

    H_drug_pathway = np.zeros((len(drug_dict), len(pathway_dict)), dtype=np.int32)
    for i in drug_pathway_processed:
        H_drug_pathway[i[1], i[0]] = 1
    # print(H_drug_pathway)
    # print(len(H_drug_pathway), len(H_drug_pathway[0]))  # 4284*105

    H_target_pathway = np.zeros((len(target_dict), len(pathway_dict)), dtype=np.int32)
    for i in target_pathway_processed:
        H_target_pathway[i[1], i[0]] = 1
    # print(len(H_target_pathway), len(H_target_pathway[0]))  # 945*105

    with open(os.path.sep.join([dataset_dir, "drug_target_interaction.txt"]), "w") as f0:
        for i in range(len(drug_target_processed)):
            s = str(drug_target_processed[i]).replace('[', ' ').replace(']', ' ')
            s = s.replace("'", ' ').replace(',', '') + '\n'
            f0.write(s)

    np.savetxt(os.path.sep.join([dataset_dir, "H_drug_pathway.txt"]), H_drug_pathway)
    np.savetxt(os.path.sep.join([dataset_dir, "H_target_pathway.txt"]), H_target_pathway)

    disease_drug = kg[kg[1].str.contains('DRUG_EFFICACY_DISEASE')]
    # print(disease_drug)
    drug_disease = []
    for row in disease_drug.itertuples():
        if getattr(row, '_1') in drug_list:
            drug_disease.append([getattr(row, '_1'), getattr(row, '_3')])
    print("drug_disease", drug_disease)

    disease_target = kg[kg[1].str.contains('GENE_DISEASE')]
    # print(disease_target)
    target_disease = []
    for row in disease_target.itertuples():
        if getattr(row, '_1') in target_list:
            target_disease.append([getattr(row, '_1'), getattr(row, '_3')])
    print("target_disease", target_disease)

    disease1 = np.array(drug_disease)[:, -1]
    disease2 = np.array(target_disease)[:, -1]
    disease = set(np.concatenate((disease1, disease2), axis=0))
    # print(len(disease))  # 360

    disease_dict = {}
    for index, i in enumerate(disease):
        disease_dict[i] = index
    print("disease_dict", disease_dict)

    drug_disease_processed = []
    for i in drug_disease:
        if i[1] in disease_dict:
            drug_disease_processed.append([disease_dict[i[1]], drug_dict[i[0]]])
    # print("drug_disease_processed", drug_disease_processed)
    # print(len(drug_disease_processed))  # 365

    target_disease_processed = []
    for i in target_disease:
        if i[1] in disease_dict:
            target_disease_processed.append([disease_dict[i[1]], target_dict[i[0]]])
    # print("target_disease_processed", target_disease_processed)
    # print(len(target_disease_processed))  # 433

    H_drug_disease = np.zeros((len(drug_dict), len(disease_dict)), dtype=np.int32)
    for i in drug_disease_processed:
        H_drug_disease[i[1], i[0]] = 1
    # print(H_drug_disease)
    # print(len(H_drug_disease), len(H_drug_disease[0]))  # 4284*360

    H_target_disease = np.zeros((len(target_dict), len(disease_dict)), dtype=np.int32)
    for i in target_disease_processed:
        H_target_disease[i[1], i[0]] = 1
    # print(len(H_target_disease), len(H_target_disease[0]))  # 945*360

    np.savetxt(os.path.sep.join([dataset_dir, "H_drug_disease.txt"]), H_drug_disease)
    np.savetxt(os.path.sep.join([dataset_dir, "H_target_disease.txt"]), H_target_disease)


# pre_processed_kegg_kg()


def load_data_KEGG_MED(dataset_train="kegg_train_0.2_0", dataset_test="kegg_test_0.2_0"):
    dataset_dir = os.path.sep.join(['KEGG_MED'])

    # build incidence matrix
    edge_train = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_train)]), dtype=np.int32)
    edge_all = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format("kegg_all")]), dtype=np.int32)
    edge_test = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_test)]), dtype=np.int32)
    # print('edge_test', len(edge_test) / 2)
    # edge_val = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_val)]), dtype=np.int32)
    # print(edge_train)

    i_m = np.genfromtxt(os.path.sep.join([dataset_dir, 'drug_target_interaction.txt']), dtype=np.int32)

    H_T = np.zeros((4284, 945), dtype=np.int32)
    H_T_all = np.zeros((4284, 945), dtype=np.int32)

    for i in edge_train:
        H_T[i[0]][i[1]] = 1

    for i in edge_all:
        H_T_all[i[0]][i[1]] = 1

    test = np.zeros(len(edge_test))
    for i in range(len(test)):
        if i <= len(edge_test) // 2:
            test[i] = 1


    H_T = torch.Tensor(H_T)
    H = H_T.t()
    H_T_all = torch.Tensor(H_T_all)
    H_all = H_T_all.t()
    print("KEGG_MED", H.size())  # 945, 4284
    drug_feat1 = torch.eye(4284)
    prot_feat1 = torch.eye(945)
    # print(drug_feat.size())  # 732, 732
    # print(prot_feat.size())  # 1915, 1915
    drugDisease1 = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'H_drug_pathway.txt']), dtype=np.int32))
    proteinDisease1 = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'H_target_pathway.txt']), dtype=np.int32))
    drugDisease = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'H_drug_disease.txt']), dtype=np.int32))
    proteinDisease = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'H_target_disease.txt']), dtype=np.int32))
    # print(drugDisease.size())  # 4284 360
    # print(proteinDisease.size())  # 945 360
    # print(drugDisease1.size())  # 4284 105
    # print(proteinDisease1.size())  # 945 105

    # return drugDisease1, proteinDisease1, drugDisease, proteinDisease, drug_feat1, prot_feat1, H, H_T, edge_test, test
    return drugDisease, proteinDisease, drug_feat1, prot_feat1, H, H_T, edge_test, test


# load_data_KEGG_MED()


def generate_data_2(dataset_str="drug_target_interaction"):
    # 将数据集分为训练集，测试集
    dataset_dir = os.path.sep.join(['KEGG_MED'])
    edge = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_str)]), dtype=np.int32)  # dtype='U75'
    # print(edge)

    data = torch.utils.data.DataLoader(edge, shuffle=True)
    # print(data)
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
                x1 = random.sample(range(0, len(drugs)), 1)[0]
                y1 = random.sample(range(0, len(targets)), 1)[0]
                if [x1, y1] not in edge_shuffled and [x1, y1] not in test_zeros:
                    test_zeros.append([x1, y1])
            print(test_zeros)
            edge_test = edge_test + test_zeros

            with open(os.path.sep.join([dataset_dir, "kegg_train_{ratio}_{fold}.txt".format(ratio=d, fold=a)]), "w") as f0:
                for i in range(len(edge_train)):
                    s = str(edge_train[i]).replace('[', ' ').replace(']', ' ')
                    s = s.replace("'", ' ').replace(',', '') + '\n'
                    f0.write(s)

            with open(os.path.sep.join([dataset_dir, "kegg_test_{ratio}_{fold}.txt".format(ratio=d, fold=a)]), "w") as f1:
                for i in range(len(edge_test)):
                    s = str(edge_test[i]).replace('[', ' ').replace(']', ' ')
                    s = s.replace("'", ' ').replace(',', '') + '\n'
                    f1.write(s)

    with open(os.path.sep.join([dataset_dir,  "kegg_all.txt"]), "w") as f3:
        for i in range(len(edge)):
            s = str(edge[i]).replace('[', ' ').replace(']', ' ')
            s = s.replace("'", ' ').replace(',', '') + '\n'
            f3.write(s)

# load_data_2()


def generate_data_3(object='target', dataset_str="drug_target_interaction"):
    # 生成冷启动训练集，测试集（药物缺失）
    dataset_dir = os.path.sep.join(['KEGG_MED'])
    # edge = np.genfromtxt("edges.txt", dtype=np.int32)
    edge = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_str)]), dtype=np.int32)  # dtype='U75'
    # print(edge)
    data = torch.utils.data.DataLoader(edge, shuffle=True)
    edge_shuffled = []
    for i in data:
        edge_shuffled.append(i[0].tolist())
    # print(edge_shuffled)

    edge_ = []
    for i in edge_shuffled:
        edge_.append(list(i))
    edge = edge_
    print(len(edge_))  #12112

    drugs = []
    targets = []
    for i in edge:
        if i[0] not in drugs:
            drugs.append(i[0])
        if i[1] not in targets:
            targets.append(i[1])
    # print(len(drugs))  #
    # print(len(targets))  #

    test_ration = 0.1

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
        # train = drugs[: a * int(len(drugs) * test_ration)] + drugs[(a + 1) * int(len(drugs) * test_ration):]
        # print(test)
        else:
            test = targets[a * int(len(targets) * test_ration): (a + 1) * int(len(targets) * test_ration)]
            for i in edge:
                if i[1] in test:
                    edge_test.append(i)
                else:
                    edge_train.append(i)
        # print(edge_train)
        # print(edge_test)

        test_zeros = []
        while len(test_zeros) != len(edge_test):
            x1 = random.sample(range(0, 4284), 1)[0]
            y1 = random.sample(range(0, 945), 1)[0]
            if [x1, y1] not in edge and [x1, y1] not in test_zeros and len(test_zeros) != len(edge_test):
                test_zeros.append([x1, y1])
        edge_test = edge_test + test_zeros

        with open(os.path.sep.join([dataset_dir, "kegg_{object}_cold_start_train_{fold}.txt".format(object=object, fold=a)]), "w") as f0:
            for i in range(len(edge_train)):
                s = str(edge_train[i]).replace('[', ' ').replace(']', ' ')
                s = s.replace("'", ' ').replace(',', '') + '\n'
                f0.write(s)

        with open(os.path.sep.join([dataset_dir, "kegg_{object}_cold_start_test_{fold}.txt".format(object=object, fold=a)]), "w") as f1:
            for i in range(len(edge_test)):
                s = str(edge_test[i]).replace('[', ' ').replace(']', ' ')
                s = s.replace("'", ' ').replace(',', '') + '\n'
                f1.write(s)


# load_data_3(object='target')


def Degree():
    dataset_dir = os.path.sep.join(['KEGG_MED'])
    edge_all = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format("kegg_all")]), dtype=np.int32)
    print(len(edge_all))
    H_T_all = np.zeros((4284, 945), dtype=np.int32)

    for i in edge_all:
        H_T_all[i[0]][i[1]] = 1

    H_all = H_T_all.T

    av_De = H_all.sum(axis=0).sum() / 4284  # 2.83
    e_Degree_List = H_all.sum(axis=0)
    av_Dv = H_all.sum(axis=1).sum() / 945  # 12.81
    v_Degree_List= H_all.sum(axis=1)
    print(av_De, av_Dv)
    e_Degree_List.sort()  # 1, 40
    v_Degree_List.sort()  # 1, 200
    print(e_Degree_List[2181], v_Degree_List[472])  # 1, 3

    e_div = 3
    v_div = 12

    sum1 = 0
    n1 = 0
    while sum1 < 6056:
        sum1 += e_Degree_List[n1]
        n1 += 1
    print(n1)  # 674
    print(e_Degree_List[3663])  # 5

    sum2 = 0
    n2 = 0
    while sum2 < 6056:
        sum2 += v_Degree_List[n2]
        n2 += 1
    print(n2)  # 1869
    print(v_Degree_List[890])  # 62


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

    with open(os.path.sep.join([dataset_dir, "kegg_more_e.txt"]), "w") as f1:
        for i in range(len(edge_more)):
            s = str(edge_more[i]).replace('[', ' ').replace(']', ' ')
            s = s.replace("'", ' ').replace(',', '') + '\n'
            f1.write(s)

    with open(os.path.sep.join([dataset_dir, "kegg_less_e.txt"]), "w") as f1:
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

    with open(os.path.sep.join([dataset_dir, "kegg_more_v.txt"]), "w") as f1:
        for i in range(len(node_more)):
            s = str(node_more[i]).replace('[', ' ').replace(']', ' ')
            s = s.replace("'", ' ').replace(',', '') + '\n'
            f1.write(s)

    with open(os.path.sep.join([dataset_dir, "kegg_less_v.txt"]), "w") as f1:
        for i in range(len(node_less)):
            s = str(node_less[i]).replace('[', ' ').replace(']', ' ')
            s = s.replace("'", ' ').replace(',', '') + '\n'
            f1.write(s)


# Degree()


def generate_data_4(dataset_str="kegg_more_e"):
    # 将数据集分为训练集，测试集
    dataset_dir = os.path.sep.join(['KEGG_MED'])

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

    test_ration = [0.1]
    for d in test_ration:
        for a in (range(10)):
            edge_test = edge_shuffled[a * int(len(edge_shuffled) * d): (a + 1) * int(len(edge_shuffled) * d)]
            edge_train = edge_shuffled[: a * int(len(edge_shuffled) * d)] + edge_shuffled[(a + 1) * int(len(edge_shuffled) * d):]

            test_zeros = []
            while len(test_zeros) != len(edge_test):
                x1 = random.sample(range(0, 4284), 1)[0]
                y1 = random.sample(range(0, 945), 1)[0]
                if [x1, y1] not in edge and [x1, y1] not in test_zeros and len(test_zeros) != len(edge_test):
                    test_zeros.append([x1, y1])

            edge_test = edge_test + test_zeros

            with open(os.path.sep.join([dataset_dir, "kegg_train_{ratio}_{fold}_more_e.txt".format(ratio=d, fold=a)]), "w") as f0:
                for i in range(len(edge_train)):
                    s = str(edge_train[i]).replace('[', ' ').replace(']', ' ')
                    s = s.replace("'", ' ').replace(',', '') + '\n'
                    f0.write(s)

            with open(os.path.sep.join([dataset_dir, "kegg_test_{ratio}_{fold}_more_e.txt".format(ratio=d, fold=a)]), "w") as f1:
                for i in range(len(edge_test)):
                    s = str(edge_test[i]).replace('[', ' ').replace(']', ' ')
                    s = s.replace("'", ' ').replace(',', '') + '\n'
                    f1.write(s)


# generate_data_4()
