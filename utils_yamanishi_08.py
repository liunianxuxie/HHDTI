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
from sklearn.cluster import KMeans, MeanShift
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.manifold import *
from sklearn.decomposition import *
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics import silhouette_samples

random.seed(1)

def pre_processed_Yamanishi_08():
    dataset_dir = os.path.sep.join(['Yamanishi_08'])
    edge = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format("dt_nr")]), dtype=np.int32)
    print(edge)
    drug_map = {}
    target_map = {}
    j = 0
    k = 0
    for i in range(len(edge)):
        if edge[i][0] not in drug_map:
            drug_map[edge[i][0]] = j
            j += 1

        if edge[i][1] not in target_map:
            target_map[edge[i][1]] = k
            k += 1
    print(drug_map)
    print(target_map)

    edge_all = []
    for i in range(len(edge)):
        edge_all.append([drug_map[edge[i][0]], target_map[edge[i][1]]])
    print(edge_all)

    with open(os.path.sep.join([dataset_dir, "Yamanishi_08_nr.txt"]), "w") as f0:
        for i in range(len(edge_all)):
            s = str(edge_all[i]).replace('[', ' ').replace(']', ' ')
            s = s.replace("'", ' ').replace(',', '') + '\n'
            f0.write(s)

    edge_all_pro = []
    for i in edge_all:
        edge_all_pro.append([i[0], i[1] + 54])

    with open(os.path.sep.join([dataset_dir, "Yamanishi_08_nr_pro.txt"]), "w") as f0:
        for i in range(len(edge_all_pro)):
            s = str(edge_all_pro[i]).replace('[', ' ').replace(']', ' ')
            s = s.replace("'", ' ').replace(',', '') + '\n'
            f0.write(s)

# pre_processed_Yamanishi_08()


def pre_processed_Yamanishi():
    dataset_dir = os.path.sep.join(['Yamanishi_08'])
    edge = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format("Yamanishi_08_all")]), dtype=np.int32)
    label = {}
    for i in range(len(edge)):
        if i < 2926:
            if edge[i][1] not in label:
                label[edge[i][1]] = 0
        if i >=2926 and i < 4402:
            if edge[i][1] not in label:
                label[edge[i][1]] = 1
        if i >=4402 and i < 5037:
            if edge[i][1] not in label:
                label[edge[i][1]] = 2
        if i >= 5037:
            if edge[i][1] not in label:
                label[edge[i][1]] = 3

    labels = []
    for i in label.keys():
        labels.append([i, label[i]])

    with open(os.path.sep.join([dataset_dir, "labels.txt"]), "w") as f0:
        for i in range(len(labels)):
            s = str(labels[i]).replace('[', ' ').replace(']', ' ')
            s = s.replace("'", ' ').replace(',', '') + '\n'
            f0.write(s)


# pre_processed_Yamanishi()


def H_yamanishi_08():
    # build incidence matrix
    dataset_dir = os.path.sep.join(['Yamanishi_08'])
    edge = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format("Yamanishi_08_all")]), dtype=np.int32)
    edge1 = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format("Yamanishi_08_e")]), dtype=np.int32)
    edge2 = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format("Yamanishi_08_ic")]), dtype=np.int32)
    edge3 = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format("Yamanishi_08_gpcr")]), dtype=np.int32)
    edge4 = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format("Yamanishi_08_nr")]), dtype=np.int32)
    H_T = np.zeros((791, 989), dtype=np.int32)
    H1_T = np.zeros((445, 664), dtype=np.int32)
    H2_T = np.zeros((210, 204), dtype=np.int32)
    H3_T = np.zeros((223, 95), dtype=np.int32)
    H4_T = np.zeros((54, 26), dtype=np.int32)
    label_t = []
    for i in range(989):
        if i < 664:
            label_t.append(0)
        if i < 868 and i >= 664 :
            label_t.append(1)
        if i < 963 and i >= 868 :
            label_t.append(2)
        if i >= 963:
            label_t.append(3)
    # print(len(label_t))

    label_d = []
    for i in range(791):
        if i < 385:
            label_d.append(0)
        if i < 595 and i >= 385:
            label_d.append(1)
        if i < 737 and i >= 595:
            label_d.append(2)
        if i >= 737:
            label_d.append(3)
    # print(len(label_t))

    for i in edge:
        H_T[i[0]][i[1]] = 1
    for i in edge1:
        H1_T[i[0]][i[1]] = 1
    for i in edge2:
        H2_T[i[0]][i[1]] = 1
    for i in edge3:
        H3_T[i[0]][i[1]] = 1
    for i in edge4:
        H4_T[i[0]][i[1]] = 1
    H = H_T.T
    H1 = H1_T.T
    H2 = H2_T.T
    H3 = H3_T.T
    H4 = H4_T.T

    color = ['#FFBD0A', '#00F0F4', '#5F5FFF', '#FF3FE6']

    target = H
    target1 = H1
    target2 = H2
    target3 = H3
    target4 = H4

    drug = H_T

    # target = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format("output")]))
    # print(len(target))

    tsne = TSNE(n_components=2, init='pca', perplexity=40, learning_rate=10)
    target_emb = tsne.fit_transform(target)
    # tsne1 = TSNE(n_components=2, init='pca')
    # target_emb1 = tsne1.fit_transform(target1)
    # tsne2 = TSNE(n_components=2, init='pca')
    # target_emb2 = tsne2.fit_transform(target2)
    # tsne3 = TSNE(n_components=2, init='pca')
    # target_emb3 = tsne3.fit_transform(target3)
    # tsne4 = TSNE(n_components=2, init='pca')
    # target_emb4 = tsne4.fit_transform(target4)

    # spectralEmbedding = SpectralEmbedding()
    # target_emb = spectralEmbedding.fit_transform(target)
    # SpectralEmbedding1 = SpectralEmbedding()
    # target_emb1 = SpectralEmbedding1.fit_transform(target1)
    # print(len(target_emb1))
    # SpectralEmbedding2 = SpectralEmbedding()
    # target_emb2 = SpectralEmbedding2.fit_transform(target2)
    # print(len(target_emb2))
    # SpectralEmbedding3 = SpectralEmbedding()
    # target_emb3 = SpectralEmbedding3.fit_transform(target3)
    # print(len(target_emb3))
    # SpectralEmbedding4 = SpectralEmbedding()
    # target_emb4 = SpectralEmbedding4.fit_transform(target4)
    # print(len(target_emb4))
    # target_emb = np.concatenate((target_emb1, target_emb2, target_emb3, target_emb4), axis=0)

    # pca = PCA(n_components=2, whiten=True)
    # target_emb = pca.fit_transform(target)
    print(target_emb)
    print(len(target_emb))
    # kmeans = KMeans(n_clusters=4, random_state=0)
    # kmeans.fit(tsne.embedding_)
    # labels = kmeans.labels_
    # label_t = label_t[: 664]
    sc_score1 = silhouette_score(target_emb, label_t, metric='euclidean')
    sc_score2 = calinski_harabasz_score(target_emb, label_t)
    print(sc_score1, sc_score2)
    legend = ['E', 'IC', 'GPCR', 'NR']
    number1 = [0, 664, 868, 963, 989]  # 0, 664, 868, 963, 989
    # number1 = [0, 445, 655, 701, 791]  # 0, 664, 868, 963, 989
    for j in range(1, 5):
        for i in range(number1[j - 1], number1[j]):
            x = target_emb[i][0]
            y = target_emb[i][1]
            plt.scatter(x, y, c=color[j - 1], s=5, alpha=0.75)
        plt.scatter(target_emb[j][0], target_emb[j][1], c=color[j - 1], s=5, alpha=0.75, label=legend[j - 1])
        plt.legend()
    # for i in range(989):
    #     x = target_emb[i][0]
    #     y = target_emb[i][1]
    #     plt.scatter(x, y, c=color[labels[i]], s=2, alpha=0.75)
    plt.show()


    # number2 = [0, 385, 595, 737, 791]
    # for j in range(1, 5):
    #     for i in range(number2[j - 1], number2[j]):
    #         x = target_emb[i][0]
    #         y = target_emb[i][1]
    #         plt.scatter(x, y, c=color[j - 1], alpha=0.5)
    # plt.show()


# H_yamanishi_08()

#
# def H_yamanishi_08_div():
#     # build incidence matrix
#     dataset_dir = os.path.sep.join(['Yamanishi_08'])
#     edge_ic = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format("Yamanishi_08_ic")]), dtype=np.int32)
#     H_T_ic = np.zeros((210, 204), dtype=np.int32)
#
#     for i in edge_ic:
#         H_T_ic[i[0]][i[1]] = 1
#
#     H_ic = H_T_ic.T
#
#     color = ['#FF0000', '#32CD32', '#00FFFF', '#8A2BE2']
#
#     target_embeddings = H_ic
#     drug_embeddings = H_T_ic
#
#     tsne1 = TSNE(n_components=2)
#     tsne2 = TSNE(n_components=2)
#     tsne1.fit_transform(target_embeddings)
#     tsne2.fit_transform(drug_embeddings)
#
#     # kmeans = KMeans(n_clusters=1, random_state=0)
#     # kmeans.fit(tsne.embedding_)
#     # labels = kmeans.labels_
#     number1 = [0, 204]
#     for j in range(1, 2):
#         for i in range(number1[j - 1], number1[j]):
#             x = tsne1.embedding_[i][0]
#             y = tsne1.embedding_[i][1]
#             plt.scatter(x, y, c=color[j - 1], alpha=0.5)
#     plt.show()
#
#     number2 = [0, 205]
#     for j in range(1, 2):
#         for i in range(number2[j - 1], number2[j]):
#             x = tsne2.embedding_[i][0]
#             y = tsne2.embedding_[i][1]
#             plt.scatter(x, y, c=color[j - 1], alpha=0.5)
#     plt.show()
#
#
# H_yamanishi_08_div()


def G_yamanishi_08():
    # build incidence matrix
    dataset_dir = os.path.sep.join(['Yamanishi_08'])
    edge = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format("Yamanishi_08_all")]), dtype=np.int32)
    G = np.eye(1780)
    for i in edge:
        G[i[0], i[1]] = 1
    G = G + G.T * (G.T > G) - G * (G.T > G)
    # print(G)
    # drug_embeddings = G[:791]
    target = G  # [791:]

    color = ['#FF0000', '#32CD32', '#00FFFF', '#8A2BE2']
    # print(G)

    label_t = []
    for i in range(989):
        if i < 664:
            label_t.append(0)
        if i < 868 and i >= 664:
            label_t.append(1)
        if i < 963 and i >= 868:
            label_t.append(2)
        if i >= 963:
            label_t.append(3)

    tsne = TSNE(n_components=2, init='pca', perplexity=40, learning_rate=10)
    target_emb = tsne.fit_transform(target)

    # SpectralEmbedding1 = SpectralEmbedding()
    # target_emb = SpectralEmbedding1.fit_transform(target)
    print(target_emb)
    # print(SpectralEmbedding1.n_neighbors)

    sc_score1 = silhouette_score(target_emb[791:], label_t, metric='euclidean')
    sc_score2 = calinski_harabasz_score(target_emb[791:], label_t)
    print(sc_score1, sc_score2)

    number = [791, 1455, 1659, 1754, 1780]  # 791, 1455, 1659, 1754, 1780  0, 664, 868, 963, 989
    for j in range(1, 5):
        for i in range(number[j - 1], number[j]):
            x = target_emb[i][0]
            y = target_emb[i][1]
            plt.scatter(x, y, c=color[j - 1], s=3, alpha=1)
    plt.show()


# G_yamanishi_08()


def generate_data_2(dataset_str="Yamanishi_08_all"):
    # 将数据集分为训练集，测试集
    dataset_dir = os.path.sep.join(['Yamanishi_08'])
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

    test_ration = [0.2]
    for d in test_ration:
        for a in (range(5)):
            edge_test = edge_shuffled[a * int(len(edge_shuffled) * d): (a + 1) * int(len(edge_shuffled) * d)]
            edge_train = edge_shuffled[: a * int(len(edge_shuffled) * d)] + edge_shuffled[(a + 1) * int(len(edge_shuffled) * d):]

            test_zeros = []
            while len(test_zeros) != len(edge_test):
                x1 = random.sample(range(0, 791), 1)[0]
                y1 = random.sample(range(0, 989), 1)[0]
                print(x1, y1)
                if [x1, y1] not in edge_shuffled and [x1, y1] not in test_zeros and len(test_zeros) != len(edge_test):
                    test_zeros.append([x1, y1])
                print([x1, y1] in edge_shuffled)

            edge_test = edge_test + test_zeros

            with open(os.path.sep.join([dataset_dir, "Yamanishi_train_{ratio}_{fold}.txt".format(ratio=d, fold=a)]), "w") as f0:
                for i in range(len(edge_train)):
                    s = str(edge_train[i]).replace('[', ' ').replace(']', ' ')
                    s = s.replace("'", ' ').replace(',', '') + '\n'
                    f0.write(s)

            with open(os.path.sep.join([dataset_dir, "Yamanishi_test_{ratio}_{fold}.txt".format(ratio=d, fold=a)]), "w") as f1:
                for i in range(len(edge_test)):
                    s = str(edge_test[i]).replace('[', ' ').replace(']', ' ')
                    s = s.replace("'", ' ').replace(',', '') + '\n'
                    f1.write(s)

    # with open(os.path.sep.join([dataset_dir,  "DTnet_all.txt"]), "w") as f3:
    #     for i in range(len(edge)):
    #         s = str(edge[i]).replace('[', ' ').replace(']', ' ')
    #         s = s.replace("'", ' ').replace(',', '') + '\n'
    #         f3.write(s)


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


# generate_data_3(object='target')


# target_embeddings1 = np.array([[1,2,3,4,2],
#                               [2,4,1,0,3],
#                               [7,8,5,4,4],
#                               [3,0,9,0,0]])
#
# target_embeddings2 = np.array([[1,2,3,4,2],
#                               [2,4,1,0,3],
#                               [7,8,5,4,4]])
# tsne1 = TSNE(n_components=2, random_state=1)
# tsne1.fit_transform(target_embeddings1)
# print(tsne1.embedding_)
# tsne2 = TSNE(n_components=2, random_state=1)
# tsne2.fit_transform(target_embeddings2)
#
#
# print(tsne2.embedding_)
