import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from utils_deepDTnet import load_data_deepDTnet  # cold_start epoch  drug 70 target 300
from utils_DTInet import load_data_DTInet  # cold_start epoch  drug 200  target 300
from utils_KEGG_MED import load_data_KEGG_MED  # cold_start epoch  drug 80  target 69
from kl_loss import kl_loss
from models import HETE1
from hypergraph_utils import generate_G_from_H

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=11, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--num_fold', type=int, default=5, help='number of fold')
parser.add_argument('--dataset', type=str, default='deepDTnet', help='dataset')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)


def train(epoch):
    t = time.time()

    if epoch != args.epochs - 1:
        model2.train()
        optimizer2.zero_grad()
        output, recover = model2(G1, G2, drug_feat, prot_feat, H, H_T)
        loss_train = loss_kl(model2.z_node_log_std, model2.z_node_mean, model2.z_edge_log_std, model2.z_edge_mean)
        loss = norm * F.binary_cross_entropy_with_logits(output.t(), H_T, pos_weight=pos_weight) + loss_train
        loss.backward()
        optimizer2.step()

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss: {:.5f}'.format(loss.data.item()),
              'time: {:.4f}s'.format(time.time() - t),
              )
    else:
        model2.eval()
        output, recover = model2(G1, G2, drug_feat, prot_feat, H, H_T)
        sim_outputs = torch.sigmoid(recover).t().cpu().detach().numpy()
        res_test = []
        for i in range(len(edge_test)):
            res_test.append(sim_outputs[edge_test[i][0]][edge_test[i][1]])
        auc_test = roc_auc_score(test, res_test)
        aupr_test = average_precision_score(test, res_test)
        # with open(("DTnet_test_result_{}.csv".format(k)), "w") as f1:
        #     for i in range(len(res_test)):
        #         s = str(res_test[i]).replace('[', ' ').replace(']', ' ')
        #         s = s.replace("'", ' ').replace(',', '') + '\n'
        #         f1.write(s)

    auc1 = 0
    aupr1 = 0
    if epoch == args.epochs - 1:
        print('auc_test: {:.5f}'.format(auc_test),
              'aupr_test: {:.5f}'.format(aupr_test),
              )
        auc1 = auc_test
        aupr1 = aupr_test

    return auc1, aupr1


auc = 0
aupr = 0
time_star = time.time()
k = 0
for i in range(args.num_fold):
    if args.dataset == 'deepDTnet':
        drugDisease, proteinDisease, drug_feat, prot_feat, H, H_T, edge_test, test = \
            load_data_deepDTnet(
                dataset_train="DTnet_train_0.2_{}".format(i),
                dataset_test="DTnet_test_0.2_{}".format(i)
            )
        parameters = [1915, 732]
    elif args.dataset == 'DTInet':
        drugDisease, proteinDisease, drug_feat, prot_feat, H, H_T, edge_test, test = \
            load_data_DTInet(
                dataset_train="DTInet_train_0.1_{}".format(i),
                dataset_test="DTInet_test_0.1_{}".format(i)
            )
        parameters = [1512, 708]
    elif args.dataset == 'KEGG_MED':
        drugDisease, proteinDisease, drug_feat, prot_feat, H, H_T, edge_test, test = \
            load_data_KEGG_MED(
                dataset_train="kegg_train_0.1_{}".format(i),
                dataset_test="kegg_test_0.1_{}".format(i)
            )
        parameters = [945, 4284]

    G1 = generate_G_from_H(drugDisease)
    G2 = generate_G_from_H(proteinDisease)
    model2 = HETE1(parameters[0], parameters[1], 256, args.hidden)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = float(H.shape[0] * H.shape[1] - H.sum()) / H.sum()
    norm = H.shape[0] * H.shape[1] / float((H.shape[0] * H.shape[1] - H.sum()) * 2)

    if args.cuda:
        model2.cuda()
        drug_feat = drug_feat.cuda()
        prot_feat = prot_feat.cuda()
        H = H.cuda()
        H_T = H_T.cuda()
        G1 = G1.cuda()
        G2 = G2.cuda()

    drug_feat, prot_feat, H, H_T = Variable(drug_feat), Variable(prot_feat), Variable(H), Variable(H_T)

    loss_kl = kl_loss(parameters[0], parameters[1])
    print("fold:", i)
    for epoch in range(args.epochs):
        auc1, aupr1 = train(epoch)
        auc = auc + auc1
        aupr = aupr + aupr1
    k += 1
    if i == args.num_fold - 1:
        print('auc: {:.5f}'.format(auc / args.num_fold),
              'aupr: {:.5f}'.format(aupr / args.num_fold),
              'total_time: {:.5f}'.format(time.time() - time_star)
             )
