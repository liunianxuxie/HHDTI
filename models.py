import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.module import Module
from layers import *
import numpy as np


class HETE1(nn.Module):
    def __init__(self, num_in_node, num_in_edge, num_hidden1, num_out):  # 1915, 732, 256, 32
        super(HETE1, self).__init__()
        self.node_encoders1 = node_encoder(num_in_edge, num_hidden1, 0.5)

        self.hyperedge_encoders1 = hyperedge_encoder(num_in_node, num_hidden1, 0.5)

        self.decoder2 = decoder2(act=lambda x: x)

        self._enc_mu_node = node_encoder(num_hidden1, num_out, 0.5, act=lambda x: x)
        self._enc_log_sigma_node = node_encoder(num_hidden1, num_out, 0.5, act=lambda x: x)

        self._enc_mu_hedge = node_encoder(num_hidden1, num_out, 0.5, act=lambda x: x)
        self._enc_log_sigma_hyedge = node_encoder(num_hidden1, num_out, 0.5, act=lambda x: x)

        self.hgnn_node = HGNN2(num_in_node, num_hidden1, num_out)
        self.hgnn_hyperedge = HGNN2(num_in_edge, num_hidden1, num_out)
        self.act = torch.sigmoid

    def sample_latent(self, z_node, z_hyperedge):
        # Return the latent normal sample z ~ N(mu, sigma^2)
        self.z_node_mean = self._enc_mu_node(z_node)  # mu
        self.z_node_log_std = self._enc_log_sigma_node(z_node)
        self.z_node_std = torch.exp(self.z_node_log_std)  # sigma
        z_node_std_ = torch.from_numpy(np.random.normal(0, 1, size=self.z_node_std.size())).float()
        self.z_node_std_ = z_node_std_.cuda()
        self.z_node_ = self.z_node_mean + self.z_node_std.mul(Variable(self.z_node_std_, requires_grad=True))

        self.z_edge_mean = self._enc_mu_hedge(z_hyperedge)
        self.z_edge_log_std = self._enc_log_sigma_hyedge(z_hyperedge)
        self.z_edge_std = torch.exp(self.z_edge_log_std)  # sigma
        z_edge_std_ = torch.from_numpy(np.random.normal(0, 1, size=self.z_edge_std.size())).float()
        self.z_edge_std_ = z_edge_std_.cuda()
        self.z_hyperedge_ = self.z_edge_mean + self.z_edge_std.mul(Variable(self.z_edge_std_, requires_grad=True))

        return self.z_node_, self.z_hyperedge_  # Reparameterization trick

    def forward(self, G1, G2, drug_vec, protein_vec, H, H_T):
        drug_feature = self.hgnn_hyperedge(drug_vec, G1)
        protein_feature = self.hgnn_node(protein_vec, G2)
        # fuse layer
        mask_drug_feature = Variable(nn.Parameter(torch.zeros(size=drug_feature.size()).float()), requires_grad=True).cuda()  # alpha
        nn.init.xavier_uniform_(mask_drug_feature.data, gain=0.5)
        mask_protein_feature = Variable(nn.Parameter(torch.zeros(size=protein_feature.size()).float()), requires_grad=True).cuda()
        nn.init.xavier_uniform_(mask_protein_feature.data, gain=0.5)

        drug_feature = torch.mul(drug_feature, mask_drug_feature)
        protein_feature = torch.mul(protein_feature, mask_protein_feature)

        z_node_encoder = self.node_encoders1(H)
        z_hyperedge_encoder = self.hyperedge_encoders1(H_T)

        self.z_node_s, self.z_hyperedge_s = self.sample_latent(z_node_encoder, z_hyperedge_encoder)

        z_node = protein_feature + self.z_node_s
        z_hyperedge = drug_feature + self.z_hyperedge_s

        H_ = self.decoder2(z_node, z_hyperedge)
        recover = torch.sigmoid(self.z_node_mean.mm(self.z_edge_mean.t()))
        # node_embedding = self.z_node_mean.cpu().detach().numpy()
        # edge_embedding = self.z_edge_mean.cpu().detach().numpy()
        # np.savetxt('node_embedding.txt', node_embedding)
        # np.savetxt('edge_embedding.txt', edge_embedding)

        return H_, recover
