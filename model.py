from dgl.nn.pytorch import GraphConv, GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from distributions import log_gaussian, log_standard_gaussian

from args import read_args
args = read_args()
device = torch.device("cuda:{}".format(args.device)
                      if torch.cuda.is_available() else "cpu")


class VGAEModel(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, feat_drop, attn_drop, num_classes, num_heads, classifier, backbone):
        super(VGAEModel, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.mean_norm = nn.BatchNorm1d(hidden2_dim)
        self.classifier = classifier
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.backbone = backbone
        self.flow = None
        self.kl_div = 0

        self.adjd_linear = nn.Linear(
            self.hidden2_dim + self.num_classes,  int(self.hidden2_dim / 2))

        if self.backbone == "gat":
            layer0 = GATConv(self.in_dim, self.hidden1_dim, self.num_heads,
                             feat_drop=self.feat_drop, attn_drop=self.attn_drop, activation=F.relu)

            layer1 = GATConv(self.hidden1_dim + num_classes, self.hidden2_dim, self.num_heads,
                             feat_drop=self.feat_drop, attn_drop=self.attn_drop, activation=F.relu)
            layer2 = GATConv(self.hidden1_dim + num_classes, self.hidden2_dim, self.num_heads,
                             feat_drop=self.feat_drop, attn_drop=self.attn_drop, activation=F.relu)

        else:
            layer0 = GraphConv(self.in_dim, self.hidden1_dim,
                               activation=F.relu, allow_zero_in_degree=True)
            layer1 = GraphConv(self.hidden1_dim + num_classes, self.hidden2_dim,
                               activation=lambda x: x, allow_zero_in_degree=True)
            layer2 = GraphConv(self.hidden1_dim + num_classes, self.hidden2_dim,
                               activation=lambda x: x, allow_zero_in_degree=True)

        layers = [layer0, layer1, layer2]

        self.linear = nn.Linear(
            self.hidden1_dim + num_classes, self.hidden1_dim + num_classes)

        self.layers = nn.ModuleList(layers)

        self.adj_d_fc1 = nn.Linear(
            self.hidden2_dim + self.num_classes, self.hidden1_dim)
        self.adj_d_fc2 = nn.Linear(self.hidden1_dim, self.in_dim)

        self.dlayers = nn.Sequential(
            nn.Linear(self.hidden2_dim + self.num_classes, self.hidden1_dim), nn.ReLU())

    def encoder(self, g, features, y_onehot):
        h = self.layers[0](g, features)
        if self.backbone == 'gat':
            h = h.mean(1)
        h = torch.cat([h, y_onehot], dim=1)
        h = F.relu(self.linear(h))
        self.mean = self.layers[1](g, h)
        self.mean = F.normalize(self.mean, dim=-1)

        self.log_std = self.layers[2](g, h)
        self.log_std = F.normalize(self.log_std, dim=-1)

        if self.backbone == 'gat':
            self.mean = self.mean.mean(1)
            self.log_std = self.log_std.mean(1)
        gaussian_noise1 = torch.randn(
            features.size(0), self.hidden2_dim).to(device)
        gaussian_noise2 = torch.randn(
            features.size(0), self.hidden2_dim).to(device)
        z1 = self.mean + gaussian_noise1 * \
            torch.exp(self.log_std).to(device)  # 2708 * 16
        z2 = self.mean + gaussian_noise2 * \
            torch.exp(self.log_std).to(device)  # 2708 * 16

        return z1, z2, h, self.mean, self.log_std

    def fea_decoder(self, z, y_onehot):
        z = torch.cat([z, y_onehot], dim=1)
        z = self.dlayers(z)
        return z

    def adj_decoder(self, z, y_onehot):
        z = torch.cat([z, y_onehot], dim=1)
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def _kld(self, z, q_param, p_param=None):
        (mu, log_var) = q_param

        if self.flow is not None:
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        else:
            qz = log_gaussian(z, mu, log_var)

        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z, mu, log_var)

        kl = qz - pz

        return kl

    def forward(self, g, features, y_onehot, indices=None):
        # z, h = self.encoder(g, features)
        z1, z2, h, mean, log_std = self.encoder(g, features, y_onehot)
        adj_rec = self.adj_decoder(z1, y_onehot)
        fea_rec1 = self.fea_decoder(z1, y_onehot)
        fea_rec2 = self.fea_decoder(z2, y_onehot)

        self.kl_div = self._kld(z1, (mean, log_std))
        return adj_rec, z1, z2, h, fea_rec1, fea_rec2,  mean, log_std, self.kl_div

    def get_embeds(self, g, feats):
        rep = self.encoder(g, feats)
        return rep.detach()


class GATClassifier(nn.Module):
    def __init__(self, in_size, hid_size1, hid_size2, nclass, heads, feat_dp, attn_dp, dropout, resi):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.gat1 = GATConv(
            in_size, hid_size1, heads[0], feat_drop=feat_dp, attn_drop=attn_dp, activation=F.relu,)
        self.gat2 = GATConv(hid_size1 * heads[0], nclass, heads[1],
                            feat_drop=feat_dp, attn_drop=attn_dp, activation=None,)

        self.resi = resi
        self.residual = nn.Linear(in_size, nclass)

    def init_weights(self):
        ms = self.modules()
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, g, inputs):
        h = inputs
        h = self.gat1(g, h)
        h = h.flatten(1)
        h = self.dropout(h)

        h = self.gat2(g, h)
        h = h.mean(1)

        if self.resi == 1:
            h = self.residual(inputs) + h
        _h = F.log_softmax(h, dim=1)

        return h, 0,  _h


class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hid1_dim, hid2_dim, nclass, dropout):
        super(GCNClassifier, self).__init__()
        self.gnn1 = GraphConv(in_dim, hid1_dim, activation=F.relu)
        self.gnn2 = GraphConv(hid1_dim, hid2_dim)
        self.init_weights()
        self.linear = nn.Linear(hid1_dim, 64)
        self.dropout = nn.Dropout(dropout)
        self.skiplinear = nn.Linear(in_dim, hid2_dim)

    def init_weights(self):
        ms = self.modules()
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, g, x):
        x0 = x
        x = self.gnn1(g, x)
        logits = self.linear(x)
        x = self.dropout(x)
        x = self.gnn2(g, x)
        x = self.skiplinear(x0) + x
        _x = F.log_softmax(x, dim=1)
        return x, logits, _x
