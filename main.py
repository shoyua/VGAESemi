import torch.nn as nn
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
import torch
from tqdm import tqdm
from pprint import pprint
import warnings
import model
from utils import *

from args import read_args


# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import logging
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')


def cvae_eval(graph, feats, test_mask, labels, model):
    model.eval()
    with torch.no_grad():
        prob, _, log_prob = model(graph, feats)

    test_prob = log_prob[test_mask]
    test_predict = torch.argmax(test_prob, dim=1).cpu().detach().numpy()
    test_labels = labels[test_mask].cpu().detach().numpy()
    acc = sklearn.metrics.accuracy_score(test_labels, test_predict)
    return acc


def cvae_eval_lr(graph, feats, train_mask, test_mask, labels, model, classifier):
    model.eval()
    with torch.no_grad():
        feas, _, log_prob = classifier(graph, feats)
    train_labels = labels[train_mask]
    train_feas = feas[train_mask]
    train_feas, train_labels = train_feas.cpu().detach(
    ).numpy(), train_labels.cpu().detach().numpy()

    test_labels = labels[test_mask].cpu().detach().numpy()
    test_feas = feas[test_mask].cpu().detach().numpy()
    learner = linear_model.LogisticRegression()

    learner.fit(train_feas, train_labels)
    test_predict = learner.predict(test_feas)

    acc = sklearn.metrics.accuracy_score(test_labels, test_predict)
    # bacc = balanced_accuracy_score(test_labels, test_predict)
    # f_score = f1_score(test_labels, test_predict, average='macro')
    return acc


def load_resources(device, dataset):
    graph, feat, (num_features, num_classes,
                  num_per_class) = load_dataset(dataset)
    graph = graph.to(device)
    feat = feat.to(device)

    return graph, feat, num_per_class


class Trainer():
    def __init__(self, args, model, classifier, tau, epochs, y_onehot):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        # self.classifier = classifier
        self.tau = tau
        self.epochs = epochs
        self.y_onehot = y_onehot

    def train(self, graph, feats, adj, optimizer, alpha, beta, gamma):
        loss_fcn = nn.CrossEntropyLoss()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            train_mask = graph.ndata['train_mask']
            test_mask = graph.ndata['test_mask']
            val_mask = graph.ndata['val_mask']
            labels = graph.ndata['label']
            adj_rec, z1, z2, h, fea_rec1, fea_rec2,  mean, log_std, kl_div = self.model(
                graph, feats, self.y_onehot)

            # labeled
            mean_labeled = mean[train_mask]
            log_std_labeled = log_std[train_mask]
            adj_labeled = adj[train_mask]

            # unlabed
            mean_unlabeled = mean[~train_mask]
            log_std_unlabeled = log_std[~train_mask]
            adj_unlabeled = adj[~train_mask]

            # classifier loss
            prob, gnn_logits, log_prob = self.model.classifier(graph, feats)
            unlabeled_log_prob = log_prob[~train_mask]

            prob_c = prob.cpu().detach().numpy()
            pred_c = np.argmax(prob_c, axis=1)
            cm = clustering_metrics(labels.cpu().detach().numpy(), pred_c)
            nmi, ari = cm.evaluationClusterModelFromLabel()

            # classifier loss
            classifier_loss = loss_fcn(prob[train_mask], labels[train_mask])

            fea_rec1_l, fea_rec2_l = fea_rec1[train_mask], fea_rec2[train_mask]
            fea_rec1_u, fea_rec2_u = fea_rec1[~train_mask], fea_rec2[~train_mask]

            # elbo loss for labeled
            rec_adj_labeled = adj_rec[train_mask]
            one_hot_labeled = self.y_onehot[train_mask]
            labeled_loss = loss_labeled(self.model, adj_labeled, one_hot_labeled, rec_adj_labeled,
                                        mean_labeled, log_std_labeled, fea_rec1_l, fea_rec2_l, self.tau)

            # elbo loss for unlabeled
            rec_adj_unlabeled = adj_rec[~train_mask]
            unlabeled_loss = loss_unlabeled(self.model, adj_unlabeled, unlabeled_log_prob,
                                            rec_adj_unlabeled, mean_unlabeled, log_std_unlabeled, fea_rec1_u, fea_rec2_u, self.tau)

            # train_acc = cvae_eval_lr(graph, feats, train_mask, train_mask, labels, self.model, self.model.classifier)
            # val_acc = cvae_eval_lr(graph, feats,train_mask,  val_mask, labels, self.model,  self.model.classifier)
            # test_acc = cvae_eval_lr(graph, feats, train_mask, test_mask, labels, self.model, self.model.classifier)

            train_acc = cvae_eval(graph, feats, train_mask,
                                  labels, self.model.classifier)
            val_acc = cvae_eval(graph, feats, val_mask,
                                labels, self.model.classifier)
            test_acc = cvae_eval(graph, feats, test_mask,
                                 labels, self.model.classifier)

            loss = alpha * classifier_loss + beta * labeled_loss + gamma * unlabeled_loss

            if epoch % 100 == 0:
                logging.info("Epoch {:05d} | Dataset {:} | Train Acc {:.4f}| Val Acc {:.4f}| Test Acc {:.4f}| Loss {:.4f}".format(
                    epoch,  self.args.dataset,   train_acc, val_acc, test_acc, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_acc = cvae_eval_lr(
            graph, feats, train_mask, test_mask, labels, self.model, self.model.classifier)

        return test_acc, nmi, ari


def main(args):
    device = torch.device("cuda:{}".format(args.device)
                          if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    feat_dp = args.feat_dp
    attn_dp = args.attn_dp
    dp_rate = args.dp_rate
    lr = args.lr
    num_heads = args.num_heads
    epochs = args.epochs
    resi = args.resi
    classifier_dim1 = args.classifier_dim1
    classifier_dim2 = args.classifier_dim2
    vae_hidden1 = args.vae_hidden1
    vae_hidden2 = args.vae_hidden2
    tau = args.tau

    accs, nmis, aris = [], [], []
    for run in range(args.repeats):
        graph,  feats, num_per_class = load_resources(device, args.dataset)
        adj = graph.adjacency_matrix().to_dense().to(device)

        train_mask = graph.ndata['train_mask']
        labels = graph.ndata['label']

        num_classes = max(graph.ndata['label']).item()+1
        labels_semi = labels[train_mask]
        y_onehot = torch.FloatTensor(len(labels), num_classes).to(device)
        y_onehot.zero_()
        y_onehot.scatter_(1, labels_semi.view(-1, 1), 1)

        # auxiliary classifier model
        if args.backbone == "gat":
            classifier = model.GATClassifier(feats.shape[-1], classifier_dim1, classifier_dim2, num_classes, [
                                             8, 1], feat_dp, attn_dp, dp_rate, resi).to(device)
        else:
            classifier = model.GCNClassifier(
                feats.shape[-1], classifier_dim1, classifier_dim2,  num_classes, dp_rate).to(device)

        # vgae model
        vgae_model = model.VGAEModel(feats.shape[-1], vae_hidden1, vae_hidden2,
                                     feat_dp, attn_dp, num_classes, num_heads, classifier, args.backbone)
        vgae_model = vgae_model.to(device)
        optimizer = torch.optim.Adam(
            vgae_model.parameters(), lr=lr, weight_decay=5e-4)
        print('Total Parameters:', sum([p.nelement()
              for p in vgae_model.parameters()]))

        trainer = Trainer(args, vgae_model, classifier, tau, epochs, y_onehot)
        # indices
        # acc, bacc, ma_fscore, acc_lr, bacc_lr, f_score_lr = trainer.train(graph, 0, feats, adj, norm, weight_tensor, optimizer, alpha, beta, gamma, kl_c1, kl_c2, grace_c1, grace_c2,  up_ratio, device)
        acc, nmi, ari = trainer.train(
            graph, feats, adj, optimizer, alpha, beta, gamma)
        logging.info("run {:05d} |acc {:04f} |nmi {:04f} |ari {:04f} |".format(
            run, acc, nmi, ari))
        accs.append(acc)
        nmis.append(nmi)
        aris.append(ari)

    accs = np.array(accs)
    nmis = np.array(nmis)
    aris = np.array(aris)

    acc_mean, acc_std, nmi_mean, nmi_std, ari_mean, ari_std, = round(np.mean(accs), 4), round(np.std(accs), 4), round(np.mean(nmis), 4), \
        round(np.std(nmis), 4),   round(
            np.mean(aris), 4), round(np.std(aris), 4)

    message = {"acc_mean": acc_mean, "acc_std": acc_std,
               "nmi_mean": nmi_mean, "nmi_std": nmi_std,
               "ari_mean": ari_mean, "ari_std": ari_std}
    print(message)


if __name__ == '__main__':
    args = read_args()
    if args.use_cfg:
        args = load_best_configs(args, "config.yml")
    main(args)
