from sklearn.metrics import roc_auc_score


import torch
import torch.nn.functional as F


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    )
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_var_loss(log_std, mean, n):
    kl_divergence = (
        0.5 / n * (1 + 2 * log_std - mean ** 2 - torch.exp(log_std) ** 2).sum(1).mean()
    )
    return kl_divergence


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    return roc_auc_score(labels, scores)


def evaluate(pred, h, test_pos_g, test_neg_g, train_pos_g, train_neg_g):
    with torch.no_grad():
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        print("Test AUC", compute_auc(pos_score, neg_score))
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        print("Train AUC", compute_auc(pos_score, neg_score))

