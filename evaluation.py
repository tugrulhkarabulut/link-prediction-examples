from sklearn.metrics import roc_auc_score


import torch
import torch.nn.functional as F
import dgl


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

def batch_evaluate(model, pred, g, sampler, batch_size=1024):
    dataloader = dgl.dataloading.DataLoader(
        g,
        torch.arange(g.number_of_edges()),
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )
    
    with torch.no_grad():
        pos_scores = torch.tensor([])
        neg_scores = torch.tensor([])
        for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
            inputs = blocks[0].srcdata['feat']

            outputs = model(blocks, inputs)
            pos_score = pred(pos_graph, outputs)
            neg_score = pred(neg_graph, outputs)
            pos_scores = torch.cat([pos_scores, pos_score])
            neg_scores = torch.cat([neg_scores, neg_score])
    
    print('Test AUC', compute_auc(pos_scores, neg_scores))