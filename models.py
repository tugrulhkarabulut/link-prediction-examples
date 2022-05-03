import itertools
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import SAGEConv, GraphConv


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.conv1 = SAGEConv(in_feats, h_feats[0], "mean")
        self.conv2 = SAGEConv(h_feats[0], h_feats[1], "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class StochasticGraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(StochasticGraphSAGE, self).__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.conv1 = SAGEConv(in_feats, h_feats[0], 'mean')
        self.conv2 = SAGEConv(h_feats[0], h_feats[1], 'mean')

    def forward(self, blocks, in_feat):
        h_dst = in_feat[:blocks[0].num_dst_nodes()]
        h = self.conv1(blocks[0], (in_feat, h_dst))
        h = F.relu(h)

        h_dst = h[:blocks[1].num_dst_nodes()]
        h = self.conv2(blocks[1], (h, h_dst))
        return h

class VGAE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(VGAE, self).__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.layer_feats = [in_feats] + h_feats

        layers = [
            GraphConv(
                self.in_feats, h_feats[0], activation=F.relu, allow_zero_in_degree=True
            ),
            GraphConv(h_feats[0], h_feats[1], allow_zero_in_degree=True),
            GraphConv(h_feats[0], h_feats[1], allow_zero_in_degree=True),
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, g, in_feat):
        h = self.layers[0](g, in_feat)
        self.mean = self.layers[1](g, h)
        self.log_std = self.layers[2](g, h)
        gaussian_noise = torch.randn(in_feat.size(0), self.h_feats[1])
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std)
        return sampled_z


class MLP(nn.Module):
    def __init__(
        self, in_feats, h_feats, hidden_activation=F.relu, out_activation=None
    ):
        super(MLP, self).__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.layer_feats = [in_feats] + h_feats
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation

        self.layers = []
        for in_, out_ in zip(self.layer_feats, self.layer_feats[1:]):
            self.layers.append(nn.Linear(in_, out_))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, _, in_feats):
        h = in_feats
        for l in self.layers[:-1]:
            h = self.hidden_activation(l(h))

        h = self.layers[-1](h)
        if self.out_activation is not None:
            h = self.out_activation(h)

        return h


class ResidualGNN(nn.Module):
    def __init__(self, *models, reduce="sum", ff_size=32):
        super(ResidualGNN, self).__init__()
        self.models = nn.ModuleList(models)
        self.reduce = reduce
        self.ff_size = ff_size

        if reduce == "sum":
            pass
        elif reduce == "ff":
            total_feat_size = sum(model.h_feats[-1] for model in self.models)
            self.ff = nn.Linear(total_feat_size, ff_size)
        else:
            raise ValueError("Invalid reduce method.")

    def forward(self, g, in_feats):
        H = tuple(model(g, in_feats) for model in self.models)

        if self.reduce == "sum":
            return functools.reduce(lambda x, y: x + y, H, 0)
        elif self.reduce == "ff":
            H = torch.concat(H, axis=-1)
            return self.ff(H)


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            return g.edata["score"][:, 0]

