import argparse

import torch

from models import *
from preprocess import *
from evaluation import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "GraphSAGE",
            "VGAE",
            "ResidualMLPVGAEReduceSum",
            "ResidualMLPVGAEReduceFF",
            "ResidualMLPGraphSAGEReduceSum",
            "ResidualMLPGraphSAGEReduceFF",
        ],
        default="GraphSAGE",
    ),

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--graph-h-feats", nargs="+", type=int, default=[16, 16])
    parser.add_argument("--mlp-h-feats", nargs="+", type=int, default=[16, 16])
    parser.add_argument("--reduce-ff-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    return parser.parse_args()


def train_loop(
    model,
    predictor,
    train_g,
    train_pos_g,
    train_neg_g,
    epochs=80,
    lr=0.01,
    variational_model=None,
):
    optimizer = torch.optim.Adam(
        itertools.chain(model.parameters(), predictor.parameters()), lr=lr
    )
    for e in range(epochs):
        # forward
        h = model(train_g, train_g.ndata["feat"])
        pos_score = predictor(train_pos_g, h)
        neg_score = predictor(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)
        if variational_model is not None:
            loss -= compute_var_loss(
                variational_model.log_std, variational_model.mean, h.shape[0]
            )

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print("In epoch {}, loss: {}".format(e, loss))

    return h


if __name__ == "__main__":
    train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = load_dataset()
    predictor = DotPredictor()
    args = parse_arguments()
    print(args.graph_h_feats)

    variational_model = None
    if args.model == "GraphSAGE":
        model = GraphSAGE(train_g.ndata["feat"].shape[1], args.graph_h_feats)
    elif args.model == "VGAE":
        model = VGAE(train_g.ndata["feat"].shape[1], args.graph_h_feats)
        variational_model = model
    elif args.model == "ResidualMLPVGAEReduceSum":
        mlp = MLP(train_g.ndata["feat"].shape[1], args.mlp_h_feats)
        vgae = VGAE(train_g.ndata["feat"].shape[1], args.graph_h_feats)
        model = ResidualGNN(mlp, vgae, reduce="sum")
        variational_model = vgae
    elif args.model == "ResidualMLPVGAEReduceFF":
        mlp = MLP(train_g.ndata["feat"].shape[1], args.mlp_h_feats)
        vgae = VGAE(train_g.ndata["feat"].shape[1], args.graph_h_feats)
        model = ResidualGNN(mlp, vgae, reduce="ff", ff_size=args.reduce_ff_size)
        variational_model = vgae
    elif args.model == "ResidualMLPGraphSAGEReduceSum":
        mlp = MLP(train_g.ndata['feat'].shape[1], args.mlp_h_feats)
        graphsage = GraphSAGE(train_g.ndata['feat'].shape[1], args.graph_h_feats)
        model = ResidualGNN(mlp, graphsage, reduce='sum')
    elif args.model == "ResidualMLPGraphSAGEReduceFF":
        mlp = MLP(train_g.ndata['feat'].shape[1], args.mlp_h_feats)
        graphsage = GraphSAGE(train_g.ndata['feat'].shape[1], args.graph_h_feats)
        model = ResidualGNN(mlp, graphsage, reduce='ff', ff_size=args.reduce_ff_size)

    print('Training {args.model}')
    h = train_loop(model, predictor, train_g, train_pos_g, train_neg_g, variational_model=variational_model, epochs=args.epochs, lr=args.lr)
    evaluate(predictor, h, test_pos_g, test_neg_g, train_pos_g, train_neg_g)


