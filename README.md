## Link Prediction Examples

Trying out different network architectures on Cora Dataset using DGL & Pytorch.

Here are the models that are included in this repo:

| Model Name | Description   |
|--|-----|
| GraphSAGE | Learns node representations by aggregating node neighborhood features |
| StochasticGraphSAGE | Stochastic version of GraphSAGE. Uses DGL's DataLoader and NeighborSampler APIs. batch-size, n-neighbors and negative-samples parameters are available for this model. |
|  VGAE  | Variational Graph Auto-Encoder. Only difference from standard VAE is that it uses GraphConv layer instead of a Linear layer.  |
|   ResidualMLPVGAEReduceSum  |  Jointly learns both graph-wise (with VGAE) and attribute-wise features and aggregates them with summation  |
|   ResidualMLPVGAEReduceFF  |  Jointly learns both graph-wise (with VGAE) and attribute-wise features and aggregates them with a feed-forward layer  |
|   ResidualMLPGraphSAGEReduceSum  |  Jointly learns both graph-wise (with GraphSAGE) and attribute-wise features and aggregates them with summation  |
|   ResidualMLPGraphSAGEReduceFF  |  Jointly learns both graph-wise (with GraphSAGE) and attribute-wise features and aggregates them with a feed-forward layer  |

### Code

Most of the preprocessing stuff is taken from [DGL documentation](https://docs.dgl.ai/tutorials/blitz/4_link_predict.html#sphx-glr-tutorials-blitz-4-link-predict-py).

You can run it with the following example command:

```bash
    python train.py \
        --model ResidualMLPVGAEReduceFF \
        --epochs 80 \ 
        --graph-h-feats 16 16 \ 
        --mlp-h-feats 16 16 \
        --reduce-ff-size 16 \
        --lr 0.01
    /
```