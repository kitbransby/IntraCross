"""
Adapted from SuperGlue (Paul-Edouard Sarlin et al., CVPR 2020).
Original code: github.com/MagicLeapResearch/SuperPointPretrainedNetwork
& 
Adapted by Philipp Lindenberger (Phil26AT)
"""

import torch
from torch import nn
from copy import deepcopy
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt


def MLP(channels, do_bn=True, dropout=0.0):
    """
    Multi-Layer Perceptron (MLP) with optional BatchNorm and Dropout.

    Args:
        channels (list of int): List specifying the number of channels in each layer.
        do_bn (bool): Whether to use InstanceNorm1d normalization.
        dropout (float): Dropout probability (0.0 means no dropout).

    Returns:
        nn.Sequential: The constructed MLP model.
    """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):  # Skip dropout and activation for the last layer
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i], affine=True))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

class KeypointEncoder_Adapter(nn.Module):
    def __init__(self, input_dim, feature_dim, layers, dropout):
        super().__init__()
        self.encoder = MLP([input_dim] + list(layers) + [feature_dim], dropout=dropout)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        return self.encoder(kpts.transpose(1,2))

class ContextEncoder_Adapter(nn.Module):
    def __init__(self, input_dim, feature_dim, layers, dropout):
        super().__init__()
        self.encoder = MLP([input_dim] + list(layers) + [feature_dim], dropout=dropout)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        return self.encoder(kpts.transpose(1,2))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim**0.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        assert d_model % h == 0
        self.dim = d_model // h
        self.h = h
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        b = query.size(0)
        query, key, value = [
            layer(x).view(b, self.dim, self.h, -1)
            for layer, x in zip(self.proj, (query, key, value))
        ]
        x, attn_scores = attention(query, key, value)
        return self.merge(x.contiguous().view(b, self.dim * self.h, -1)), attn_scores


class AttentionalPropagation(nn.Module):
    def __init__(self, num_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, num_dim)
        self.mlp = MLP([num_dim * 2, num_dim * 2, num_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message, attn_scores = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1)), attn_scores


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim, layer_names):
        super().__init__()
        self.layers = nn.ModuleList(
            [AttentionalPropagation(feature_dim, 4) for _ in range(len(layer_names))]
        )
        self.names = layer_names

    def forward(self, desc0, desc1):
        attn_maps = {}
        for i, (layer, name) in enumerate(zip(self.layers, self.names)):
            layer.attn.prob = []
            if self.training:
                (delta0, attn0), (delta1, attn1) = checkpoint(
                    self._forward, layer, desc0, desc1, name, preserve_rng_state=False
                )
            else:
                (delta0, attn0), (delta1, attn1) = self._forward(layer, desc0, desc1, name)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
            del delta0, delta1
            attn_maps['{}_{}'.format(i, name)] = [attn0, attn1]
        return desc0, desc1, attn_maps

    def _forward(self, layer, desc0, desc1, name):
        if name == "self":
            return layer(desc0, desc0), layer(desc1, desc1)
        elif name == "cross":
            return layer(desc0, desc1), layer(desc1, desc0)
        else:
            raise ValueError(name)


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters):
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters):
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat(
        [torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 1
    )

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class IntraCross(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.conf = {
            "descriptor_dim": config['EMB_DIM'],
            "ctx_only": config['CTX_ONLY'],
            "pos_only": config['POS_ONLY'],
            "lambda": config['LAMBDA'],
            "weights": False,
            "keypoint_encoder": [32, 64, 128, 256],
            "GNN_layers": ["self", "cross"] * 9,
            "num_sinkhorn_iterations": 50,
            "filter_threshold": config['TAU'],
            "use_scores": False,
            "loss": {
                "nll_balancing": config['GAMMA'],
            },
        }

        self.kenc = KeypointEncoder_Adapter(
            config['POS_DIM'], 
            self.conf['descriptor_dim'], 
            self.conf['keypoint_encoder'], 
            dropout=config['DROPOUT']['POS'])

        self.cenc = ContextEncoder_Adapter(
            config['CTX_DIM'], 
            self.conf['descriptor_dim'], 
            self.conf['keypoint_encoder'], 
            dropout=config['DROPOUT']['CTX'])

        self.gnn = AttentionalGNN(self.conf['descriptor_dim'], self.conf['GNN_layers'])

        self.final_proj = nn.Conv1d(self.conf['descriptor_dim'], self.conf['descriptor_dim'], kernel_size=1, bias=True)
        bin_score = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter("bin_score", bin_score)


    def forward(self, kpts0, kpts1, ctx0, ctx1):

        assert torch.all(kpts0 >= -0.5) and torch.all(kpts0 <= 1.5)
        assert torch.all(kpts1 >= -0.5) and torch.all(kpts1 <= 1.5)

        if self.conf['pos_only']:
            desc0 = self.kenc(kpts0, adapter='0')  
            desc1 = self.kenc(kpts1, adapter='1') 
        elif self.conf['ctx_only']:
            desc0 = self.cenc(ctx0, adapter='0') 
            desc1 = self.cenc(ctx1, adapter='1')
        else:
            desc0 = self.kenc(kpts0, adapter='0') + self.cenc(ctx0, adapter='0') 
            desc1 = self.kenc(kpts1, adapter='1') + self.cenc(ctx1, adapter='1')

        desc0, desc1, attn_maps = self.gnn(desc0, desc1)

        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        scores = torch.einsum("bdn,bdm->bnm", mdesc0, mdesc1) # matmul over the d dim. similarity scores.
        cost = scores / self.conf['descriptor_dim']**0.5


        temporal_penalty = self.compute_temporal_penalty(
            kpts0[0], 
            kpts1[0], 
            scaler=self.conf['lambda'])

        cost += temporal_penalty

        scores = log_optimal_transport(
            cost, self.bin_score, iters=self.conf['num_sinkhorn_iterations']
        )

        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        m0, m1 = max0.indices, max1.indices
        mutual0 = arange_like(m0, 1)[None] == m1.gather(1, m0)
        mutual1 = arange_like(m1, 1)[None] == m0.gather(1, m1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
        valid0 = mutual0 & (mscores0 > self.conf['filter_threshold'])
        valid1 = mutual1 & valid0.gather(1, m1)
        m0 = torch.where(valid0, m0, m0.new_tensor(-1))
        m1 = torch.where(valid1, m1, m1.new_tensor(-1))

        mscores0[mscores0 < self.conf['filter_threshold']] = 0
        mscores1[mscores1 < self.conf['filter_threshold']] = 0

        return {
            "sinkhorn_cost": cost,
            "temporal_penalty": temporal_penalty,
            "log_assignment": scores,
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "emb0": mdesc0, 
            "emb1": mdesc1,
            "attn_maps": attn_maps
        }

    def loss(self, pred, data):
        losses = {"total": 0}

        positive = data["gt_assignment"].float()

        num_pos = torch.max(positive.sum((1, 2)), positive.new_tensor(1))
        neg0 = (data["gt_matches0"] == -1).float()
        neg1 = (data["gt_matches1"] == -1).float()
        num_neg = torch.max(neg0.sum(1) + neg1.sum(1), neg0.new_tensor(1))

        log_assignment = pred["log_assignment"]

        nll_pos = -(log_assignment[:, :-1, :-1] * positive).sum((1, 2))
        nll_pos /= num_pos
        nll_neg0 = -(log_assignment[:, :-1, -1] * neg0).sum(1)
        nll_neg1 = -(log_assignment[:, -1, :-1] * neg1).sum(1)
        nll_neg = (nll_neg0 + nll_neg1) / num_neg
        nll = (
            self.conf['loss']['nll_balancing'] * nll_pos
            + (1 - self.conf['loss']['nll_balancing']) * nll_neg
        )
        losses["assignment_nll"] = nll
        losses["total"] = nll

        losses["nll_pos"] = nll_pos
        losses["nll_neg"] = nll_neg

        # Some statistics
        losses["num_matchable"] = num_pos
        losses["num_unmatchable"] = num_neg
        losses["bin_score"] = self.bin_score[None]

        return losses

    def metrics(self, pred, data):
        raise NotImplementedError

    def compute_temporal_penalty(self, x, y, scaler):

        # far away point receive large penalty, close points receive small penalty

        # Separate features
        points1_f0 = x[:, 0:1]  # temporal position 1, shape (N, 1), in interval [0, 1]
        points2_f0 = y[:, 0:1]  # temporal position 2, shape (M, 1), in interval [0, 1]

        # Compute pairwise distances for temporal position
        diff_f0 = points1_f0 - points2_f0.T  # Shape (N, M)
        dist_f0 = torch.abs(diff_f0) # Absolute difference for temporal position, shape (N, M) in interval [0, 1]

        # take negative distance, so that far away point receive number close to -1, and close points receive number close to 0
        dist_f0 = - dist_f0

        # scale by gamma
        dist_f0 = dist_f0 * scaler

        return dist_f0