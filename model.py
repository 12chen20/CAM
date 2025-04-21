import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import erdos_renyi_graph, remove_self_loops, add_self_loops, degree, add_remaining_self_loops
from data_utils import sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor
from torch_sparse import SparseTensor, matmul

from backbone import *



def gcn_conv(x, edge_index):
    N = x.shape[0]
    row, col = edge_index
    d = degree(col, N).float()
    d_norm_in = (1. / d[col]).sqrt()
    d_norm_out = (1. / d[row]).sqrt()
    value = torch.ones_like(row) * d_norm_in * d_norm_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    return matmul(adj, x)


class GraphConvolutionBase(nn.Module):
    def __init__(self, in_features, out_features, residual=False):
        super(GraphConvolutionBase, self).__init__()
        self.residual = residual
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if self.residual:
            self.weight_r = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_r.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, x0):
        hi = gcn_conv(x, adj)
        output = torch.mm(hi, self.weight)
        if self.residual:
            output = output + torch.mm(x, self.weight_r)
        return output


class CamConv(nn.Module):
    def __init__(self, in_features, out_features, K, residual=True, backbone_type='gcn', variant=False, device=None):
        super(CamConv, self).__init__()
        self.backbone_type = backbone_type
        self.out_features = out_features
        self.residual = residual
        if backbone_type == 'gcn':
            self.weights = Parameter(torch.FloatTensor(K, in_features * 2, out_features))
        elif backbone_type == 'gat':
            self.leakyrelu = nn.LeakyReLU()
            self.weights = nn.Parameter(torch.zeros(K, in_features, out_features))
            self.a = nn.Parameter(torch.zeros(K, 2 * out_features, 1))
        self.K = K
        self.device = device
        self.variant = variant
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weights.data.uniform_(-stdv, stdv)
        if self.backbone_type == 'gat':
            nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def specialspmm(self, adj, spm, size, h):
        adj = SparseTensor(row=adj[0], col=adj[1], value=spm, sparse_sizes=size)
        return matmul(adj, h)

    def forward(self, x, adj, e, weights=None):
        if weights == None:
            weights = self.weights
        if self.backbone_type == 'gcn':
            if not self.variant:
                hi = gcn_conv(x, adj)
            else:
                adj = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1]).to(self.device),
                                              size=(x.shape[0], x.shape[0])).to(self.device)
                hi = torch.sparse.mm(adj, x)
            hi = torch.cat([hi, x], 1)
            hi = hi.unsqueeze(0).repeat(self.K, 1, 1)
            outputs = torch.matmul(hi, weights)
            outputs = outputs.transpose(1, 0)
        elif self.backbone_type == 'gat':
            xi = x.unsqueeze(0).repeat(self.K, 1, 1)
            h = torch.matmul(xi, weights)
            N = x.size()[0]
            adj, _ = remove_self_loops(adj)
            adj, _ = add_self_loops(adj, num_nodes=N)
            edge_h = torch.cat((h[:, adj[0, :], :], h[:, adj[1, :], :]), dim=2)
            logits = self.leakyrelu(torch.matmul(edge_h, self.a)).squeeze(2)
            logits_max, _ = torch.max(logits, dim=1, keepdim=True)
            edge_e = torch.exp(logits - logits_max)

            outputs = []
            eps = 1e-8
            for k in range(self.K):
                edge_e_k = edge_e[k, :]
                e_expsum_k = self.specialspmm(adj, edge_e_k, torch.Size([N, N]),
                                              torch.ones(N, 1).cuda()) + eps
                assert not torch.isnan(e_expsum_k).any()

                hi_k = self.specialspmm(adj, edge_e_k, torch.Size([N, N]), h[k])
                hi_k = torch.div(hi_k, e_expsum_k)
                outputs.append(hi_k)
            outputs = torch.stack(outputs, dim=1)

        es = e.unsqueeze(2).repeat(1, 1, self.out_features)
        output = torch.sum(torch.mul(es, outputs), dim=1)

        if self.residual:
            output = output + x

        return output


class Cam(nn.Module):
    def __init__(self, d, c, args, device):
        super(Cam, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(args.num_layers):
            self.convs.append(
                CamConv(args.hidden_channels, args.hidden_channels, args.K,
                          backbone_type=args.backbone_type, residual=True,
                          device=device, variant=args.variant))
        self.W_gate = nn.ModuleList()
        self.lambda_sparse = args.lambda_sparse
        self.theta = args.theta
        for _ in range(args.num_layers):
            self.W_gate.append(nn.Linear(args.hidden_channels, 1))
        self.gru = nn.ModuleList()
        for _ in range(args.num_layers - 1):
            self.gru.append(nn.GRUCell(args.K, args.hidden_channels))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(d, args.hidden_channels))
        self.fcs.append(nn.Linear(args.hidden_channels, c))
        self.env_enc = nn.ModuleList()
        for _ in range(args.num_layers):
            if args.env_type == 'node':
                in_features = args.hidden_channels if args.num_layers == 0 else args.hidden_channels * 2
                self.env_enc.append(nn.Linear(in_features, args.K))
            elif args.env_type == 'graph':
                self.env_enc.append(GraphConvolutionBase(args.hidden_channels, args.K, residual=True))
            else:
                raise NotImplementedError

        self.act_fn = nn.ReLU()
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.tau = args.tau
        self.env_type = args.env_type
        self.device = device

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()
        for enc in self.env_enc:
            enc.reset_parameters()
        for gate in self.W_gate:
            gate.reset_parameters()

    def forward(self, x, adj, idx=None, training=False):
        self.training = training
        x = F.dropout(x, self.dropout, training=self.training)
        h = self.act_fn(self.fcs[0](x))
        h0 = h.clone()
        h_env = torch.zeros(x.size(0), self.gru[0].hidden_size).to(self.device) if self.gru else None
        reg = 0
        for i, con in enumerate(self.convs):
            h = F.dropout(h, self.dropout, training=self.training)
            if self.training:
                if i > 0 and h_env is not None:
                    env_input = torch.cat([h, h_env], dim=1)
                else:
                    env_input = torch.cat([h, h], dim=1)
                if self.env_type == 'node':
                    raw_logit = self.env_enc[i](env_input)
                    gate_logit = self.W_gate[i](h)
                    threshold = torch.sigmoid(gate_logit)
                    sparse_penalty = self.lambda_sparse * torch.sigmoid(raw_logit) * (raw_logit != 0).float()
                    sparse_logit = raw_logit - sparse_penalty
                    e = F.gumbel_softmax(sparse_logit, tau=self.tau, dim=-1)
                elif self.env_type == 'graph':
                    raw_logit = self.env_enc[i](h, adj, h0)
                    gate_logit = self.W_gate[i](h)
                    threshold = torch.sigmoid(gate_logit)
                    sparse_penalty = self.lambda_sparse * torch.sigmoid(raw_logit) * (raw_logit != 0).float()
                    sparse_logit = raw_logit - sparse_penalty
                    e = F.gumbel_softmax(sparse_logit, tau=self.tau, dim=-1)
                else:
                    raise NotImplementedError
                if i < len(self.convs) - 1 and h_env is not None:
                    h_env = self.gru[i](e, h_env)
                    h_env = torch.clamp(h_env, -5, 5)
                    h_env = h_env.detach()
                reg += self.reg_loss(e, sparse_logit)
            else:
                if self.env_type == 'node':
                    raw_logit = self.env_enc[i](torch.cat([h, h_env], dim=1) if h_env is not None else h)
                    pi = F.softmax(raw_logit, dim=-1)
                    mask = (pi > self.theta).float()
                    e = pi * mask
                elif self.env_type == 'graph':
                    raw_logit = self.env_enc[i](h, adj, h0)
                    pi = F.softmax(raw_logit / self.tau, dim=-1)
                    mask = (pi > self.theta).float()
                    e = pi * mask
                else:
                    raise NotImplementedError
            h = self.act_fn(con(h, adj, e))
        h = F.dropout(h, self.dropout, training=self.training)
        out = self.fcs[-1](h)
        return (out, reg / self.num_layers) if self.training else out

    def reg_loss(self, z, logit):
        log_pi = logit - torch.logsumexp(logit, dim=-1, keepdim=True).repeat(1, logit.size(1))
        return torch.mean(torch.sum(z * log_pi, dim=1))

    def sup_loss_calc(self, y, pred, criterion, args):
        if args.dataset in ('twitch', 'elliptic'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            loss = criterion(out, target)
        return loss

    def loss_compute(self, d, criterion, args):
        logits, reg_loss = self.forward(d.x, d.edge_index, idx=d.train_idx, training=True)
        sup_loss = self.sup_loss_calc(d.y[d.train_idx], logits[d.train_idx], criterion, args)
        loss = sup_loss + args.lamda * reg_loss
        return loss
