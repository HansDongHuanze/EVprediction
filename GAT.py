import torch
import torch.nn as nn
import models
import torch.nn.functional as F
import functions as fn
import copy

from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer

use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
fn.set_seed(seed=2023, flag=True)

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.kaiming_normal_(self.W, mode='fan_out', nonlinearity='leaky_relu')
        self.a = nn.Parameter(torch.empty(2*out_features, 1))
        nn.init.xavier_normal_(self.a, gain=nn.init.calculate_gain('leaky_relu', param=alpha))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        batch_size, N, _ = input.size()
        if adj.dim() == 3:
            adj = adj[:,:,1].unsqueeze(2).repeat(1,1,adj.shape[1])  # 扩展为 [batch_size, N, N]
        elif adj.size(0) != batch_size:
            adj = adj[:,:].unsqueeze(0).repeat(batch_size, 1, 1)

        h = torch.matmul(input, self.W)  # [batch_size, N, out_features]

        h_repeated1 = h.unsqueeze(2).expand(-1, -1, N, -1)  # [batch_size, N, N, out_features]
        h_repeated2 = h.unsqueeze(1).expand(-1, N, -1, -1)  # [batch_size, N, N, out_features]
        a_input = torch.cat([h_repeated1, h_repeated2], dim=-1)  # [batch_size, N, N, 2*out_features]

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # [batch_size, N, N]

        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)  # 扩展为 [batch_size, N, N]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # [batch_size, N, N]

        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h)  # [batch_size, N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        
class GAT_Multi(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT_Multi, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid//nheads, dropout, alpha) 
            for _ in range(nheads)
        ])
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.norm = nn.LayerNorm(nhid)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        residual = x
        x = F.dropout(x, self.dropout, training=self.training)
        multi_head_outputs = []
        for att in self.attentions:
            att_output = att(x, adj)  # [batch_size, N, nhid]
            att_output = self.norm(att_output)
            multi_head_outputs.append(att_output)

        x = torch.cat(multi_head_outputs, dim=-1)  # [batch_size, N, nhid * nheads]
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj)) + residual  # [batch_size, N, nclass]
        return x[:,:,-1]