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

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, num_nodes=247):
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
        self.node_weights = nn.Parameter(torch.randn(num_nodes))
        self.node_bias = nn.Parameter(torch.zeros(num_nodes))
        # self.NodeWiseTransform = NodeWiseTransform(num_nodes)
        self.norm = nn.LayerNorm(out_features)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # input = node_wise_operation(input)
        input = self.node_wise_matrix(input) + input
        # input = self.NodeWiseTransform(input)
        batch_size, N, _ = input.size()
        if adj.dim() == 3:
            adj = adj[:,:,1].unsqueeze(2).repeat(1,1,adj.shape[1])  # 扩展为 [batch_size, N, N]
        elif adj.size(0) != batch_size:
            adj = adj[:,:].unsqueeze(0).repeat(batch_size, 1, 1)

        h = torch.matmul(input, self.W)  # [batch_size, N, out_features]
        h = self.norm(h)

        residential = h

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

        h_prime = torch.matmul(attention, h) + residential  # [batch_size, N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        
    def node_wise_matrix(self, x):
        return x * self.node_weights.view(1, -1, 1) + self.node_bias.view(1, -1, 1)

def node_wise_operation(x):
    mean = x.mean(dim=-1, keepdim=True)  # [batch, nodes, 1]
    std = x.std(dim=-1, keepdim=True)    # [batch, nodes, 1]
    return (x - mean) / (std + 1e-8)

class GAT_Multi(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT_Multi, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid//nheads, dropout, alpha) 
            # GraphAttentionLayer(nfeat, nhid, dropout, alpha) 
            for _ in range(nheads)
        ])
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.norm = nn.LayerNorm(nhid)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.encoder = nn.Linear(nfeat, 64)
        self.activate = nn.LeakyReLU(0.01)
        self.decoder = nn.Linear(64, 1)

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
        x = self.encoder(x)
        x = self.activate(x)
        x = self.decoder(x) + residual
        return x[:,:,-1]

class NodeWiseTransform(nn.Module):
    def __init__(self, 
                 num_nodes: int, 
                 use_weights: bool = True,
                 use_bias: bool = True,
                 activation: str = None,
                 init_method: str = 'xavier'):
        """
        高级节点级变换层
        参数：
            num_nodes: 节点数量
            use_weights: 是否启用可学习权重
            use_bias: 是否启用可学习偏置
            activation: 激活函数类型（'relu','sigmoid','tanh'等）
            init_method: 参数初始化方法 ('xavier', 'kaiming', 'normal')
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.use_weights = use_weights
        self.use_bias = use_bias

        # 权重参数初始化
        if use_weights:
            self.weights = nn.Parameter(torch.Tensor(num_nodes))
            self._init_parameter(self.weights, init_method)
        else:
            self.register_parameter('weights', None)

        # 偏置参数初始化
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(num_nodes))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        # 激活函数配置
        self.activation = None
        if activation:
            self.activation = getattr(nn, activation.capitalize() + '()', None)
            if not self.activation:
                raise ValueError(f"Unsupported activation: {activation}")

    def _init_parameter(self, tensor, method):
        """修正后的初始化方法"""
        if tensor.dim() == 1:
            if method == 'xavier':
                # 一维参数改用均匀分布初始化
                nn.init.uniform_(tensor, a=-0.1, b=0.1)
            elif method == 'kaiming':
                nn.init.normal_(tensor, mean=0, std=1.0/math.sqrt(tensor.size(0)))
            elif method == 'normal':
                nn.init.normal_(tensor, mean=0, std=0.01)
        else:
            # 保留原有二维初始化逻辑
            if method == 'xavier':
                nn.init.xavier_normal_(tensor)
            elif method == 'kaiming':
                nn.init.kaiming_normal_(tensor)
            elif method == 'normal':
                nn.init.normal_(tensor, mean=0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入形状: [batch, nodes, time]
        输出形状: [batch, nodes, time]
        """
        assert x.size(1) == self.num_nodes, \
            f"节点数量不匹配，预期{self.num_nodes}，实际输入{x.size(1)}"

        # 应用权重
        if self.use_weights:
            weight_matrix = self.weights.view(1, -1, 1)  # 广播维度
            x = x * weight_matrix

        # 应用偏置
        if self.use_bias:
            bias_matrix = self.bias.view(1, -1, 1)
            x = x + bias_matrix

        # 应用激活函数
        if self.activation is not None:
            x = self.activation(x)

        return x

    def extra_repr(self) -> str:
        """打印配置信息"""
        return f"nodes={self.num_nodes}, weight={self.use_weights}, bias={self.use_bias}"