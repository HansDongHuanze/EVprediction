import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")

class CoupFourGAT(nn.Module):
    def __init__(self, nfeat, noutput, nclass, dropout, alpha, nheads, adj, num_nodes=247, embed_size=32, sparsity_threshold=0.01):
        super(CoupFourGAT, self).__init__()
        self.adj = adj
        self.nfeat = nfeat
        self.dropout = dropout
        self.nheads = nheads
        self.sparsity_threshold = sparsity_threshold
        self.noutput = noutput

        self.attentions = nn.ModuleList([
            CFGATLayer(nfeat, nfeat, dropout, alpha) 
            for _ in range(nheads)
        ])
            
        self.norm = nn.LayerNorm(nfeat)
        self.encoder = nn.Linear(2, 1)
        self.activate = nn.LeakyReLU(0.01)
        self.decoder = nn.Linear(nfeat, noutput)
        self.mapping = nn.Linear(129, num_nodes)
        self.norm_2 = nn.LayerNorm(self.nfeat)
        
        self.freq_conv = nn.Conv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.gate_fusion = GateFusion(in_dim=nfeat)

        self.W_q = nn.ParameterList([
            nn.Parameter(torch.randn(self.nfeat, self.nfeat)),
            nn.Parameter(torch.randn(self.nfeat, self.nfeat))
        ])
        self.W_k = nn.ParameterList([
            nn.Parameter(torch.randn(self.nfeat, self.nfeat)),
            nn.Parameter(torch.randn(self.nfeat, self.nfeat))
        ])
        self.W_v = nn.ParameterList([
            nn.Parameter(torch.randn(self.nfeat, self.nfeat)),
            nn.Parameter(torch.randn(self.nfeat, self.nfeat))
        ])
        
        self.complexMapping = nn.Linear(self.nfeat, self.nfeat)

    def forward(self, x, prc):
        assert prc.shape == x.shape, f"Shape mismatch: prc {prc.shape} vs x {x.shape}"
        residual = x

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.norm_2(self.FGCN(x)) + residual

        x = torch.stack([x, prc], dim=3)
        x = self.encoder(x)
        x = torch.squeeze(x, dim=-1)  # safer squeeze
        x = self.activate(x)
        x = self.decoder(x) + residual[:,:,:self.noutput]
        return x

    def atten_com(self, x):
        res = x
        multi_head_outputs = []
        for att in self.attentions:
            att_output = att(x, self.adj)
            att_output = self.norm(att_output)
            multi_head_outputs.append(att_output)

        heads_stack = torch.stack(multi_head_outputs, dim=1)

        fused_features = []
        for i in range(heads_stack.size(2)):
            node_features = heads_stack[:, :, i, :]
            fused_node = self.gate_fusion(node_features.reshape(self.nheads, -1, self.nfeat))
            fused_features.append(fused_node + node_features.mean(dim=1))

        x = torch.stack(fused_features, dim=1)
        return x

    def FGCN(self, x):
        res = x
        B, N, L = x.shape
        x_reshaped = x.reshape(B, -1)

        n_fft = 256
        target_frames = 12
        seq_len = 2964
        hop_length = (seq_len - n_fft) // (target_frames - 1)

        x_stft = torch.stft(x_reshaped, n_fft=n_fft, hop_length=hop_length,
                            win_length=n_fft, return_complex=True)

        x_stft = x_stft[..., :12]
        x_stft = x_stft.reshape(B, L, -1)
        x_stft_real = self.mapping(x_stft.real)
        x_stft_imag = self.mapping(x_stft.imag)
        x_stft = torch.stack([x_stft_real, x_stft_imag], dim=-1)
        x_stft = torch.view_as_complex(x_stft)
        x_stft = x_stft.reshape(B, N, -1)

        x = checkpoint(self.freq_convolution, x_stft, B, N, self.nfeat, preserve_rng_state=False, use_reentrant=False)
        x = checkpoint(self.fourierGC, x, use_reentrant=False, preserve_rng_state=False)

        x_vec = torch.stack([x.real, x.imag], dim=-1)
        x = self.encoder(x_vec).squeeze(dim=-1)
        return x

    def fourierGC(self, x):
        res = x
        o1_real = self.atten_com(x.real)
        o1_imag = self.atten_com(x.imag)

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)

        x = torch.view_as_complex(y)
        return x

    def freq_convolution(self, x, B, N, L):
        x_real = x.real
        x_imag = x.imag
        vec = torch.stack([x_real, x_imag], dim=-1)
        res = vec
        vec = torch.reshape(vec, (B, 2, N, L))
        x = self.freq_conv(vec)
        x = torch.reshape(x, (B, N, L, 2))
        x = self.activate(x) + res
        x = torch.view_as_complex(x)
        return x

    # Optional future use: frequency attention
    def freq_attention(self, x, B, N, L):
        x_real = x.real
        x_imag = x.imag

        Q_real = torch.einsum('bli,io->blo', x_real, self.W_q[0]) - torch.einsum('bli,io->blo', x_imag, self.W_q[1])
        Q_imag = torch.einsum('bli,io->blo', x_imag, self.W_q[0]) + torch.einsum('bli,io->blo', x_real, self.W_q[1])
        Q = torch.stack([Q_real, Q_imag], dim=-1)

        K_real = torch.einsum('bli,io->blo', x_real, self.W_k[0]) - torch.einsum('bli,io->blo', x_imag, self.W_k[1])
        K_imag = torch.einsum('bli,io->blo', x_imag, self.W_k[0]) + torch.einsum('bli,io->blo', x_real, self.W_k[1])
        K = torch.stack([K_real, K_imag], dim=-1)

        V_real = torch.einsum('bli,io->blo', x_real, self.W_v[0]) - torch.einsum('bli,io->blo', x_imag, self.W_v[1])
        V_imag = torch.einsum('bli,io->blo', x_imag, self.W_v[0]) + torch.einsum('bli,io->blo', x_real, self.W_v[1])
        V = torch.stack([V_real, V_imag], dim=-1)

        Q_complex = torch.view_as_complex(Q)
        K_complex = torch.view_as_complex(K)
        V_complex = torch.view_as_complex(V)

        scale = 1 / math.sqrt(N)
        scores = torch.einsum('bik,bjk->bij', Q_complex, K_complex) * scale

        mask = torch.triu(torch.ones(scores.size(2), scores.size(2), dtype=torch.bool, device=x.device), diagonal=1)
        mask = mask.unsqueeze(0).expand(scores.size(0), -1, -1)
        scores = scores.masked_fill(mask, -float('inf'))

        real_softmax = torch.softmax(scores.real, dim=-1)
        imag_softmax = torch.softmax(scores.imag, dim=-1)

        real_temp = real_softmax @ V_complex.real
        imag_temp = imag_softmax @ V_complex.imag

        attention = torch.stack([real_temp, imag_temp], dim=-1)
        return torch.view_as_complex(attention)

class CFGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(CFGATLayer, self).__init__()
        
        # Define the attention mechanism parameters
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Weight matrices for each head
        self.W_q = nn.Parameter(torch.randn(in_features, out_features))
        self.W_k = nn.Parameter(torch.randn(in_features, out_features))
        self.W_v = nn.Parameter(torch.randn(in_features, out_features))
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: input features, adj: adjacency matrix
        
        # Compute Query, Key, and Value
        Q = torch.matmul(x, self.W_q)  # Query computation
        K = torch.matmul(x, self.W_k)  # Key computation
        V = torch.matmul(x, self.W_v)  # Value computation
        
        # Calculate attention scores using the adjacency matrix
        scores = torch.matmul(Q, K.transpose(1, 2))  # Q*K^T, resulting in [B, N, N]
        
        # Scale attention scores to prevent extremely large values
        scale = 1 / torch.sqrt(torch.tensor(self.out_features, dtype=torch.float))
        scores = scores * scale
        
        # Masking the attention matrix using adjacency matrix
        scores = scores.masked_fill(adj == 0, -1e9)  # Mask invalid edges with a large negative number

        # Apply softmax to get the attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply dropout to the attention weights
        attention_weights = self.dropout_layer(attention_weights)
        
        # Aggregate the features with the attention weights
        out = torch.matmul(attention_weights, V)  # [B, N, F_out]
        
        return out

class GateFusion(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gate_block = nn.Sequential(
            nn.Linear(in_dim*2, in_dim),
            nn.Dropout(p=.5),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )
         
        self._initialize_weights()

    def _initialize_weights(self):  
        for m in self.modules():
            if isinstance(m, nn.Linear):                      
                nn.init.xavier_normal_(m.weight)
    
    def forward(self, heads):
        avg_pool = torch.mean(heads, dim=0)
        max_pool = torch.max(heads, dim=0)[0]
        gate = self.gate_block(torch.cat([avg_pool,max_pool],dim=-1))
        return gate * avg_pool + (1-gate) * max_pool