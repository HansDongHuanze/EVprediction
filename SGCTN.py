import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SGCTN(nn.Module):
    def __init__(self, adj_matrix, lap_matrix, num_nodes=247, 
                 feat_dim=2, embed_dim=64, n_heads=4, 
                 pred_steps=12, window_size=24 * 4, lambda1=0.001, lambda2=0.01):
        super().__init__()
        self.register_buffer('adj', adj_matrix)
        self.register_buffer('L', lap_matrix)
        self.num_nodes = num_nodes
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        # Spectral embedding module
        self.spectral_encoder = SpectralEncoder(feat_dim, embed_dim, window_size)
        
        # Spatio-temporal attention
        self.attn_heads = nn.ModuleList([
            GraphAwareAttentionLayer(embed_dim, adj_matrix, lap_matrix)
            for _ in range(n_heads)
        ])
        
        # Multi-scale fusion
        self.fusion = MultiScaleFusion(embed_dim, n_heads)
        
        # Prediction modules
        self.temporal_decoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, pred_steps)
        )
        
        # Learnable parameters for regularization
        self.theta = nn.Parameter(torch.randn(num_nodes, embed_dim))
        
    def forward(self, x, covariates):
        # x: [B, T, N, F], covariates: [B, T, D]
        B, T, N, F = x.shape
        
        # 1. Spectral embedding
        Z = self.spectral_encoder(x)  # [B, N, embed_dim]
        
        # 2. Spatio-temporal attention
        head_outputs = []
        for head in self.attn_heads:
            h = head(Z, covariates)
            head_outputs.append(h)
        
        # 3. Multi-scale fusion
        fused = self.fusion(head_outputs)  # [B, N, embed_dim]
        
        # 4. Temporal decoding
        predictions = self.temporal_decoder(fused)  # [B, N, H]
        
        # 5. Compute regularization terms
        graph_reg = torch.trace(self.theta.T @ self.L @ self.theta)
        param_reg = torch.norm(self.theta, p='fro')**2
        
        return predictions.transpose(1,2), graph_reg + param_reg

class SpectralEncoder(nn.Module):
    def __init__(self, feat_dim, embed_dim, window_size):
        super().__init__()
        self.window_size = window_size
        self.hop_length = window_size // 2
        
        # Complex-valued processing
        self.real_proj = nn.Sequential(
            nn.Linear(window_size//2 + 1, embed_dim//2),
            nn.LayerNorm(embed_dim//2),
            nn.GELU()
        )
        self.imag_proj = nn.Sequential(
            nn.Linear(window_size//2 + 1, embed_dim//2),
            nn.LayerNorm(embed_dim//2),
            nn.GELU()
        )
        
        # Time embedding for covariates
        self.time_embed = nn.Embedding(24 * 7, embed_dim)
        
    def forward(self, x):
        # x: [B, T, N, F]
        B, T, N, F = x.shape
        x = x.permute(0, 2, 1, 3)  # [B, N, T, F]
        
        # STFT processing
        stft = torch.stft(
            x.reshape(B*N, T, F), 
            n_fft=self.window_size,
            hop_length=self.hop_length,
            return_complex=True
        )  # [B*N, F, Freq, Time]
        
        # Split real/imaginary parts
        real = self.real_proj(stft.real.permute(0,3,2,1))  # [B*N, T', embed/2]
        imag = self.imag_proj(stft.imag.permute(0,3,2,1))
        
        # Combine features
        Z = torch.cat([real, imag], dim=-1)  # [B*N, T', embed]
        Z = Z.view(B, N, -1, Z.shape[-1]).mean(dim=2)  # [B, N, embed]
        
        return F.layer_norm(Z, Z.shape[-1:])

class GraphAwareAttentionLayer(nn.Module):
    def __init__(self, embed_dim, adj_matrix, lap_matrix):
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer('adj_mask', adj_matrix.bool())
        
        # Complex projection matrices
        self.W_q = nn.Parameter(torch.randn(embed_dim, embed_dim//2, dtype=torch.cfloat))
        self.W_k = nn.Parameter(torch.randn(embed_dim, embed_dim//2, dtype=torch.cfloat))
        
        # Gating mechanism
        self.gate_mlp = nn.Sequential(
            nn.Linear(2*embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
        # Structure-aware transformation
        self.L = nn.Parameter(lap_matrix.clone().float(), requires_grad=False)
        
    def forward(self, Z, covariates):
        # Z: [B, N, D], covariates: [B, T, D]
        B, N, D = Z.shape
        
        # Complex projections
        Z_complex = torch.view_as_complex(
            torch.stack([Z, torch.zeros_like(Z)], dim=-1)
        )  # [B, N, D] complex
        
        Q = torch.einsum('bnd,dk->bnk', Z_complex, self.W_q)
        K = torch.einsum('bnd,dk->bnk', Z_complex, self.W_k)
        
        # Sparse attention scores (real part only)
        scores = torch.real(Q @ K.conj().transpose(1,2)) / math.sqrt(D)
        scores = scores.masked_fill(~self.adj_mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        
        # Structure-aware gating
        LH = torch.einsum('nm,bmd->bnd', self.L, Z)  # Laplacian transform
        gate_input = torch.cat([Z, LH], dim=-1)
        gate = self.gate_mlp(gate_input)
        
        return gate * torch.einsum('bnm,bmd->bnd', attn, Z)

class MultiScaleFusion(nn.Module):
    def __init__(self, embed_dim, n_scales):
        super().__init__()
        self.scale_weights = nn.Parameter(torch.ones(n_scales))
        self.context_mlp = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            nn.GELU(),
            nn.Linear(2*embed_dim, n_scales),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, features_list):
        # features_list: list of [B, N, D] tensors
        stacked = torch.stack(features_list, dim=-1)  # [B, N, D, S]
        
        # Context-aware weighting
        context = stacked.mean(dim=1)  # [B, D, S]
        weights = self.context_mlp(context.transpose(1,2))  # [B, S, S]
        
        # Adaptive fusion
        fused = torch.einsum('bnds,bs->bnd', stacked, weights)
        return fused