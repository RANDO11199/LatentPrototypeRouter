import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from .Metrics import *


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class DiversityRegularizer(nn.Module):
    def __init__(self, diversity_type='cosine', lambda_div=0.1):
        super().__init__()
        self.diversity_type = diversity_type
        self.lambda_div = lambda_div  # 多样性损失权重

    def forward(self, expert_keys):

        M, D = expert_keys.size()

        if M < 2:
            return torch.tensor(0.0, device=expert_keys.device)

        if self.diversity_type == 'cosine':
            norm_keys = F.normalize(expert_keys, p=2, dim=-1)  # [M, D]
            sim_matrix = torch.matmul(norm_keys, norm_keys.T)  # [M, M]
            diversity_loss = self._off_diagonal_mean(sim_matrix)

        elif self.diversity_type == 'euclidean':
            dist_matrix = torch.cdist(expert_keys, expert_keys, p=2)  # [M, M]
            diversity_loss = -self._off_diagonal_mean(dist_matrix)

        elif self.diversity_type == 'orthogonal':
            norm_keys = F.normalize(expert_keys, p=2, dim=-1)
            proj_matrix = torch.matmul(norm_keys, norm_keys.T)  # [M, M]
            identity = torch.eye(M, device=expert_keys.device)
            diversity_loss = F.mse_loss(proj_matrix, identity)

        else:
            raise ValueError(f"Unsupported diversity type: {self.diversity_type}")

        return self.lambda_div * diversity_loss

    def _off_diagonal_mean(self, matrix):
        M = matrix.size(0)
        off_diag_sum = matrix.sum() - matrix.diag().sum()
        return off_diag_sum / (M * (M - 1))
    
def GetSimilarityMetrics(config):
    if config.routing_metrics =="Cosine":
        return CosineSimilarity(config.router_latent_dim)
    elif config.routing_metrics =="CrossAttention":
        return AttentionSimilarity(config.router_latent_dim)
    elif config.routing_metrics =="GaussianKernel":
        return GaussianKernelSimilarity(config.router_latent_dim)
    elif config.routing_metrics =="Mahalanobis":
        return MahalanobisSimilarity(config.router_latent_dim)
    elif config.routing_metrics =="DotProducts":
        return compute_dot_similarity
    else:
        raise NotImplementedError()

def GetDistributionMetrics(config):
    if config.routing_metrics =="kl":
        return kl_divergence_gaussian
    elif config.routing_metrics =="js":
        return js_divergence_gaussian
    elif config.routing_metrics =="hellinger":
        return hellinger_distance_gaussian
    elif config.routing_metrics =="wasserstein":
        return wasserstein_distance_gaussian
    else:
        raise NotImplementedError()

class TokenDistributionRouter(nn.Module):
    def __init__(self, config,layer_id):
        super().__init__()
        self.layer_id = layer_id
        input_dim = config.hidden_size 
        try:
            self.num_experts = config.num_experts 
        except:
            try:
                self.num_experts = config.n_routed_experts 
            except:
                self.num_experts = config.num_local_experts 
        self.top_k = config.num_experts_per_tok 
        latent_dim = config.router_latent_dim
        self.model_type = config.model_type
        # Encode token into latent space
        self.norm = nn.LayerNorm(input_dim) # Encoded latent scale ≈ expert key
        self.encoder = nn.Linear(input_dim, latent_dim*2)  # output mu and logvar
        self.act = nn.SiLU()
        # Aligner 
        self.out_proj = nn.Linear(latent_dim, input_dim)
        # Regulation Setting
        self.diversity_reg = DiversityRegularizer(config.diversity_type,config.diversity_lambda)
        # Similarity Metrics
        self.SimilarityMetrics = config.SimilarityMetrics
        if config.SimilarityMetrics =="VectorSimilarity":
            self.similarity_func = GetSimilarityMetrics(config)
            if config.unit_ball:
                self.expert_keys = nn.Parameter(F.normalize(torch.randn(self.num_experts,latent_dim,dtype=torch.float)))
            else:
                self.expert_keys = nn.Parameter(torch.randn(self.num_experts,latent_dim,dtype=torch.float))
        elif config.SimilarityMetrics =="DistributionDistance":
            self.similarity_func = GetDistributionMetrics(config)
            keys = torch.randn(self.num_experts, latent_dim)
            keys = F.normalize(keys, dim=-1)      # each row has norm=1
            self.expert_keys_mu = nn.Parameter(keys,dtype=torch.float)
            self.expert_keys_logvar = nn.Parameter(torch.zeros(self.num_experts, latent_dim,dtype=torch.float))
            self.proj = nn.Linear(latent_dim, latent_dim)
        # other hyper
        self.ema_sums = nn.Parameter(self.expert_keys.data.clone())
        self.ema_counts = nn.Parameter(torch.zeros(self.num_experts))
        self.ema_decay = config.router_ema_decay 
        self._forward_count = 0
        self.kl_weight = config.kl_weight
        self.align_weight = config.align_weight
        self.div_weight = config.div_weight
    def encode(self,x):
        x = self.norm(x)
        x = self.act(x)
        mu_logvar = self.encoder(x)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # [B, D]
        return z, mu, logvar

    def compute_kl(self,mu,logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    
    def compute_diversity_loss(self,x):
        """鼓励专家原型多样化"""
        if self.SimilarityMetrics == "VectorSimilarity":
            return self.diversity_reg(x)
        elif self.SimilarityMetrics == "DistributionDistance":
            return self.diversity_reg(x)
    
    def forward(self, x):
        
        z, mu, logvar = self.encode(x)

        self._forward_count+=1

        if self.SimilarityMetrics == "VectorSimilarity":
            scores, weighted_v = self.similarity_func(mu, self.expert_keys)  # [B, N]
            scores = scores
            topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)
            self._last_z = z.detach()

        elif self.SimilarityMetrics == "DistributionDistance":
            scores = self.similarity_func(mu.unsqueeze(1), logvar.unsqueeze(1),self.expert_keys_mu.unsqueeze(0),self.expert_keys_logvar.unsqueeze(0))  # [B, N]
            scores = -(scores )
            weighted_v = torch.softmax(scores,dim=-1,  dtype=torch.float) @ self.proj(self.expert_keys_mu)
            topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)
            # self._last_mu = mu.detach()
            # self._last_log_var = logvar.detach()

        div_loss = self.compute_diversity_loss(mu) + self.compute_diversity_loss(self.expert_keys)
        kl = self.compute_kl(mu,logvar)

        routing_weights = F.softmax(topk_vals, dim=-1, dtype=torch.float)  # [B, K]
        routing_scores = torch.zeros_like(scores).scatter(1, topk_idx, topk_vals)
        # hard version ema
        self._last_routing = F.softmax(routing_scores, dim=-1, dtype=torch.float).detach()  # [B, M]
        ## soft version ema
        # self._last_routing = F.softmax(scores, dim=-1, dtype=torch.float).detach()  # [B, M]

        z_decoded =  self.out_proj(weighted_v)
        weighted_keys_target = self._last_routing.T @ z
        sim_loss = torch.abs(self.expert_keys - weighted_keys_target.detach()).mean()

        return routing_weights, self.div_weight*div_loss+ self.kl_weight*kl + self.align_weight*sim_loss, topk_idx, scores, z_decoded

    @torch.no_grad()
    def update_expert_keys(self):
        """
        Use EMA to update expert_keys based on token latents and routing weights.
        """
        if self.SimilarityMetrics == "VectorSimilarity":
            z = self._last_z  # [B, D]
            w = self._last_routing  # [B, M]

            # Sum over batch: weighted token latents per expert 
            token_sum = torch.matmul(w.T, z)  # [M, D]
            token_count = w.sum(dim=0)  # [M]

            # EMA update
            self.ema_sums.mul_(self.ema_decay).add_(token_sum, alpha=1-self.ema_decay)
            self.ema_counts.mul_(self.ema_decay).add_(token_count, alpha=1-self.ema_decay)
            # Avoid divide-by-zero
            norm = self.ema_counts.unsqueeze(1).clamp(min=1e-6)  # [M, 1]

            # Update expert keys
            new_keys = self.ema_sums / norm  # [M, D]
            self.expert_keys.data.copy_(new_keys)
        elif self.SimilarityMetrics == "DistributionDistance":
            w = self._last_routing  # [B, M]
            mu = self._last_mu.to(w.dtype)
            logvar = self._last_log_var.to(w.dtype)
            weighted_mu =  torch.matmul(w.T, mu)
            weighted_var = torch.matmul(w.T, mu**2 + logvar.exp())  # [M, D]
            # EMA update
            self.ema_counts.mul_(self.ema_decay).add_((1 - self.ema_decay) * w.sum(dim=0))     # [M]
            self.ema_mu.mul_(self.ema_decay).add_((1 - self.ema_decay) * weighted_mu)          # [M, D]
            self.ema_var.mul_(self.ema_decay).add_((1 - self.ema_decay) * weighted_var) 
            # Avoid divide-by-zero
            counts = self.ema_counts.unsqueeze(1).clamp(min=1e-6)  # [M, 1]
            mu_new = self.ema_mu / counts
            var_new = self.ema_var / counts - mu_new ** 2          # [M, D]
            logvar_new = torch.log(var_new.clamp(min=1e-6))
            # Update expert keys
            self.expert_keys_mu.data.copy_(mu_new)
            self.expert_keys_logvar.data.copy_(logvar_new)
