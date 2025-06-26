import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


def kl_divergence_gaussian(mu1, logvar1, mu2, logvar2):
    """
    KL( N(mu1, var1) || N(mu2, var2) ) for diagonal Gaussians.
    Analytic formula:
      0.5 * sum( log(var2/var1) + (var1 + (mu1-mu2)^2)/var2 - 1 )
    Inputs:
      mu1, logvar1: tensors [..., D]
      mu2, logvar2: tensors [..., D]
    Returns:
      kl: tensor [...] (sum over D)
    """
    var1 = logvar1.exp().clamp(min=1e-6)
    var2 = logvar2.exp().clamp(min=1e-6)
    term1 = (logvar2 - logvar1)  # log(var2/var1)
    term2 = (var1 + (mu1 - mu2).pow(2)) / var2
    kl = 0.5 * (term1 + term2 - 1).sum(dim=-1)
    return kl

def js_divergence_gaussian(mu1, logvar1, mu2, logvar2):
    """
    Jensen-Shannon divergence between two diagonal Gaussians:
      JS(P||Q) = 0.5 KL(P||M) + 0.5 KL(Q||M),  M = 0.5(P+Q)
    where P+Q means mixture of the two distributions.
    For diagonal Gaussians, M is not Gaussian, but we approximate M by 
    the Gaussian with mu_M=(mu1+mu2)/2, var_M=(var1+var2)/2.
    """
    mu_m = 0.5 * (mu1 + mu2)
    var_m = 0.5 * (logvar1.exp() + logvar2.exp())
    var_m = var_m.clamp(min=1e-6)
    logvar_m = torch.log(var_m)
    kl1 = kl_divergence_gaussian(mu1, logvar1, mu_m, logvar_m)
    kl2 = kl_divergence_gaussian(mu2, logvar2, mu_m, logvar_m)
    return 0.5 * (kl1 + kl2)

def hellinger_distance_gaussian(mu1, logvar1, mu2, logvar2):
    """
    Hellinger distance H(P,Q) for diagonal Gaussians:
      H^2 = 1 - prod_i [ (2 * sqrt(var1_i*var2_i) / (var1_i+var2_i))^(1/2) 
                         * exp( - (mu1_i-mu2_i)^2 / [4*(var1_i+var2_i)] )
                       ]
    Returns H (not squared).
    """
    var1 = logvar1.exp()
    var2 = logvar2.exp()
    # term for each dim
    denom = var1 + var2 + 1e-6
    sqrt_term = torch.sqrt(2.0 * torch.sqrt(var1 * var2) / denom)
    exp_term = torch.exp(- (mu1 - mu2).pow(2) / (4.0 * denom))
    per_dim = sqrt_term * exp_term
    per_dim = per_dim.clamp(min=1e-6,max=1.0)
    prod = per_dim.prod(dim=-1)
    prod = prod.clamp(min=0.,max=1.)
    h2 = 1.0 - prod
    return torch.sqrt(h2.clamp(min=0.0))

def wasserstein_distance_gaussian(mu1, logvar1, mu2, logvar2):
    """
    2-Wasserstein distance W2 between two diagonal Gaussians:
      W2^2 = ||mu1 - mu2||^2 + ||sqrt(var1) - sqrt(var2)||^2
    Returns W2 (not squared).
    """
    var1 = logvar1.exp()
    var2 = logvar2.exp()
    mean_diff2 = (mu1 - mu2).pow(2).sum(dim=-1)
    std_diff2  = (var1.sqrt() - var2.sqrt()).pow(2).sum(dim=-1)
    w2_sq = mean_diff2 + std_diff2
    w2_sq =w2_sq.clamp(min=0.)
    return torch.sqrt(w2_sq + 1e-6)

class AttentionSimilarity(nn.Module):
    def __init__(self, hidden_size, num_heads=1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        # Linear Combination
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.q_norm = RMSNorm(hidden_size)
        self.k_norm = RMSNorm(hidden_size)
        if num_heads!=1:
            self.v_proj = nn.Linear(hidden_size,hidden_size)
        else:
            self.v_proj = nn.Identity()
        self.head_dim = hidden_size//num_heads
        self.num_heads = num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # nn.init.xavier_uniform_(self.q_proj.weight)
        # nn.init.xavier_uniform_(self.k_proj.weight)
        # # nn.init.xavier_uniform_(self.v_proj.weight)
        # # nn.init.zeros_(self.v_proj.bias)
        # nn.init.zeros_(self.q_proj.bias)
        # nn.init.zeros_(self.k_proj.bias)

    def forward(self, z, expert_keys):
        B, D = z.shape
        M, _ = expert_keys.shape


        q = self.q_proj(z)  # [B, D]
        q = self.q_norm(q)
        k = self.k_proj(expert_keys)  # [M, D]
        k = self.k_norm(k)
        v = self.v_proj(expert_keys)

        q = q.view(B, self.num_heads, self.head_dim)  # [B, H, Dh]
        k = k.view(M, self.num_heads, self.head_dim)  # [M, H, Dh]
        v = v.view(M, self.num_heads, self.head_dim)  # [M, H, Dh]

        # q: [B, H, Dh]
        # k: [M, H, Dh] -> [H, Dh, M]
        attn_scores = torch.einsum('bhd,mhd->bhm', q, k) * self.scale  # [B, H, M]

        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, H, M]

        # attn_weights: [B, H, M]
        # v: [M, H, Dh] -> [M, H, Dh]
        weighted_v = torch.einsum('bhm,mhd->bhd', attn_weights, v)  # [B, H, Dh]

        
        weighted_v = weighted_v.reshape(-1,self.num_heads*self.head_dim)  # [B, D]

        similarity = attn_scores.mean(dim=1)  # [B, M]
        
        return similarity, weighted_v
    
class GaussianKernelSimilarity(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
    def forward(self, z, expert_keys, sigma=1.0):
        euclidean_dist_squared = torch.cdist(z, expert_keys, p=2).pow(2)  # [B, M]
    
        similarity = torch.exp(-euclidean_dist_squared / (2 * sigma**2)) # [B , M]
        
        weighted_expert = torch.softmax(similarity,dim=-1) @ expert_keys #[B,M] [M,D] -> [B,D] 

        return similarity, weighted_expert

class MahalanobisSimilarity(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.log_sigma = nn.Parameter(torch.zeros(latent_dim))  
    def forward(self, z, expert_keys):

        sigma_inv = torch.diag(torch.exp(-self.log_sigma))  # [D, D]
        
        
        diff = z.unsqueeze(1) - expert_keys.unsqueeze(0)  # [B, M, D]
        
        mahalanobis_dist = torch.sum(
            torch.matmul(diff, sigma_inv) * diff, 
            dim=-1
        )
        

        similarity = 1.0 / (1.0 + mahalanobis_dist)
        weighted_expert = torch.softmax(similarity,dim=-1) @expert_keys
        return similarity, weighted_expert

class CosineSimilarity(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
    def forward(self, z, expert_keys):
        similarity = F.cosine_similarity(z.unsqueeze(1), expert_keys.unsqueeze(0), dim=-1) 
        weighted_expert = torch.softmax(similarity, dim=-1) @ expert_keys
        return similarity, weighted_expert


def adaptive_similarity(z, expert_keys, alpha=0.5):
 
    cos_sim = F.cosine_similarity(z.unsqueeze(1), expert_keys.unsqueeze(0), dim=-1)  # [B, M]
    

    euclidean_dist = torch.cdist(z, expert_keys, p=2)  # [B, M]
    eucl_sim = 1.0 / (1.0 + euclidean_dist)  
    

    similarity = alpha * cos_sim + (1 - alpha) * eucl_sim
    
    return similarity


def compute_cosine_similarity(z, expert_key):
    return F.cosine_similarity(z.unsqueeze(1), expert_key.unsqueeze(0),dim=-1)

def compute_dot_similarity(z,expert_key):
    return torch.matmul(z,expert_key.T)
    