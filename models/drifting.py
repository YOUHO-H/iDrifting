import torch
import math

def compute_drift(gen: torch.Tensor, pos: torch.Tensor, temp: float = 0.05):
    """
    Compute drift field V with attention-based kernel.

    Args:
        gen: Generated samples [G, D]
        pos: Data samples [P, D]
        temp: Temperature for softmax kernel

    Returns:
        V: Drift vectors [G, D]
    """
    targets = torch.cat([gen, pos], dim=0)
    G = gen.shape[0]

    dist = torch.cdist(gen, targets)
    dist[:, :G].fill_diagonal_(1e6)  # mask self
    kernel = (-dist / temp).exp() # unnormalized kernel

    normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True) 
    # normalize along both dimensions, which we found to slightly improve performance
    normalizer = normalizer.clamp_min(1e-12).sqrt()
    normalized_kernel = kernel / normalizer

    pos_coeff = normalized_kernel[:, G:] * normalized_kernel[:, :G].sum(dim=-1, keepdim=True)
    pos_V = pos_coeff @ targets[G:]
    neg_coeff = normalized_kernel[:, :G] * normalized_kernel[:, G:].sum(dim=-1, keepdim=True)
    neg_V = neg_coeff @ targets[:G]

    return pos_V - neg_V

def sample_hybrid_omega(
    m: int,
    dim: int,
    sigma: float = 1.0,
    laplace_scale: float = 1.0,
    wc: float = 1.0,
    device="cuda"
):
    """
    Sample omega from hybrid spectral distribution.
    """
    half = m // 2

    # Gaussian part (low frequency)
    omega_g = torch.randn(half, dim, device=device) / sigma
    mask_g = omega_g.norm(dim=-1) <= wc
    omega_g = omega_g[mask_g]

    # Laplacian (Cauchy) part (high frequency)
    omega_l = torch.distributions.Cauchy(
        torch.zeros(dim, device=device),
        torch.ones(dim, device=device) / laplace_scale
    ).sample((half,))
    mask_l = omega_l.norm(dim=-1) > wc
    omega_l = omega_l[mask_l]

    omega = torch.cat([omega_g, omega_l], dim=0)

    # If not enough samples, resample
    while omega.shape[0] < m:
        extra = torch.randn(m, dim, device=device) / sigma
        omega = torch.cat([omega, extra], dim=0)

    return omega[:m]


def compute_rff_features(x, omega, b):
    """
    Compute random Fourier features.
    """
    proj = x @ omega.T + b
    return math.sqrt(2.0 / omega.shape[0]) * torch.cos(proj)


def compute_drift_hybrid(
    gen: torch.Tensor,
    pos: torch.Tensor,
    m: int = 512,
    sigma: float = 1.0,
    laplace_scale: float = 1.0,
    wc: float = 1.0,
):
    """
    Drift using hybrid kernel via random Fourier features.

    Args:
        gen: [G, D]
        pos: [P, D]

    Returns:
        V: [G, D]
    """
    device = gen.device
    targets = torch.cat([gen, pos], dim=0)
    G = gen.shape[0]

    # ---- 1. sample hybrid frequencies ----
    omega = sample_hybrid_omega(
        m=m,
        dim=gen.shape[1],
        sigma=sigma,
        laplace_scale=laplace_scale,
        wc=wc,
        device=device
    )

    # random phase
    b = 2 * math.pi * torch.rand(m, device=device)

    # ---- 2. compute features ----
    phi_gen = compute_rff_features(gen, omega, b)        # [G, m]
    phi_targets = compute_rff_features(targets, omega, b)  # [G+P, m]

    # ---- 3. kernel approximation ----
    kernel = phi_gen @ phi_targets.T  # [G, G+P]

    # mask self interaction
    kernel[:, :G].fill_diagonal_(0.0)

    # ---- 4. same normalization trick ----
    normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True)
    normalizer = normalizer.clamp_min(1e-12).sqrt()
    normalized_kernel = kernel / normalizer

    # ---- 5. compute drift ----
    pos_coeff = normalized_kernel[:, G:] * normalized_kernel[:, :G].sum(dim=-1, keepdim=True)
    pos_V = pos_coeff @ targets[G:]

    neg_coeff = normalized_kernel[:, :G] * normalized_kernel[:, G:].sum(dim=-1, keepdim=True)
    neg_V = neg_coeff @ targets[:G]

    return pos_V - neg_V