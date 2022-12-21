from enum import Enum, auto

import torch
import torch.nn as nn
from torch.distributions import Normal, kl


def init_weights_xavier(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_normal_(module.weight)
        torch.nn.init.constant_(module.bias, 0.)


class NoiseType(Enum):
    DIAGONAL = auto()
    ISOTROPIC = auto()
    FIXED = auto()


class NormalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, noise_type, fixed_noise_level=None):
        super().__init__()
        assert (fixed_noise_level is not None) == (noise_type is NoiseType.FIXED)
        num_sigma_channels = {
            NoiseType.DIAGONAL: out_dim,
            NoiseType.ISOTROPIC: 1,
            NoiseType.FIXED: 0,
        }[noise_type]
        self.in_dim, self.out_dim = in_dim, out_dim
        self.noise_type, self.fixed_noise_level = noise_type, fixed_noise_level
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim + num_sigma_channels),
        )
        self.apply(init_weights_xavier)

    def forward(self, x, eps=1e-6):
        params = self.network(x)
        mu, log_sigma_param = params[..., :self.out_dim], params[..., self.out_dim:]
        if self.noise_type is NoiseType.DIAGONAL:
            sigma = torch.exp(log_sigma_param + eps)
        if self.noise_type is NoiseType.ISOTROPIC:
            sigma = torch.exp(log_sigma_param + eps).repeat(1, self.out_dim)
        if self.noise_type is NoiseType.FIXED:
            sigma = torch.full_like(mu, fill_value=self.fixed_noise_level)
        return Normal(loc=mu, scale=sigma)


class VAE(nn.Module):

    def __init__(self, input_dim, prior_dim, hidden_dim):
        super().__init__()
        self.prior = Normal(torch.zeros(prior_dim), torch.ones(prior_dim))
        self.z_given_x = NormalNetwork(input_dim, prior_dim, hidden_dim, NoiseType.ISOTROPIC)
        self.x_given_z = NormalNetwork(prior_dim, input_dim, hidden_dim, NoiseType.ISOTROPIC)

    def loss(self, x):
        pred_z = self.z_given_x(x)
        kl_div = kl.kl_divergence(pred_z, self.prior)
        monte_carlo_z = pred_z.rsample()
        monte_carlo_x = self.x_given_z(monte_carlo_z)
        rec_loss = torch.sum(monte_carlo_x.log_prob(x), dim=1, keepdim=True)
        return -(rec_loss - kl_div).squeeze(dim=1)

    def sample_z(self, n_samples):
        return self.prior.rsample((n_samples,))
