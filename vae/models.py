import torch
import torch.nn as nn
from torch.distributions import Normal, kl


class VAE(nn.Module):

    def __init__(self, input_dim, prior_dim, hidden_dim):
        super().__init__()
        self.prior = Normal(torch.zeros(prior_dim), torch.ones(prior_dim))
        self.encoder = DiagNormalNetwork(input_dim, prior_dim, hidden_dim)
        self.decoder = DiagNormalNetwork(prior_dim, input_dim, hidden_dim)

    def loss(self, x):
        pred_z = self.compute_z_given_x(x)
        kl_div = kl.kl_divergence(pred_z, self.prior).squeeze(1)
        monte_carlo_z = pred_z.rsample()
        monte_carlo_x = self.compute_x_given_z(monte_carlo_z)
        rec_loss = -torch.sum(monte_carlo_x.log_prob(x), dim=1)
        return kl_div + rec_loss

    def sample_z(self, n_samples):
        return self.prior.rsample((n_samples,))

    def compute_x_given_z(self, z):
        return self.decoder(z)

    def compute_z_given_x(self, x):
        return self.encoder(x)


class MMDVAE(VAE):

    def __init__(self, input_dim, prior_dim, hidden_dim, alpha, lambd):
        super().__init__(input_dim, prior_dim, hidden_dim)
        self.prior_dim = prior_dim
        self.alpha = alpha
        self.lambd = lambd

    def loss(self, x):
        pred_z = self.compute_z_given_x(x)
        kl_div = kl.kl_divergence(pred_z, self.prior).squeeze(1)
        monte_carlo_z = pred_z.rsample()
        monte_carlo_x = self.compute_x_given_z(monte_carlo_z)
        rec_loss = -torch.sum(monte_carlo_x.log_prob(x), dim=1)
        mmd = self.compute_mmd(monte_carlo_z)
        return rec_loss + (1 - self.alpha) * kl_div + \
               (self.alpha + self.lambd - 1) * mmd

    def compute_mmd(self, sampled_z):
        n_batch = sampled_z.shape[0]
        sampled_z_prior = self.prior.rsample((n_batch,))
        return self.compute_kernel(sampled_z, sampled_z).mean() + \
               self.compute_kernel(sampled_z_prior, sampled_z_prior).mean() - \
               2 * self.compute_kernel(sampled_z, sampled_z_prior).mean()

    def compute_kernel(self, x, y):
        tiled_x = x.unsqueeze(1).expand(x.shape[0], y.shape[0], self.prior_dim)
        tiled_y = y.unsqueeze(0).expand(x.shape[0], y.shape[0], self.prior_dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)
        return torch.exp(-kernel_input)


class DiagNormalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 * out_dim),
        )

    def forward(self, x):
        params = self.network(x)
        mean, sd = torch.split(params, params.shape[1] // 2, dim=1)
        return Normal(loc=mean, scale=torch.exp(sd))
