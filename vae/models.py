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
        pred_z = self.sample_z_given_x(x)
        kl_div = kl.kl_divergence(pred_z, self.prior).squeeze(1)
        monte_carlo_z = pred_z.rsample()
        monte_carlo_x = self.sample_x_given_z(monte_carlo_z)
        rec_loss = -torch.sum(monte_carlo_x.log_prob(x), dim=1)
        return kl_div + rec_loss

    def sample_z(self, n_samples):
        return self.prior.rsample((n_samples,))

    def sample_x_given_z(self, z):
        return self.decoder(z)

    def sample_z_given_x(self, x):
        return self.encoder(x)


class DiagNormalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * out_dim),
        )

    def forward(self, x):
        params = self.network(x)
        mean, sd = torch.split(params, params.shape[1] // 2, dim=1)
        return Normal(loc=mean, scale=torch.exp(sd))
