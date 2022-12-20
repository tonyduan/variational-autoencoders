import torch
import torch.nn as nn
from torch.distributions import Normal, kl


def init_weights_xavier(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_normal_(module.weight)
        torch.nn.init.constant_(module.bias, 0.)


class DiagNormalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * out_dim),
        )
        self.apply(init_weights_xavier)

    def forward(self, x, eps=1e-6):
        params = self.network(x)
        mu, log_sigma = torch.split(params, params.shape[1] // 2, dim=-1)
        return Normal(loc=mu, scale=torch.exp(log_sigma + eps))


class VAE(nn.Module):

    def __init__(self, input_dim, prior_dim, hidden_dim):
        super().__init__()
        self.prior = Normal(torch.zeros(prior_dim), torch.ones(prior_dim))
        self.z_given_x = DiagNormalNetwork(input_dim, prior_dim, hidden_dim)
        self.x_given_z = DiagNormalNetwork(prior_dim, input_dim, hidden_dim)

    def loss(self, x):
        pred_z = self.z_given_x(x)
        kl_div = kl.kl_divergence(pred_z, self.prior)
        monte_carlo_z = pred_z.rsample()
        monte_carlo_x = self.x_given_z(monte_carlo_z)
        rec_loss = torch.sum(monte_carlo_x.log_prob(x), dim=1, keepdim=True)
        return -(rec_loss - kl_div).squeeze(dim=1)

    def sample_z(self, n_samples):
        return self.prior.rsample((n_samples,))


class InfoVAE(VAE):

    def __init__(self, input_dim, prior_dim, hidden_dim, alpha, lambd, div):
        super().__init__(input_dim, prior_dim, hidden_dim)
        self.prior_dim = prior_dim
        self.alpha = alpha
        self.lambd = lambd
        self.div = div

    def loss(self, x):
        pred_z = self.z_given_x(x)
        kl_div = kl.kl_divergence(pred_z, self.prior).squeeze(1)
        monte_carlo_z = pred_z.rsample()
        monte_carlo_x = self.x_given_z(monte_carlo_z)
        rec_loss = -torch.sum(monte_carlo_x.log_prob(x), dim=1)
        monte_carlo_prior = self.prior.rsample((200,))
        div = self.div(monte_carlo_prior, monte_carlo_z)
        return rec_loss + (1 - self.alpha) * kl_div + (self.alpha + self.lambd - 1) * div

