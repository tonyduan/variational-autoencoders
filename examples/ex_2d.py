import numpy as np
import scipy as sp
import scipy.stats
import logging
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.distributions import MultivariateNormal

from vae.models import VAE, InfoVAE
from vae.divergences import mmd_divergence, energy_distance


def gen_mixture_data(n=512):
    return (np.r_[np.random.randn(n // 2, 2) + np.array([-5, 3]),
                  np.random.randn(n // 2, 2) + np.array([5, 3])],
            np.r_[np.zeros(n // 2), np.ones(n //2)])

def density_plot(x, color, label):
    density = sp.stats.kde.gaussian_kde(x)
    axis = np.linspace(-4, 4, 100)
    plt.plot(axis, density(axis), color=color, label=label)
    plt.fill(axis, density(axis), color=color, alpha=0.5)
    plt.axis("off")

def plot_data(x):
    plt.hist2d(x[:,0].numpy(), x[:,1].numpy(), bins=100,
               range=np.array([(-10, 10), (-5, 12)]))
    plt.axis("off")


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n", default=512, type=int)
    argparser.add_argument("--iterations", default=2000, type=int)
    argparser.add_argument("--vanilla-vae", action="store_true")
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    if args.vanilla_vae:
        model = VAE(input_dim=2, prior_dim=1, hidden_dim=100)
    else:
        model = InfoVAE(input_dim=2, prior_dim=1, hidden_dim=100,
                        alpha=1, lambd=10, div=mmd_divergence)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    x, labels = gen_mixture_data(args.n)
    x = torch.Tensor(x)

    for i in range(args.iterations):
        optimizer.zero_grad()
        loss = model.loss(x).mean()
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            logger.info(f"Iter: {i}\t" + f"Loss: {loss.data:.2f}\t")

    plot_data(x)
    plt.show()
    z = model.compute_z_given_x(x).sample().squeeze(1).numpy()
    density_plot(z[labels == 0], color="darkblue", label="Left")
    density_plot(z[labels == 1], color="darkgreen", label="Right")
    plt.legend()
    plt.show()

    z = model.sample_z(512)
    x = model.compute_x_given_z(z)
    plot_data(x.sample())
    plt.show()
