import numpy as np
import scipy as sp
import scipy.stats
import logging

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser

from src.blocks import VAE


def gen_mixture_data(n=512):
    x = np.r_[np.random.randn(n // 2, 2) + np.array([-5, 0]),
              np.random.randn(n // 2, 2) + np.array([5, 0])]
    y = np.r_[np.zeros(n // 2), np.ones(n //2)]
    return (x - np.mean(x)) / np.std(x, ddof=1), y

def density_plot(x, color, label):
    density = sp.stats.kde.gaussian_kde(x)
    axis = np.linspace(-3, 3, 100)
    plt.plot(axis, density(axis), color=color, label=label)
    plt.fill(axis, density(axis), color=color, alpha=0.5)
    plt.axis("off")

def plot_data(x):
    plt.hist2d(x[:,0].numpy(), x[:,1].numpy(), bins=100,
               range=np.array([(-3, 3), (-3, 3)]))
    plt.axis("off")


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n", default=512, type=int)
    argparser.add_argument("--iterations", default=2000, type=int)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model = VAE(input_dim=2, prior_dim=1, hidden_dim=100)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iterations)
    x, labels = gen_mixture_data(args.n)
    x = torch.Tensor(x)

    for i in range(args.iterations):
        optimizer.zero_grad()
        loss = model.loss(x).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            logger.info(f"Iter: {i}\t" + f"Loss: {loss.data:.2f}\t")

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    plot_data(x)
    plt.title("Actual data")

    plt.subplot(1, 3, 3)
    z = model.z_given_x(x).sample().squeeze(1).numpy()
    density_plot(z[labels == 0], color="darkblue", label="Left")
    density_plot(z[labels == 1], color="darkgreen", label="Right")
    plt.title("Latent space")
    plt.legend()

    plt.subplot(1, 3, 2)
    z_sample = model.sample_z(512)
    x_sample = model.x_given_z(z_sample)
    plot_data(x_sample.sample())
    plt.title("Sampled data")

    plt.tight_layout()
    plt.show()
