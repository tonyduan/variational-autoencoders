import numpy as np
import scipy as sp
import scipy.stats
import logging

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser
from sklearn.datasets import load_digits

from src.blocks import VAE


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n", default=512, type=int)
    argparser.add_argument("--iterations", default=2000, type=int)
    args = argparser.parse_args()

    x, _ = load_digits(return_X_y=True)
    input_mean = np.mean(x, axis=0, keepdims=True)
    input_sd = np.std(x, axis=0, keepdims=True, ddof=1).clip(min=0.01)
    x = (x - input_mean) / input_sd

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model = VAE(input_dim=64, latent_dim=2, hidden_dim=100)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iterations)
    x = torch.tensor(x, dtype=torch.float32)

    for i in range(args.iterations):
        optimizer.zero_grad()
        loss = model.loss(x).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            logger.info(f"Iter: {i}\t" + f"Loss: {loss.data:.2f}\t")

    x_vis = x * input_sd + input_mean
    z_sample = model.sample_z(64)
    x_sample = model.x_given_z(z_sample)
    rvs = x_sample.sample().data.numpy()
    rvs_vis = rvs * input_sd + input_mean

    plt.figure(figsize=(12, 6))
    raster = np.zeros((64, 128))
    for i in range(64):
        row, col = i // 8, i % 8
        raster[8 * row: 8 * (row + 1) , 8 * col : 8 * (col + 1)] = x_vis[i].reshape(8, 8)
    for i in range(64):
        row, col = i // 8, i % 8
        raster[8 * row: 8 * (row + 1) , 64 + 8 * col : 64 + 8 * (col + 1)] = rvs_vis[i].reshape(8, 8)
    plt.imshow(raster, vmin=0, vmax=16)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
