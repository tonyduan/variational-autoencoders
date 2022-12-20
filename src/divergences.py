import torch


def compute_diffs(x, y):
    tiled_x = x.unsqueeze(dim=1).expand(x.shape[0], y.shape[0], x.shape[1])
    tiled_y = y.unsqueeze(dim=0).expand(x.shape[0], y.shape[0], y.shape[1])
    return tiled_x - tiled_y


def mmd_divergence(p_samples, q_samples):
    k_pq = compute_diffs(p_samples, q_samples)
    k_pp = compute_diffs(p_samples, p_samples)
    k_qq = compute_diffs(q_samples, q_samples)
    return torch.exp(-torch.mean(torch.pow(k_pp, 2), dim=2)).mean() + \
           torch.exp(-torch.mean(torch.pow(k_qq, 2), dim=2)).mean() - \
           2 * torch.exp(-torch.mean(torch.pow(k_pq, 2), dim=2)).mean()


def energy_distance(p_samples, q_samples):
    k_pq = compute_diffs(p_samples, q_samples)
    k_pp = compute_diffs(p_samples, p_samples)
    k_qq = compute_diffs(q_samples, q_samples)
    return 2 * torch.norm(k_pq.mean(dim=2), dim=2).mean() - \
           torch.norm(k_pp.mean(dim=2), dim=2).mean() - \
           torch.norm(k_qq.mean(dim=2), dim=2).mean()

