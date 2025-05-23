import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.stats import gaussian_kde

def target_density(z):
    X_item = torch.tensor([[2 * np.pi * (2. + 2.)]], dtype=torch.float32)
    sigma = 1.
    return (torch.distributions.Normal(loc=2 * np.pi * (torch.sqrt(torch.sum(torch.pow(z, 2), dim=1, keepdim=True)) + 2.), scale=sigma)
            .log_prob(X_item.repeat(z.shape[0], 1)).sum(-1) + torch.distributions.Normal(
                loc=torch.tensor(0., dtype=torch.float32),
                scale=torch.tensor(1., dtype=torch.float32))
            .log_prob(z).sum(-1))

def generate_data(n_samples):
    grid_points = int(np.sqrt(n_samples))
    z1 = torch.linspace(-4, 4, grid_points)
    z2 = torch.linspace(-4, 4, grid_points)
    z_grid = torch.stack(torch.meshgrid(z1, z2), -1).reshape(-1, 2)
    probs = torch.exp(target_density(z_grid))
    probs = probs / probs.sum()
    indices = torch.multinomial(probs, n_samples, replacement=True)
    return z_grid[indices]

def target_density(z):
    X_item = torch.tensor([[2 * np.pi * (2. + 2.)]], dtype=torch.float32)
    sigma = 1.
    return (torch.distributions.Normal(loc=2 * np.pi * (torch.sqrt(torch.sum(torch.pow(z, 2), dim=1, keepdim=True)) + 2.), scale=sigma)
            .log_prob(X_item.repeat(z.shape[0], 1)).sum(-1) + torch.distributions.Normal(
                loc=torch.tensor(0., dtype=torch.float32),
                scale=torch.tensor(1., dtype=torch.float32))
            .log_prob(z).sum(-1))

def prepare_data(batch_size=128):
    n_samples = 20000
    train_data = generate_data(n_samples)[:10000]
    val_data = generate_data(n_samples)[:2000]
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def plot_kde(samples, title, x_limits=(-4, 4), y_limits=(-4, 4)):
    plt.figure(figsize=(5,5))
    x = samples[:, 0]
    y = samples[:, 1]
    nbins = 300
    k = gaussian_kde([x, y], bw_method=0.1)
    
    xi, yi = np.mgrid[x_limits[0]:x_limits[1]:nbins*1j, 
                     y_limits[0]:y_limits[1]:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    cmap = plt.cm.viridis
    cmap.set_bad(alpha=0)

    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), 
                  cmap=cmap)
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.xlim((x_limits[0], x_limits[1]))
    plt.ylim((y_limits[0], y_limits[1]))
    plt.show()

