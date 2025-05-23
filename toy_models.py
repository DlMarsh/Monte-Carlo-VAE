import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import math
import numpy as np

class VAE_Toy(pl.LightningModule):
    def __init__(self, num_samples, hidden_dim, learning_rate=1e-3):
        super(VAE_Toy, self).__init__()
        self.num_samples = num_samples
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(),
        )
        self.fc_mu = nn.Linear(128, self.hidden_dim)
        self.fc_logvar = nn.Linear(128, self.hidden_dim)
        self.decoder_part = nn.Sequential(nn.Linear(self.hidden_dim, 128), nn.LeakyReLU(), nn.Linear(128, 2))

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _, _ = self.step(batch)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, z):
        return self.decoder_part(z)
    
    def loss_function(self, recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x, reduction='none').view(
            (self.num_samples, -1, 2)).mean(0).sum(-1).mean()
        KLD = -0.5 * torch.mean((1 + logvar - mu.pow(2) - logvar.exp()).view(
            (self.num_samples, -1, 2)).mean(0).sum(-1))
        loss = MSE + KLD
        return loss, MSE

    def step(self, batch):
        x = batch[0]
        enc_exit = self.encoder(x.view(-1, 2))
        mu, logvar = self.fc_mu(enc_exit), self.fc_logvar(enc_exit)
        mu = mu.repeat(self.num_samples, 1)
        logvar = logvar.repeat(self.num_samples, 1)
        z = self.reparameterize(mu, logvar)
        x_hat = self(z)
        loss, MSE = self.loss_function(x_hat, x.repeat(self.num_samples, 1), mu, logvar)
        return loss, x_hat, z, MSE
    
class IWAE_Toy(VAE_Toy):
    def step(self, batch):
        x = batch[0]
        enc_exit = self.encoder(x.view(-1, 2))
        mu, logvar = self.fc_mu(enc_exit), self.fc_logvar(enc_exit)
        mu = mu.repeat(self.num_samples, 1)
        logvar = logvar.repeat(self.num_samples, 1)
        z = self.reparameterize(mu, logvar)
        x_hat = self(z)
        loss, MSE = self.loss_function(x_hat, x, mu, logvar, z)
        return loss, x_hat, z, MSE

    def loss_function(self, recon_x, x, mu, logvar, z):
        batch_size = x.shape[0]
        x_repeated = x.repeat(self.num_samples, 1)

        log_pxz = -0.5 * ((recon_x - x_repeated).pow(2).sum(-1) + 2 * np.log(2 * np.pi))
        log_pxz = log_pxz.view(self.num_samples, batch_size)

        log_pz = -0.5 * (z.pow(2).sum(-1) + self.hidden_dim * np.log(2 * np.pi))
        log_pz = log_pz.view(self.num_samples, batch_size)
        std = torch.exp(0.5 * logvar)
        eps = (z - mu) / std
        log_qzx = -0.5 * (eps.pow(2).sum(-1) + logvar.sum(-1) + self.hidden_dim * np.log(2 * np.pi))
        log_qzx = log_qzx.view(self.num_samples, batch_size)

        log_weights = log_pxz + log_pz - log_qzx
        elbo = torch.logsumexp(log_weights, dim=0) - np.log(self.num_samples)
        loss = -elbo.mean()

        MSE_avg = F.mse_loss(recon_x, x_repeated).mean()
        return loss, MSE_avg
    
class SISVAE_Toy(VAE_Toy):
    def __init__(self, num_samples, hidden_dim, K=4, eta=0.001, learning_rate=1e-3):
        super(SISVAE_Toy, self).__init__(num_samples, hidden_dim, learning_rate)
        self.K = K
        self.eta = eta
        self.beta_schedule = torch.linspace(0, 1, K)

    def log_gamma_k(self, z, x, beta_k):
        batch_size = x.shape[0]
        x_repeated = x.repeat(self.num_samples, 1)

        enc_exit = self.encoder(x.view(-1, 2))
        mu, logvar = self.fc_mu(enc_exit), self.fc_logvar(enc_exit)
        mu = mu.repeat(self.num_samples, 1)
        logvar = logvar.repeat(self.num_samples, 1)
        std = torch.exp(0.5 * logvar)

        log_qzx = -0.5 * ( ((z - mu) / std).pow(2).sum(-1) + logvar.sum(-1) + self.hidden_dim * np.log(2 * np.pi) )
        log_qzx = log_qzx.view(self.num_samples, batch_size)
        x_hat = self(z)
        log_pxz = -0.5 * ((x_hat - x_repeated).pow(2).sum(-1) + 2 * np.log(2 * np.pi))
        log_pxz = log_pxz.view(self.num_samples, batch_size)
        log_pz = -0.5 * (z.pow(2).sum(-1) + self.hidden_dim * np.log(2 * np.pi))
        log_pz = log_pz.view(self.num_samples, batch_size)
        log_p_xz = log_pxz + log_pz

        return beta_k * log_p_xz + (1 - beta_k) * log_qzx
        
    def langevin_step(self, z, x, beta_k):
        z = z.requires_grad_(True)
        log_gamma = self.log_gamma_k(z, x, beta_k).sum()
        grad_log_gamma = torch.autograd.grad(log_gamma, z, create_graph=True)[0]
        
        u_k = torch.randn_like(z)
        z_k = z + self.eta * grad_log_gamma + math.sqrt(2 * self.eta) * u_k
        z_k.requires_grad_(True)
        return z_k, grad_log_gamma
    
    def log_m_k(self, z_k, z_km1, grad_log_gamma_k, grad_log_gamma_km1):
        m_k_forward = dist.Normal(z_km1 + self.eta * grad_log_gamma_km1, math.sqrt(2 * self.eta))
        log_m_forward = m_k_forward.log_prob(z_k).sum(dim=-1)
        m_k_backward = dist.Normal(z_k + self.eta * grad_log_gamma_k, math.sqrt(2 * self.eta))
        log_m_backward = m_k_backward.log_prob(z_km1).sum(dim=-1)
        return log_m_forward, log_m_backward
    
    def loss_function(self, x, z_K, log_weights, x_hat):
        batch_size = x.shape[0]
        elbo = torch.logsumexp(log_weights, dim=0) - np.log(self.num_samples)
        loss = -elbo.mean()
        
        x_repeated = x.repeat(self.num_samples, 1)
        MSE_avg = F.mse_loss(x_hat, x_repeated).mean()
        return loss, MSE_avg
    
    def step(self, batch):
        x = batch[0]
        batch_size = x.shape[0]

        enc_exit = self.encoder(x.view(-1, 2))
        mu, logvar = self.fc_mu(enc_exit), self.fc_logvar(enc_exit)
        mu = mu.repeat(self.num_samples, 1)
        logvar = logvar.repeat(self.num_samples, 1)
        std = torch.exp(0.5 * logvar)
        z = self.reparameterize(mu, logvar)

        log_qzx = -0.5 * ( ((z - mu) / std).pow(2).sum(-1) + logvar.sum(-1) + self.hidden_dim * np.log(2 * np.pi) )
        log_qzx = log_qzx.view(self.num_samples, batch_size)
        log_weights = -log_qzx

        z_km1 = z
        for k in range(0, self.K):
            beta_k = self.beta_schedule[k]
            z_k, grad_log_gamma_km1 = self.langevin_step(z_km1, x, beta_k)
            
            log_gamma_k = self.log_gamma_k(z_k, x, beta_k).sum()
            grad_log_gamma_k = torch.autograd.grad(log_gamma_k, z_k, create_graph=True)[0]
            
            log_m_forward, log_m_backward = self.log_m_k(z_k, z_km1, grad_log_gamma_k, grad_log_gamma_km1)
            log_w_k = log_m_backward - log_m_forward
            log_weights += log_w_k.view(self.num_samples, batch_size)
            
            z_km1 = z_k
        
        log_p_xz = self.log_gamma_k(z_k, x, beta_k=1.0)
        log_weights += log_p_xz.view(self.num_samples, batch_size)

        x_hat = self(z_k)
        loss, MSE = self.loss_function(x, z_k, log_weights, x_hat)
        return loss, x_hat, z_k, MSE
    
    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            loss, _, _, MSE = self.step(batch)
        return loss
    
class VAE_MALA_Toy(pl.LightningModule):
    def __init__(self, num_samples, hidden_dim, K=5, h=0.01, beta=0.01, alpha_star=0.574,
                 learning_rate=1e-3, pretrain_epochs=25):
        super(VAE_MALA_Toy, self).__init__()
        self.num_samples = num_samples
        self.hidden_dim = hidden_dim
        self.K = K
        self.h = nn.Parameter(torch.tensor(h))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.alpha_star = alpha_star
        self.learning_rate = learning_rate
        self.pretrain_epochs = pretrain_epochs
        self.encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(),
        )
        self.fc_mu = nn.Linear(128, self.hidden_dim)
        self.fc_logvar = nn.Linear(128, self.hidden_dim)
        self.decoder_part = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )
        self.C_diag = nn.Parameter(torch.eye(self.hidden_dim).tril() + 0.01 * torch.randn(self.hidden_dim, self.hidden_dim).tril())
        self.pretrain = True
        self.current_alpha = 0.0

    def training_step(self, batch, batch_idx):
        if self.current_epoch < self.pretrain_epochs:
            loss, x_hat, z = self.step(batch, phase="pretrain")
        else:
            if self.pretrain:
                self.pretrain = False
                for param in self.encoder.parameters():
                    param.requires_grad = False
                for param in self.fc_mu.parameters():
                    param.requires_grad = False
                for param in self.fc_logvar.parameters():
                    param.requires_grad = False
            loss, x_hat, z = self.step(batch, phase="adapt")
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            loss, x_hat, z = self.step(batch, phase="pretrain" if self.pretrain else "adapt")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, z):
        return self.decoder_part(z)

    def U_theta(self, z, x, sigma2=0.1):
        x_hat = self(z)
        log_pxz = -0.5 * ((x_hat - x).pow(2).sum(-1) / sigma2 + 2 * np.log(2 * np.pi * sigma2))
        log_pz = -0.5 * (z.pow(2).sum(-1) + self.hidden_dim * np.log(2 * np.pi))
        return -(log_pxz + log_pz)

    def mala_step(self, z, x, sigma2=0.1):
        z = z.requires_grad_(True)
        with torch.enable_grad():
            U = self.U_theta(z, x, sigma2=sigma2)
            grad_U = torch.autograd.grad(U.sum(), z, retain_graph=True)[0]
            grad_U = torch.clamp(grad_U, -100, 100)

        C = torch.tril(self.C_diag)
        C_diag = torch.diag(C).abs().clamp(min=1e-3)
        cov = self.h**2 * C @ C.T
        scale = torch.sqrt(torch.diag(cov)).clamp(min=1e-6)

        with torch.no_grad():
            v = torch.randn_like(z)
            drift = (self.h**2 / 2) * torch.matmul(grad_U, C @ C.T)
            z_prime = z - drift + v * scale

        z_prime = z_prime.requires_grad_(True)
        with torch.enable_grad():
            U_prime = self.U_theta(z_prime, x, sigma2=sigma2)
            grad_U_prime = torch.autograd.grad(U_prime.sum(), z_prime, retain_graph=True)[0]
            grad_U_prime = torch.clamp(grad_U_prime, -100, 100)

        q_forward = dist.Normal(z - drift, scale)
        log_q_forward = q_forward.log_prob(z_prime).sum(-1)

        q_backward = dist.Normal(z_prime - (self.h**2 / 2) * torch.matmul(grad_U_prime, C @ C.T), scale)
        log_q_backward = q_backward.log_prob(z).sum(-1)

        log_alpha = -U_prime + U + log_q_backward - log_q_forward
        alpha = torch.exp(log_alpha.clamp(max=0))

        with torch.no_grad():
            entropy = torch.log(torch.abs(self.h * C_diag)).sum()
            F = torch.mean(log_alpha) + self.beta * entropy

        with torch.no_grad():
            u = torch.rand_like(alpha)
            accept = u < alpha
            z_new = torch.where(accept.unsqueeze(-1), z_prime, z).detach()
        return z_new, F, alpha.mean()

    def step(self, batch, phase="train"):
        x = batch[0]
        batch_size = x.shape[0]

        enc_out = self.encoder(x.view(-1, 2))
        mu, logvar = self.fc_mu(enc_out), self.fc_logvar(enc_out)
        mu = mu.repeat(self.num_samples, 1)
        logvar = logvar.repeat(self.num_samples, 1)
        z = self.reparameterize(mu, logvar)

        total_F = 0.0
        total_alpha = 0.0
        x_repeated = x.repeat(self.num_samples, 1)
        if phase != "pretrain":
            with torch.no_grad():
                for k in range(self.K):
                    z, _, alpha_k = self.mala_step(z, x_repeated, sigma2=0.1)
                    total_alpha += alpha_k
            self.current_alpha = total_alpha / self.K
            z = z.detach().requires_grad_(True)
            _, total_F, _ = self.mala_step(z, x_repeated, sigma2=0.1)

        x_hat = self(z)

        if phase == "pretrain":
            loss = self.loss_function_elbo(x_hat, x_repeated, mu, logvar)
        else:
            loss = self.loss_function_adapt(x_hat, x_repeated, z, total_F, total_alpha)

        return loss, x_hat, z

    def loss_function_elbo(self, recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x, reduction='sum') / (x.size(0) / self.num_samples)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (x.size(0) / self.num_samples)
        return MSE + KLD

    def loss_function_adapt(self, recon_x, x, z_K, total_F, total_alpha, sigma2=0.1):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / (x.size(0) / self.num_samples)

        log_pxz = -0.5 * ((recon_x - x).pow(2).sum(-1) / sigma2 + 2 * np.log(2 * np.pi * sigma2))
        log_pz = -0.5 * (z_K.pow(2).sum(-1) + self.hidden_dim * np.log(2 * np.pi))
        G = torch.mean(log_pxz + log_pz)

        F_loss = -total_F / self.K
        beta_loss = self.beta * (total_alpha / self.K - self.alpha_star)**2
        C_reg = 0.1 * (self.C_diag.pow(2).sum() + 1e-4 / self.C_diag.pow(2).sum())

        return recon_loss - 2.0 * G + F_loss + 0.1 * beta_loss + 0.1 * C_reg

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self.pretrain and hasattr(self, 'current_alpha'):
            if self.current_alpha < self.alpha_star - 0.1:
                self.h.data *= 0.9
            elif self.current_alpha > self.alpha_star + 0.1:
                self.h.data *= 1.1
            if self.current_epoch == self.pretrain_epochs + 2:
                for param in self.encoder.parameters():
                    param.requires_grad = True
                for param in self.fc_mu.parameters():
                    param.requires_grad = True
                for param in self.fc_logvar.parameters():
                    param.requires_grad = True

    def configure_optimizers(self):
        return torch.optim.Adam([
            {'params': self.decoder_part.parameters(), 'lr': 1e-3},
            {'params': list(self.encoder.parameters()) + list(self.fc_mu.parameters()) + list(self.fc_logvar.parameters()), 'lr': 1e-3},
            {'params': [self.h, self.C_diag, self.beta], 'lr': 1e-4}
        ])
    
class VAE_HMC_Toy(pl.LightningModule):
    def __init__(self, num_samples, hidden_dim, K=20, L=10, h=0.01, beta=0.01, alpha_star=0.574,
                 learning_rate=1e-3, pretrain_epochs=25):
        super(VAE_HMC_Toy, self).__init__()
        self.num_samples = num_samples
        self.hidden_dim = hidden_dim
        self.K = K
        self.L = L
        self.h = nn.Parameter(torch.tensor(h))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.alpha_star = alpha_star
        self.learning_rate = learning_rate
        self.pretrain_epochs = pretrain_epochs
        self.encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(),
        )
        self.fc_mu = nn.Linear(128, self.hidden_dim)
        self.fc_logvar = nn.Linear(128, self.hidden_dim)
        self.decoder_part = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )
        self.C_diag = nn.Parameter(torch.eye(self.hidden_dim).tril() + 0.01 * torch.randn(self.hidden_dim, self.hidden_dim).tril())
        self.pretrain = True
        self.current_alpha = 0.0

    def training_step(self, batch, batch_idx):
        if self.current_epoch < self.pretrain_epochs:
            loss, x_hat, z = self.step(batch, phase="pretrain")
        else:
            if self.pretrain:
                self.pretrain = False
                for param in self.encoder.parameters():
                    param.requires_grad = False
                for param in self.fc_mu.parameters():
                    param.requires_grad = False
                for param in self.fc_logvar.parameters():
                    param.requires_grad = False

            loss, x_hat, z = self.step(batch, phase="adapt")
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            loss, x_hat, z = self.step(batch, phase="pretrain" if self.pretrain else "adapt")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam([
            {'params': self.decoder_part.parameters(), 'lr': 1e-3},
            {'params': list(self.encoder.parameters()) + list(self.fc_mu.parameters()) + list(self.fc_logvar.parameters()), 'lr': 1e-3},
            {'params': [self.h, self.C_diag, self.beta], 'lr': 1e-4}
        ])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, z):
        return self.decoder_part(z)

    def U_theta(self, z, x, sigma2=0.1):
        x_hat = self(z)
        log_pxz = -0.5 * ((x_hat - x).pow(2).sum(-1) / sigma2 + 2 * np.log(2 * np.pi * sigma2))
        log_pz = -0.5 * (z.pow(2).sum(-1) + self.hidden_dim * np.log(2 * np.pi))
        return -(log_pxz + log_pz)

    def hmc_step(self, z, x, sigma2=0.1):
        z = z.requires_grad_(True)
        C = torch.tril(self.C_diag)
        C_diag = torch.diag(C).abs().clamp(min=1e-3)
        M_inv = C @ C.T

        v = torch.randn_like(z)
        p = torch.matmul(v, torch.inverse(C).T)

        z_current = z.clone()
        p_current = p.clone()
        U_current = self.U_theta(z_current, x, sigma2=sigma2)

        z_current = z_current.requires_grad_(True)
        for ell in range(self.L):
            with torch.enable_grad():
                U = self.U_theta(z_current, x, sigma2=sigma2)
                grad_U = torch.autograd.grad(U.sum(), z_current, retain_graph=True)[0]
                grad_U = torch.clamp(grad_U, -100, 100)

            p_current = p_current - (self.h / 2) * grad_U
            z_current = z_current + self.h * torch.matmul(p_current, M_inv)
            z_current = z_current.requires_grad_(True)

            with torch.enable_grad():
                U = self.U_theta(z_current, x, sigma2=sigma2)
                grad_U = torch.autograd.grad(U.sum(), z_current, retain_graph=True)[0]
                grad_U = torch.clamp(grad_U, -100, 100)

            p_current = p_current - (self.h / 2) * grad_U

        z_prime = z_current
        p_prime = p_current
        U_prime = self.U_theta(z_prime, x, sigma2=sigma2)

        H_current = U_current + 0.5 * torch.sum(p * torch.matmul(p, M_inv), dim=-1)
        H_prime = U_prime + 0.5 * torch.sum(p_prime * torch.matmul(p_prime, M_inv), dim=-1)

        log_alpha = -(H_prime - H_current)
        alpha = torch.exp(log_alpha.clamp(max=0))

        with torch.enable_grad():
            U_mid = self.U_theta(z, x, sigma2=sigma2)
            grad_U_mid = torch.autograd.grad(U_mid.sum(), z, retain_graph=True)[0]
            grad_U_mid = torch.clamp(grad_U_mid, -100, 100)

        identity = torch.eye(self.hidden_dim, device=self.device)
        det_term = identity - ((self.L**2 - 1) / 6) * torch.matmul(grad_U_mid.unsqueeze(-1), grad_U_mid.unsqueeze(-2))
        log_det = torch.slogdet(det_term).logabsdet
        entropy = torch.log(torch.abs(self.h * C_diag)).sum() - self.hidden_dim * np.log(self.L) - log_det
        entropy = entropy.mean()

        with torch.no_grad():
            F = torch.mean(log_alpha) + self.beta * entropy

        with torch.no_grad():
            u = torch.rand_like(alpha)
            accept = u < alpha
            z_new = torch.where(accept.unsqueeze(-1), z_prime, z).detach()
        return z_new, F, alpha.mean()

    def step(self, batch, phase="train"):
        x = batch[0]
        batch_size = x.shape[0]

        enc_out = self.encoder(x.view(-1, 2))
        mu, logvar = self.fc_mu(enc_out), self.fc_logvar(enc_out)
        mu = mu.repeat(self.num_samples, 1)
        logvar = logvar.repeat(self.num_samples, 1)
        z = self.reparameterize(mu, logvar)

        total_F = 0.0
        total_alpha = 0.0
        x_repeated = x.repeat(self.num_samples, 1)
        if phase != "pretrain":
            with torch.no_grad():
                for k in range(self.K):
                    z, _, alpha_k = self.hmc_step(z, x_repeated, sigma2=0.1)
                    total_alpha += alpha_k
            self.current_alpha = total_alpha / self.K
            z = z.detach().requires_grad_(True)
            _, total_F, _ = self.hmc_step(z, x_repeated, sigma2=0.1)

        x_hat = self(z)
        if phase == "pretrain":
            loss = self.loss_function_elbo(x_hat, x_repeated, mu, logvar)
        else:
            loss = self.loss_function_adapt(x_hat, x_repeated, z, total_F, total_alpha)

        return loss, x_hat, z

    def loss_function_elbo(self, recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x, reduction='sum') / (x.size(0) / self.num_samples)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (x.size(0) / self.num_samples)
        return MSE + KLD

    def loss_function_adapt(self, recon_x, x, z_K, total_F, total_alpha, sigma2=0.1):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / (x.size(0) / self.num_samples)

        log_pxz = -0.5 * ((recon_x - x).pow(2).sum(-1) / sigma2 + 2 * np.log(2 * np.pi * sigma2))
        log_pz = -0.5 * (z_K.pow(2).sum(-1) + self.hidden_dim * np.log(2 * np.pi))
        G = torch.mean(log_pxz + log_pz)

        F_loss = -total_F / self.K
        beta_loss = self.beta * (total_alpha / self.K - self.alpha_star)**2
        C_reg = 0.1 * (self.C_diag.pow(2).sum() + 1e-4 / self.C_diag.pow(2).sum())

        return recon_loss - 2.0 * G + F_loss + 0.1 * beta_loss + 0.1 * C_reg

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self.pretrain and hasattr(self, 'current_alpha'):
            if self.current_alpha < self.alpha_star - 0.1:
                self.h.data *= 0.9
            elif self.current_alpha > self.alpha_star + 0.1:
                self.h.data *= 1.1
            if self.current_epoch == self.pretrain_epochs + 2:
                for param in self.encoder.parameters():
                    param.requires_grad = True
                for param in self.fc_mu.parameters():
                    param.requires_grad = True
                for param in self.fc_logvar.parameters():
                    param.requires_grad = True