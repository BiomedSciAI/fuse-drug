# type: ignore
"""
(C) Copyright 2021 IBM Corp.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
----

Based on Payel Das et al. Accelerated antimicrobial discovery via deep generative models and molecular dynamics simulations.
Some parts of the code are based in repo https://github.com/IBM/controlled-peptide-generation
"""
from collections import namedtuple

import torch
import torch.nn.functional as F
import math


def recon_dec(sequences, logits, ignore_index):
    """compute reconstruction error (NLL of next-timestep predictions)"""
    # dec_inputs: '<start> I want to fly <eos>'
    # dec_targets: 'I want to fly <eos> <pad>'
    # sequences: [mbsize x seq_len]
    # logits: [mbsize x seq_len x vocabsize]
    mbsize = sequences.size(0)
    pad_words = torch.LongTensor(mbsize, 1).fill_(ignore_index).to(sequences.device)
    dec_targets = torch.cat([sequences[:, 1:], pad_words], dim=1)
    recon_loss = F.cross_entropy(  # this is log_softmax + nll
        logits.view(-1, logits.size(2)),
        dec_targets.view(-1),
        reduction="mean",
        ignore_index=ignore_index,  # padding doesnt contribute to recon loss & gradient
    )
    return recon_loss


def wae_mmd_gaussianprior_rf(z, sigma=7.0, rf_dim=500):
    """compute MMD with samples from unit gaussian.
    MMD parametrization from cfg loaded here."""
    z_prior = torch.randn_like(z)  # shape and device
    return mmd_rf(z, z_prior, sigma=sigma, rf_dim=rf_dim)


def mmd_rf(z1, z2, **mmd_kwargs):
    mu1 = compute_mmd_mean_rf(z1, **mmd_kwargs)
    mu2 = compute_mmd_mean_rf(z2, **mmd_kwargs)
    loss = ((mu1 - mu2) ** 2).sum()
    return loss


def compute_mmd_mean_rf(z, sigma, rf_dim):
    # random features approx of gaussian kernel mmd.
    # Then just loss = |mu_real - mu_fake|_H
    rf_w = torch.randn((z.shape[1], rf_dim), device=z.device)
    rf_b = math.pi * 2 * torch.rand((rf_dim,), device=z.device)

    z_rf = compute_gaussian_rf(z, rf_w, rf_b, sigma, rf_dim)
    mu_rf = z_rf.mean(0, keepdim=False)
    return mu_rf


def compute_gaussian_rf(z, rf_w, rf_b, sigma, rf_dim):
    z_emb = (z @ rf_w) / sigma + rf_b
    z_emb = torch.cos(z_emb) * (2.0 / rf_dim) ** 0.5
    return z_emb


def kl_gaussian_sharedmu(logvar):
    """analytically compute kl divergence N(mu,sigma) with N(mu, I)."""
    return torch.mean(0.5 * torch.sum((logvar.exp() - 1 - logvar), 1))


class LossRecon(torch.nn.Module):
    def __init__(self, input: str, logits: str, pad_index: int) -> None:
        super().__init__()

        self.input = input
        self.logits = logits
        self.pad_index = pad_index

    def forward(self, batch_dict):
        logits = batch_dict[self.logits]
        inputs = batch_dict[self.input].to(device=logits.device)

        # compute loss
        recon_loss = recon_dec(inputs, logits, self.pad_index)

        loss = recon_loss
        return loss


class LossWAE(torch.nn.Module):
    def __init__(self, z: str, n_iter: int) -> None:
        super().__init__()
        self.z = z
        Beta = namedtuple("Beta", ["start_val", "end_val", "start_iter", "end_iter"])
        self.beta_cfg = Beta(1.0, 2.0, 0, n_iter)

    def forward(self, batch_dict):

        z = batch_dict[self.z]

        it = batch_dict["global_step"]

        # compute loss
        beta = anneal(self.beta_cfg, it)

        wae_mmdrf_loss = wae_mmd_gaussianprior_rf(z)

        z_regu_loss = wae_mmdrf_loss

        loss = beta * z_regu_loss
        return loss


# Linearly interpolate between start and end val depending on current iteration
def interpolate(start_val, end_val, start_iter, end_iter, current_iter):
    if current_iter < start_iter:
        return start_val
    elif current_iter >= end_iter:
        return end_val
    else:
        return start_val + (end_val - start_val) * (current_iter - start_iter) / (end_iter - start_iter)


def anneal(cfgan, it):
    return interpolate(cfgan.start_val, cfgan.end_val, cfgan.start_iter, cfgan.end_iter, it)
