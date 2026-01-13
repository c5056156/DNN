# utils.py
import torch
import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def adversarial_loss(preds, targets):
    """
    BCE with logits for PatchGAN
    """
    return torch.nn.functional.binary_cross_entropy_with_logits(preds, targets)


def recon_loss(preds, targets):
    """
    L1 reconstruction loss
    """
    return torch.nn.functional.l1_loss(preds, targets)
