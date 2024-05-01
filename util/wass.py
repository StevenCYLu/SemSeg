from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
import time
from torchvision.transforms.functional import normalize

def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections = projections / (torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True)))
    return projections


def sliced_wasserstein_distance(first_samples, second_samples, num_projections=1000, p=2, device="cuda"):
    dim = second_samples.size(1)
    projections = rand_projections(dim, num_projections).to(device)
    first_projections = first_samples.matmul(projections.transpose(0, 1))
    second_projections = second_samples.matmul(projections.transpose(0, 1))
    wasserstein_distance = torch.abs(
        (
            torch.sort(first_projections.transpose(0, 1), dim=1)[0]
            - torch.sort(second_projections.transpose(0, 1), dim=1)[0]
        )
    )

    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1.0 / p)
    
    return torch.pow(torch.pow(wasserstein_distance, p).mean(), 1.0 / p)

def cov(m, y=None):
    '''
    Estimate a covariance matrix given data
    '''
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov

def m_2(X):
    return torch.mean(torch.pow(X, 2), dim=0)


def fast_swd_loss(x, y) -> float:    
    
    # x = x.view(x.shape[0], -1)
    # y = y.view(y.shape[0], -1)

    n, dim = x.shape
    
    meanx = torch.mean(x, dim=0)
    xc = x - meanx
    gamma_xc = torch.mean(torch.linalg.norm(xc, dim=1) ** 2) / dim

    meany = torch.mean(y, dim=0)
    yc = y - meany
    gamma_yc = torch.mean(torch.linalg.norm(yc, dim=1) ** 2) / dim

    mean_term = 1 / dim * torch.linalg.norm(meanx - meany) ** 2
    sw2 = mean_term + (gamma_xc ** (1/2) - gamma_yc ** (1/2)) ** 2
    return sw2

class FastApproxSWDLoss(nn.Module):
    def __init__(self, lmda=1.0, reduction = 'mean', ignore_index=255):
        super(FastApproxSWDLoss, self).__init__()
        
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.lmda = lmda

    def forward(self, inputs, targets):
        
        inputs = inputs.view(inputs.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)

        swd_loss = fast_swd_loss(inputs, targets)
        
        cov_in = cov(inputs)
        cov_tar = cov(targets)
        diag_in = torch.diag(cov_in)
        diag_tar = torch.diag(cov_tar)
        swd_loss += self.lmda * (
            torch.linalg.norm(cov_in - diag_in) ** 2 
            + torch.linalg.norm(cov_tar - diag_tar) ** 2
        )
        
        return swd_loss
