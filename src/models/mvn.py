import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal


class MVN(nn.Module):
    def __init__(self, mean, logvar):
        super().__init__()
        self.reset(mean, logvar)

    def reset(self, mean, logvar):
        self.mean = nn.Parameter(mean)
        self.logvar = nn.Parameter(logvar)
        self.params = {'mean': mean, 'logvar': logvar}
        self.model = MultivariateNormal(mean, logvar.exp())
        return

    def pdf(self, x):
        pdf_ = self.model.log_prob(x).exp()
        return pdf_

    def cdf(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).to(self.mean.device)
        cdf_ = self.model.cdf(x)
        return cdf_

    def cdf_numpy(self, x):
        return self.cdf(x).cpu().numpy()

    def score(self, x):
        score_ = -1 * torch.matmul((x - self.mean), torch.linalg.inv(self.logvar.exp()))
        return score_

    def hscore(self, x):
        mean = self.mean
        invcov = torch.linalg.inv(self.logvar.exp())
        t1 = 0.5 * (x - mean).matmul(invcov).matmul(invcov).matmul((x - mean).transpose(-1, -2))
        t2 = - invcov.diagonal().sum()
        t1 = t1.diagonal(dim1=-2, dim2=-1)
        hscore_ = t1 + t2
        return hscore_


def mvn(params):
    mean = params['mean']
    logvar = params['logvar']
    model = MVN(mean, logvar)
    return model
