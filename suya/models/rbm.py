import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from utils import make_optimizer

class RBM(nn.Module):
    def __init__(self, W, v, h):
        super().__init__()
        self.reset(W, v, h)

    def reset(self, W, v, h):
        self.W = nn.Parameter(W)
        self.v = nn.Parameter(v)
        self.h = nn.Parameter(h)
        self.params = {'W': W, 'v': v, 'h': h}
        return

    def visible_to_hidden(self, v):
        mean_h_cond_v = torch.sigmoid(F.linear(v, self.W.t(), self.h))
        h = mean_h_cond_v.bernoulli()
        return h

    def hidden_to_visible(self, h):
        v_cond_h = F.linear(h, self.W, self.v)
        v = torch.randn((h.size()[0], self.v.size(0)), device=h.device) + v_cond_h
        return v

    def free_energy(self, v):
        """Free energy function
            F(x) = -x`b+.5*(x`x+b`b)-\sum_hdim log(1+exp(self.v+W`x))
        """
        v_term = torch.matmul(v, self.v) - 0.5 * torch.sum(v ** 2, dim=-1) - 0.5 * torch.sum(self.v ** 2, dim=-1)
        w_v_h = F.linear(v, self.W.t(), self.h)
        h_term = torch.sum(F.softplus(w_v_h), dim=-1)
        return -v_term - h_term  # (n, )

    def pdf(self, x):
        """Unnormalized probability
            \tilde(p)(x) = exp(-F(x))
        """
        return torch.exp(-self.free_energy(x))  # (n, )

    def score(self, v):
        """Score function for Gaussian-Bernoulli RBM,
            s(x) = b-x+sigmoid(xW+c)W`
        """
        sig = torch.sigmoid(F.linear(v, self.W.t(), self.h))
        _x_px = F.linear(sig, self.W, self.v - v)
        return _x_px

    def hscore(self, v):
        """Hyvarinen score for Gaussian-Bernoulli RBM,
            s_H(x) = 0.5*||grad_log_px||^2+trace(grad_grad_log_px)
        """
        sig = torch.sigmoid(F.linear(v, self.W.t(), self.h))
        _x_px = F.linear(sig, self.W, self.v - v)
        _xx_px = -torch.sum(torch.matmul((1 - sig) * sig, self.W.t() ** 2), dim=-1) + self.v.shape[0]
        hs = 0.5 * torch.sum(_x_px ** 2, dim=-1) - _xx_px
        return hs

    def forward(self, v, num_iters):
        for _ in range(num_iters):
            h = self.visible_to_hidden(v)
            v = self.hidden_to_visible(h)
        return v

    def fit(self, v):
        self.train(True)
        optimizer = make_optimizer([self.W], 'hst')
        for epoch in range(cfg['hst']['num_iters']):
            optimizer.zero_grad()
            v_gibbs = self(v, 1)
            loss = self.free_energy(v).mean() - self.free_energy(v_gibbs).mean()
            loss = loss.mean()
            loss.backward()
            optimizer.step()
        self.reset(self.W.data, self.v.data, self.h.data)
        return


def rbm(params):
    W = params['W']
    v = params['v']
    h = params['h']
    model = RBM(W, v, h)
    return model