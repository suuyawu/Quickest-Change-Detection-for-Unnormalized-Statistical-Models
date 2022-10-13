from abc import ABC, abstractmethod
from typing import Optional, List
import torch


def zero_diag(mat: torch.tensor) -> torch.tensor:
    return mat - torch.diag(mat.diag())


class Kernel(ABC):
    params: dict
    max: float

    @abstractmethod
    def __call__(self, x: torch.tensor) -> torch.tensor:
        raise NotImplementedError()


class GaussianRBF(Kernel):
    """
    Note convention with variance
    """

    def __init__(self, var: float = 1.) -> None:
        self.params = {'var': var}
        self.max = 1.

    def __call__(self,
                 x: torch.tensor, y: Optional[torch.tensor] = None, configure: bool = False
                 ) -> torch.tensor:
        y = x if y is None else y
        l2_mat_squared = (x[:, None, :] - y[None, :, :]).square().sum(-1)

        if configure:
            n = x.shape[0]
            sorted_squared_distances = l2_mat_squared.flatten().sort().values
            median_index = torch.tensor(n * (n - 1) / 2 + n).int()  # +n to account for diagonal
            median_squared_dist = sorted_squared_distances[median_index]
            self.params['var'] = median_squared_dist.item()

        var = torch.tensor(self.params['var'])
        kernel_mat = (-1 * (l2_mat_squared / (2 * var))).exp()

        return kernel_mat


def load_kernel(name) -> Kernel:
    if name == 'gaussian_rbf':
        kernel = GaussianRBF()
    else:
        raise NotImplementedError(f"No implementation found for: {name}")
    return kernel


def quantile(sample: torch.tensor, p: float, types: List[int] = [7], sorted: bool = False) -> torch.tensor:
    """
    See https://wikipedia.org/wiki/Quantile#Estimating_quantiles_from_a_sample
    Averages estimates corresponding to each type in list
    """
    N = len(sample)
    if not 1 / N <= p <= (N - 1) / N:
        raise ValueError(f"The {p}-quantile should not be estimated using only {N} samples.")
    if not sorted:
        sorted_sample = sample.sort().values

    quantiles = []
    for type in types:
        if type == 6:  # With M = k*ert - 1 this one is exact
            h = (N + 1) * p
        elif type == 7:
            h = (N - 1) * p + 1
        elif type == 8:
            h = (N + 1 / 3) * p + 1 / 3
        h_floor = int(h)
        quantile = sorted_sample[h_floor - 1]
        if h_floor != h:
            quantile += (h - h_floor) * (sorted_sample[h_floor] - sorted_sample[h_floor - 1])
        quantiles.append(quantile)
    return torch.stack(quantiles).mean()
