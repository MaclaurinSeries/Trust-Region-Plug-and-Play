from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from .utils import rademacher


class Scheduler(ABC):
    def __init__(self, sigma):
        super().__init__()

        self.sigma = sigma

    @abstractmethod
    def __call__(self, current_iter: int = 0): ...


class ConstantScheduler(Scheduler):
    def __call__(self, current_iter: int = 0):
        return self.sigma


class LinearScheduler(Scheduler):
    def __init__(self, start_sigma: float, end_sigma: float, max_iter: int):
        super().__init__(start_sigma)
        self.end_sigma = end_sigma
        self.max_iter = max_iter

    def __call__(self, current_iter: int):
        return self.sigma + (self.end_sigma - self.sigma) * current_iter / self.max_iter


class ExponentialScheduler(Scheduler):
    def __init__(self, start_sigma: float, factor: float, min_sigma: float = 1e-5):
        super().__init__(start_sigma)

        assert 0 < factor < 1, "Exponential factor should be positive and lower than 1"

        self.factor = factor
        self.min_sigma = min_sigma

    def __call__(self, current_iter: int = 0):
        sigma = self.sigma * self.factor**current_iter
        return max(sigma, self.min_sigma)


class CosineScheduler(Scheduler):
    def __init__(self, start_sigma: float, end_sigma: float, max_iter: int):
        super().__init__(start_sigma)
        self.end_sigma = end_sigma
        self.max_iter = max_iter

    def __call__(self, current_iter: int) -> float:
        fraction = min(current_iter / self.max_iter, 1.0)
        coeff = 0.5 * (1.0 + np.cos(np.pi * fraction))
        return self.end_sigma + (self.sigma - self.end_sigma) * coeff
