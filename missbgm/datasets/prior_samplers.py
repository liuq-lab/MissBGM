"""Prior samplers used by MissBGM."""

from __future__ import annotations

import numpy as np


class Gaussian_sampler(object):
    """Multivariate Gaussian sampler."""

    def __init__(self, mean, sd=1, N=20000):
        self.total_size = int(N)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.sd = float(sd)
        np.random.seed(1024)
        self.X = np.random.normal(self.mean, self.sd, (self.total_size, len(self.mean))).astype("float32")

    def train(self, batch_size, label=False):
        indices = np.random.randint(low=0, high=self.total_size, size=int(batch_size))
        return self.X[indices, :]

    def get_batch(self, batch_size):
        return np.random.normal(self.mean, self.sd, (int(batch_size), len(self.mean))).astype("float32")

    def load_all(self):
        return self.X
