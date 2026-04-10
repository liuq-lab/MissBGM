"""Mini-batch samplers used by MissBGM."""

from __future__ import annotations

import math

import numpy as np
from sklearn.preprocessing import StandardScaler


class Base_sampler(object):
    """Simple infinite mini-batch iterator over a tabular dataset."""

    def __init__(self, x, y, v, batch_size=32, normalize=False, random_seed=123):
        assert len(x) == len(y) == len(v)
        np.random.seed(random_seed)
        self.data_x = np.asarray(x, dtype="float32")
        self.data_y = np.asarray(y, dtype="float32")
        self.data_v = np.asarray(v, dtype="float32")
        if self.data_x.ndim == 1:
            self.data_x = self.data_x.reshape(-1, 1)
        if self.data_y.ndim == 1:
            self.data_y = self.data_y.reshape(-1, 1)
        self.batch_size = int(batch_size)
        if normalize:
            self.data_v = StandardScaler().fit_transform(self.data_v)
        self.sample_size = len(x)
        self.full_index = np.arange(self.sample_size)
        np.random.shuffle(self.full_index)
        self.idx_gen = self.create_idx_generator(sample_size=self.sample_size)

    def create_idx_generator(self, sample_size):
        while True:
            for step in range(math.ceil(sample_size / self.batch_size)):
                start = step * self.batch_size
                end = (step + 1) * self.batch_size
                if end <= sample_size:
                    yield self.full_index[start:end]
                else:
                    yield np.hstack([self.full_index[start:], self.full_index[: end - sample_size]])
                    np.random.shuffle(self.full_index)

    def next_batch(self):
        indices = next(self.idx_gen)
        return self.data_x[indices, :], self.data_y[indices, :], self.data_v[indices, :]

    def load_all(self):
        return self.data_x, self.data_y, self.data_v
