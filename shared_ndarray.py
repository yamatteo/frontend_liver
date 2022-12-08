import multiprocessing as mp
from dataclasses import dataclass

import numpy as np


@dataclass
class SharedNdarray:
    shared_array: mp.Array
    shape: tuple

    @classmethod
    def from_numpy(cls, ndarray: np.ndarray):
        return cls(
            mp.Array('d', ndarray.reshape((-1,))),
            tuple(ndarray.shape),
        )

    @property
    def as_numpy(self):
        return np.frombuffer(self.shared_array.get_obj()).reshape(self.shape)

    def update(self, ndarray: np.ndarray):
        self.shared_array[:] = ndarray.reshape((-1,))
