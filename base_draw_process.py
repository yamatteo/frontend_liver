import ctypes
import multiprocessing
import time
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
import queue

from .shared_ndarray import SharedNdarray


class Worker:
    def __init__(
            self,
            request_queue: multiprocessing.Queue,
            return_queue: multiprocessing.Queue, 
            data: np.ndarray,
            ):
        self._is_alive = multiprocessing.Value(ctypes.c_bool, True)
        self.request_queue = request_queue
        self.return_queue = return_queue
        self.data = data

        self._process = multiprocessing.Process(target=self.run)
        self._process.start()


        # self.flip_x = flip_x
        # self.flip_y = flip_y
        # self.swap_xy = swap_xy
        # self.phase = phase
        # self.z = z
        # self.resolution = resolution

    def get_is_alive(self):
        with self._is_alive.get_lock():
            return self._is_alive.value

    def set_is_alive(self, value):
        with self._is_alive.get_lock():
            self._is_alive.value = value

    is_alive = property(get_is_alive, set_is_alive)

    def run(self):
        while self.is_alive:
            request = None
            while True:
                try:
                    request = self.request_queue.get_nowait()
                except queue.Empty:
                    break
            if request:
                if self.data is None:
                    slice = np.random.randint(0, 256, (512, 512)).astype(np.uint8)
                else:
                    slice = np.uint8(self.data[request.phase, :, :, request.z])
                if request.swap_xy:
                    slice = slice.transpose()
                if request.flip_x:
                    slice = np.flip(slice, axis=0)
                if request.flip_y:
                    slice = np.flip(slice, axis=1)

                img = Image.fromarray(slice).convert('RGB').resize(
                    (request.resolution, request.resolution))
                self.return_queue.put(img, timeout=5)
            else:
                time.sleep(0.5)
    
    def stop(self):
        self.is_alive = False
        self._process.join(timeout=2)
