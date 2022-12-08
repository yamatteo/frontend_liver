import ctypes
import multiprocessing
import time
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from .shared_ndarray import SharedNdarray


# from . import nibabel_utils


class SliceGenWorker(object):
    def __init__(self, *, queue, path, z):
        self._is_alive = multiprocessing.Value(ctypes.c_bool, True)
        self.i = 0
        self.queue = queue
        self.path = path
        self.z = z

    def get_is_alive(self):
        with self._is_alive.get_lock():
            return self._is_alive.value

    def set_is_alive(self, value):
        with self._is_alive.get_lock():
            self._is_alive.value = value

    is_alive = property(get_is_alive, set_is_alive)

    def run(self):
        data = nibabelio.load(self.path, scan=True, segm=True, clip=(0, 255))
        scan = data["scan"]
        slice = np.uint8(scan[2, :, :, self.z])
        # slice = np.random.randint(0, 256, (512, 512)).astype(np.uint8)
        self.queue.put(slice)
        self.is_alive = False


class DrawBaseImageWorker:
    def __init__(self, *, queue: multiprocessing.Queue, data: np.ndarray, z: int, resolution: int, phase: int,
                 swap_xy: bool, flip_x: bool, flip_y: bool):
        self._is_alive = multiprocessing.Value(ctypes.c_bool, True)
        self.queue = queue
        self.data = data
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.swap_xy = swap_xy
        self.phase = phase
        self.z = z
        self.resolution = resolution

    def get_is_alive(self):
        with self._is_alive.get_lock():
            return self._is_alive.value

    def set_is_alive(self, value):
        with self._is_alive.get_lock():
            self._is_alive.value = value

    is_alive = property(get_is_alive, set_is_alive)

    def run(self):
        if self.data is None:
            slice = np.random.randint(0, 256, (512, 512)).astype(np.uint8)
        else:
            slice = np.uint8(self.data[self.phase, :, :, self.z])
        if self.swap_xy:
            slice = slice.transpose()
        if self.flip_x:
            slice = np.flip(slice, axis=0)
        if self.flip_y:
            slice = np.flip(slice, axis=1)

        img = Image.fromarray(slice).convert('RGB').resize((self.resolution, self.resolution))
        self.queue.put(img, timeout=4)
        self.is_alive = False


class DrawOverImageWorker:
    def __init__(self, *, queue: multiprocessing.Queue, data: SharedNdarray, z: int, resolution: int,
                 swap_xy: bool, flip_x: bool, flip_y: bool):
        self._is_alive = multiprocessing.Value(ctypes.c_bool, True)
        self.queue = queue
        self.shared_data = data
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.swap_xy = swap_xy
        self.z = z
        self.resolution = resolution

    def get_is_alive(self):
        with self._is_alive.get_lock():
            return self._is_alive.value

    def set_is_alive(self, value):
        with self._is_alive.get_lock():
            self._is_alive.value = value

    is_alive = property(get_is_alive, set_is_alive)

    def run(self):
        if self.shared_data is None:
            slice = np.random.randint(0, 3, (512, 512)).astype(np.uint8)
        else:
            slice = np.uint8(self.shared_data.as_numpy[:, :, self.z])
        if self.swap_xy:
            slice = slice.transpose()
        if self.flip_x:
            slice = np.flip(slice, axis=0)
        if self.flip_y:
            slice = np.flip(slice, axis=1)

        red = 255 * np.uint8(slice == 1)
        green = 255 * np.uint8(slice == 2)
        blue = 255 * np.uint8(slice == 0)
        alpha = np.uint8(np.zeros_like(slice) + 255 * 0.6 * (slice > 0))

        img = Image.fromarray(
            np.stack([red, green, blue, alpha], axis=-1)
        ).convert('RGBA').resize((self.resolution, self.resolution))
        self.queue.put(img, timeout=4)
        self.is_alive = False


class EditWorker:
    def __init__(self, *, flag: multiprocessing.Value, call_stack):
        self._is_alive = multiprocessing.Value(ctypes.c_bool, True)
        self.flag = flag
        self.call_stack = call_stack
        self.shared_data = call_stack[0][1]["data"]

    def get_is_alive(self):
        with self._is_alive.get_lock():
            return self._is_alive.value

    def set_is_alive(self, value):
        with self._is_alive.get_lock():
            self._is_alive.value = value

    is_alive = property(get_is_alive, set_is_alive)

    def run(self):
        segm = self.shared_data.as_numpy
        for (args, kwargs) in self.call_stack:
            event = kwargs["event"]
            scan_size = kwargs["scan_size"]
            swap_xy = kwargs["swap_xy"]
            flip_x = kwargs["flip_x"]
            flip_y = kwargs["flip_y"]
            action = kwargs["action"]
            brush = kwargs["brush"]
            r = kwargs["r"]
            z = kwargs["z"]
            canvas_size = event.widget.winfo_width(), event.widget.winfo_height()
            n = max(*canvas_size) / scan_size
            x, y = int(event.x / n), int(event.y / n)
            if not swap_xy:
                x, y = y, x
            if not flip_x:
                x = (scan_size - 1) - x
            if not flip_y:
                y = (scan_size - 1) - y
            xa, xb, xo, ya, yb, yo = max(0, x - r - 1), min(x + r, scan_size), abs(min(0, x - r - 1)), max(0,
                                                                                                           y - r - 1), min(
                y + r, scan_size), abs(min(0, y - r - 1))

            segm[xa:xb, ya:yb, z] = action(brush[xo:xo + xb - xa, yo:yo + yb - ya], segm[xa:xb, ya:yb, z])
        self.shared_data.update(segm)
        self.flag.value = True
        self.is_alive = False


class ReplacementHandler:
    def __init__(self, return_queue: multiprocessing.Queue, worker_class):
        self.worker = None
        self.worker_class = worker_class
        self.process = None
        self.return_queue = return_queue
        self.next_call = None

    def __delete__(self):
        self.stop()

    def schedule(self, *args, **kwargs):
        self.next_call = (args, kwargs)
        # if self.worker and self.worker.is_alive:
        #     self.scheduled = True
        #     self.next_args = args
        #     self.next_kwargs = kwargs
        # else:
        #     self.stop()
        #     self.scheduled = False
        #     self.worker = self.worker_class(*args, queue=self.return_queue, **kwargs)
        #     self.process = multiprocessing.Process(target=self.worker.run)
        #     self.process.start()

    def wake(self):
        # print("-- wake? --")
        if self.next_call and (self.worker is None or not self.worker.is_alive):
            # print("-- wake! --")
            self.stop()
            args, kwargs = self.next_call
            self.next_call = None
            self.worker = self.worker_class(*args, queue=self.return_queue, **kwargs)
            self.process = multiprocessing.Process(target=self.worker.run)
            self.process.start()

    def stop(self):
        if self.is_alive:
            self.worker.is_alive = False
            self.process.join(timeout=0.5)
            # try:
            #    self.process.close()
            # except ValueError:
            #    self.process.terminate()
            self.worker = None
            self.process = None

    def get_is_alive(self):
        return bool(self.worker and self.process)

    is_alive = property(get_is_alive)


class QueueHandler:
    def __init__(self, flag: multiprocessing.Value, worker_class):
        self.worker = None
        self.worker_class = worker_class
        self.process = None
        self.flag = flag
        self.next_calls = []

    def __delete__(self):
        self.stop()

    def schedule(self, *args, **kwargs):
        # print(" ** scheduled", args, kwargs)
        self.next_calls.append((args, kwargs))
        # if self.worker and self.worker.is_alive:
        #     self.scheduled = True
        #     self.next_args = args
        #     self.next_kwargs = kwargs
        # else:
        #     self.stop()
        #     self.scheduled = False
        #     self.worker = self.worker_class(*args, queue=self.return_queue, **kwargs)
        #     self.process = multiprocessing.Process(target=self.worker.run)
        #     self.process.start()

    def wake(self):
        try:
            assert self.worker is None or not self.worker.is_alive
            assert self.next_calls
            self.worker = self.worker_class(flag=self.flag, call_stack=self.next_calls)
            self.process = multiprocessing.Process(target=self.worker.run)
            self.process.start()
            self.next_calls = []
        except (AttributeError, AssertionError, IndexError) as err:
            pass

    def stop(self):
        pass
        # if self.is_alive:
        #     self.worker.is_alive = False
        #     self.process.join(timeout=0.5)
        #     #try:
        #     #    self.process.close()
        #     #except ValueError:
        #     #    self.process.terminate()
        #     self.worker = None
        #     self.process = None

    def get_is_alive(self):
        return bool(self.worker and self.process)

    is_alive = property(get_is_alive)
