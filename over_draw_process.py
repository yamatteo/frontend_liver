import ctypes
import multiprocessing
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
import torch
import numpy as np
from PIL import Image
import queue

from .shared_ndarray import SharedNdarray


class Worker:
    def __init__(
            self,
            draw_queue: multiprocessing.Queue,
            edit_queue: multiprocessing.Queue,
            return_queue: multiprocessing.Queue, 
            save_queue: multiprocessing.Queue, 
            data: np.ndarray,
            ):
        self._is_alive = multiprocessing.Value(ctypes.c_bool, True)
        self.draw_queue = draw_queue
        self.edit_queue = edit_queue
        self.save_queue = save_queue
        self.return_queue = return_queue
        if data is None:
            data = np.zeros((512, 512, 1)).astype(np.uint8)
        self.data = data
        import lovely_tensors as lt
        print("New OIWorker with", lt.lovely(torch.tensor(data)))

        self._process = multiprocessing.Process(target=self.run)
        self._process.start()

    def get_is_alive(self):
        with self._is_alive.get_lock():
            return self._is_alive.value

    def set_is_alive(self, value):
        with self._is_alive.get_lock():
            self._is_alive.value = value

    is_alive = property(get_is_alive, set_is_alive)

    def run(self):
        segm = self.data
        draw_parameters = SimpleNamespace(
            swap_xy=True,
            flip_x=True,
            flip_y=False,
            resolution=800,
            z=0,
        )
        self_request = False
        self_action = lambda b, s: s + (1 - 0) * b * (s == 0)

        def put():
            try:
                slice = np.uint8(segm[:, :, draw_parameters.z])
            except:
                slice = np.random.randint(0, 3, (512, 512)).astype(np.uint8)
            if draw_parameters.swap_xy:
                slice = slice.transpose()
            if draw_parameters.flip_x:
                slice = np.flip(slice, axis=0)
            if draw_parameters.flip_y:
                slice = np.flip(slice, axis=1)

            red = 255 * np.uint8(slice == 1)
            green = 255 * np.uint8(slice == 2)
            blue = 255 * np.uint8(slice == 0)
            alpha = np.uint8(np.zeros_like(slice) + 255 * 0.6 * (slice > 0))

            img = Image.fromarray(
                np.stack([red, green, blue, alpha], axis=-1)
            ).convert('RGBA').resize((draw_parameters.resolution, draw_parameters.resolution))
            self.return_queue.put(img)

        while self.is_alive:
            try:
                edit_request = self.edit_queue.get_nowait()
            except queue.Empty:
                edit_request = None
            if hasattr(edit_request, 'event'):
                event = edit_request.event

                scan_size = edit_request.scan_size
                swap_xy = edit_request.swap_xy
                flip_x = edit_request.flip_x
                flip_y = edit_request.flip_y
                action = self_action
                brush = edit_request.brush
                r = edit_request.r
                z = edit_request.z
                canvas_size = event.canvas_size
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
                self_request = True
                continue
            elif hasattr(edit_request, "flipaxis"):
                segm = np.flip(segm, axis=edit_request.flipaxis)
                self_request = True
                continue
            elif hasattr(edit_request, "translate"):
                back = np.zeros_like(segm)
                delta = edit_request.translate
                if delta > 0:
                    back[..., delta:] = segm[..., :-delta]
                else:
                    back[..., :delta] = segm[..., -delta:]
                segm = back
                self_request = True
                continue
            elif hasattr(edit_request, "set_action"):
                action = edit_request.set_action
                from_index, to_index = action // 10, action % 10
                self_action = lambda b, s: s + \
                    (to_index - from_index) * b * (s == from_index)
            elif hasattr(edit_request, "mask"):
                shape = edit_request.shape
                mask = np.zeros(shape)
                ed_mask = np.clip(edit_request.mask, 0, 1)
                top = min(ed_mask.shape[-1], mask.shape[-1])
                mask[..., :top] = ed_mask[..., :top]
                segm = mask * edit_request.index + (1-mask) * segm
                self_request = True
                continue
            elif hasattr(edit_request, "save"):
                self.save_queue.put(segm)
            while True:
                try:
                    draw_parameters = self.draw_queue.get_nowait()
                    self_request = True
                except queue.Empty:
                    break
            if self_request:
                put()
                self_request = False
            else:
                time.sleep(0.5)
    
    def stop(self):
        self.is_alive = False
        self._process.join(timeout=2)
