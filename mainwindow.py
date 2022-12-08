import multiprocessing
import queue
import tempfile
import tkinter as tk
from dataclasses import dataclass
from functools import partial
from sys import platform
from types import SimpleNamespace
from tkinter import filedialog
from .shared_ndarray import SharedNdarray

import numpy as np
from PIL import ImageTk
from . import nibabel_utils as nu
from pathlib import Path


@dataclass
class Store:
    brush_action: tk.IntVar
    brush_radius: tk.IntVar
    flip_x: tk.BooleanVar
    flip_y: tk.BooleanVar
    phase: tk.IntVar
    scan_height: tk.IntVar
    selected_case: tk.StringVar
    swap_xy: tk.BooleanVar
    z: tk.IntVar


class MainWindow(tk.Tk):
    def __init__(self):
        from .gdrive_screen import GDriveScreen
        from .menubar import Menubar
        from .main import args

        from .worker import ReplacementHandler, DrawBaseImageWorker, DrawOverImageWorker, QueueHandler, EditWorker
        super(MainWindow, self).__init__()
        self.loaded_scan = None
        self.loaded_segm = None
        self.edit_process = None
        self.action = None
        self.brush = None
        self.edit_flag = multiprocessing.Value('i', 0)
        self.draw_queue = multiprocessing.Queue(100)
        self.overdraw_queue = multiprocessing.Queue(100)
        self.draw_process = ReplacementHandler(return_queue=self.draw_queue, worker_class=DrawBaseImageWorker)
        self.overdraw_process = ReplacementHandler(return_queue=self.overdraw_queue, worker_class=DrawOverImageWorker)
        self.edit_process = QueueHandler(flag=self.edit_flag, worker_class=EditWorker)
        self.vars = Store(
            brush_action=tk.IntVar(value=1),
            brush_radius=tk.IntVar(value=5),
            flip_x=tk.BooleanVar(value=True),
            flip_y=tk.BooleanVar(value=False),
            phase=tk.IntVar(value=2),
            scan_height=tk.IntVar(value=1),
            selected_case=tk.StringVar(value=""),
            swap_xy=tk.BooleanVar(value=True),
            z=tk.IntVar(value=0),
        )
        self.vars.flip_x.trace_add("write", self.trigger_draw)
        self.vars.flip_y.trace_add("write", self.trigger_draw)
        self.vars.swap_xy.trace_add("write", self.trigger_draw)
        self.vars.phase.trace_add("write", self.trigger_draw)
        self.vars.z.trace_add("write", self.trigger_draw)
        self.vars.brush_action.trace_add("write", self.set_action)
        self.vars.brush_radius.trace_add("write", self.set_brush)

        self.gdrive_screen = GDriveScreen(self)
        self.menubar = Menubar(self)
        self.resolution = 800
        self.canvas = tk.Canvas(self, bg="black", height=self.resolution, width=self.resolution)

        self.canvas.bind("<MouseWheel>", mouse_wheel(self))
        self.canvas.bind("<Button-4>", mouse_wheel(self))
        self.canvas.bind("<Button-5>", mouse_wheel(self))
        self.canvas.bind(
            "<Button-1>",
            lambda event: self.edit_process.schedule(
                event=event,
                data=self.loaded_segm,
                action=self.action,
                brush=self.brush,
                scan_size=self.loaded_segm.shape[0],
                swap_xy=self.vars.swap_xy.get(),
                flip_x=self.vars.flip_x.get(),
                flip_y=self.vars.flip_y.get(),
                r=self.vars.brush_radius.get(),
                z=self.vars.z.get(),
                resolution=self.resolution,
            )
        )
        self.canvas.bind(
            "<B1-Motion>",
            lambda event: self.edit_process.schedule(
                event=event,
                data=self.loaded_segm,
                action=self.action,
                brush=self.brush,
                scan_size=self.loaded_segm.shape[0],
                swap_xy=self.vars.swap_xy.get(),
                flip_x=self.vars.flip_x.get(),
                flip_y=self.vars.flip_y.get(),
                r=self.vars.brush_radius.get(),
                z=self.vars.z.get(),
                resolution=self.resolution,
            )
        )

        self.bind("<Up>", lambda e: self.move(1))
        self.bind("<Down>", lambda e: self.move(-1))
        
        self.bind("<Shift-Up>", partial(self.translate, delta=1))
        self.bind("<Shift-Down>", partial(self.translate,delta=-1))

        self.canvas.pack()
        self.base_imgtk = None
        self.base_image_id = None
        self.over_imgtk = None
        self.over_image_id = None

        if args.debug:
            self.tmpdir_path = Path("/home/yamatteo/tmpdir")
        else:
            self.tmpdir = tempfile.TemporaryDirectory()
            self.tmpdir_path = Path(self.tmpdir.name)

        self.trigger_draw()
        self.set_action()
        self.set_brush()

        # End process on exit
        self.protocol("WM_DELETE_WINDOW", self.on_window_deleted)

        self.after_idle(self.process_queues)

    def __delete__(self):
        self.stop()
        self.draw_process = None
        self.edit_process = None
        self.overdraw_process = None

    def move(self, delta):
        try:
            self.vars.z.set(max(0, min(self.vars.z.get() + delta, self.vars.scan_height.get() - 1)))
        except tk.TclError:
            self.vars.z.set(0)

    def on_window_deleted(self):
        self.stop()
        self.destroy()

    def process_queues(self):
        try:
            img = self.draw_queue.get_nowait()
            # print("-- got base image --")
            self.base_imgtk = ImageTk.PhotoImage(img)
            if self.base_image_id:
                self.canvas.itemconfig(self.base_image_id, image=self.base_imgtk)
            else:
                self.base_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.base_imgtk)
            self.canvas.update()
            self.after(10, self.process_queues)
            return
        except queue.Empty:
            pass
        try:
            img = self.overdraw_queue.get_nowait()
            # print("-- got over image --")
            self.over_imgtk = ImageTk.PhotoImage(img)
            if self.over_image_id:
                self.canvas.itemconfig(self.over_image_id, image=self.over_imgtk)
            else:
                self.over_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.over_imgtk)
            self.canvas.update()
            self.after(10, self.process_queues)
            return
        except queue.Empty:
            pass
        if self.edit_flag.value == True:
            self.trigger_overdraw()
            self.edit_flag.value = False

        self.draw_process.wake()
        self.overdraw_process.wake()
        self.edit_process.wake()
        self.after(50, self.process_queues)

    def stop(self):
        if self.edit_process:
            self.edit_process.stop()
        if self.draw_process:
            self.draw_process.stop()

    def trigger_draw(self, *args):
        # print("-- trigger draw --")
        self.draw_process.schedule(
            data=self.loaded_scan,
            flip_x=self.vars.flip_x.get(),
            flip_y=self.vars.flip_y.get(),
            resolution=self.resolution,
            phase=self.vars.phase.get(),
            swap_xy=self.vars.swap_xy.get(),
            z=self.vars.z.get(),
        )
        self.trigger_overdraw()

    def trigger_overdraw(self, *args):
        # print("-- trigger overdraw --")
        self.overdraw_process.schedule(
            data=self.loaded_segm,
            flip_x=self.vars.flip_x.get(),
            flip_y=self.vars.flip_y.get(),
            resolution=self.resolution,
            swap_xy=self.vars.swap_xy.get(),
            z=self.vars.z.get(),
        )

    def load_local_segm(self):
        filetypes = (
            ('Compressed nifti', '*.nii.gz'),
        )

        filename = filedialog.askopenfilename(
            title='Load a segmentation',
            initialdir='/',
            filetypes=filetypes,
        )

        affine, bottom, top, height = nu.load_registration_data(self.tmpdir_path)
        segm = nu.load_ndarray(Path(filename))
        segm = segm[..., bottom:top]
        segm = segm.astype(np.int64)
        self.loaded_segm = SharedNdarray.from_numpy(segm)

    def overwrite_drive_segm(self):
        from .gdrive_screen import states
        self.gdrive_screen.set_state(states.UPLOADING)

    def set_action(self, *args):
        action = self.vars.brush_action.get()
        from_index, to_index = action // 10, action % 10
        self.action = lambda b, s: s + (to_index - from_index) * b * (s == from_index)

    def set_brush(self, *args):
        radius = self.vars.brush_radius.get()
        side = 2 * radius + 1
        self.brush = np.array([
            [int((i - radius) ** 2 + (j - radius) ** 2 < (radius + 1) ** 2) for j in range(side)]
            for i in range(side)
        ])

    def clear_segm(self, *args):
        self.loaded_segm.update(
            np.zeros_like(self.loaded_segm.as_numpy)
        )
        self.trigger_overdraw()

    def flip_segm(self, *args, axis=0):
        segm = self.loaded_segm.as_numpy
        segm = np.flip(segm, axis=axis)
        self.loaded_segm.update(segm)
        self.trigger_overdraw()

    def translate(self, *args, delta=0):
        segm = self.loaded_segm.as_numpy
        back = np.zeros_like(segm)
        if delta > 0:
            back[..., delta:] = segm[..., :-delta]
        else:
            back[..., :delta] = segm[..., -delta:]
        self.loaded_segm.update(back)
        self.trigger_overdraw()

    def merge_mask(self, *args, index: int):
        segm = self.loaded_segm.as_numpy
        mask = np.zeros_like(segm)
        filetypes = (
            ('Nifti', '*.nii'),
        )
        filename = filedialog.askopenfilename(
            title='Load a mask',
            initialdir='/',
            filetypes=filetypes,
        )
        affine, bottom, top, height = nu.load_registration_data(self.tmpdir_path)

        mask[..., :(top-bottom)] = nu.load_ndarray(Path(filename))[..., :(top-bottom)]
        mask = np.clip(mask, 0, 1)
        segm = (1 - mask) * segm + mask * index
        self.loaded_segm.update(segm)
        self.trigger_overdraw()

def mouse_wheel(root):
    if platform == "linux" or platform == "linux2":
        def _mouse_wheel(event):
            if event.num == 4:
                root.move(-1)
            elif event.num == 5:
                root.move(1)
    elif platform == "darwin":
        def _mouse_wheel(event):
            root.move(event.delta)
    else:
        def _mouse_wheel(root, event):
            root.move(event.delta // 120)
    return _mouse_wheel
