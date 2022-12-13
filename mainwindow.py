import multiprocessing
import queue
import tempfile
import tkinter as tk
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from sys import platform
from tkinter import filedialog
from types import SimpleNamespace

import numpy as np
from PIL import ImageTk

from . import over_draw_process

from . import base_draw_process

from . import nibabel_utils as nu
from .shared_ndarray import SharedNdarray


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
        from .main import args
        from .menubar import Menubar
        from .worker import (DrawBaseImageWorker, DrawOverImageWorker,
                             EditWorker, QueueHandler, ReplacementHandler)
        super(MainWindow, self).__init__()
        self.loaded_segm = None
        self.edit_process = None
        self.action = None
        self.brush = None
        self.selected_case = None
        self.case_shape = None

        self.base_image_reqque = multiprocessing.Queue(100)
        self.base_image_retque = multiprocessing.Queue(100)
        self.base_image_process = base_draw_process.Worker(self.base_image_reqque, self.base_image_retque, None)
        self.base_image_id = None
        self.base_imgtk = None

        self.over_image_editque = multiprocessing.Queue(100)
        self.over_image_drawque = multiprocessing.Queue(100)
        self.over_image_retque = multiprocessing.Queue(100)
        self.over_image_saveque = multiprocessing.Queue(100)
        self.over_image_process = over_draw_process.Worker(
            self.over_image_drawque,
            self.over_image_editque,
            self.over_image_retque,
            self.over_image_saveque,
            None,
        )
        self.over_imgtk = None
        self.over_image_id = None

        if args.debug:
            self.tmpdir_path = Path("/home/yamatteo/tmpdir")
        else:
            self.tmpdir = tempfile.TemporaryDirectory()
            self.tmpdir_path = Path(self.tmpdir.name)

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
        self.canvas = tk.Canvas(
            self, bg="black", height=self.resolution, width=self.resolution)

        self.canvas.bind("<MouseWheel>", mouse_wheel(self))
        self.canvas.bind("<Button-4>", mouse_wheel(self))
        self.canvas.bind("<Button-5>", mouse_wheel(self))
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<B1-Motion>", self.click)

        self.bind("<Up>", lambda e: self.move(1))
        self.bind("<Down>", lambda e: self.move(-1))

        self.bind("<Shift-Up>", partial(self.translate, delta=1))
        self.bind("<Shift-Down>", partial(self.translate, delta=-1))

        self.canvas.pack()

        self.trigger_draw()
        self.set_action()
        self.set_brush()

        # End process on exit
        self.protocol("WM_DELETE_WINDOW", self.on_window_deleted)

        self.after_idle(self.process_queues)

    def __delete__(self):
        self.stop()
        self.base_image_process = None
        self.over_image_process = None

    def move(self, delta):
        try:
            self.vars.z.set(
                max(0, min(self.vars.z.get() + delta, self.vars.scan_height.get() - 1)))
        except tk.TclError:
            self.vars.z.set(0)

    def on_window_deleted(self):
        self.stop()
        self.destroy()

    def process_queues(self):
        try:
            img = self.base_image_retque.get_nowait()
            # print("-- got base image --")
            self.base_imgtk = ImageTk.PhotoImage(img)
            if self.base_image_id:
                self.canvas.itemconfig(
                    self.base_image_id, image=self.base_imgtk)
            else:
                self.base_image_id = self.canvas.create_image(
                    0, 0, anchor=tk.NW, image=self.base_imgtk)
            self.canvas.update()
            self.after(5, self.process_queues)
            return
        except queue.Empty:
            pass
        try:
            img = self.over_image_retque.get_nowait()
            # print("-- got over image --")
            self.over_imgtk = ImageTk.PhotoImage(img)
            if self.over_image_id:
                self.canvas.itemconfig(
                    self.over_image_id, image=self.over_imgtk)
            else:
                self.over_image_id = self.canvas.create_image(
                    0, 0, anchor=tk.NW, image=self.over_imgtk)
            self.canvas.update()
            self.after(5, self.process_queues)
            return
        except queue.Empty:
            pass
        
        self.after(50, self.process_queues)

    def stop(self):
        if self.base_image_process:
            self.base_image_process.stop()
        if self.over_image_process:
            self.over_image_process.stop()
    
    def start_base_image_process(self, data: np.ndarray):
        self.base_image_process.stop()
        self.base_image_process = base_draw_process.Worker(self.base_image_reqque, self.base_image_retque, data)
        self.trigger_draw()
    
    def start_over_image_process(self, data: np.ndarray):
        self.over_image_process.stop()
        self.over_image_process = over_draw_process.Worker(
            self.over_image_drawque,
            self.over_image_editque,
            self.over_image_retque,
            self.over_image_saveque,
            data,
        )
        self.trigger_draw()

    def trigger_draw(self, *args):
        # print("-- trigger draw --")
        self.base_image_reqque.put(
            SimpleNamespace(
                flip_x=self.vars.flip_x.get(),
                flip_y=self.vars.flip_y.get(),
                resolution=self.resolution,
                phase=self.vars.phase.get(),
                swap_xy=self.vars.swap_xy.get(),
                z=self.vars.z.get(),
            )
        )
        self.trigger_overdraw()

    def trigger_overdraw(self, *args):
        # print("-- trigger overdraw --")
        self.over_image_drawque.put(
            SimpleNamespace(
                flip_x=self.vars.flip_x.get(),
                flip_y=self.vars.flip_y.get(),
                resolution=self.resolution,
                swap_xy=self.vars.swap_xy.get(),
                z=self.vars.z.get(),
            )
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

        affine, bottom, top, height = nu.load_registration_data(
            self.tmpdir_path)
        segm = nu.load_ndarray(Path(filename))
        import torch
        import lovely_tensors as lt
        print("segm", filename, (bottom, top), lt.lovely(torch.tensor(segm)))
        segm = segm[..., bottom:top]
        segm = segm.astype(np.int64)
        self.start_over_image_process(segm)
        self.trigger_overdraw()

    def overwrite_drive_segm(self):
        from .gdrive_screen import states
        self.gdrive_screen.set_state(states.UPLOADING)

    def set_action(self, *args):
        action = self.vars.brush_action.get()
        self.over_image_editque.put(
            SimpleNamespace(
                set_action=action
            )
        )

    def set_brush(self, *args):
        radius = self.vars.brush_radius.get()
        side = 2 * radius + 1
        self.brush = np.array([
            [int((i - radius) ** 2 + (j - radius) ** 2 < (radius + 1) ** 2)
             for j in range(side)]
            for i in range(side)
        ])

    def clear_segm(self, *args):
        self.start_over_image_process(np.zeros(self.case_shape))

    def flip_segm(self, *args, axis=0):
        self.over_image_editque.put(
            SimpleNamespace(
                flipaxis=axis
            )
        )

    def translate(self, *args, delta=0):
        self.over_image_editque.put(
            SimpleNamespace(
                translate=delta
            )
        )

    def merge_mask(self, *args, index: int):
        filetypes = (
            ('Nifti', '*.nii'),
        )
        filename = filedialog.askopenfilename(
            title='Load a mask',
            initialdir='/',
            filetypes=filetypes,
        )
        self.over_image_editque.put(SimpleNamespace(
            mask=nu.load_ndarray(Path(filename)),
            shape=self.case_shape,
            index=index,
        ))
    
    def click(self, event, *args):
        self.over_image_editque.put(
                SimpleNamespace(
                    event=SimpleNamespace(
                        canvas_size=(event.widget.winfo_width(), event.widget.winfo_height()),
                        x=event.x,
                        y=event.y
                    ),
                    scan_size=self.case_shape[-2],
                    brush=self.brush,
                    swap_xy=self.vars.swap_xy.get(),
                    flip_x=self.vars.flip_x.get(),
                    flip_y=self.vars.flip_y.get(),
                    r=self.vars.brush_radius.get(),
                    z=self.vars.z.get(),
                    resolution=self.resolution,
                )
            )


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
