import tempfile
import time
from enum import Enum
from types import SimpleNamespace

import numpy as np
from functools import partial
from pathlib import Path
import tkinter as tk

from .shared_ndarray import SharedNdarray
from . import nibabel_utils as nu
from . import pydrive_utils as pu
from .mainwindow import MainWindow
from .main import args


class states(Enum):
    DISABLED = 0
    CONNECTING = 1
    SELECTING = 2
    DOWNLOADING = 3
    UPLOADING = 4


class GDriveScreen(tk.Toplevel):
    def __init__(self, mw: MainWindow):
        super(GDriveScreen, self).__init__(mw)
        self.root = mw
        self.geometry("640x480")
        self.title("GDrive Interface")
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_rowconfigure(3, weight=0)
        self.grid_rowconfigure(4, weight=1)

        self.connecting_label = tk.Label(self, text='Wait while connecting to Google Drive')

        self.select_label = tk.Label(self, text="Select the case to load.")
        self.case_choices_var = tk.StringVar(value=[])
        self.cases_listbox = tk.Listbox(self, listvariable=self.case_choices_var)
        self.select_button = tk.Button(self, text="Load selected case",
                                       command=lambda: self.set_state(states.DOWNLOADING))

        self.downloading_label = tk.Label(self, text=f"Downloading...")

        self.uploading_label = tk.Label(self, text=f"Uploading...")
        self.set_state(states.CONNECTING)
        self.protocol("WM_DELETE_WINDOW", self.on_window_deleted)

    def on_window_deleted(self):
        self.root.on_window_deleted()
        self.destroy()

    def set_state(self, state: states):
        if state == states.CONNECTING:
            self.root.withdraw()
            self.deiconify()
            self.connecting_label.grid(column=1, row=2)
            self.select_label.grid_forget()
            self.cases_listbox.grid_forget()
            self.select_button.grid_forget()
            self.downloading_label.grid_forget()
            self.uploading_label.grid_forget()
            self.root.update()

            files = self.connect_to_gdrive()
            self.case_choices_var.set(files)
            self.set_state(states.SELECTING)
        elif state == states.SELECTING:
            self.root.withdraw()
            self.deiconify()
            self.connecting_label.grid_forget()
            self.select_label.grid(column=1, row=1)
            self.cases_listbox.grid(column=1, row=2)
            self.select_button.grid(column=1, row=3)
            self.downloading_label.grid_forget()
            self.uploading_label.grid_forget()
            self.root.update()
        elif state == states.DOWNLOADING:
            self.root.withdraw()
            self.deiconify()
            self.connecting_label.grid_forget()
            self.select_label.grid_forget()
            self.cases_listbox.grid_forget()
            self.select_button.grid_forget()
            self.downloading_label.grid(column=1, row=2)
            self.uploading_label.grid_forget()

            self.load_selected()
            self.set_state(states.DISABLED)
        elif state == states.UPLOADING:
            self.root.withdraw()
            self.deiconify()
            self.connecting_label.grid_forget()
            self.select_label.grid_forget()
            self.cases_listbox.grid_forget()
            self.select_button.grid_forget()
            self.downloading_label.grid_forget()
            self.uploading_label.grid(column=1, row=2)
            self.update()
            self.overwrite()
            self.set_state(states.DISABLED)
        else:
            if self.root.selected_case is None:
                self.set_state(states.SELECTING)
                return
            self.connecting_label.grid_forget()
            self.select_label.grid_forget()
            self.cases_listbox.grid_forget()
            self.select_button.grid_forget()
            self.downloading_label.grid_forget()
            self.uploading_label.grid_forget()
            self.withdraw()
            self.root.trigger_draw()
            self.root.deiconify()

    def load_selected(self):
        case = self.cases_listbox.get(tk.ACTIVE)
        self.root.vars.selected_case.set(str(case))
        if args.debug:
            self.root.tmpdir_path = self.root.tmpdir_path / case
            for i in range(5):
                self.downloading_label.config(text=f"Downloading nothing ({i + 1}/{5})...")
                self.downloading_label.update()
                time.sleep(0.1)
        else:
            case = pu.DrivePath(["sources"], root="1N5UQx2dqvWy1d6ve1TEgEFthE8tEApxq") / case
            files = list(case.iterdir())
            for i, file in enumerate(files):
                self.downloading_label.config(text=f"Downloading {case.name} ({i + 1}/{len(files)})...")
                self.downloading_label.update()
                file.resolve().obj.GetContentFile(self.root.tmpdir_path / file.name)
        self.downloading_label.config(text="Converting to numpy...")
        self.downloading_label.update()
        data = nu.load(self.root.tmpdir_path, scan=True, segm=True, clip=(0, 255))
        self.root.selected_case = str(case)
        self.root.vars.scan_height.set(data["scan"].shape[-1])
        self.root.vars.z.set(0)
        self.root.start_base_image_process(data["scan"])
        self.root.case_shape = data["scan"].shape[1:]
        self.root.start_over_image_process(data["segm"])

    def overwrite(self):
        target_case = pu.DrivePath(["sources"], root="1N5UQx2dqvWy1d6ve1TEgEFthE8tEApxq")\
                      / self.root.vars.selected_case.get()
        self.root.over_image_editque.put(
            SimpleNamespace(
                save=True,
            )
        )
        segm = self.root.over_image_saveque.get(block=True)
        nu.save_segmentation(segm, self.root.tmpdir_path)

        source_file = self.root.tmpdir_path / "segmentation.nii.gz"
        target_file = target_case / "segmentation.nii.gz"

        if not target_file.exists():
            f = pu.drive.CreateFile(dict(title=source_file.name, parents=[{"id": target_case.id}]))
            print("  Uploading", source_file.name)
        else:
            f = pu.drive.CreateFile(
                dict(id=target_file.id, title=source_file.name, parents=[{"id": target_case.id}])
            )
            print("  Overwriting", source_file.name)
        f.SetContentFile(str(source_file))
        f.Upload()
        print(f"  ...done!")
    
    def connect_to_gdrive(self):
        if args.debug:
            time.sleep(0.5)
            return [p.relative_to(self.root.tmpdir_path) for p in self.root.tmpdir_path.iterdir()]
        sources = pu.DrivePath(["sources"], root="1N5UQx2dqvWy1d6ve1TEgEFthE8tEApxq")
        files = sorted([path.relative_to(sources) for path in sources.iterdir()])
        # files = [path.relative_to(sources) for path in pu.iter_registered(sources)]
        # root.store.available_cases = files
        return files

    # root.store.available_cases = []
    # root.add_task(connect_to_gdrive(root))
    #
    # root.store.trace_add("available_cases", partial(display_listbox, root, gds, pool_factor=pool_factor))
    # gds.protocol("WM_DELETE_WINDOW", partial(close_both, root, gds))
    #
    # root.store.new("selected_case", None, partial(selected_case_trigger, root, gds))
    # root.store.new("temp_folder", tempfile.TemporaryDirectory())


# def display_listbox(root, cst, pool_factor):
#     for child in cst.winfo_children():
#         child.destroy()
#     cst_title.grid(column=1, row=1)
#     listbox.grid(column=1, row=2)
#     cst_load_button.grid(column=1, row=3)
#
#

#
#
# def selected_case_trigger(root, cst):
#     if root.store.selected_case:
#         root.deiconify()
#         cst.withdraw()
#     else:
#         root.withdraw()
#         cst.deiconify()
#
#
# def close_both(root, cst):
#     cst.destroy()
#     root.destroy()


    


# def avgpool(array, pool_factor):
#     C, X, Y, Z = array.shape
#     return array.reshape(C, X // pool_factor, pool_factor, Y // pool_factor, pool_factor, Z).mean(axis=(2, 4))
#
#
# def maxpool(array, pool_factor):
#     X, Y, Z = array.shape
#     return array.reshape(X // pool_factor, pool_factor, Y // pool_factor, pool_factor, Z).max(axis=(1, 3))


def unmaxpool(array, pool_factor):
    X, Y, Z = array.shape
    array = np.repeat(array, pool_factor, axis=2)
    array = np.repeat(array, pool_factor, axis=1)
    array = np.repeat(array, pool_factor, axis=0)
    return array
