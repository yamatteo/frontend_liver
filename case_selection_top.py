import tempfile
import numpy as np
from functools import partial
from pathlib import Path
from tkinter import *

from . import nibabelio
from . import pydrive_utils as pu
from .asynctk import AsyncTk
from .shared_tasks import SharedNdarray


def build(root: AsyncTk, pool_factor=1):
    cst = Toplevel(root)
    cst.geometry("640x480")
    cst.title("Select case")
    cst.grid_columnconfigure(0, weight=1)
    cst.grid_columnconfigure(1, weight=0)
    cst.grid_columnconfigure(2, weight=1)
    cst.grid_rowconfigure(0, weight=1)
    cst.grid_rowconfigure(1, weight=0)
    cst.grid_rowconfigure(2, weight=0)
    cst.grid_rowconfigure(3, weight=0)
    cst.grid_rowconfigure(4, weight=1)
    cst_wait_label = Label(cst, text='Wait while connecting to GDrive')
    cst_wait_label.pack(expand=1, fill="both")
    root.store.available_cases = []
    root.add_task(connect_to_gdrive(root))

    root.store.trace_add("available_cases", partial(display_listbox, root, cst, pool_factor=pool_factor))
    cst.protocol("WM_DELETE_WINDOW", partial(close_both, root, cst))

    root.store.new("selected_case", None, partial(selected_case_trigger, root, cst))
    root.store.new("temp_folder", tempfile.TemporaryDirectory())
    return cst


def display_listbox(root, cst, pool_factor):
    for child in cst.winfo_children():
        child.destroy()
    cst_title = Label(cst, text="Select the case to load.")
    choicesvar = StringVar(value=sorted(root.store.available_cases))
    listbox = Listbox(cst, listvariable=choicesvar)
    cst_load_button = Button(cst, text="Load selected case",
                             command=partial(load_selected, root.store, cst, listbox, pool_factor))
    cst_title.grid(column=1, row=1)
    listbox.grid(column=1, row=2)
    cst_load_button.grid(column=1, row=3)


def load_selected(store, cst, listbox, pool_factor):
    case = listbox.get(ACTIVE)
    for child in cst.winfo_children():
        child.destroy()
    label = Label(cst, text=f"Downloading {case}...")
    label.pack(expand=1, fill="both")
    # DEBUG
    # files = [1, 2, 3, 4, 5]
    # REAL
    case = pu.DrivePath(["sources"], root="1N5UQx2dqvWy1d6ve1TEgEFthE8tEApxq") / case
    files = list(case.iterdir())
    for i, file in enumerate(files):
        label.config(text=f"Downloading {case.name} ({i + 1}/{len(files)})...")
        label.update()
        # DEBUG
        # time.sleep(0.1)
        # REAL
        file.resolve().obj.GetContentFile(Path(store.temp_folder.name) / file.name)
    data = nibabelio.load(Path(store.temp_folder.name), scan=True, segm=True, clip=(0, 255))
    scan = avgpool(data["scan"], pool_factor)
    if data["segm"] is None:
        segm = np.zeros_like(scan[0])
    else:
        segm = maxpool(data["segm"], pool_factor)
    store.height = scan.shape[-1]
    store.loaded_scan = SharedNdarray.from_numpy(scan)
    store.loaded_segm = SharedNdarray.from_numpy(segm)
    store.redraw = True
    store.selected_case = case


def selected_case_trigger(root, cst):
    if root.store.selected_case:
        root.deiconify()
        cst.withdraw()
    else:
        root.withdraw()
        cst.deiconify()


def close_both(root, cst):
    cst.destroy()
    root.destroy()


async def connect_to_gdrive(root):
    # DEBUG
    # time.sleep(1)
    # files = ["one", "two", "three"]
    # REAL
    sources = pu.DrivePath(["sources"], root="1N5UQx2dqvWy1d6ve1TEgEFthE8tEApxq")
    files = [ path.relative_to(sources) for path in sources.iterdir() ]
    root.store.available_cases = files


def avgpool(array, pool_factor):
    C, X, Y, Z = array.shape
    return array.reshape(C, X // pool_factor, pool_factor, Y // pool_factor, pool_factor, Z).mean(axis=(2, 4))


def maxpool(array, pool_factor):
    X, Y, Z = array.shape
    return array.reshape(X // pool_factor, pool_factor, Y // pool_factor, pool_factor, Z).max(axis=(1, 3))


def unmaxpool(array, pool_factor):
    X, Y, Z = array.shape
    array = np.repeat(array, pool_factor, axis=2)
    array = np.repeat(array, pool_factor, axis=1)
    array = np.repeat(array, pool_factor, axis=0)
    return array
