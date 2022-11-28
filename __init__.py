import asyncio
from functools import partial
from pathlib import Path
from sys import platform
from tkinter import *
from tkinter import filedialog

import numpy as np

from . import case_selection_top, canvas as canvas_module, nibabelio
from .asynctk import AsyncTk
from .shared_tasks import SharedNdarray
from . import pydrive_utils as pu

pool_factor = 2

def build_root():
    root = AsyncTk()
    store = root.store
    store.height = 0

    root.geometry("1280x800")
    root.minsize(1280, 800)
    root.option_add('*tearOff', FALSE)

    case_selection_top.build(root, pool_factor=pool_factor)
    canvas, draw = canvas_module.build(root)

    canvas.bind(
        "<Button-1>",
        lambda event: root.spawn_event_task(
            click_action,
            event,
            store.loaded_segm,
            action=store.action,
            brush=store.brush,
            scan_size=store.loaded_segm.shape[0],
            swap=store.swap_xy,
            flipx=store.flip_x,
            flipy=store.flip_y,
            r=store.brush_radius,
            z=store.z,
        )
    )
    canvas.bind(
        "<B1-Motion>",
        lambda event: root.spawn_event_task(
            click_action,
            event,
            store.loaded_segm,
            action=store.action,
            brush=store.brush,
            scan_size=store.loaded_segm.shape[0],
            swap=store.swap_xy,
            flipx=store.flip_x,
            flipy=store.flip_y,
            r=store.brush_radius,
            z=store.z,
        )
    )

    root.set_trigger("redraw", draw)

    store.new("z", value=1, callback=lambda: setattr(store, "redraw", True))
    store.trace_add("loaded_scan", root.task_factory(draw))
    root.bind_all("<MouseWheel>", partial(mouse_wheel, root))
    root.bind_all("<Button-4>", partial(mouse_wheel, root))
    root.bind_all("<Button-5>", partial(mouse_wheel, root))

    menubar = Menu(root)
    root['menu'] = menubar

    menu_segmentation = Menu(menubar)
    menubar.add_cascade(menu=menu_segmentation, label='Segmentation')
    menu_segmentation.add_command(label="Load local segmentation", command=partial(load_local, store))
    menu_segmentation.add_separator()
    menu_segmentation.add_command(label="Upload and overwrite on LIVERS", command=partial(overwrite, store))

    menu_view = Menu(menubar)
    menubar.add_cascade(menu=menu_view, label='View options')
    menu_view.add_checkbutton(label="Transpose", onvalue=1, offvalue=0,
                              variable=store.new("swap_xy", BooleanVar(value=False)))
    store.trace_add("swap_xy", root.task_factory(draw))
    menu_view.add_checkbutton(label="Flip left-right", onvalue=1, offvalue=0,
                              variable=store.new("flip_x", BooleanVar(value=False)))
    store.trace_add("flip_x", root.task_factory(draw))
    menu_view.add_checkbutton(label="Flip front-back", onvalue=1, offvalue=0,
                              variable=store.new("flip_y", BooleanVar(value=True)))
    store.trace_add("flip_y", root.task_factory(draw))
    menu_view.add_separator()
    menu_view.add_command(label="Which phase to show", state="disabled")
    phases = [
        ("Basale", 0),
        ("Arteriosa", 1),
        ("Venosa", 2),
        ("Tardiva", 3),
    ]
    intvar = store.new("phase", IntVar(value=2))
    for phase_name, val in phases:
        menu_view.add_radiobutton(label=phase_name, variable=intvar, value=val)
    store.trace_add("phase", root.task_factory(draw))

    menu_brush = Menu(menubar)
    menubar.add_cascade(menu=menu_brush, label='Brush options')
    menu_brush.add_command(label="The brush action", state="disabled")
    actions = [
        ("Paint liver", 1),
        ("Delete liver", 10),
        ("Paint tumor", 2),
        ("Delete tumor", 20),
        ("Change tumor -> liver", 21),
        ("Change liver -> tumor", 12),
    ]
    intvar = store.new("brush_action", IntVar(value=1), callback=partial(set_action, root))
    for label, val in actions:
        menu_brush.add_radiobutton(label=label, variable=intvar, value=val)
    menu_brush.add_separator()
    menu_brush.add_command(label="The size of the brush", state="disabled")
    brushes = [
        ("Single point", 0),
        ("Radius 2", 2),
        ("Radius 5", 5),
        ("Radius 10", 10),
        ("Radius 20", 20),
    ]
    intvar = store.new("brush_radius", IntVar(value=5), callback=partial(set_brush, root))
    for label, val in brushes:
        menu_brush.add_radiobutton(label=label, variable=intvar, value=val)

    return root


def set_action(root):
    store = root.store
    action = store.brush_action
    from_index, to_index = action // 10, action % 10
    store.action = lambda b, s: s + (to_index - from_index) * b * (s == from_index)


def set_brush(root):
    store = root.store
    radius = store.brush_radius
    side = 2 * radius + 1
    store.brush = np.array([
        [int((i - radius) ** 2 + (j - radius) ** 2 < (radius + 1) ** 2) for j in range(side)]
        for i in range(side)
    ])


if platform == "linux" or platform == "linux2":
    def mouse_wheel(root, event):
        if event.num == 4:
            root.store.z = max(0, root.store.z - 1)
        elif event.num == 5:
            root.store.z = min(root.store.z + 1, root.store.height - 1)
elif platform == "darwin":
    def mouse_wheel(root, event):
        root.store.z = max(0, min(root.store.z + event.delta, root.store.height - 1))
else:
    def mouse_wheel(root, event):
        root.store.z = max(0, min(root.store.z + event.delta // 120, root.store.height - 1))


def click_action(event, shared_segm: SharedNdarray, *, action, brush, scan_size, swap, flipx, flipy, r, z):
    segm = shared_segm.as_numpy
    canvas_size = event.widget.winfo_width(), event.widget.winfo_height()
    n = max(*canvas_size) / scan_size
    x, y = int(event.x / n), int(event.y / n)
    if swap:
        x, y = y, x
    if flipx:
        x = (scan_size - 1) - x
    if flipy:
        y = (scan_size - 1) - y
    xa, xb, xo, ya, yb, yo = max(0, x - r - 1), min(x + r, scan_size), abs(min(0, x - r - 1)), max(0, y - r - 1), min(
        y + r, scan_size), abs(min(0, y - r - 1))

    segm[xa:xb, ya:yb, z] = action(brush[xo:xo + xb - xa, yo:yo + yb - ya], segm[xa:xb, ya:yb, z])
    shared_segm.update(segm)


def overwrite(store):
    target_case = store.selected_case
    segm = store.loaded_segm.as_numpy
    segm = case_selection_top.unmaxpool(segm, pool_factor)
    nibabelio.save_segmentation(segm, Path(store.temp_folder.name))

    source_file = Path(store.temp_folder.name) / "segmentation.nii.gz"
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


def load_local(store):
    filetypes = (
        ('Compressed nifti', '*.nii.gz'),
    )

    filename = filedialog.askopenfilename(
        title='Load a segmentation',
        initialdir='/',
        filetypes=filetypes,
    )

    affine, bottom, top, height = nibabelio.load_registration_data(Path(store.temp_folder.name))
    segm = nibabelio.load_ndarray(Path(filename))
    segm = segm[..., bottom:top]
    segm = segm.astype(np.int64)
    segm = case_selection_top.maxpool(segm, pool_factor)
    store.loaded_segm = SharedNdarray.from_numpy(segm)
    store.redraw = True


def stub(*args, **kwargs):
    print("Stub function", args, kwargs)
