import tkinter as tk
from functools import partial

from .mainwindow import MainWindow

class Menubar(tk.Menu):
    def __init__(self, root: MainWindow):
        super(Menubar, self).__init__(root)
        root['menu'] = self

        menu_segmentation = tk.Menu(self)
        self.add_cascade(menu=menu_segmentation, label='Segmentation')
        menu_segmentation.add_command(
            label="Load local segmentation",
            command=root.load_local_segm
        )
        menu_segmentation.add_separator()
        menu_segmentation.add_command(
            label="Upload and overwrite on LIVERS",
            command=root.overwrite_drive_segm
        )

        menu_editsegmentation = tk.Menu(self)
        self.add_cascade(menu=menu_editsegmentation, label='Edit segmentation')
        menu_editsegmentation.add_command(
            label="Clear segmentation",
            command=root.clear_segm
        )
        menu_editsegmentation.add_command(
            label="Merge nifti mask (liver)",
            command=partial(root.merge_mask, index=1)
        )
        menu_editsegmentation.add_command(
            label="Merge nifti mask (tumor)",
            command=partial(root.merge_mask, index=2)
        )
        menu_editsegmentation.add_command(
            label="Flip left-right",
            command=partial(root.flip_segm, axis=0)
        )
        menu_editsegmentation.add_command(
            label="Flip front-back",
            command=partial(root.flip_segm, axis=1)
        )
        menu_editsegmentation.add_command(
            label="Flip up-down",
            command=partial(root.flip_segm, axis=2)
        )

        menu_view = tk.Menu(self)
        self.add_cascade(menu=menu_view, label='View options')
        menu_view.add_checkbutton(
            label="Transpose",
            onvalue=1,
            offvalue=0,
            variable=root.vars.swap_xy
        )
        menu_view.add_checkbutton(
            label="Flip front-back",
            onvalue=1,
            offvalue=0,
            variable=root.vars.flip_x
        )
        menu_view.add_checkbutton(
            label="Flip left-right",
            onvalue=1,
            offvalue=0,
            variable=root.vars.flip_y
        )
        menu_view.add_separator()
        menu_view.add_command(label="Which phase to show", state="disabled")
        phases = [
            ("Basale", 0),
            ("Arteriosa", 1),
            ("Venosa", 2),
            ("Tardiva", 3),
        ]
        for phase_name, val in phases:
            menu_view.add_radiobutton(label=phase_name, variable=root.vars.phase, value=val)

        menu_brush = tk.Menu(self)
        self.add_cascade(menu=menu_brush, label='Brush options')
        menu_brush.add_command(label="The brush action", state="disabled")
        actions = [
            ("Paint liver", 1),
            ("Delete liver", 10),
            ("Paint tumor", 2),
            ("Delete tumor", 20),
            ("Change tumor -> liver", 21),
            ("Change liver -> tumor", 12),
        ]
        for label, val in actions:
            menu_brush.add_radiobutton(label=label, variable=root.vars.brush_action, value=val)
        menu_brush.add_separator()
        menu_brush.add_command(label="The size of the brush", state="disabled")
        brushes = [
            ("Single point", 0),
            ("Radius 2", 2),
            ("Radius 5", 5),
            ("Radius 10", 10),
            ("Radius 20", 20),
        ]
        for label, val in brushes:
            menu_brush.add_radiobutton(label=label, variable=root.vars.brush_radius, value=val)