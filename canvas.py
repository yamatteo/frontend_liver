from tkinter import *

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

matplotlib.use('TkAgg')

def build(root):
    tv = Frame(root)
    store = root.store
    figure = Figure(figsize=(800, 800), dpi=1)
    figure.set_layout_engine("tight", pad=0)
    axes = figure.add_subplot(xticks=[], yticks=[])
    figure_canvas = FigureCanvasTkAgg(figure, tv)
    canvas = figure_canvas.get_tk_widget()
    canvas.pack(anchor="center", fill="none", expand=0)

    async def draw(*args):
        try:
            white = store.loaded_scan.as_numpy[store.phase, :, :, store.z]
            if store.loaded_segm is None:
                segm = np.zeros_like(white)
            else:
                segm = store.loaded_segm.as_numpy[:, :, store.z]
            red = white + 180 * (segm == 1)
            green = white + 180 * (segm == 2)
            rgb_x_y = np.clip(np.stack([red, green, white]) / 280, 0, 1)
            if store.swap_xy:
                rgb_x_y = rgb_x_y.transpose((0, 2, 1))
            if store.flip_x:
                rgb_x_y = np.flip(rgb_x_y, 1)
            if store.flip_y:
                rgb_x_y = np.flip(rgb_x_y, 2)
            axes.imshow(np.transpose(rgb_x_y), cmap='gray', interpolation='nearest')
        except Exception as err:
            print(err)
            axes.imshow(np.zeros((256, 256)), cmap='gray', interpolation='nearest')
        figure_canvas.draw()

    tv.pack(anchor="center", fill="both", expand=1)
    return canvas, draw