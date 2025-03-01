from typing import List
from examples.seismic import Model

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np


def plot_density(model: Model, show: bool = True):
    """
    Plot a two-dimensional density field from a seismic Model
    object.

    Parameters
    ----------
    model : Model
        Object that holds the density model.
    """
    rho = 1/model.b.data
    rho = rho[
        model.nbl:-model.nbl, 
        model.nbl:-model.nbl
    ]
    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]

    plt.figure()
    plot = plt.imshow(np.transpose(rho), cmap='cividis', vmin=np.min(rho), vmax=np.max(rho), 
                      extent=extent)
    plt.xlabel('X position (km)')
    plt.ylabel('Depth (km)')

    ax = plt.gca()
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', 0.05)
    cbar = plt.colorbar(plot, cax=cax)
    cbar.set_label('Density ($g/$cm$^3$)')

    if show:
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_image2(data, model, vmin=None, vmax=None, colorbar=True, cmap="gray", 
                show: bool = True, clip_percent=None, clip_low=None):
    """
    Works like plot_image, but adjusts the image aspect ratio and clips high values 
    based on a percentile, while keeping the range dynamic for better contrast.

    Parameters
    ----------
    data : ndarray
        Image data to plot.
    model : SeismicModel
        Reference model to get the aspect ratio.
    cmap : str
        Choice of colormap. Defaults to gray scale for images as a seismic convention.
    clip_percent : float, optional
        Percentage threshold for clipping high values.
    clip_low : float, optional
        Percentage threshold for clipping low values.
    """
    plt.figure()

    # Calculate domain size and extent for the plot
    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [
        model.origin[0], model.origin[0] + domain_size[0],
        model.origin[1] + domain_size[1], model.origin[1]
    ]

    print("Original data shape:", data.shape)
    data = np.asarray(data)

    
    if clip_percent is not None or clip_low is not None:
        low_value = np.percentile(data, clip_low) if clip_low is not None else np.min(data)
        high_value = np.percentile(data, clip_percent) if clip_percent is not None else np.max(data)
        print(f"Clipping low values below {low_value:.5f} and high values above {high_value:.5f}.")
        data = np.clip(data, low_value, high_value)

        if vmin is None:
            vmin = low_value
        if vmax is None:
            vmax = high_value
    else:
        print("Clipping not applied; plotting data as is.")

    plot = plt.imshow(
        np.transpose(data),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extent=extent
    )

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)

    if show:
        plt.show()

def plot_shotrecord(rec, model, t0, tn, colorbar=True, clip_percent=None, clip_low=None):
    """
    Plot a shot record (receiver values over time) with optional percentile clipping.

    Parameters
    ----------
    rec : ndarray
        Receiver data with shape (time, points).
    model : Model
        Object that holds the velocity model.
    t0 : int
        Start of time dimension to plot.
    tn : int
        End of time dimension to plot.
    clip_percent : float, optional
        Percentage threshold for clipping high values.
    clip_low : float, optional
        Percentage threshold for clipping low values.
    """
    rec = np.asarray(rec)
    
    if clip_percent is not None or clip_low is not None:
        low_value = np.percentile(rec, clip_low) if clip_low is not None else np.min(rec)
        high_value = np.percentile(rec, clip_percent) if clip_percent is not None else np.max(rec)
        print(f"Clipping low values below {low_value:.5f} and high values above {high_value:.5f}.")
        rec = np.clip(rec, low_value, high_value)
    
    scale = np.max(np.abs(rec)) / 10.
    extent = [
        model.origin[0], model.origin[0] + 1e-3 * model.domain_size[0],
        1e-3 * tn, t0
    ]

    plot = plt.imshow(rec, vmin=-scale, vmax=scale, cmap=plt.get_cmap("gray"), extent=extent)
    plt.xlabel('X position (km)')
    plt.ylabel('Time (s)')

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)
    plt.show()




def plot_graph(xdata: np.ndarray, ydata: np.ndarray, xlabel: str, ylabel: str, 
               title: str = '', show: bool = True):
    plt.figure()
    plt.plot(xdata, ydata)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if show:
        plt.show()


def plot_array(arr: np.ndarray, xlabel: str, ylabel: str, 
               title: str = '', show: bool = True):
    x = np.arange(1, 1 + len(arr))
    plot_graph(x, arr, xlabel, ylabel, title, show)
    plt.xticks(x[::int(np.ceil(len(arr)/10))])


def plot_trace(traces: List[np.ndarray], ymin, ymax, styles: List[str], legends: List[str], 
                xlabel: str, ylabel: str, show: bool = True):
    plt.figure(figsize = (8, 9))

    for i in range(len(traces)):
        y = np.linspace(ymin, ymax, traces[i].size)
        plt.plot(traces[i], y, styles[i], linewidth=2)

    plt.legend(legends, fontsize=15)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.gca().invert_yaxis()

    if show:
        plt.show()