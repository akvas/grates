# Copyright (c) 2020-2021 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Classes and functions for visualizing data.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.collections
import cartopy as ctp
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import grates.utilities
import grates.grid
import grates.gravityfield


class StyleContext:

    def __init__(self, name):

        font_size_small = 12
        font_size_medium = 14
        font_size_large = 16
        linewidth = 2
        figure_size = (12 / 2.54, 6 / 2.54)

        style_dict = {}
        if name == 'presentation_calibri':
            font_size_small = 12
            font_size_medium = 14
            font_size_large = 16

            style_dict['font.family'] = 'Calibri'
            style_dict['figure.dpi'] = 600

        elif name == 'presentation_arial':
            font_size_small = 10
            font_size_medium = 12
            font_size_large = 14

            style_dict['font.family'] = 'Arial'
            style_dict['figure.dpi'] = 600

        elif name == 'article_arial':
            font_size_small = 8
            font_size_medium = 10
            font_size_large = 11
            linewidth = 2
            figure_size = (10 / 2.54, 6 / 2.54)

            style_dict['font.family'] = 'Arial'
            style_dict['figure.dpi'] = 600

        style_dict.update(**{'font.size': font_size_small, 'axes.titlesize': font_size_large,
                      'axes.labelsize': font_size_medium, 'figure.titlesize': font_size_large,
                      'xtick.labelsize': font_size_small, 'legend.fontsize': font_size_small,
                      'lines.linewidth': linewidth, 'figure.figsize': figure_size})

        self.__context = mpl.rc_context(style_dict)

    def __enter__(self):
        self.__context.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__context.__exit__(exc_type, exc_val, exc_tb)


def __cell2patch(cell):
    """
    Convert surface elements to matplotlib patches.

    Parameters
    ----------
    cell : grates.grid.SurfaceElement
        instance of SurfaceElement subclass

    Returns
    -------
    patch : matplotlib.patches.Patch
        corresponding Patch subclass
    """
    if isinstance(cell, grates.grid.RectangularSurfaceElement):
        return matplotlib.patches.Rectangle((cell.x*180/np.pi, cell.y*180/np.pi),
                                            cell.width*180/np.pi, cell.height*180/np.pi)
    elif isinstance(cell, grates.grid.PolygonSurfaceElement):
        return matplotlib.patches.Polygon(cell.xy*180/np.pi)
    else:
        raise ValueError('no known conversion for type ' + str(type(cell)) + '.')


def surface_tiles(grid, ax=None, vmin=None, vmax=None, transform=ctp.crs.PlateCarree(), **kwargs):
    """
    Make a 2D plot of the surface tiles (Voronoi cells) of a grid.

    Parameters
    ----------
    grid : grates.grid.Grid
        point distribution
    ax : matplotlib.axes.Axes
        axes into which to plot, if None (default) the current axes are used
    vmin : float
        lower colorbar limit
    vmax : float
        upper colorbar limit
    **kwargs
        forwarded to PatchCollection

    Returns
    -------
    p : matplotlib.collections.PatchCollection
        handle of the PatchCollection

    """
    patches = [__cell2patch(cell) for cell in grid.voronoi_cells()]

    p = matplotlib.collections.PatchCollection(patches, transform=transform, **kwargs)
    if ax is None:
        ax = plt.gca()
    if grid.values is not None:
        p.set_array(grid.values)
        p.set_clim(vmin, vmax)
    ax.add_collection(p)
    return p


def voronoi_bin(lon, lat, C=None, ax=None, grid=grates.grid.GeodesicGrid(25), mincnt=0, reduce_C_function=np.mean,
                vmin=None, vmax=None, **kwargs):
    """
    Make a 2D plot of points lon, lat which are binned into the Voronoi cells of grid.

    Parameters
    ----------
    lon : ndarray(m,)
        point longitude in radians
    lat : ndarray(m,)
        point latitude in radians
    C : ndarray(m,)
        if given these values are accumulated in the bins, otherwise the point count per bin is used
    ax : matplotlib.axes.Axes
        axes into which to plot, if None (default) the current axes are used
    grid : grates.grid.Grid
        the base grid for the Voronoi diagram into which the points are sorted
    mincnt : int
        only draw bins with at least mincnt entries
    reduce_C_function: callable
        the function to aggregate the C values for each bin, ignored if C is not given
    vmin : float
        lower colorbar limit
    vmax : float
        upper colorbar limit
    **kwargs
        forwarded to PatchCollection

    Returns
    -------
    p : matplotlib.collections.PatchCollection
        handle of the PatchCollection

    """
    idx = grid.nn_index(lon, lat)
    patches = [__cell2patch(cell) for cell in grid.voronoi_cells()]

    if C is None:
        values = np.array([len(points) for points in idx], dtype=float)
        values[values < mincnt] = np.nan
    else:
        values = np.full(len(idx), np.nan)
        for k, points in enumerate(idx):
            if len(points) > mincnt:
                values[k] = reduce_C_function(C[points])

    p = matplotlib.collections.PatchCollection(patches, transform=ctp.crs.PlateCarree(), **kwargs)
    if ax is None:
        ax = plt.gca()
    p.set_array(values)
    ax.add_collection(p)
    p.set_clim(vmin, vmax)
    return p


def colorbar(mappable, ax=None, width=0.75, height=0.05, offset=0.1, **kwargs):
    """
    Add a horizontal colorbar to an existing axes.

    Parameters
    ----------
    mappable : handle
        the mappable (AxesImage, ContourSet, ...) described by this colorbar
    ax : matplotlib.axes.Axes
        parent axes
    width : float
        colorbar width (normalized)
    height : float
        colorbar height (normalized)
    offset : float
        offset between plot and colorbar axes
    kwargs :
        passed onto matplotlib.figure.Figure.colorbar

    Returns
    -------
    cbar : Colorbar
        handle of the created colorbar
    """
    if ax is None:
        ax = plt.gca()

    cbaxes = inset_axes(ax, width='{0:f}%'.format(width*100), height='{0:f}%'.format(height*100), loc='lower center',
                        bbox_to_anchor=(0, -offset, 1, 1), bbox_transform=ax.transAxes, borderpad=0, )
    cbar = ax.figure.colorbar(mappable, ax=ax, cax=cbaxes, orientation='horizontal', **kwargs)

    return cbar


def vertical_colorbar(mappable, ax=None, width=0.1, height=1, **kwargs):
    """
    Add a vertical mappable to an existing axes.

    Parameters
    ----------
    mappable : handle
        the mappable (AxesImage, ContourSet, ...) described by this colorbar
    ax : matplotlib.axes.Axes
        parent axes
    width : float
        colorbar width (normalized)
    height : float
        colorbar height (normalized)
    kwargs :
        passed onto matplotlib.figure.Figure.colorbar

    Returns
    -------
    cbar : Colorbar
        handle of the created colorbar
    """
    if ax is None:
        ax = plt.gca()

    cbaxes = inset_axes(ax, width='{0:f}%'.format(width*100), height='{0:f}%'.format(height*100), loc='center left',
                        bbox_to_anchor=(1.05, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0, )
    cbar = ax.figure.colorbar(mappable, ax=ax, cax=cbaxes, orientation='vertical', **kwargs)

    return cbar


def set_axes_width(ax=None, width=None):
    """
    Changes figure width so that ax is exactly width wide.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        axes to scale, if None current axes will be used
    width : float
        axes width in figure units

    """
    if ax is None:
        ax = plt.gca()

    if width is None:
        return

    aw = ax.figure.subplotpars.right - ax.figure.subplotpars.left
    ah = ax.figure.subplotpars.top - ax.figure.subplotpars.bottom

    aspect_ratio = aw / ah

    fw = width / aw
    fh = width / aspect_ratio / ah

    ax.figure.set_size_inches(fw, fh)
    ax.figure.canvas.draw()


def contour_colors(cmap, levels, insignificance_bound=None, insignificance_color=None):
    """
    Compute colors and ticks for contour plots from a colormap and levels.

    Parameters
    ----------
    cmap : Colormap
        a Colormap instance
    levels : list or ndarray
        containes with the level boundaries
    insignificance_bound : float
        levels with midpoints below insignifance bound will be colored with insignificance_color
    insignificance_bound : name or rgb/a tuple
        color for levels which are below the insignificance_bound
    """
    normalized_levels = (levels - np.min(levels)) / (np.max(levels) - np.min(levels))
    colors = []
    ticks = set()
    for k in range(len(levels) - 1):

        level_mid = levels[k] * 0.5 + levels[k + 1] * 0.5
        if insignificance_bound is not None and np.abs(level_mid) < insignificance_bound:
            colors.append(insignificance_color)
        else:
            colors.append(cmap(normalized_levels[k] * 0.5 + normalized_levels[k + 1] * 0.5))
            ticks.update((levels[k], levels[k + 1]))

    return colors, sorted(list(ticks))
