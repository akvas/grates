import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.collections
import grates.utilities
import grates.grid
import grates.gravityfield
import cartopy as ctp
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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


def preview_gravityfield(x, vmin=-25, vmax=25, min_degree=2, max_degree=None):

    array = grates.utilities.unravel_coefficients(x, min_degree, max_degree)
    gf = grates.gravityfield.PotentialCoefficients()
    gf.anm = array

    grid = gf.to_grid()

    plt.figure()
    ax = plt.axes(projection=ctp.crs.Mollweide())

    ax.imshow(grid.values[::-1, :]*100, vmin=vmin, vmax=vmax, cmap='RdBu', transform=ctp.crs.PlateCarree())
    ax.coastlines()
    plt.show()


def colorbar(image, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    cbaxes = inset_axes(ax,
                       width="75%",  # width = 5% of parent_bbox width
                       height="5%",  # height : 50%
                       loc='lower center',
                       bbox_to_anchor=(0, -0.1, 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0,
                       )

    cbar = ax.figure.colorbar(image, ax=ax, cax=cbaxes, orientation='horizontal', **kwargs)

    return cbar


class GlobalFigure:


    def __init__(self, file_name=None, width=12, height=None):

        self.__width = width
        self.__height = height

        self.__figure = plt.figure()

        self.__axes = plt.axes(projection=ctp.crs.Mollweide())
        self.__axes.set_global()
        self.__dpi = 300
        self.__file_name = file_name

        self.__cblabel = None

    def imshow(self, values, **kwargs):

        self.__im = self.__axes.imshow(values[::-1, :], transform=ctp.crs.PlateCarree(), **kwargs)

    def plot(self, x, y, **kwargs):

        self.__axes.plot(x, y, transform=ctp.crs.Geodetic(), **kwargs)

    def coastlines(self, **kwargs):

        self.__axes.coastlines(**kwargs)

    def colorbar(self, label, **kwargs):

        self.__cblabel = label
        self.__cbargs = kwargs

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        width = self.__axes.figure.subplotpars.right - self.__axes.figure.subplotpars.left
        height = self.__axes.figure.subplotpars.top - self.__axes.figure.subplotpars.bottom

        aspect_ratio = width / height

        if self.__height is None:
            self.__height = self.__width / aspect_ratio

        fw = self.__width / 2.54 / width
        fh = self.__height / 2.54 / height

        self.__axes.figure.set_size_inches(fw, fh)
        self.__figure.canvas.draw()

        if self.__cblabel:
            cbaxes = self.__figure.add_axes(
                [self.__axes.figure.subplotpars.left + width * 0.125, self.__axes.figure.subplotpars.bottom + 0.15,
                 width * 0.75, 0.025])
            self.__cbar = self.__figure.colorbar(self.__im, label=self.__cblabel, ax=self.__axes, cax=cbaxes,
                                                 orientation='horizontal', **self.__cbargs)

        if self.__file_name is not None:
            self.__figure.savefig(self.__file_name, dpi=self.__dpi, transparent=True, bbox_inches='tight')
        else:
           plt.show()
