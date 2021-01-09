# Copyright (c) 2020-2021 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Functions for importing package data.
"""

import numpy as np
import pkg_resources


def import_load_love_numbers(max_degree=None, frame='CE'):
    """
    Load Love numbers computed by Wang et al. (2012) [1]_ for the elastic Earth model ak135 [2]_.
    from degree 0 to 46340.

    Parameters
    ----------
    max_degree : int
        maximum degree of the load love numbers to be returned (default: return all love numbers).
    frame : str
        frame of the load love numbers (CM, CE, CF).

    Returns
    -------
    k : array_like
        gravity change load love numbers
    h : array_like
        radial displacement load love numbers
    l : array_like
        horizontal displacement load love numbers

    References
    ----------

    .. [1] Hansheng Wang, Longwei Xiang, Lulu Jia, Liming Jiang, Zhiyong Wang, Bo Hu, and Peng Gao. 2012. Load Love
           numbers and Green's functions for elastic Earth models PREM, iasp91, ak135, and modified models with refined
           crustal structure from Crust 2.0. Comput. Geosci. 49 (December, 2012), 190–199.
           DOI:https://doi.org/10.1016/j.cageo.2012.06.022

    .. [2] B. L. N. Kennett, E. R. Engdahl, R. Buland, Constraints on seismic velocities in the Earth from traveltimes,
           Geophysical Journal International, Volume 122, Issue 1, July 1995, Pages 108–124,
           https://doi.org/10.1111/j.1365-246X.1995.tb03540.x

    """
    if max_degree is not None and max_degree < 1:
        return np.zeros(1), np.zeros(1), np.zeros(1)

    hlk = np.vstack((np.zeros((1, 3)), np.loadtxt(pkg_resources.resource_filename('grates', 'data/ak135-LLNs-complete.dat.gz'),
                                                  skiprows=1, usecols=(1, 2, 3), max_rows=max_degree)))

    if frame.lower() == 'cm':
        hlk[1, :] -= 1
    elif frame.lower() == 'cf':
        hlk_ce = hlk[1, :].copy()
        hlk[1, 0] = (hlk_ce[0] - hlk_ce[1]) * 2 / 3
        hlk[1, 1] = (hlk_ce[0] - hlk_ce[1]) * -1 / 3
        hlk[1, 2] = (-1 / 3 * hlk_ce[0] - 2 / 3 * hlk_ce[1])
    elif frame.lower() == 'ce':
        pass
    else:
        raise ValueError('frame of load love numbers must be one of CM, CE, or CF (got <' + frame + '>)')

    return hlk[:, 2], hlk[:, 0], hlk[:, 1]


love_numbers_ce = import_load_love_numbers(frame='CE')
love_numbers_cm = import_load_love_numbers(frame='CM')
love_numbers_cf = import_load_love_numbers(frame='CF')


def load_love_numbers(max_degree=None, frame='CE'):
    """
    Wrapper for load love numbers. Return love numbers in different frames loaded on module import.

    Parameters
    ----------
    max_degree : int
        maximum degree of the load love numbers to be returned (default: return all love numbers).
    frame : str
        frame of the load love numbers (CM, CE, CF).

    Returns
    -------
    k : array_like
        gravity change load love numbers
    h : array_like
        radial displacement load love numbers
    l : array_like
        horizontal displacement load love numbers
    """
    if frame.lower() == 'cm':
        return love_numbers_cm[0:None]
    elif frame.lower() == 'cf':
        return love_numbers_cf[0:None]
    elif frame.lower() == 'ce':
        return love_numbers_ce[0:None]
    else:
        raise ValueError('frame of load love numbers must be one of CM, CE, or CF (got <' + frame + '>)')


def ddk_normal_blocks():
    """
    Return the orderwise normal equation blocks of the DDK normal equation matrix.

    Returns
    -------
    block_matrix : list of ndarrays
        orderwise matrix blocks (alternating cosine/sine per order, order 0 only contains cosine coefficients)
    """
    with np.load(pkg_resources.resource_filename('grates', 'data/ddk_normal_blocks.npz')) as f:
        blocks = [f['order0_cos']]
        for m in range(1, 120 + 1):
            blocks.append(f['order{0:d}_cos'.format(m)])
            blocks.append(f['order{0:d}_sin'.format(m)])

        return blocks
