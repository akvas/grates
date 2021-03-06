# Copyright (c) 2020-2021 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Experimental features with frequent interface changes.
"""

import abc
import numpy as np
import scipy.integrate
import scipy.linalg as la
import scipy.signal as sig
import grates.kernel
import grates.utilities
import netCDF4
import grates.grid
import grates.filter


class BlockedVDK(grates.filter.OrderWiseFilter):
    """
    Experimental version of the VDK filter [1]_: Construct the filter matrix from a full normal equation matrix but then
    use only orderwise blocks for filtering.

    Parameters
    ----------
    normal_equation_matrix : ndarray
        normal equation matrix in degree wise coefficient order
        min_degree : int
        minimum degree contained in the normal equation matrix
    max_degree : int
        maximum degree contained in the normal equation matrix
    kaula_scale : float
        scale factor for the Kaula regularization used (scale factor for degree wise weights)
    kaula_power : float
        power for the Kaula regularization used (scale factor for degree wise weights)

    References
    ----------

    .. [1] Horvath, A., Murböck, M., Pail, R., & Horwath, M. (2018). Decorrelation of GRACE time variable gravity field
           solutions using full covariance information. Geosciences, 8(9), 323. https://doi.org/10.3390/geosciences8090323


    """
    def __init__(self, normal_equation_matrix, min_degree, max_degree, kaula_scale, kaula_power):

        parameter_count = normal_equation_matrix.shape[0]

        coefficient_weights = np.empty((max_degree + 1, max_degree + 1))
        for n in range(min_degree, max_degree + 1):
            row_idx, col_idx = grates.gravityfield.degree_indices(n)
            coefficient_weights[row_idx, col_idx] = kaula_scale * float(n)**kaula_power

        NP = normal_equation_matrix.copy()
        NP.flat[::NP.shape[0]+1] = np.diag(normal_equation_matrix) + grates.utilities.ravel_coefficients(coefficient_weights, min_degree, max_degree)

        filter_matrix = np.linalg.solve(NP, normal_equation_matrix)

        coefficient_meta = np.zeros((3, parameter_count), dtype=int)
        idx = 0
        for n in range(min_degree, max_degree + 1):
            coefficient_meta[1, idx] = n
            idx += 1
            for m in range(1, n + 1):
                coefficient_meta[1, idx] = n
                coefficient_meta[2, idx] = m

                coefficient_meta[0, idx + 1] = 1
                coefficient_meta[1, idx + 1] = n
                coefficient_meta[2, idx + 1] = m
                idx += 2

        index_array = coefficient_meta[2, :] == 0
        array = [np.zeros((max_degree + 1, max_degree + 1))]
        array[0][min_degree:, min_degree:] = filter_matrix[np.ix_(index_array, index_array)]
        for m in range(1, max_degree + 1):
            index_array_cosine = np.logical_and(coefficient_meta[2, :] == m, coefficient_meta[0, :] == 0)
            index_array_sine = np.logical_and(coefficient_meta[2, :] == m, coefficient_meta[0, :] == 1)

            if m >= min_degree:
                array.append(filter_matrix[np.ix_(index_array_cosine, index_array_cosine)])
                array.append(filter_matrix[np.ix_(index_array_sine, index_array_sine)])
            else:
                coefficient_count = max_degree + 1 - m

                array.append(np.zeros((coefficient_count, coefficient_count)))
                array[-1][min_degree - m:, min_degree - m:] = filter_matrix[np.ix_(index_array_cosine,
                                                                                              index_array_cosine)]
                array.append(np.zeros((coefficient_count, coefficient_count)))
                array[-1][min_degree - m:, min_degree - m:] = filter_matrix[np.ix_(index_array_sine,
                                                                                             index_array_sine)]

        super(BlockedVDK, self).__init__(array)


def lsa_psd(x, y, nperseg=256, window='boxcar'):

    nperseg = min(nperseg, x.size)

    dx = float(np.median(np.diff(x)))
    interval_bounds = [0]
    while interval_bounds[-1] < x.size:
        interval_bounds.append(min(interval_bounds[-1] + nperseg, x.size))
    interval_bounds.append(x.size)

    frequencies = np.fft.rfftfreq(nperseg, dx)
    is_even = (nperseg % 2) == 0
    loop_count = frequencies.size - 2 if is_even else frequencies.size - 1

    N = [np.zeros((1, 1))]
    n = [np.zeros((1, 1))]
    for k in range(loop_count):
        N.append(np.zeros((2, 2)))
        n.append(np.zeros((2, 1)))
    if is_even:
        N.append(np.zeros((1, 1)))
        n.append(np.zeros((1, 1)))

    for idx_start, idx_end in zip(interval_bounds[0:-1], interval_bounds[1:]):
        interval_length = idx_end - idx_start
        if interval_length < 3:
            continue

        w = sig.get_window(window, interval_length)[:, np.newaxis]

        t = x[idx_start:idx_end]
        l = y[idx_start:idx_end, np.newaxis]*w

        A = np.ones((interval_length, 1))*w
        N[0] += A.T@A
        n[0] += A.T@l

        for k in range(1, loop_count + 1):
            A = np.vstack((np.cos(2*np.pi*frequencies[k]*t),
                           np.sin(2*np.pi*frequencies[k]*t))).T*w
            N[k] += A.T @ A
            n[k] += A.T @ l

        if is_even:
            A = np.ones((interval_length, 1))*w
            A[1::2, 0] = -1
            N[-1] += A.T @ A
            n[-1] += A.T @ l

    x_hat = []
    for k in range(len(N)):
        x_hat.append(np.linalg.solve(N[k], n[k]))

    psd = np.zeros(frequencies.size)
    for idx_start, idx_end in zip(interval_bounds[0:-1], interval_bounds[1:]):
        interval_length = idx_end - idx_start
        if interval_length < 3:
            continue

        t = x[idx_start:idx_end]

        A = np.ones((interval_length, 1))
        l_hat = A@x_hat[0]
        psd[0] = np.sum(l_hat**2)

        for k in range(1, loop_count + 1):
            A = np.vstack((np.cos(2*np.pi*frequencies[k]*t),
                           np.sin(2*np.pi*frequencies[k]*t))).T
            l_hat = A @ x_hat[k]
            psd[k] = np.sum(l_hat ** 2)

        if is_even:
            A = np.ones((interval_length, 1))
            A[1::2, 0] = -1
            l_hat = A @ x_hat[-1]
            psd[-1] = np.sum(l_hat ** 2)

    return frequencies, psd/dx*np.sqrt(2)


def vce_psd(x, y, nperseg=256, initial_variance=1, max_iter=5, detrend=False, window='boxcar'):

    nperseg = min(nperseg, x.size)

    dx = np.median(np.diff(x))
    interval_bounds = [0]
    segment_length = 0
    for k in range(1, x.size):
        segment_length += 1
        if x[k]-x[k-1] > dx*1.5 or segment_length == nperseg:
            interval_bounds.append(min(interval_bounds[-1] + segment_length, x.size))
            segment_length = 0
    interval_bounds.append(x.size)

    dummy = np.array(interval_bounds)
    nperseg = np.max(dummy[1:]-dummy[0:-1])

    dct_matrix = np.fromfunction(lambda i, j: 2*np.cos(np.pi*i*j/(nperseg - 1)), (nperseg, nperseg))
    dct_matrix[:, (0, -1)] *= 0.5
    dct_matrix *= 1.0/np.sqrt(2*(nperseg-1))

    initial_covariance = np.zeros(nperseg)
    initial_covariance[0] = initial_variance
    variance_components = (dct_matrix@initial_covariance[:, np.newaxis]).squeeze()

    for iteration in range(max_iter):
        covariance_function = (dct_matrix@variance_components[:, np.newaxis]).squeeze()
        covariance_matrix = la.toeplitz(covariance_function, covariance_function)

        square_sum = np.zeros(covariance_function.size)
        redundancy = np.zeros(covariance_function.size)

        for idx_start, idx_end in zip(interval_bounds[0:-1], interval_bounds[1:]):
            interval_length = idx_end - idx_start
            if interval_length < 2:
                continue

            projection_matrix = np.linalg.inv(covariance_matrix[0:interval_length, 0:interval_length])
            e = y[idx_start:idx_end, np.newaxis]

            residuals = (projection_matrix @ e).squeeze()

            for k in range(residuals.size):
                square_sum[k] += np.sum(residuals[0:residuals.size-k]*residuals[k:])
                redundancy[k] += np.sum(np.diag(projection_matrix, k))

        redundancy[1:] *= 2
        square_sum[1:] *= 2

        ePe = (square_sum[np.newaxis, :] @ dct_matrix).squeeze()
        r = (redundancy[np.newaxis, :] @ dct_matrix).squeeze()

        variance_components *= ePe/r

    return np.linspace(0, 0.5/dx, variance_components.size),  variance_components*dx*np.sqrt(2*(nperseg-1)), dummy[1:]-dummy[0:-1], dct_matrix


def legendre_matern(sigma0, alpha, nu, psi, max_degree=1024, min_degree=2):

    n = np.arange(max_degree + 1)

    coefficients = sigma0**2 * (alpha**2 + n**2)**-(nu + 0.5)
    coefficients[0:min_degree] = 0

    return grates.utilities.legendre_summation(coefficients, psi)







