# Copyright (c) 2020-2021 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Classes and functions for solving overdetermined systems of equations.
"""

import numpy as np
import scipy.linalg as la


class AutoregressiveModel:
    """
    Class representation of vector-autoregressive (VAR) model.

    Parameters
    ----------
    coefficients : list, tuple, ndarray
        VAR model coefficients
    covariance_matrix : ndarray
        covariance matrix of the white noise sequence
    """
    def __init__(self, coefficients, covariance_matrix):

        if isinstance(coefficients, np.ndarray):
            self.__coefficients = tuple(coefficients)
        else:
            self.__coefficients = coefficients

        self.__covariance_matrix = covariance_matrix
        self.__normal_equation = None

    @property
    def dimension(self):
        """
        Dimension of the VAR model.

        Returns
        -------
        dim : int
            dimension of the VAR model
        """
        return self.__covariance_matrix.shape[0]

    @property
    def order(self):
        """
        Order of the VAR model.

        Returns
        -------
        order : int
            order of the VAR model
        """
        return len(self.__coefficients)

    @property
    def white_noise_covariance(self):
        """
        Return a view of the white noise covariance matrix.

        Returns
        -------
        covariance_matrix : ndarray
            covariance matrix of the white noise sequence
        """
        return self.__covariance_matrix

    @property
    def coefficients(self):
        """
        Return the coefficients of the VAR model as a list of ndarrays.

        Returns
        -------
        coefficients : list of ndarrays
            VAR model coefficients
        """
        return self.__coefficients

    def order_one_representation(self):
        """
        Compute the order-one representation of the VAR model.

        Returns
        -------
        armodel : AutoregressiveModel
            autoregressive model of order one
        """
        if self.order == 1:
            return self
        else:
            B = np.eye(self.dimension * self.order)
            for k in range(self.order):
                B[0:self.dimension, k * self.dimension:(k + 1) * self.dimension] = self.__coefficients[k].copy()
            Q = np.zeros(B.shape)
            Q[0:self.dimension, 0:self.dimension] = self.__covariance_matrix.copy()

            return AutoregressiveModel(B, Q)

    @staticmethod
    def from_transformed_coefficients(transformed_coefficients):
        """
        Generate a VAR model from a matrix containing transformed coefficients and white noise covariance matrix.

        Parameters
        ----------
        transformed_coefficients : ndarray(m, (p+1)*m)
             transformed coefficients and white noise covariance matrix

        Returns
        -------
        armodel : AutoregressiveModel
            autoregressive model instance
        """
        W = np.linalg.pinv(transformed_coefficients[:, -transformed_coefficients.shape[0]:])
        dim = transformed_coefficients.shape[0]
        coefficient_count = int(transformed_coefficients.shape[1] / transformed_coefficients.shape[0] - 1)

        coefficients = tuple(-transformed_coefficients[:, k * dim:(k + 1) * dim] for k in range(coefficient_count))

        covariance_matrix = W @ W.T

        return AutoregressiveModel(coefficients[::-1], covariance_matrix)

    @staticmethod
    def from_covariance_function(covariance_function):
        """
        Generate a VAR model from a multivariate covariance function using Yule-Walker equations.

        Parameters
        ----------
        covariance_function : list of ndarrays
             covariance function of the underlying process as a list of ndarrays (index k corresponds to the coviarance
             matrix of lag k)

        Returns
        -------
        armodel : AutoregressiveModel
            autoregressive model instance
        """
        if isinstance(covariance_function, np.ndarray):
            covariance_function = tuple(covariance_function)

        model_order = len(covariance_function) - 1
        if model_order == 0:
            return AutoregressiveModel((), covariance_function[0])

        dimension = covariance_function[0].shape[0]
        block_index = [0]
        while block_index[-1] < model_order * dimension:
            block_index.append(block_index[-1] + dimension)

        coefficient_matrix = BlockMatrix(block_index, block_index)
        right_hand_side = np.empty((dimension * model_order, dimension))
        for row in range(coefficient_matrix.shape[0]):
            right_hand_side[row * dimension:(row + 1) * dimension, :] = covariance_function[row + 1]
            for column in range(row, coefficient_matrix.shape[1]):
                coefficient_matrix[row, column] = covariance_function[column - row].T

        coefficient_matrix.cholesky()
        x1 = coefficient_matrix.solve_triangular(right_hand_side, transpose=True)
        x2 = coefficient_matrix.solve_triangular(x1)

        Q = covariance_function[0] - x2.T @ right_hand_side

        return AutoregressiveModel(np.split(x2.T, model_order, axis=1), Q)

    @staticmethod
    def from_sample(sample, order):
        """
        Generate a VAR model from a multivariate covariance function using Yule-Walker equations.

        Parameters
        ----------
        sample : ndarray(n, m)
            time series of process realization of dimension m
        order : int
            order of the VAR model to be computed

        Returns
        -------
        armodel : AutoregressiveModel
            autoregressive model instance
        """
        covariance_function = []
        for k in range(order + 1):
            covariance_function.append((sample.T @ sample) / (sample.shape[0] - k))

        return AutoregressiveModel.from_covariance_function(covariance_function)

    def __compute_normals(self):
        """
        Compute the system of normal equations corresponding to the pseudo observations represented by the VAR
        model.
        """
        W = np.linalg.cholesky(self.__covariance_matrix).T

        observation_equations = [np.linalg.solve(W.T, B) for B in self.__coefficients[::-1]]
        observation_equations.append(-np.linalg.inv(W.T))

        block_index = [0]
        while block_index[-1] < (self.order + 1) * self.dimension:
            block_index.append(block_index[-1] + self.dimension)

        self.__normal_equation = BlockMatrix(block_index, block_index)
        for row in range(self.__normal_equation.shape[0]):
            for column in range(row, self.__normal_equation.shape[1]):
                self.__normal_equation[row, column] = observation_equations[row].T @ observation_equations[column]

    def normal_equation_block(self, row, column):
        """
        Return the normal equation coefficient matrix block at a specific row and column.

        Parameters
        ----------
        row : int
            block row index
        column : int
            block column index

        Returns
        -------
        block : ndarray(m, m)
            normal equation coefficient matrix block
        """
        if self.__normal_equation is None:
            self.__compute_normals()

        return self.__normal_equation[row, column]

    def to_transformed_coefficients(self):
        """
        Return the VAR model as transformed coefficients.

        Returns
        -------
        coeffs : ndarray(dim, dim*(order+1))
        """
        W_inv = la.inv(la.cholesky(self.__covariance_matrix))

        transformed_coefficients = []
        for B in self.__coefficients[::-1]:
            transformed_coefficients.append(-W_inv @ B)
        transformed_coefficients.append(W_inv)

        return np.hstack(transformed_coefficients)


class AutoregressiveModelSequence:
    """
    Class representation of a sequence of vector autoregressive models of increasing orders.

    Parameters
    ----------
    armodels : list of AutoregressiveModel instances
        list of AutoregressiveModel instances of increasing model order (starting from order 0)
    """
    def __init__(self, armodels):

        self.__armodels = armodels

    @staticmethod
    def from_covariance_function(covariance_function):
        """
        Generate a VAR model sequence from a multivariate covariance function using Yule-Walker equations.

        Parameters
        ----------
        covariance_function : list of ndarrays
             covariance function of the underlying process as a list of ndarrays (index k corresponds to the coviarance
             matrix of lag k)

        Returns
        -------
        armodel_sequence : AutoregressiveModelSequence
            autoregressive model sequence
        """
        armodels = []
        for k in range(len(covariance_function)):
            armodels.append(AutoregressiveModel.from_covariance_function(covariance_function[0:k + 1]))

        return AutoregressiveModelSequence(armodels)

    @staticmethod
    def from_sample(sample, maximum_order):
        """
        Generate a VAR model sequence from a sample time series. This function estimates the empirical covariance
        function and computes the VAR models using Yule-Walker equations.

        Parameters
        ----------
        sample : ndarray(n, m)
            sample time series of dimension m
        maximum_order : int
            maximum order of the VAR model sequence

        Returns
        -------
        armodel_sequence : AutoregressiveModelSequence
            autoregressive model sequence
        """
        armodels = []
        for order in range(maximum_order + 1):
            armodels.append(AutoregressiveModel.from_sample(sample, order))

        return AutoregressiveModelSequence(armodels)

    @property
    def maximum_order(self):
        """
        Return the maximum order of the VAR model sequence.

        Returns
        -------
        max_order : int
            maximum order of the VAR model sequence
        """
        return self.__armodels[-1].order

    @property
    def dimension(self):
        """
        Dimension of the VAR model sequence.

        Returns
        -------
        dim : int
            dimension of the VAR model sequence
        """
        return self.__armodels[-1].dimension

    def __normals_block(self, epoch_count, row, column):
        """
        Compute the normal equation coefficients matrix block (row, column) for a least squares adjustment with
        epoch_count epochs. Note: only the upper triangle is supported.

        Parameters
        ----------
        epoch_count : int
            number of epoch to be estimated
        row : int
            row coordinate of matrix block
        column : int
            column coordinate of matrix block

        Returns
        -------
        normals_block : ndarray(dim, dim)
            normal equation coefficients matrix block (row, column)
        """
        N = np.zeros((self.dimension, self.dimension))

        for index in range(epoch_count - self.maximum_order):
            if row >= index and column <= (self.maximum_order + index):
                N += self.__armodels[-1].normal_equation_block(row - index, column - index)

        for order in range(self.maximum_order):
            if row <= order and column <= order:
                N += self.__armodels[order].normal_equation_block(row, column)

        return N

    def normal_equations(self, epoch_count):
        """
        Generate the inverse covariance matrix (normal equation coefficient matrix) represented by the
        AutoregressiveModelSequence. The resulting system of normal equations represents zero constraint
        temporal variations.

        Parameters
        ----------
        epoch_count : int
            number of epoch to be constrained

        Returns
        -------
        normals : NormalEquations
            blocked normal equation with zero right hand side
        """
        parameter_count = epoch_count * self.dimension
        block_index = np.arange(0, parameter_count + self.dimension, self.dimension, dtype=int)

        normals_matrix = BlockMatrix(block_index, block_index)
        right_hand_side = np.zeros((parameter_count, 1))
        observation_count = parameter_count
        observation_square_sum = 0.0

        for row in range(epoch_count):
            for column in range(row, min(epoch_count, row + self.maximum_order + 1)):
                normals_matrix[row, column] = self.__normals_block(epoch_count, row, column)

        return NormalEquations(normals_matrix, right_hand_side, observation_square_sum, observation_count)

    def covariance_function(self, maximum_lag):
        """
        Compute the covariance function of the AR model sequence.

        Parameters
        ----------
        maximum_lag : int
            maximum lag up to which to compute the covariance function.

        Returns
        -------
        covariance_func : list of ndarray
            vector valued covariance function of the AR sequence with increasing lags
        """
        normals = self.normal_equations(max(maximum_lag + 1, self.maximum_order + 1))
        normals.compute_covariance(sparse=False)

        return [normals.matrix[0, k] for k in range(maximum_lag + 1)]


class BlockMatrix:
    """
    Implementation of a sparse rectangular block matrix.
    """
    def __init__(self, row_index, column_index):

        self.shape = (len(row_index) - 1, len(column_index) - 1)
        self.__row_index = row_index
        self.__column_index = column_index
        self.__data = np.empty(self.shape, dtype=np.ndarray)
        self.__is_nonzero = np.full(self.shape, False, dtype=bool)

    def copy(self):
        """Deep copy of BlockMatrix"""
        output = BlockMatrix(self.__row_index, self.__column_index)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.__is_nonzero[i, j]:
                    output[i, j] = self.__data[i, j]

        return output

    @staticmethod
    def compute_block_index(array_shape, block_size):
        """
        Compute the row and column block index given an array shape and block size.

        Parameters
        ----------
        array_shape : tuple
            2-tuple containing row and column count
        block_size : int
            target block size

        Returns
        -------
        row_index : ndarray
            index bounds for block rows
        column_index  : ndarray
            index bounds for block columns
        """
        row_index = [0]
        while row_index[-1] < array_shape[0]:
            row_index.append(min(array_shape[0], row_index[-1] + block_size))

        column_index = [0]
        while column_index[-1] < array_shape[1]:
            column_index.append(min(array_shape[1], column_index[-1] + block_size))

        return np.array(row_index), np.array(column_index)

    @staticmethod
    def from_array(array, row_index, column_index):
        """
        Create a block matrix from a 2D ndarray. The contents of the array are copied.

        Parameters
        ----------
        array : ndarray(m, n)
            array from which the block matrix is generated
        row_index : list, tuple
            index bounds for block rows
        column_index  : list, tuple
            index bounds for block columns
        """
        if not isinstance(array, np.ndarray):
            raise ValueError('array must be of type ' + str(np.ndarray))
        if array.ndim != 2:
            raise ValueError('array must be a two-dimensional ' + str(np.ndarray))
        if row_index[-1] != array.shape[0]:
            raise ValueError("mismatch in array shape in dimension 0 and row block index")
        if column_index[-1] != array.shape[1]:
            raise ValueError("mismatch in array shape in dimension 1 and column block index")

        array_copy = array.copy()

        block_matrix = BlockMatrix(row_index, column_index)
        for row in range(len(row_index) - 1):
            for column in range(len(column_index) - 1):
                if np.count_nonzero(array_copy[row_index[row]:row_index[row + 1], column_index[column]:column_index[column + 1]]):
                    block_matrix[row, column] = array_copy[row_index[row]:row_index[row + 1], column_index[column]:column_index[column + 1]]
                    block_matrix.__is_nonzero[row, column] = True

        return block_matrix

    def to_array(self):
        """
        Create a 2D ndarray from the BlockMatrix.

        Returns
        -------
        array : ndarray(m, n)
            ndarray representation of the BlockMatrix
        """
        array = np.zeros((self.__row_index[-1], self.__column_index[-1]))
        for row in range(self.shape[0]):
            for column in range(self.shape[1]):
                if self.__is_nonzero[row, column]:
                    array[self.__row_slice(row), self.__column_slice(column)] = self.__data[row, column]

        return array

    def __check_bounds(self, i, j):
        """
        Check of block row and block column indices are within bounds.

        Parameters
        ---------
        i : int
            block row index
        j : int
            block column index

        """
        if i > self.shape[0]:
            raise IndexError("block index {0} is out of bounds for axis 0 with size {1}".format(i, self.shape[0]))
        if j > self.shape[1]:
            raise IndexError("block index {0} is out of bounds for axis 1 with size {1}".format(j, self.shape[1]))

    def __block_shape(self, i, j):
        """
        Return the expected shape of block :math:`A_{ij}`.

        Parameters
        ----------
        i : int
            block row index
        j : int
            block column index

        Returns
        -------
        shape : tuple
            2-element tuple with row and column count
        """
        return self.__row_index[i + 1] - self.__row_index[i], self.__column_index[j + 1] - self.__column_index[j]

    def __row_slice(self, i):
        """
        Return an index slice for all rows represented by block row i.

        Parameters
        ----------
        i : int
            block row index

        Returns
        -------
        s : slice
            slice for all rows represented by block row i
        """
        return slice(self.__row_index[i], self.__row_index[i + 1], 1)

    def __column_slice(self, i):
        """
        Return an index slice for all columns represented by block column i.

        Parameters
        ----------
        i : int
            block column index

        Returns
        -------
        s : slice
            slice for all columns represented by block column i
        """
        return slice(self.__column_index[i], self.__column_index[i + 1], 1)

    def __check_item(self, i, j, item):
        """
        Check if an ndarray is compatible with block :math:`A_{ij}`.

        Parameters
        ----------
        i : int
            block row index
        j : int
            block column index
        item : ndarray
            ndarray to be assigned to block i, j

        Raises
        ------
        ValueError:
            if dimension or shape of item is not compatible
        """
        if not isinstance(item, np.ndarray):
            raise ValueError('Block matrix item must be of type ' + str(np.ndarray))
        if item.ndim != 2:
            raise ValueError('Block matrix item must be a two-dimensional ' + str(np.ndarray))
        if item.shape != self.__block_shape(i, j):
            raise ValueError('Block matrix item at position ({0:d}, {1:d}) must be of size ({2:d}, {3:d}). '
                             'Got ({4:d}, {5:d}).'.format(i, j, self.__row_index[i], self.__row_index[j],
                                                          item.shape[0], item.shape[1]))

    def __setitem__(self, key, value):
        """
        Assign a 2D ndarray to block :math:`A_{ij}`.

        Parameters
        ----------
        key : tuple
            2-element tuple with row and column index
        value : ndarray
            view of block i, j
        """
        if not isinstance(key, tuple) and len(key) != 2:
            raise IndexError("Indices to block matrix must be tuples of length 2")

        self.__check_bounds(key[0], key[1])
        self.__check_item(key[0], key[1], value)

        self.__data[key] = value.copy()
        self.__is_nonzero[key] = True

    def __getitem__(self, key):
        """
        Return a view of block :math:`A_{ij}`.

        Parameters
        ----------
        key : tuple
            2-element tuple with row and column index

        Returns
        -------
        block : ndarray
            view of block i, j
        """
        if not isinstance(key, tuple) and len(key) != 2:
            raise IndexError("Indices to block matrix must be tuples of length 2")

        self.__check_bounds(key[0], key[1])

        return self.__data[key]

    def __matmul__(self, other):
        """
        Compute the matrix product :math:`\mathbf{C} = \mathbf{A}\mathbf{B}`.

        Parameters
        ----------
        other : BlockMatrix
            matrix B

        Returns
        -------
        result : BlockMatrix
            matrix C
        """
        if not isinstance(other, BlockMatrix):
            raise ValueError("Matrix multiplication not implemented for type {0}".format(type(other)))

        result = BlockMatrix(self.__row_index, other.__column_index)

        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                has_nonzero_products = np.any([self.__is_nonzero[i, k] and other.__is_nonzero[k, j]
                                               for k in range(self.shape[1])])

                if has_nonzero_products:
                    result.__set_block(i, j)
                    for k in range(self.shape[1]):
                        if self.__is_nonzero[i, k] and other.__is_nonzero[k, j]:
                            result[i, j] += self.__data[i, k] @ other.__data[k, j]

        return result

    def __set_block(self, i, j):
        """
        Initialize block :math:`A_{ij}` with zeros of appropriate size.

        Parameters
        ----------
        i : int
            block row index
        j : int
            block column index
        """
        if self.__data[i, j] is None:
            self.__data[i, j] = np.zeros(self.__block_shape(i, j))
            self.__is_nonzero[i, j] = True

    def cholesky(self):
        """
        Compute the Cholesky factorization :math:`\mathbf{N} = \mathbf{W}^T\mathbf{W}`. The block matrix is
        assumed to be symmetric and positive definite. The operation is computed in-place and only the upper triangle
        is referenced. After successful termination, the block matrix holds the upper triangular cholesky factor
        :math:`\mathbf{W}`.
        """
        for row in range(self.shape[0]):

            for r in range(row):
                for c in range(row, self.shape[1]):
                    if self.__is_nonzero[r, row] and self.__is_nonzero[r, c]:
                        self.__set_block(row, c)
                        self.__data[row, c] -= self.__data[r, row].T @ self.__data[r, c]

            self.__data[row, row] = la.cholesky(self.__data[row, row], overwrite_a=True, lower=False)
            for column in range(row + 1, self.shape[1]):
                if self.__is_nonzero[row, column]:
                    self.__data[row, column] = la.solve_triangular(self.__data[row, row], self.__data[row, column],
                                                                   trans='T', overwrite_b=True, lower=False)

    def multiply_triangular(self, b, transpose=False):
        """
        Compute the matrix product :math:`\mathbf{v} = \mathbf{W} \cdot \mathbf{b}` or
        :math:`\mathbf{v} = \mathbf{W}^T \cdot \mathbf{b}`. The block matrix is assumed to be upper triangular.

        Parameters
        ----------
        b : ndarray(m, n)
            multiplicator as 2D ndarray
        transpose : Bool
            Boolean flag whether to compute :math:`\mathbf{v} = \mathbf{W} \cdot \mathbf{b}` or
            :math:`\mathbf{v} = \mathbf{W}^T \cdot \mathbf{b}` (Default: False)

        Returns
        -------
        v : ndarray(m, n)
            multiplication result as 2D ndarray.
        """
        v = np.zeros(b.shape)

        if transpose:
            for i in range(self.shape[0]):
                for j in range(i + 1):
                    if self.__is_nonzero[j, i]:
                        v[self.__row_slice(i), :] = self.__data[j, i].T @ b[self.__row_slice(j), :]
        else:
            for i in range(self.shape[0]):
                for j in range(i, self.shape[1]):
                    if self.__is_nonzero[i, j]:
                        v[self.__row_slice(i), :] += self.__data[i, j] @ b[self.__row_slice(j), :]

        return v

    def multiply_symmetric(self, b):
        """
        Compute the matrix product :math:`\mathbf{v} = \mathbf{N} \cdot \mathbf{b}`. The block matrix is assumed to be symmetric.
        Only the upper triangle is accessed.

        Parameters
        ----------
        b : ndarray(m, n)
            multiplicator as 2D ndarray

        Returns
        -------
        v : ndarray(m, n)
            multiplication result as 2D ndarray.
        """
        v = np.zeros(b.shape)
        for i in range(self.shape[0]):
            if self.__is_nonzero[i, i]:
                v[self.__row_slice(i)] += self.__data[i, i] @ b[self.__row_slice(i)]
            for j in range(i + 1, self.shape[1]):
                if self.__is_nonzero[i, j]:
                    v[self.__row_slice(i)] += self.__data[i, j] @ b[self.__row_slice(j)]
                    v[self.__row_slice(j)] += self.__data[i, j].T @ b[self.__row_slice(i)]

        return v

    def solve_triangular(self, b, transpose=False):
        """
       Solve the system of equations :math:`\mathbf{W} \cdot \mathbf{x} = \mathbf{b}` or
       :math:`\mathbf{W}^T \cdot \mathbf{x} = \mathbf{b}`. The block matrix is assumed to be upper triangular and
       invertible.

       Parameters
       ----------
       b : ndarray(m, n)
           right-hand-side as 2D ndarray
       transpose : Bool
           Boolean flag whether to compute :math:`\mathbf{W} \cdot \mathbf{x} = \mathbf{b}` or
           :math:`\mathbf{W}^T \cdot \mathbf{x} = \mathbf{b}` (Default: False)

       Returns
       -------
       x : ndarray(m, n)
           solution as 2D ndarray.
       """
        b_copy = np.atleast_2d(b.copy())
        x = np.zeros(b_copy.shape)

        if transpose:
            for row in range(self.shape[0]):

                for column in range(row):
                    if self.__is_nonzero[column, row]:
                        b_copy[self.__row_slice(row), :] -= self.__data[column, row].T @ x[self.__row_slice(column), :]

                x[self.__row_slice(row), :] = la.solve_triangular(self.__data[row, row],
                                                                  b_copy[self.__row_slice(row), :],
                                                                  trans='T', overwrite_b=False, lower=False)
        else:
            for row in range(self.shape[0] - 1, -1, -1):

                for column in range(self.shape[0] - 1, row, -1):
                    if self.__is_nonzero[row, column]:
                        b_copy[self.__row_slice(row), :] -= self.__data[row, column] @ x[self.__row_slice(column), :]

                x[self.__row_slice(row), :] = la.solve_triangular(self.__data[row, row],
                                                                  b_copy[self.__row_slice(row), :],
                                                                  trans='N', overwrite_b=False, lower=False)

        return x

    def sparse_inverse(self):
        """
        Compute the sparse inverse :math:`\mathbf{N}^{-1} = \mathbf{W}^{-1}\mathbf{W}^{-T}` of the block matrix based
        on the cholesky factor :math:`\mathbf{W}`. This method assumes that the block matrix holds the upper triangular
        Cholesky factor :math:`\mathbf{W}`. The operation is computed in-place and only the upper triangle is
        referenced. The sparsity structure of the Cholesky factor is retained.
        """
        for i in range(self.shape[0] - 1, -1, -1):

            temporary_row = np.empty(self.shape[0] - i - 1, dtype=np.ndarray)
            for k in range(i + 1, self.shape[1]):
                if self.__is_nonzero[i, k]:
                    temporary_row[k - i - 1] = la.solve_triangular(self.__data[i, i], self.__data[i, k],
                                                                   trans='N', overwrite_b=True, lower=False)
                    self.__data[i, k] = np.zeros(self.__data[i, k].shape)

            self.__data[i, i] = la.inv(self.__data[i, i].T @ self.__data[i, i])

            for j in range(self.shape[0] - 1, i - 1, -1):
                if self.__is_nonzero[i, j]:
                    for k in range(i + 1, self.shape[0]):
                        if self.__is_nonzero[min(k, j), max(k, j)] and temporary_row[k - i - 1] is not None:
                            matrix_block = self.__data[k, j] if k < j else self.__data[j, k].T
                            self.__data[i, j] -= temporary_row[k - i - 1] @ matrix_block

    def inverse(self):
        """
        Compute the inverse of the block matrix :math:`\mathbf{N}^{-1} = \mathbf{W}^{-1}\mathbf{W}^{-T}`
        based on the cholesky factor :math:`\mathbf{W}`. This method assumes that the block matrix holds the upper
        triangular Cholesky factor :math:`\mathbf{W}`. The operation is computed in-place and only the upper triangle is
        referenced. In general, the resulting inverse will be fully populated.
        """
        for j in range(self.shape[0] - 1, -1, -1):
            self.__data[j, j] = la.inv(self.__data[j, j], overwrite_a=True)

            for i in range(j - 1, -1, -1):
                if self.__is_nonzero[i, j]:
                    self.__data[i, j] = self.__data[i, j] @ self.__data[j, j]

                for k in range(i + 1, j):
                    if self.__is_nonzero[i, k] and self.__is_nonzero[k, j]:
                        self.__set_block(i, j)
                        self.__data[i, j] += self.__data[i, k] @ self.__data[k, j]

                if self.__is_nonzero[i, j]:
                    self.__data[i, j] = -la.solve_triangular(self.__data[i, i], self.__data[i, j],
                                                             overwrite_b=False, lower=False)

        for i in range(self.shape[0]):

            self.__data[i, i] = self.__data[i, i] @ self.__data[i, i].T
            for j in range(i + 1, self.shape[0]):
                if self.__is_nonzero[i, j]:
                    self.__data[i, i] += self.__data[i, j] @ self.__data[i, j].T
                    self.__data[i, j] = self.__data[i, j] @ self.__data[j, j].T

                for k in range(j + 1, self.shape[0]):
                    if self.__is_nonzero[i, k] and self.__is_nonzero[j, k]:
                        self.__set_block(i, j)
                        self.__data[i, j] += self.__data[i, k] @ self.__data[j, k].T

    @property
    def is_nonzero(self, row, column):
        """Returns whether block (row, column) is non-zero."""
        return self.__is_nonzero[row, column]

    def _scale(self, value):
        """Scale whole matrix with a factor."""
        for row in range(self.shape[0]):
            for column in range(self.shape[1]):
                if self.__is_nonzero[row, column]:
                    self.__data[row, column] *= value

    def _axpy(self, factor, other):
        """Perform self += factor * other."""
        for row in range(self.shape[0]):
            for column in range(self.shape[1]):
                if self.__is_nonzero[row, column] and other.__is_nonzero[row, column]:
                    self.__data[row, column] += other[row, column] * factor
                elif not self.__is_nonzero[row, column] and other.__is_nonzero[row, column]:
                    self[row, column] = other[row, column] * factor

    def diag(self):
        """Return copy of main diagonal."""
        d = np.zeros(min(self.__row_index[-1], self.__column_index[-1]))
        for idx in range(min(len(self.__row_index), len(self.__column_index)) - 1):
            if self.__is_nonzero[idx, idx]:
                d[self.__row_index[idx]:self.__row_index[idx + 1]] = np.diag(self.__data[idx, idx])

        return d


class NormalEquations:
    """
    Class representation of a system of normal equations.

    Parameters
    ----------
    normal_matrix : BlockMatrix
        normal equation coefficient matrix
    right_hand_side : ndarray(n, 1)
        normal equation right hand side
    observation_square_sum : float
        weighted square sum ob observations :math:`\mathbf{l}^T\mathbf{P}\mathbf{l}`
    observation_count : int
        observation count
    """
    def __init__(self, normal_matrix, right_hand_side, observation_square_sum, observation_count):

        self.matrix = normal_matrix
        self.right_hand_side = right_hand_side
        self.observation_square_sum = observation_square_sum
        self.observation_count = observation_count
        self.status = 'normal_matrix'

    def __cholesky(self):
        """
        Compute the Cholesky decomposition of the coefficient matrix.
        """
        if self.status == 'cholesky_factor':
            pass
        elif self.status == 'normal_matrix':
            self.matrix.cholesky()
            self.status = 'cholesky_factor'
        else:
            raise ValueError('Cholesky factor can only be computed from the normal matrix')

    def solve(self):
        """
        Solve the system of normal equations. The coefficient matrix now holds the upper triangular Cholesky
        factor.

        Returns
        -------
        x : ndarray(n, 1)
            estimated parameter vector
        """
        self.__cholesky()

        h = self.matrix.solve_triangular(self.right_hand_side, transpose=True)
        xi = np.random.randint(0, 2, size=(h.shape[0], 100))
        xi[xi == 0] = -1
        x = self.matrix.solve_triangular(np.hstack((h, xi)))
        self.monte_carlo_vectors = x[:, 1:]

        return x[:, 0:1]

    def redundancy(self, combined_normals, variance_factor):
        """
        Compute the redudancy given a combined system of normal equations and the variance factor.

        Parameters
        ----------
        combined_normals : NormalEquation
            accumulated normal equation system
        variance_factor : float
            corresponding variance factor

        Returns
        -------
        r : float
            redudancy of the normal equation
        """
        estimated_trace = np.trace(combined_normals.monte_carlo_vectors.T @ (self.matrix.multiply_symmetric(combined_normals.monte_carlo_vectors))) / combined_normals.monte_carlo_vectors.shape[1]

        return (self.observation_count - estimated_trace / variance_factor).squeeze()

    def residual_square_sum(self, solution):
        """
        Compute the residual square sum for a given solution.

        Parameters
        ----------
        solution : ndarray
            ndarray containing the solution vector

        Returns
        -------
        ePe : float
            residual square sum
        """
        Nx = self.matrix.multiply_symmetric(solution)
        return (self.observation_square_sum - 2 * np.sum(self.right_hand_side * solution) + np.sum(solution * Nx)).squeeze()

    def posterior_sigma(self, solution):
        """
        Compute the estimated posterior sigma from a given solution.

        Parameters
        ----------
        solution : ndarray(n, 1)
            estimated parameter vector

        Returns
        -------
        sigma : float
            estimated posterior sigma
        """
        Wx = self.matrix.multiply_triangular(solution)
        ePe = self.observation_square_sum - 2 * np.sum(self.right_hand_side * solution) + np.sum(Wx * Wx)

        return np.sqrt(ePe / (self.observation_count - solution.shape[0])).squeeze()

    def compute_covariance(self, sparse=True):
        """
        Compute the (sparse) inverse of the normal equation coefficient matrix.

        Parameters
        ----------
        sparse : bool
            flag whether to compute the sparse (incomplete) or dense inverse
        """
        self.__cholesky()

        if sparse:
            self.matrix.sparse_inverse()
        else:
            self.matrix.inverse()

        self.status = 'covariance_matrix'

    def to_array(self):
        """
        Return normal equation system as ndarrays. Sparsity is not taken into account.

        Returns
        -------
        N : ndarray
            normal equation coefficient matrix
        n : ndarray
            right hand side
        observation_square_sum : float
            observation square sum
        observation_count : int
            observation count
        """
        return self.matrix.to_array(), self.right_hand_side, self.observation_square_sum, self.observation_count


class TikhonovRegularization(NormalEquations):
    """
    Class representation of the normal equation system corresponding to a Thikonov regularization.

    Parameters
    ----------
    regularization_vector : ndarray
        1d ndarray containing the diagonal of the regularization matrix
    block_index : array_like
        block index for construction of the normal equation matrix
    right_hand_side : ndarray or None
        right hand side (bias vector) of the normal equation system (if None, a zero vector is generated)
    """
    def __init__(self, regularization_vector, block_index, right_hand_side=None):

        if right_hand_side is None:
            right_hand_side = np.zeros((block_index[-1], 1))
            lPl = 0
        else:
            lPl = np.sum(right_hand_side**2 * regularization_vector[:, np.newaxis])
            right_hand_side = right_hand_side * regularization_vector[:, np.newaxis]

        matrix = BlockMatrix(block_index, block_index)
        for i in range(matrix.shape[0]):
            matrix[i, i] = np.diag(regularization_vector[block_index[i]:block_index[i + 1]])

        super(TikhonovRegularization, self).__init__(matrix, right_hand_side, lPl, right_hand_side.size)


def accumulate_normals(normal_equations, variance_factors):
    """
    Accumulate multiple normal equations with given variance factors.

    Parameters
    ----------
    normal_equations : list of NormalEquation instances
        list or tuple of NormalEquation objects
    variance_factors : array_like
        container of corresponding variance factors

    Returns
    -------
    combined_normals : NormalEquation
        accumulated normal equation system
    """
    output_matrix = normal_equations[0].matrix.copy()
    output_matrix._scale(1 / variance_factors[0])
    output_rhs = normal_equations[0].right_hand_side.copy() / variance_factors[0]
    lPl = normal_equations[0].observation_square_sum / variance_factors[0]
    obs_count = normal_equations[0].observation_count

    for k in range(1, len(normal_equations)):
        output_matrix._axpy(1 / variance_factors[k], normal_equations[k].matrix)
        output_rhs += normal_equations[k].right_hand_side.copy() / variance_factors[k]
        lPl += normal_equations[k].observation_square_sum / variance_factors[k]
        obs_count += normal_equations[k].observation_count

    return NormalEquations(output_matrix, output_rhs, lPl, obs_count)


def compute_variance_factors(normal_equations, combined_normals, solution, variance_factors):
    """
    Compute the variance factors for a list of normal equations given the combined normals, solution and variance factors.

    Parameters
    ----------
    normal_equations : list of NormalEquation instances
        list or tuple of NormalEquation objects
    combined_normals : NormalEquation
        accumulated normal equation system
    solution : ndarray
        2d ndarray containing the solution vector
    variance_factors : array_like
        container of prior variance factors

    Returns
    -------
    estimated_variance_factors : ndarray
        ndarray containing the estimated variance factors
    """
    vc = []
    for normals, sigma2 in zip(normal_equations, variance_factors):

        ePe = normals.residual_square_sum(solution)
        r = normals.redundancy(combined_normals, sigma2)
        vc.append(ePe / r)

    return np.array(vc)
