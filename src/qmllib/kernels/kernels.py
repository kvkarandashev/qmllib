
#

#






#


#








import numpy as np
from numpy import empty, asfortranarray, ascontiguousarray, zeros

from .fkernels import fgaussian_kernel
from .fkernels import flaplacian_kernel
from .fkernels import fget_vector_kernels_gaussian
from .fkernels import fget_vector_kernels_laplacian


def laplacian_kernel(A, B, sigma):
    """ Calculates the Laplacian kernel matrix K, where K_ij:

            K_ij = exp(-1 * sigma**(-1) * || A_i - B_j ||_1)

        Where A_i and B_j are descriptor vectors.

        K is calculated using an OpenMP parallel Fortran routine.

        NOTE: A and B need not be input as Fortran contiguous arrays.

        Arguments:
        ==============
        A -- np.array of np.array of descriptors.
        B -- np.array of np.array of descriptors.
        sigma -- The value of sigma in the kernel matrix.

        Returns:
        ==============
        K -- The Laplacian kernel matrix.
    """

    na = A.shape[0]
    nb = B.shape[0]

    K = empty((na, nb), order='F')

    # Note: Transposed for Fortran
    flaplacian_kernel(A.T, na, B.T, nb, K, sigma)

    return K


def gaussian_kernel(A, B, sigma):
    """ Calculates the Gaussian kernel matrix K, where K_ij:

            K_ij = exp(-0.5 * sigma**(-2) * || A_i - B_j ||_2)

        Where A_i and B_j are descriptor vectors.

        K is calculated using an OpenMP parallel Fortran routine.

        NOTE: A and B need not be input as Fortran contiguous arrays.

        Arguments:
        ==============
        A -- np.array of np.array of descriptors.
        B -- np.array of np.array of descriptors.
        sigma -- The value of sigma in the kernel matrix.

        Returns:
        ==============
        K -- The Gaussian kernel matrix.
    """

    na = A.shape[0]
    nb = B.shape[0]

    K = empty((na, nb), order='F')

    # Note: Transposed for Fortran
    fgaussian_kernel(A.T, na, B.T, nb, K, sigma)

    return K
