from typing import List

import numpy as np
from numpy import ndarray

from qmllib.utils.alchemy import PTP

from .farad_kernels import (
    fget_atomic_kernels_arad,
    fget_atomic_symmetric_kernels_arad,
    fget_global_kernels_arad,
    fget_global_symmetric_kernels_arad,
    fget_local_kernels_arad,
    fget_local_symmetric_kernels_arad,
)


def getAngle(sp: ndarray, norms: ndarray) -> ndarray:
    epsilon = 10.0 * np.finfo(float).eps
    angles = np.zeros(sp.shape)
    mask1 = np.logical_and(np.abs(sp - norms) > epsilon, np.abs(norms) > epsilon)
    angles[mask1] = np.arccos(np.clip(sp[mask1] / norms[mask1], -1.0, 1.0))
    return angles


def generate_arad(
    nuclear_charges: ndarray, coordinates: ndarray, size: int = 23, cut_distance: float = 5.0
) -> ndarray:
    """Generates a representation for the ARAD kernel module.

    :param coordinates: Input coordinates.
    :type coordinates: numpy array
    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param size: Max number of atoms in representation.
    :type size: integer
    :param cut_distance: Spatial cut-off distance.
    :type cut_distance: float
    :return: ARAD representation, shape = (size,5,size).
    :rtype: numpy array
    """

    # PBC is not supported by the kernels currently
    cell = None
    maxAts = size
    maxMolSize = size

    coords = coordinates
    occupationList = nuclear_charges
    cut = cut_distance

    L = coords.shape[0]
    occupationList = np.asarray(occupationList)
    M = np.zeros((maxMolSize, 5, maxAts))

    if cell is not None:
        coords = np.dot(coords, cell)
        nExtend = (np.floor(cut / np.linalg.norm(cell, 2, axis=0)) + 1).astype(int)
        for i in range(-nExtend[0], nExtend[0] + 1):
            for j in range(-nExtend[1], nExtend[1] + 1):
                for k in range(-nExtend[2], nExtend[2] + 1):
                    if i == -nExtend[0] and j == -nExtend[1] and k == -nExtend[2]:
                        coordsExt = coords + i * cell[0, :] + j * cell[1, :] + k * cell[2, :]
                        occupationListExt = occupationList.copy()
                    else:
                        occupationListExt = np.append(occupationListExt, occupationList)
                        coordsExt = np.append(
                            coordsExt,
                            coords + i * cell[0, :] + j * cell[1, :] + k * cell[2, :],
                            axis=0,
                        )

    else:
        coordsExt = coords.copy()
        occupationListExt = occupationList.copy()

    M[:, 0, :] = 1e100

    for i in range(L):
        # Calculate Distance
        cD = coordsExt[:] - coords[i]
        ocExt = np.asarray([PTP[o] for o in occupationListExt])

        # Obtaining angles
        sp = np.sum(cD[:, np.newaxis] * cD[np.newaxis, :], axis=2)
        D1 = np.sqrt(np.sum(cD**2, axis=1))
        D2 = D1[:, np.newaxis] * D1[np.newaxis, :]
        angs = getAngle(sp, D2)

        # Obtaining cos and sine terms
        cosAngs = np.cos(angs) * (1.0 - np.sin(np.pi * D1[np.newaxis, :] / (2.0 * cut)))
        sinAngs = np.sin(angs) * (1.0 - np.sin(np.pi * D1[np.newaxis, :] / (2.0 * cut)))

        args = np.argsort(D1)

        D1 = D1[args]

        ocExt = np.asarray([ocExt[l] for l in args])

        sub_indices = np.ix_(args, args)
        cosAngs = cosAngs[sub_indices]
        sinAngs = sinAngs[sub_indices]

        args = np.where(D1 < cut)[0]

        D1 = D1[args]

        ocExt = np.asarray([ocExt[l] for l in args])

        sub_indices = np.ix_(args, args)
        cosAngs = cosAngs[sub_indices]
        sinAngs = sinAngs[sub_indices]

        norm = np.sum(1.0 - np.sin(np.pi * D1[np.newaxis, :] / (2.0 * cut)))
        M[i, 0, : len(D1)] = D1
        M[i, 1, : len(D1)] = ocExt[:, 0]
        M[i, 2, : len(D1)] = ocExt[:, 1]
        M[i, 3, : len(D1)] = np.sum(cosAngs, axis=1) / norm
        M[i, 4, : len(D1)] = np.sum(sinAngs, axis=1) / norm

    return M


def get_global_kernels_arad(
    X1: ndarray,
    X2: ndarray,
    sigmas: List[float],
    width: float = 0.2,
    cut_distance: float = 5.0,
    r_width: float = 1.0,
    c_width: float = 0.5,
) -> ndarray:
    """Calculates the global Gaussian kernel matrix K for atomic ARAD
    descriptors for a list of different sigmas. Each kernel element
    is the sum of all kernel elements between pairs of atoms in two molecules.

    K is calculated using an OpenMP parallel Fortran routine.

    :param X1: ARAD descriptors for molecules in set 1.
    :type X1: numpy array
    :param X2: Array of ARAD descriptors for molecules in set 2.
    :type X2: numpy array
    :param sigmas: List of sigmas for which to calculate the Kernel matrices.
    :type sigmas: list

    :return: The kernel matrices for each sigma - shape (number_sigmas, number_molecules1, number_molecules2)
    :rtype: numpy array
    """

    amax = X1.shape[1]

    if not X1.shape[3] == amax:
        raise ValueError("Check ARAD decriptor sizes")
    if not X2.shape[1] == amax:
        raise ValueError("Check ARAD decriptor sizes")
    if not X2.shape[3] == amax:
        raise ValueError("Check ARAD decriptor sizes")

    nm1 = X1.shape[0]
    nm2 = X2.shape[0]

    N1 = np.empty(nm1, dtype=np.int32)
    Z1_arad = np.zeros((nm1, amax, 2))
    for i in range(nm1):
        N1[i] = len(np.where(X1[i, :, 2, 0] > 0)[0])
        Z1_arad[i] = X1[i, :, 1:3, 0]

    N2 = np.empty(nm2, dtype=np.int32)
    Z2_arad = np.zeros((nm2, amax, 2))
    for i in range(nm2):
        N2[i] = len(np.where(X2[i, :, 2, 0] > 0)[0])
        Z2_arad[i] = X2[i, :, 1:3, 0]

    sigmas = np.array(sigmas)
    nsigmas = sigmas.size

    return fget_global_kernels_arad(
        X1,
        X2,
        Z1_arad,
        Z2_arad,
        N1,
        N2,
        sigmas,
        nm1,
        nm2,
        nsigmas,
        width,
        cut_distance,
        r_width,
        c_width,
    )


def get_global_symmetric_kernels_arad(
    X1: ndarray,
    sigmas: List[float],
    width: float = 0.2,
    cut_distance: float = 5.0,
    r_width: float = 1.0,
    c_width: float = 0.5,
) -> ndarray:
    """Calculates the global Gaussian kernel matrix K for atomic ARAD
    descriptors for a list of different sigmas. Each kernel element
    is the sum of all kernel elements between pairs of atoms in two molecules.

    K is calculated using an OpenMP parallel Fortran routine.

    :param X1: ARAD descriptors for molecules in set 1.
    :type X1: numpy array
    :param sigmas: List of sigmas for which to calculate the Kernel matrices.
    :type sigmas: list

    :return: The kernel matrices for each sigma - shape (number_sigmas, number_molecules1, number_molecules1)
    :rtype: numpy array
    """

    nm1 = X1.shape[0]
    amax = X1.shape[1]

    N1 = np.empty(nm1, dtype=np.int32)
    Z1_arad = np.zeros((nm1, amax, 2))
    for i in range(nm1):
        N1[i] = len(np.where(X1[i, :, 2, 0] > 0)[0])
        Z1_arad[i] = X1[i, :, 1:3, 0]

    sigmas = np.array(sigmas)
    nsigmas = sigmas.size

    return fget_global_symmetric_kernels_arad(
        X1, Z1_arad, N1, sigmas, nm1, nsigmas, width, cut_distance, r_width, c_width
    )


def get_local_kernels_arad(
    X1: ndarray,
    X2: ndarray,
    sigmas: List[float],
    width: float = 0.2,
    cut_distance: float = 5.0,
    r_width: float = 1.0,
    c_width: float = 0.5,
) -> ndarray:
    """Calculates the Gaussian kernel matrix K for atomic ARAD
    descriptors for a list of different sigmas. Each kernel element
    is the sum of all kernel elements between pairs of atoms in two molecules.

    K is calculated using an OpenMP parallel Fortran routine.

    :param X1: ARAD descriptors for molecules in set 1.
    :type X1: numpy array
    :param X2: Array of ARAD descriptors for molecules in set 2.
    :type X2: numpy array
    :param sigmas: List of sigmas for which to calculate the Kernel matrices.
    :type sigmas: list

    :return: The kernel matrices for each sigma - shape (number_sigmas, number_molecules1, number_molecules2)
    :rtype: numpy array
    """

    amax = X1.shape[1]

    if not X1.shape[3] == amax:
        raise ValueError("Check ARAD decriptor sizes")
    if not X2.shape[1] == amax:
        raise ValueError("Check ARAD decriptor sizes")
    if not X2.shape[3] == amax:
        raise ValueError("Check ARAD decriptor sizes")

    nm1 = X1.shape[0]
    nm2 = X2.shape[0]

    N1 = np.empty(nm1, dtype=np.int32)
    Z1_arad = np.zeros((nm1, amax, 2))
    for i in range(nm1):
        N1[i] = len(np.where(X1[i, :, 2, 0] > 0)[0])
        Z1_arad[i] = X1[i, :, 1:3, 0]

    N2 = np.empty(nm2, dtype=np.int32)
    Z2_arad = np.zeros((nm2, amax, 2))
    for i in range(nm2):
        N2[i] = len(np.where(X2[i, :, 2, 0] > 0)[0])
        Z2_arad[i] = X2[i, :, 1:3, 0]

    sigmas = np.array(sigmas)
    nsigmas = sigmas.size

    return fget_local_kernels_arad(
        X1,
        X2,
        Z1_arad,
        Z2_arad,
        N1,
        N2,
        sigmas,
        nm1,
        nm2,
        nsigmas,
        width,
        cut_distance,
        r_width,
        c_width,
    )


def get_local_symmetric_kernels_arad(
    X1: ndarray,
    sigmas: List[float],
    width: float = 0.2,
    cut_distance: float = 5.0,
    r_width: float = 1.0,
    c_width: float = 0.5,
) -> ndarray:
    """Calculates the Gaussian kernel matrix K for atomic ARAD
    descriptors for a list of different sigmas. Each kernel element
    is the sum of all kernel elements between pairs of atoms in two molecules.

    K is calculated using an OpenMP parallel Fortran routine.

    :param X1: ARAD descriptors for molecules in set 1.
    :type X1: numpy array
    :param sigmas: List of sigmas for which to calculate the Kernel matrices.
    :type sigmas: list

    :return: The kernel matrices for each sigma - shape (number_sigmas, number_molecules1, number_molecules1)
    :rtype: numpy array
    """

    nm1 = X1.shape[0]
    amax = X1.shape[1]

    N1 = np.empty(nm1, dtype=np.int32)
    Z1_arad = np.zeros((nm1, amax, 2))
    for i in range(nm1):
        N1[i] = len(np.where(X1[i, :, 2, 0] > 0)[0])
        Z1_arad[i] = X1[i, :, 1:3, 0]

    sigmas = np.array(sigmas)
    nsigmas = sigmas.size

    return fget_local_symmetric_kernels_arad(
        X1, Z1_arad, N1, sigmas, nm1, nsigmas, width, cut_distance, r_width, c_width
    )


def get_atomic_kernels_arad(
    X1: ndarray,
    X2: ndarray,
    sigmas: List[float],
    width: float = 0.2,
    cut_distance: float = 5.0,
    r_width: float = 1.0,
    c_width: float = 0.5,
) -> ndarray:
    """Calculates the Gaussian kernel matrix K for atomic ARAD
    descriptors for a list of different sigmas. For atomic properties, e.g.
    partial charges, chemical shifts, etc.

    K is calculated using an OpenMP parallel Fortran routine.

    :param X1: ARAD descriptors for molecules in set 1. shape=(number_atoms,5,size)
    :type X1: numpy array
    :param X2: ARAD descriptors for molecules in set 1. shape=(number_atoms,5,size)
    :type X2: numpy array
    :param sigmas: List of sigmas for which to calculate the Kernel matrices.
    :type sigmas: list

    :return: The kernel matrices for each sigma - shape (number_sigmas, number_atoms1, number_atoms2)
    :rtype: numpy array
    """

    if not len(X1.shape) == 3:
        raise ValueError("Expected different shape")
    if not len(X2.shape) == 3:
        raise ValueError("Expected different shape")

    na1 = X1.shape[0]
    na2 = X2.shape[0]

    N1 = np.empty(na1, dtype=np.int32)
    N2 = np.empty(na2, dtype=np.int32)

    Z1_arad = np.zeros((na1, 2))
    Z2_arad = np.zeros((na2, 2))

    for i in range(na1):
        N1[i] = len(np.where(X1[i, 0, :] < cut_distance)[0])
        Z1_arad[i] = X1[i, 1:3, 0]

    for i in range(na2):
        N2[i] = len(np.where(X2[i, 0, :] < cut_distance)[0])
        Z2_arad[i] = X2[i, 1:3, 0]

    sigmas = np.array(sigmas)
    nsigmas = sigmas.size

    return fget_atomic_kernels_arad(
        X1,
        X2,
        Z1_arad,
        Z2_arad,
        N1,
        N2,
        sigmas,
        na1,
        na2,
        nsigmas,
        width,
        cut_distance,
        r_width,
        c_width,
    )


def get_atomic_symmetric_kernels_arad(
    X1: ndarray,
    sigmas: List[float],
    width: float = 0.2,
    cut_distance: float = 5.0,
    r_width: float = 1.0,
    c_width: float = 0.5,
) -> ndarray:
    """Calculates the Gaussian kernel matrix K for atomic ARAD
    descriptors for a list of different sigmas. For atomic properties, e.g.
    partial charges, chemical shifts, etc.

    K is calculated using an OpenMP parallel Fortran routine.

    :param X1: ARAD descriptors for molecules in set 1. shape=(number_atoms,5,size)
    :type X1: numpy array
    :param sigmas: List of sigmas for which to calculate the Kernel matrices.
    :type sigmas: list

    :return: The kernel matrices for each sigma - shape (number_sigmas, number_atoms1, number_atoms1)
    :rtype: numpy array
    """

    if not len(X1.shape) == 3:
        raise ValueError("Expected different shape")
    na1 = X1.shape[0]

    N1 = np.empty(na1, dtype=np.int32)
    Z1_arad = np.zeros((na1, 2))

    for i in range(na1):
        N1[i] = len(np.where(X1[i, 0, :] < cut_distance)[0])
        Z1_arad[i] = X1[i, 1:3, 0]

    sigmas = np.array(sigmas)
    nsigmas = sigmas.size

    return fget_atomic_symmetric_kernels_arad(
        X1, Z1_arad, N1, sigmas, na1, nsigmas, width, cut_distance, r_width, c_width
    )
