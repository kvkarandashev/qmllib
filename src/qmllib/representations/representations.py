import itertools
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import int64, ndarray

from qmllib.constants.periodic_table import NUCLEAR_CHARGE

from .facsf import (
    fgenerate_acsf,
    fgenerate_acsf_and_gradients,
    fgenerate_fchl_acsf,
    fgenerate_fchl_acsf_and_gradients,
)
from .frepresentations import (
    fgenerate_atomic_coulomb_matrix,
    fgenerate_bob,
    fgenerate_coulomb_matrix,
    fgenerate_eigenvalue_coulomb_matrix,
    fgenerate_local_coulomb_matrix,
    fgenerate_unsorted_coulomb_matrix,
)
from .slatm import get_boa, get_sbop, get_sbot


def vector_to_matrix(v):
    """Converts a representation from 1D vector to 2D square matrix.
    :param v: 1D input representation.
    :type v: numpy array
    :return: Square matrix representation.
    :rtype: numpy array
    """

    if not (np.sqrt(8 * v.shape[0] + 1) == int(np.sqrt(8 * v.shape[0] + 1))):
        raise ValueError("Can not make a square matrix.")

    n = v.shape[0]
    l = (-1 + int(np.sqrt(8 * n + 1))) // 2
    M = np.empty((l, l))

    index = 0
    for i in range(l):
        for j in range(l):
            if j > i:
                continue

            M[i, j] = v[index]
            M[j, i] = M[i, j]

            index += 1
    return M


def generate_coulomb_matrix(
    nuclear_charges: ndarray, coordinates: ndarray, size: int = 23, sorting: str = "row-norm"
) -> ndarray:
    """ Creates a Coulomb Matrix representation of a molecule.
        Sorting of the elements can either be done by ``sorting="row-norm"`` or ``sorting="unsorted"``.
        A matrix :math:`M` is constructed with elements

        .. math::

            M_{ij} =
              \\begin{cases}
                 \\tfrac{1}{2} Z_{i}^{2.4} & \\text{if } i = j \\\\
                 \\frac{Z_{i}Z_{j}}{\\| {\\bf R}_{i} - {\\bf R}_{j}\\|}       & \\text{if } i \\neq j
              \\end{cases},

        where :math:`i` and :math:`j` are atom indices, :math:`Z` is nuclear charge and
        :math:`\\bf R` is the coordinate in euclidean space.
        If ``sorting = 'row-norm'``, the atom indices are reordered such that

            :math:`\\sum_j M_{1j}^2 \\geq \\sum_j M_{2j}^2 \\geq ... \\geq \\sum_j M_{nj}^2`

        The upper triangular of M, including the diagonal, is concatenated to a 1D
        vector representation.

        If ``sorting = 'unsorted``, the elements are sorted in the same order as the input coordinates
        and nuclear charges.

        The representation is calculated using an OpenMP parallel Fortran routine.

        :param nuclear_charges: Nuclear charges of the atoms in the molecule
        :type nuclear_charges: numpy array
        :param coordinates: 3D Coordinates of the atoms in the molecule
        :type coordinates: numpy array
        :param size: The size of the largest molecule supported by the representation
        :type size: integer
        :param sorting: How the atom indices are sorted ('row-norm', 'unsorted')
        :type sorting: string

        :return: 1D representation - shape (size(size+1)/2,)
        :rtype: numpy array
    """

    if sorting == "row-norm":
        return fgenerate_coulomb_matrix(nuclear_charges, coordinates, size)

    elif sorting == "unsorted":
        return fgenerate_unsorted_coulomb_matrix(nuclear_charges, coordinates, size)

    else:
        raise ValueError("Unknown sorting scheme requested")


def generate_coulomb_matrix_atomic(
    nuclear_charges: ndarray,
    coordinates: ndarray,
    size: int = 23,
    sorting: str = "distance",
    central_cutoff: float = 1e6,
    central_decay: Union[float, int] = -1,
    interaction_cutoff: float = 1e6,
    interaction_decay: Union[float, int] = -1,
    indices: Optional[List[int]] = None,
) -> ndarray:
    """ Creates a Coulomb Matrix representation of the local environment of a central atom.
        For each central atom :math:`k`, a matrix :math:`M` is constructed with elements

        .. math::

            M_{ij}(k) =
              \\begin{cases}
                 \\tfrac{1}{2} Z_{i}^{2.4} \\cdot f_{ik}^2 & \\text{if } i = j \\\\
                 \\frac{Z_{i}Z_{j}}{\\| {\\bf R}_{i} - {\\bf R}_{j}\\|} \\cdot f_{ik}f_{jk}f_{ij} & \\text{if } i \\neq j
              \\end{cases},

        where :math:`i`, :math:`j` and :math:`k` are atom indices, :math:`Z` is nuclear charge and
        :math:`\\bf R` is the coordinate in euclidean space.

        :math:`f_{ij}` is a function that masks long range effects:

        .. math::

            f_{ij} =
              \\begin{cases}
                 1 & \\text{if } \\|{\\bf R}_{i} - {\\bf R}_{j} \\| \\leq r - \\Delta r \\\\
                 \\tfrac{1}{2} \\big(1 + \\cos\\big(\\pi \\tfrac{\\|{\\bf R}_{i} - {\\bf R}_{j} \\|
                    - r + \\Delta r}{\\Delta r} \\big)\\big)
                    & \\text{if } r - \\Delta r < \\|{\\bf R}_{i} - {\\bf R}_{j} \\| \\leq r - \\Delta r \\\\
                 0 & \\text{if } \\|{\\bf R}_{i} - {\\bf R}_{j} \\| > r
              \\end{cases},

        where the parameters ``central_cutoff`` and ``central_decay`` corresponds to the variables
        :math:`r` and :math:`\\Delta r` respectively for interactions involving the central atom,
        and ``interaction_cutoff`` and ``interaction_decay`` corresponds to the variables
        :math:`r` and :math:`\\Delta r` respectively for interactions not involving the central atom.

        if ``sorting = 'row-norm'``, the atom indices are ordered such that

            :math:`\\sum_j M_{1j}(k)^2 \\geq \\sum_j M_{2j}(k)^2 \\geq ... \\geq \\sum_j M_{nj}(k)^2`

        if ``sorting = 'distance'``, the atom indices are ordered such that

        .. math::

            \\|{\\bf R}_{1} - {\\bf R}_{k}\\| \\leq \\|{\\bf R}_{2} - {\\bf R}_{k}\\|
                \\leq ... \\leq \\|{\\bf R}_{n} - {\\bf R}_{k}\\|

        The upper triangular of M, including the diagonal, is concatenated to a 1D
        vector representation.

        The representation can be calculated for a subset by either specifying
        ``indices = [0,1,...]``, where :math:`[0,1,...]` are the requested atom indices,
        or by specifying ``indices = 'C'`` to only calculate central carbon atoms.

        The representation is calculated using an OpenMP parallel Fortran routine.

        :param nuclear_charges: Nuclear charges of the atoms in the molecule
        :type nuclear_charges: numpy array
        :param coordinates: 3D Coordinates of the atoms in the molecule
        :type coordinates: numpy array
        :param size: The size of the largest molecule supported by the representation
        :type size: integer
        :param sorting: How the atom indices are sorted ('row-norm', 'distance')
        :type sorting: string
        :param central_cutoff: The distance from the central atom, where the coulomb interaction
            element will be zero
        :type central_cutoff: float
        :param central_decay: The distance over which the the coulomb interaction decays from full to none
        :type central_decay: float
        :param interaction_cutoff: The distance between two non-central atom, where the coulomb interaction
            element will be zero
        :type interaction_cutoff: float
        :param interaction_decay: The distance over which the the coulomb interaction decays from full to none
        :type interaction_decay: float
        :param indices: Subset indices or atomtype
        :type indices: Nonetype/array/string


        :return: nD representation - shape (:math:`N_{atoms}`, size(size+1)/2)
        :rtype: numpy array
    """

    if indices is None:
        nindices = len(nuclear_charges)
        indices = np.arange(1, 1 + nindices, 1, dtype=int)
    elif isinstance(indices, str):
        if indices in NUCLEAR_CHARGE:
            indices = np.where(nuclear_charges == NUCLEAR_CHARGE[indices])[0] + 1
            nindices = indices.size
            if nindices == 0:
                return np.zeros((0, 0))

        else:
            raise ValueError("Unknown value %s given for 'indices' variable" % indices)
    else:
        indices = np.asarray(indices, dtype=int) + 1
        nindices = indices.size

    if sorting == "row-norm":
        return fgenerate_local_coulomb_matrix(
            indices,
            nindices,
            nuclear_charges,
            coordinates,
            nuclear_charges.size,
            size,
            central_cutoff,
            central_decay,
            interaction_cutoff,
            interaction_decay,
        )

    elif sorting == "distance":
        return fgenerate_atomic_coulomb_matrix(
            indices,
            nindices,
            nuclear_charges,
            coordinates,
            nuclear_charges.size,
            size,
            central_cutoff,
            central_decay,
            interaction_cutoff,
            interaction_decay,
        )

    else:
        raise ValueError("Unknown sorting scheme requested")


def generate_coulomb_matrix_eigenvalue(
    nuclear_charges: ndarray, coordinates: ndarray, size: int = 23
) -> ndarray:
    """ Creates an eigenvalue Coulomb Matrix representation of a molecule.
        A matrix :math:`M` is constructed with elements

        .. math::

            M_{ij} =
              \\begin{cases}
                 \\tfrac{1}{2} Z_{i}^{2.4} & \\text{if } i = j \\\\
                 \\frac{Z_{i}Z_{j}}{\\| {\\bf R}_{i} - {\\bf R}_{j}\\|}       & \\text{if } i \\neq j
              \\end{cases},

        where :math:`i` and :math:`j` are atom indices, :math:`Z` is nuclear charge and
        :math:`\\bf R` is the coordinate in euclidean space.
        The molecular representation of the molecule is then the sorted eigenvalues of M.
        The representation is calculated using an OpenMP parallel Fortran routine.

        :param nuclear_charges: Nuclear charges of the atoms in the molecule
        :type nuclear_charges: numpy array
        :param coordinates: 3D Coordinates of the atoms in the molecule
        :type coordinates: numpy array
        :param size: The size of the largest molecule supported by the representation
        :type size: integer

        :return: 1D representation - shape (size, )
        :rtype: numpy array
    """
    return fgenerate_eigenvalue_coulomb_matrix(nuclear_charges, coordinates, size)


def generate_bob(
    nuclear_charges: ndarray,
    coordinates: ndarray,
    atomtypes: ndarray,
    size: int = 23,
    asize: Dict[str, Union[int64, int]] = {"O": 3, "C": 7, "N": 3, "H": 16, "S": 1},
) -> ndarray:
    """Creates a Bag of Bonds (BOB) representation of a molecule.
    The representation expands on the coulomb matrix representation.
    For each element a bag (vector) is constructed for self interactions
    (e.g. ('C', 'H', 'O')).
    For each element pair a bag is constructed for interatomic interactions
    (e.g. ('CC', 'CH', 'CO', 'HH', 'HO', 'OO')), sorted by value.
    The self interaction of element :math:`I` is given by

        :math:`\\tfrac{1}{2} Z_{I}^{2.4}`,

    with :math:`Z_{i}` being the nuclear charge of element :math:`i`
    The interaction between atom :math:`i` of element :math:`I` and
    atom :math:`j` of element :math:`J` is given by

        :math:`\\frac{Z_{I}Z_{J}}{\\| {\\bf R}_{i} - {\\bf R}_{j}\\|}`

    with :math:`R_{i}` being the euclidean coordinate of atom :math:`i`.
    The sorted bags are concatenated to an 1D vector representation.
    The representation is calculated using an OpenMP parallel Fortran routine.

    :param nuclear_charges: Nuclear charges of the atoms in the molecule
    :type nuclear_charges: numpy array
    :param coordinates: 3D Coordinates of the atoms in the molecule
    :type coordinates: numpy array
    :param size: The maximum number of atoms in the representation
    :type size: integer
    :param asize: The maximum number of atoms of each element type supported by the representation
    :type asize: dictionary

    :return: 1D representation
    :rtype: numpy array
    """

    # TODO Moving between str and int is _, should translate everything to use int

    n = 0
    atoms = sorted(asize, key=asize.get)
    nmax = [asize[key] for key in atoms]
    ids = np.zeros(len(nmax), dtype=int)
    for i, (key, value) in enumerate(zip(atoms, nmax)):
        n += value * (1 + value)
        ids[i] = NUCLEAR_CHARGE[key]
        for j in range(i):
            v = nmax[j]
            n += 2 * value * v
    n /= 2

    return fgenerate_bob(nuclear_charges, coordinates, nuclear_charges, ids, nmax, n)


def get_slatm_mbtypes(nuclear_charges: List[ndarray], pbc: str = "000") -> List[List[int64]]:
    """
    Get the list of minimal types of many-body terms in a dataset. This resulting list
    is necessary as input in the ``generate_slatm()`` function.

    :param nuclear_charges: A list of the nuclear charges for each compound in the dataset.
    :type nuclear_charges: list of numpy arrays
    :param pbc: periodic boundary condition along x,y,z direction, defaulted to '000', i.e., molecule
    :type pbc: string
    :return: A list containing the types of many-body terms.
    :rtype: list
    """

    zs = nuclear_charges

    nm = len(zs)
    zsmax = set()
    nas = []
    zs_ravel = []
    for zsi in zs:
        na = len(zsi)
        nas.append(na)
        zsil = list(zsi)
        zs_ravel += zsil
        zsmax.update(zsil)

    zsmax = np.array(list(zsmax))
    nass = []
    for i in range(nm):
        zsi = np.array(zs[i], np.int32)
        nass.append([(zi == zsi).sum() for zi in zsmax])

    nzmax = np.max(np.array(nass), axis=0)
    nzmax_u = []
    if pbc != "000":
        # the PBC will introduce new many-body terms, so set
        # nzmax to 3 if it's less than 3
        for nzi in nzmax:
            if nzi <= 2:
                nzi = 3
            nzmax_u.append(nzi)
        nzmax = nzmax_u

    boas = [
        [
            zi,
        ]
        for zi in zsmax
    ]

    bops = [[zi, zi] for zi in zsmax] + [list(x) for x in itertools.combinations(zsmax, 2)]

    bots = []
    for i in zsmax:
        for bop in bops:
            j, k = bop
            tas = [[i, j, k], [i, k, j], [j, i, k]]
            for tasi in tas:
                if (tasi not in bots) and (tasi[::-1] not in bots):
                    nzsi = [(zj == tasi).sum() for zj in zsmax]
                    if np.all(nzsi <= nzmax):
                        bots.append(tasi)
    mbtypes = boas + bops + bots

    return mbtypes  # , np.array(zs_ravel), np.array(nas)


def generate_slatm(
    nuclear_charges: ndarray,
    coordinates: ndarray,
    mbtypes: List[List[int64]],
    unit_cell: None = None,
    local: bool = False,
    sigmas: List[float] = [0.05, 0.05],
    dgrids: List[float] = [0.03, 0.03],
    rcut: float = 4.8,
    alchemy: bool = False,
    pbc: str = "000",
    rpower: int = 6,
) -> Union[ndarray, List[ndarray]]:
    """
    Generate Spectrum of London and Axillrod-Teller-Muto potential (SLATM) representation.
    Both global (``local=False``) and local (``local=True``) SLATM are available.

    A version that works for periodic boundary conditions will be released soon.

    NOTE: You will need to run the ``get_slatm_mbtypes()`` function to get the ``mbtypes`` input (or generate it manually).

    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param coordinates: Input coordinates
    :type coordinates: numpy array
    :param mbtypes: Many-body types for the whole dataset, including 1-, 2- and 3-body types. Could be obtained by calling ``get_slatm_mbtypes()``.
    :type mbtypes: list
    :param local: Generate a local representation. Defaulted to False (i.e., global representation); otherwise, atomic version.
    :type local: bool
    :param sigmas: Controlling the width of Gaussian smearing function for 2- and 3-body parts, defaulted to [0.05,0.05], usually these do not need to be adjusted.
    :type sigmas: list
    :param dgrids: The interval between two sampled internuclear distances and angles, defaulted to [0.03,0.03], no need for change, compromised for speed and accuracy.
    :type dgrids: list
    :param rcut: Cut-off radius, defaulted to 4.8 Angstrom.
    :type rcut: float
    :param alchemy: Swith to use the alchemy version of SLATM. (default=False)
    :type alchemy: bool
    :param pbc: defaulted to '000', meaning it's a molecule; the three digits in the string corresponds to x,y,z direction
    :type pbc: string
    :param rpower: The power of R in 2-body potential, defaulted to London potential (=6).
    :type rpower: float
    :return: 1D SLATM representation
    :rtype: numpy array
    """

    c = unit_cell
    # UNUSED iprt = False
    if c is None:
        c = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    if pbc != "000":
        if c is None:
            raise ValueError("Please specify unit cell for SLATM")

        # =======================================================================
        # PBC may introduce new many-body terms, so at the stage of get statistics
        # info from db, we've already considered this point by letting maximal number
        # of nuclear charges being 3.
        # =======================================================================

    zs = nuclear_charges
    na = len(zs)
    coords = coordinates
    obj = [zs, coords, c]

    iloc = local

    if iloc:
        mbs = []
        X2Ns = []
        for ia in range(na):
            n1 = 0
            n2 = 0
            n3 = 0
            mbs_ia = np.zeros(0)
            # UNUSED icount = 0
            for mbtype in mbtypes:
                if len(mbtype) == 1:
                    mbsi = get_boa(
                        mbtype[0],
                        np.array(
                            [
                                zs[ia],
                            ]
                        ),
                    )
                    if alchemy:
                        n1 = 1
                        n1_0 = mbs_ia.shape[0]
                        if n1_0 == 0:
                            mbs_ia = np.concatenate((mbs_ia, mbsi), axis=0)
                        elif n1_0 == 1:
                            mbs_ia += mbsi
                        else:
                            raise ValueError()
                    else:
                        n1 += len(mbsi)
                        mbs_ia = np.concatenate((mbs_ia, mbsi), axis=0)
                elif len(mbtype) == 2:
                    mbsi = get_sbop(
                        mbtype,
                        obj,
                        iloc=iloc,
                        ia=ia,
                        sigma=sigmas[0],
                        dgrid=dgrids[0],
                        rcut=rcut,
                        pbc=pbc,
                        rpower=rpower,
                    )
                    mbsi *= 0.5  # only for the two-body parts, local rpst
                    if alchemy:
                        n2 = len(mbsi)
                        n2_0 = mbs_ia.shape[0]
                        if n2_0 == n1:
                            mbs_ia = np.concatenate((mbs_ia, mbsi), axis=0)
                        elif n2_0 == n1 + n2:
                            t = mbs_ia[n1 : n1 + n2] + mbsi
                            mbs_ia[n1 : n1 + n2] = t
                        else:
                            raise ValueError()
                    else:
                        n2 += len(mbsi)
                        mbs_ia = np.concatenate((mbs_ia, mbsi), axis=0)
                else:  # len(mbtype) == 3:
                    mbsi = get_sbot(
                        mbtype,
                        obj,
                        iloc=iloc,
                        ia=ia,
                        sigma=sigmas[1],
                        dgrid=dgrids[1],
                        rcut=rcut,
                        pbc=pbc,
                    )

                    if alchemy:
                        n3 = len(mbsi)
                        n3_0 = mbs_ia.shape[0]
                        if n3_0 == n1 + n2:
                            mbs_ia = np.concatenate((mbs_ia, mbsi), axis=0)
                        elif n3_0 == n1 + n2 + n3:
                            t = mbs_ia[n1 + n2 : n1 + n2 + n3] + mbsi
                            mbs_ia[n1 + n2 : n1 + n2 + n3] = t
                        else:
                            raise ValueError()
                    else:
                        n3 += len(mbsi)
                        mbs_ia = np.concatenate((mbs_ia, mbsi), axis=0)

            mbs.append(mbs_ia)
            X2N = [n1, n2, n3]
            if X2N not in X2Ns:
                X2Ns.append(X2N)

        if len(X2Ns) != 1:
            raise ValueError("multiple `X2N ???")

    else:
        n1 = 0
        n2 = 0
        n3 = 0
        mbs = np.zeros(0)
        for mbtype in mbtypes:
            if len(mbtype) == 1:
                mbsi = get_boa(mbtype[0], zs)
                if alchemy:
                    n1 = 1
                    n1_0 = mbs.shape[0]
                    if n1_0 == 0:
                        mbs = np.concatenate((mbs, [sum(mbsi)]), axis=0)
                    elif n1_0 == 1:
                        mbs += sum(mbsi)
                    else:
                        raise ValueError()
                else:
                    n1 += len(mbsi)
                    mbs = np.concatenate((mbs, mbsi), axis=0)
            elif len(mbtype) == 2:
                mbsi = get_sbop(
                    mbtype, obj, sigma=sigmas[0], dgrid=dgrids[0], rcut=rcut, rpower=rpower
                )

                if alchemy:
                    n2 = len(mbsi)
                    n2_0 = mbs.shape[0]
                    if n2_0 == n1:
                        mbs = np.concatenate((mbs, mbsi), axis=0)
                    elif n2_0 == n1 + n2:
                        t = mbs[n1 : n1 + n2] + mbsi
                        mbs[n1 : n1 + n2] = t
                    else:
                        raise ValueError()
                else:
                    n2 += len(mbsi)
                    mbs = np.concatenate((mbs, mbsi), axis=0)
            else:  # len(mbtype) == 3:
                mbsi = get_sbot(mbtype, obj, sigma=sigmas[1], dgrid=dgrids[1], rcut=rcut)

                if alchemy:
                    n3 = len(mbsi)
                    n3_0 = mbs.shape[0]
                    if n3_0 == n1 + n2:
                        mbs = np.concatenate((mbs, mbsi), axis=0)
                    elif n3_0 == n1 + n2 + n3:
                        t = mbs[n1 + n2 : n1 + n2 + n3] + mbsi
                        mbs[n1 + n2 : n1 + n2 + n3] = t
                    else:
                        raise ValueError()
                else:
                    n3 += len(mbsi)
                    mbs = np.concatenate((mbs, mbsi), axis=0)

    return mbs


def generate_acsf(
    nuclear_charges: List[int],
    coordinates: ndarray,
    elements: List[int] = [1, 6, 7, 8, 16],
    nRs2: int = 3,
    nRs3: int = 3,
    nTs: int = 3,
    eta2: int = 1,
    eta3: int = 1,
    zeta: int = 1,
    rcut: int = 5,
    acut: int = 5,
    bin_min: float = 0.8,
    gradients: bool = False,
    pad: Optional[int] = None,
) -> Union[Tuple[ndarray, ndarray], ndarray]:
    """
    Generate the variant of atom-centered symmetry functions used in https://doi.org/10.1039/C7SC04934J

    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param coordinates: Input coordinates
    :type coordinates: numpy array
    :param elements: list of unique nuclear charges (atom types)
    :type elements: numpy array
    :param nRs2: Number of gaussian basis functions in the two-body terms
    :type nRs2: integer
    :param nRs3: Number of gaussian basis functions in the three-body radial part
    :type nRs3: integer
    :param nTs: Number of basis functions in the three-body angular part
    :type nTs: integer
    :param eta2: Precision in the gaussian basis functions in the two-body terms
    :type eta2: float
    :param eta3: Precision in the gaussian basis functions in the three-body radial part
    :type eta3: float
    :param zeta: Precision parameter of basis functions in the three-body angular part
    :type zeta: float
    :param rcut: Cut-off radius of the two-body terms
    :type rcut: float
    :param acut: Cut-off radius of the three-body terms
    :type acut: float
    :param bin_min: the value at which to start binning the distances
    :type bin_min: positive float
    :param gradients: To return gradients or not
    :type gradients: boolean
    :param pad: `None` if no padding is to be applied other, otherwise an integer corresponding to the desired size
    :type gradients: NoneType or integer
    :return: Atom-centered symmetry functions representation
    :rtype: numpy array
    """

    Rs2 = np.linspace(bin_min, rcut, nRs2)
    Rs3 = np.linspace(bin_min, acut, nRs3)
    Ts = np.linspace(0, np.pi, nTs)
    n_elements = len(elements)
    natoms = len(coordinates)

    descr_size = n_elements * nRs2 + (n_elements * (n_elements + 1)) // 2 * nRs3 * nTs

    if gradients is False:

        rep = fgenerate_acsf(
            coordinates,
            nuclear_charges,
            elements,
            Rs2,
            Rs3,
            Ts,
            eta2,
            eta3,
            zeta,
            rcut,
            acut,
            natoms,
            descr_size,
        )

        if pad is not None:

            rep_pad = np.zeros((pad, descr_size))
            rep_pad[:natoms, :] += rep

            return rep_pad

        else:
            return rep

    else:

        (rep, grad) = fgenerate_acsf_and_gradients(
            coordinates,
            nuclear_charges,
            elements,
            Rs2,
            Rs3,
            Ts,
            eta2,
            eta3,
            zeta,
            rcut,
            acut,
            natoms,
            descr_size,
        )

        if pad is not None:
            rep_pad = np.zeros((pad, descr_size))
            grad_pad = np.zeros((pad, descr_size, pad, 3))

            rep_pad[:natoms, :] += rep
            grad_pad[:natoms, :, :natoms, :] += grad

            return rep_pad, grad_pad
        else:
            return rep, grad


def generate_fchl19(
    nuclear_charges: ndarray,
    coordinates: ndarray,
    elements: List[int] = [1, 6, 7, 8, 16],
    nRs2: int = 24,
    nRs3: int = 20,
    nFourier: int = 1,
    eta2: float = 0.32,
    eta3: float = 2.7,
    zeta: float = np.pi,
    rcut: float = 8.0,
    acut: float = 8.0,
    two_body_decay: float = 1.8,
    three_body_decay: float = 0.57,
    three_body_weight: float = 13.4,
    pad: Union[int, bool] = False,
    gradients: bool = False,
    cell: Union[ndarray, None] = None,
) -> Union[Tuple[ndarray, ndarray], ndarray]:
    """
    FCHL-ACSF

    https://pubs.aip.org/aip/jcp/article/152/4/044107/1064737/FCHL-revisited-Faster-and-more-accurate-quantum

    Reasonable hyperparameters:

    Sigma ~ 21.0
    Lambda ~ 1e-8
    Max singular value ~ 1e-12

    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param coordinates: Input coordinates
    :type coordinates: numpy array
    :param elements: list of unique nuclear charges (atom types)
    :type elements: numpy array
    :param nRs2: Number of gaussian basis functions in the two-body terms
    :type nRs2: integer
    :param nRs3: Number of gaussian basis functions in the three-body radial part
    :type nRs3: integer
    :param nFourier: Order of Fourier expansion
    :type nFourier: integer
    :param eta2: Precision in the gaussian basis functions in the two-body terms
    :type eta2: float
    :param eta3: Precision in the gaussian basis functions in the three-body radial part
    :type eta3: float
    :param zeta: Precision parameter of basis functions in the three-body angular part
    :type zeta: float
    :param rcut: Cut-off radius of the two-body terms
    :type rcut: float
    :param acut: Cut-off radius of the three-body terms
    :type acut: float
    :param gradients: To return gradients or not
    :type gradients: boolean
    :param cell: parameters of the unit cell if periodic boundary conditions are used.
    :type cell: numpy array
    :return: Atom-centered symmetry functions representation
    :rtype: numpy array
    """

    Rs2 = np.linspace(0, rcut, 1 + nRs2)[1:]
    Rs3 = np.linspace(0, acut, 1 + nRs3)[1:]

    Ts = np.linspace(0, np.pi, 2 * nFourier)
    n_elements = len(elements)
    natoms = len(coordinates)

    descr_size = n_elements * nRs2 + (n_elements * (n_elements + 1)) * nRs3 * nFourier

    # Normalization constant for three-body
    three_body_weight = np.sqrt(eta3 / np.pi) * three_body_weight

    # If periodic boundary conditions are used add neighboring cells that can influence the center cell.
    natoms_tot = natoms
    if cell is not None:
        nExtend = (np.floor(max(rcut, acut) / np.linalg.norm(cell, 2, axis=0)) + 1).astype(int)
        true_coords = coordinates
        for i in range(-nExtend[0], nExtend[0] + 1):
            for j in range(-nExtend[1], nExtend[1] + 1):
                for k in range(-nExtend[2], nExtend[2] + 1):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    true_coords = np.append(
                        true_coords,
                        coordinates + i * cell[0, :] + j * cell[1, :] + k * cell[2, :],
                        axis=0,
                    )
                    natoms_tot += natoms
        coordinates = true_coords

    if gradients is False:

        rep = fgenerate_fchl_acsf(
            coordinates,
            nuclear_charges,
            elements,
            Rs2,
            Rs3,
            Ts,
            eta2,
            eta3,
            zeta,
            rcut,
            acut,
            natoms,
            natoms_tot,
            descr_size,
            two_body_decay,
            three_body_decay,
            three_body_weight,
        )

        if pad is not False:

            rep_pad = np.zeros((pad, descr_size))
            rep_pad[:natoms, :] += rep

            return rep_pad

        else:
            return rep

    else:

        if nFourier > 1:
            raise ValueError(f"FCHL-ACSF only supports nFourier=1, requested {nFourier}")

        (rep, grad) = fgenerate_fchl_acsf_and_gradients(
            coordinates,
            nuclear_charges,
            elements,
            Rs2,
            Rs3,
            Ts,
            eta2,
            eta3,
            zeta,
            rcut,
            acut,
            natoms,
            natoms_tot,
            descr_size,
            two_body_decay,
            three_body_decay,
            three_body_weight,
        )

        if pad is not False:
            rep_pad = np.zeros((pad, descr_size))
            grad_pad = np.zeros((pad, descr_size, pad, 3))

            rep_pad[:natoms, :] += rep
            grad_pad[:natoms, :, :natoms, :] += grad

            return rep_pad, grad_pad
        else:
            return rep, grad
