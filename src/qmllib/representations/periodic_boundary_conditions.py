import numpy as np
from numba import njit, prange
from numba.types import bool_


@njit(fastmath=True)
def get_atom_environment_ranges(natoms_arr):
    nreps = natoms_arr.shape[0]
    ubound_arr = np.empty((nreps + 1,), dtype=np.int64)
    ubound_arr[0] = 0
    for rep_id in range(nreps):
        ubound_arr[rep_id + 1] = ubound_arr[rep_id] + natoms_arr[rep_id]
    return ubound_arr


@njit(fastmath=True)
def generate_cell_int_coords(
    cell_int_coords, nExtend, ndim, is_on_lower_border, is_on_upper_border, started
):
    if started:
        cell_int_coords[:] = -nExtend[:]
        is_on_lower_border[:] = True
        is_on_upper_border[:] = False
        return True
    for dim in range(ndim):
        if cell_int_coords[dim] == nExtend[dim]:
            cell_int_coords[dim] = -nExtend[dim]
            is_on_lower_border[dim] = True
            is_on_upper_border[dim] = False
        else:
            cell_int_coords[dim] += 1
            is_on_lower_border[dim] = False
            if cell_int_coords[dim] == nExtend[dim]:
                is_on_upper_border[dim] = True
            return True
    return False


@njit(fastmath=True)
def should_be_added(
    atom_coords,
    cell_dir_lower_reach,
    cell_dir_upper_reach,
    is_on_lower_border,
    is_on_upper_border,
    ndim,
):
    if not (np.any(is_on_lower_border) or np.any(is_on_upper_border)):
        return True
    for dim in range(ndim):
        if is_on_lower_border[dim] and (atom_coords[dim] > cell_dir_upper_reach[dim]):
            return True
        if is_on_upper_border[dim] and (atom_coords[dim] < cell_dir_lower_reach[dim]):
            return True
    return False


@njit(fastmath=True, parallel=True)
def count_atom_copies(
    cell_dir_coordinates, cell_dir_lower_reach, cell_dir_upper_reach, natoms, nExtend, ndim
):
    # Number of created cells not on the supercell's border and thus containing copies of all atoms.
    num_inside_cells = 1
    for nex in nExtend:
        num_inside_cells *= 2 * nex - 1
    # Not counting the original cell.
    num_inside_cells -= 1
    # We know that at least num_inside_cells copies of atoms are added.
    atom_additional_copy_number = np.repeat(num_inside_cells, natoms)
    for atom_id in prange(natoms):
        atom_coords = cell_dir_coordinates[atom_id]
        cell_int_coords = np.empty((ndim,), dtype=np.int64)
        is_on_lower_border = np.empty((ndim,), dtype=bool_)
        is_on_upper_border = np.empty((ndim,), dtype=bool_)
        started = True
        while generate_cell_int_coords(
            cell_int_coords, nExtend, ndim, is_on_lower_border, is_on_upper_border, started
        ):
            started = False
            if not (np.any(is_on_lower_border) or np.any(is_on_upper_border)):
                continue
            if should_be_added(
                atom_coords,
                cell_dir_lower_reach,
                cell_dir_upper_reach,
                is_on_lower_border,
                is_on_upper_border,
                ndim,
            ):
                atom_additional_copy_number[atom_id] += 1

    return atom_additional_copy_number


@njit(fastmath=True, parallel=True)
def extend_for_pbc(coordinates, nuclear_charges, natoms, rcut, cell):
    # Cartesian space dimensionality.
    ndim = coordinates.shape[1]
    # Normalized directions corresponding to different cells.
    cell_directions = np.copy(cell)
    cell_lengths = np.empty((ndim,))
    for dim in range(ndim):
        cell_lengths[dim] = np.linalg.norm(cell_directions[dim])
        cell_directions[dim] /= cell_lengths[dim]

    # Transform coordinates to cell.
    cell_dir_coordinates = coordinates @ cell_directions.T
    # maximum and minimum cell coordinates of atoms that can be reached from atoms in neighboring cells.
    cell_dir_lower_reach = np.empty((ndim,))
    cell_dir_upper_reach = np.empty((ndim,))
    # how many cells in different directions we need to create
    nExtend = np.empty((ndim,), dtype=np.int64)
    for dim in range(ndim):
        cell_dir_lower_reach[dim] = np.max(cell_dir_coordinates[:, dim]) + rcut - cell_lengths[dim]
        cell_dir_upper_reach[dim] = np.min(cell_dir_coordinates[:, dim]) - rcut + cell_lengths[dim]
        nExtend[dim] = np.floor(rcut / cell_lengths[dim]) + 1
    # how many copies of each atom do we need to make
    atom_additional_copy_number = count_atom_copies(
        cell_dir_coordinates, cell_dir_lower_reach, cell_dir_upper_reach, natoms, nExtend, ndim
    )
    # total number of atoms
    natoms_tot = natoms + np.sum(atom_additional_copy_number)
    # extended arrays
    extended_coordinates = np.empty((natoms_tot, ndim))
    extended_nuclear_charges = np.empty((natoms_tot,), dtype=np.int32)

    # the oridinal cell is just copied.
    extended_coordinates[:natoms, :] = coordinates[:, :]
    extended_nuclear_charges[:natoms] = nuclear_charges[:]

    atom_copy_start_ids = get_atom_environment_ranges(atom_additional_copy_number) + natoms

    for atom_id in prange(natoms):
        atom_coords = coordinates[atom_id]
        cell_atom_coords = cell_dir_coordinates[atom_id]
        cur_copy_id = atom_copy_start_ids[atom_id]
        extended_nuclear_charges[cur_copy_id : atom_copy_start_ids[atom_id + 1]] = nuclear_charges[
            atom_id
        ]
        # create new coordinates
        cell_int_coords = np.empty((ndim,), dtype=np.int64)
        is_on_lower_border = np.empty((ndim,), dtype=bool_)
        is_on_upper_border = np.empty((ndim,), dtype=bool_)
        started = True
        while generate_cell_int_coords(
            cell_int_coords, nExtend, ndim, is_on_lower_border, is_on_upper_border, started
        ):
            started = False
            if np.all(cell_int_coords == 0):
                continue
            if should_be_added(
                cell_atom_coords,
                cell_dir_lower_reach,
                cell_dir_upper_reach,
                is_on_lower_border,
                is_on_upper_border,
                ndim,
            ):
                extended_coordinates[cur_copy_id, :] = atom_coords
                for dim in range(ndim):
                    extended_coordinates[cur_copy_id, :] += cell_int_coords[dim] * cell[dim]
                cur_copy_id += 1
    return extended_coordinates, extended_nuclear_charges, natoms_tot
