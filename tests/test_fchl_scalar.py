import numpy as np
from conftest import ASSETS, get_energies, shuffle_arrays
from scipy.special import binom, factorial, jn

from qmllib.representations.fchl import (
    generate_fchl18,
    get_atomic_kernels,
    get_atomic_symmetric_kernels,
    get_global_kernels,
    get_global_symmetric_kernels,
    get_local_kernels,
    get_local_symmetric_kernels,
)
from qmllib.solvers import cho_solve
from qmllib.utils.xyz_format import read_xyz


def _get_training_data(n_points, representation_options={}):

    _representation_options = {
        **dict(
            cut_distance=1e6,
            max_size=23,
        ),
        **representation_options,
    }

    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
    data = get_energies(ASSETS / "hof_qm7.txt")

    all_representations = []
    all_properties = []
    all_atoms = []

    for xyz_file in sorted(data.keys())[:n_points]:

        filename = ASSETS / "qm7" / xyz_file
        coord, atoms = read_xyz(filename)

        # Associate a property (heat of formation) with the object
        all_properties.append(data[xyz_file])

        representation = generate_fchl18(atoms, coord, **_representation_options)

        assert (
            representation.shape[0] == _representation_options["max_size"]
        ), "ERROR: Check FCHL descriptor size!"

        all_representations.append(representation)
        all_atoms.append(atoms)

    # Convert to arrays
    all_representations = np.array(all_representations)
    all_properties = np.array(all_properties)

    return all_properties, all_representations, all_atoms


def test_krr_fchl_local():

    # Test that all kernel arguments work
    kernel_args = {
        "cut_distance": 1e6,
        "cut_start": 0.5,
        "two_body_width": 0.1,
        "two_body_scaling": 2.0,
        "two_body_power": 6.0,
        "three_body_width": 3.0,
        "three_body_scaling": 2.0,
        "three_body_power": 3.0,
        "alchemy": "periodic-table",
        "alchemy_period_width": 1.0,
        "alchemy_group_width": 1.0,
        "fourier_order": 2,
        "kernel": "gaussian",
        "kernel_args": {
            "sigma": [2.5],
        },
    }

    max_size = 23
    representation_options = dict(cut_distance=1e6, max_size=max_size)

    n_points = 100
    all_properties, all_representations, all_atoms = _get_training_data(
        n_points, representation_options=representation_options
    )
    shuffle_arrays(all_representations, all_atoms, all_properties, seed=666)

    # Make training and test sets
    n_points = len(all_representations)
    n_test = n_points // 3
    n_train = n_points - n_test
    indicies = list(range(n_points))
    train_indicies = indicies[:n_train]
    test_indicies = indicies[-n_test:]

    train_representations = all_representations[train_indicies]
    train_properties = all_properties[train_indicies]
    test_representations = all_representations[test_indicies]
    test_properties = all_properties[test_indicies]

    # Set hyper-parameters
    llambda = 1e-8

    K_symmetric = get_local_symmetric_kernels(train_representations, **kernel_args)[0]
    K = get_local_kernels(train_representations, train_representations, **kernel_args)[0]

    assert np.allclose(K, K_symmetric), "Error in FCHL symmetric local kernels"
    assert np.invert(np.all(np.isnan(K_symmetric))), "FCHL local symmetric kernel contains NaN"
    assert np.invert(np.all(np.isnan(K))), "FCHL local kernel contains NaN"

    del K_symmetric

    # Solve alpha
    K[np.diag_indices_from(K)] += llambda
    alpha = cho_solve(K, train_properties)

    # Calculate prediction kernel
    Ks = get_local_kernels(test_representations, train_representations, **kernel_args)[0]
    assert np.invert(np.all(np.isnan(Ks))), "FCHL local testkernel contains NaN"

    predicted_properties = np.dot(Ks, alpha)

    mae = np.mean(np.abs(test_properties - predicted_properties))
    assert abs(2 - mae) < 1.0, "Error in FCHL local kernel-ridge regression"


def test_krr_fchl_global():

    # Test that all kernel arguments work
    kernel_args = {
        "kernel": "gaussian",
        "kernel_args": {
            "sigma": [100.0],
        },
    }

    max_size = 23
    representation_options = dict(cut_distance=1e6, max_size=max_size)

    n_points = 100
    all_properties, all_representations, all_atoms = _get_training_data(
        n_points, representation_options=representation_options
    )

    shuffle_arrays(all_representations, all_atoms, all_properties, seed=666)

    # Make training and test sets
    n_points = len(all_representations)
    n_test = n_points // 3
    n_train = n_points - n_test
    indicies = list(range(n_points))
    train_indicies = indicies[:n_train]
    test_indicies = indicies[-n_test:]

    train_representations = all_representations[train_indicies]
    train_properties = all_properties[train_indicies]
    test_representations = all_representations[test_indicies]
    test_properties = all_properties[test_indicies]

    # Set hyper-parameters
    llambda = 1e-8

    K_symmetric = get_global_symmetric_kernels(train_representations, **kernel_args)[0]
    K = get_global_kernels(train_representations, train_representations, **kernel_args)[0]

    assert np.allclose(K, K_symmetric), "Error in FCHL symmetric global kernels"
    assert np.invert(np.all(np.isnan(K_symmetric))), "FCHL global symmetric kernel contains NaN"
    assert np.invert(np.all(np.isnan(K))), "FCHL global kernel contains NaN"

    # Solve alpha
    K[np.diag_indices_from(K)] += llambda
    alpha = cho_solve(K, train_properties)

    Ks = get_global_kernels(test_representations, train_representations, **kernel_args)[0]
    assert np.invert(np.all(np.isnan(Ks))), "FCHL global testkernel contains NaN"

    predicted_properties = np.dot(Ks, alpha)

    print(test_properties, predicted_properties)

    mae = np.mean(np.abs(test_properties - predicted_properties))
    assert abs(2 - mae) < 1.0, "Error in FCHL global kernel-ridge regression"


def test_krr_fchl_atomic():

    kernel_args = {
        "kernel": "gaussian",
        "kernel_args": {
            "sigma": [2.5],
        },
    }

    max_size = 23
    representation_options = dict(cut_distance=1e6, max_size=max_size)

    n_points = 10
    all_properties, all_representations, all_atoms = _get_training_data(
        n_points, representation_options=representation_options
    )

    # Generate kernel
    K = get_local_symmetric_kernels(all_representations, **kernel_args)[0]
    K_test = np.zeros((n_points, n_points))

    for i, Xi in enumerate(all_representations):
        for j, Xj in enumerate(all_representations):

            K_atomic = get_atomic_kernels(
                Xi[: len(all_atoms[i])], Xj[: len(all_atoms[j])], **kernel_args
            )[0]
            K_test[i, j] = np.sum(K_atomic)

            assert np.invert(np.all(np.isnan(K_atomic))), "FCHL atomic kernel contains NaN"

            if i == j:
                K_atomic_symmetric = get_atomic_symmetric_kernels(
                    Xi[: len(all_atoms[i])], **kernel_args
                )[0]
                assert np.allclose(
                    K_atomic, K_atomic_symmetric
                ), "Error in FCHL symmetric atomic kernels"
                assert np.invert(
                    np.all(np.isnan(K_atomic_symmetric))
                ), "FCHL atomic symmetric kernel contains NaN"

    assert np.allclose(K, K_test), "Error in FCHL atomic kernels"


def test_fchl_local_periodic():
    kernel_args = {
        "cut_distance": 7.0,
        "cut_start": 0.7,
        "kernel": "gaussian",
        "kernel_args": {
            "sigma": [2.5],
        },
    }

    nuclear_charges = [
        np.array(
            [
                13,
                13,
                58,
                58,
                58,
                58,
                58,
                58,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                23,
                23,
            ]
        ),
        np.array([34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 73, 73, 73, 73, 81, 81, 81, 81]),
        np.array([48, 8, 8, 8, 8, 8, 8, 51, 51]),
        np.array(
            [
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                30,
                30,
                30,
                30,
                30,
                30,
                30,
                30,
                30,
                30,
                30,
                30,
                30,
                30,
                30,
                30,
                30,
                30,
            ]
        ),
        np.array([58, 58, 8, 8, 8, 8, 8, 8, 8, 8, 23, 23]),
    ]

    cells = np.array(
        [
            [
                [1.01113290e01, -1.85000000e-04, 0.00000000e00],
                [-5.05582400e00, 8.75745400e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 6.15100100e00],
            ],
            [[9.672168, 0.0, 0.0], [0.0, 3.643786, 0.0], [0.0, 0.0, 14.961818]],
            [
                [5.28208000e00, -1.20000000e-05, 1.50000000e-05],
                [-2.64105000e00, 4.57443400e00, -3.00000000e-05],
                [1.40000000e-05, -2.40000000e-05, 4.77522000e00],
            ],
            [[-1.917912, 3.321921, 0.0], [3.835824, 0.0, 0.0], [1.917912, -1.107307, -56.423542]],
            [
                [3.699168, 3.699168, -3.255938],
                [3.699168, -3.699168, 3.255938],
                [-3.699168, -3.699168, -3.255938],
            ],
        ]
    )

    fractional_coordinates = [
        [
            [0.6666706, 0.33333356, 0.15253127],
            [0.33332896, 0.66666655, 0.65253119],
            [0.14802736, 0.375795, 0.23063888],
            [0.62422269, 0.77225019, 0.23063888],
            [0.22775607, 0.85196133, 0.23063888],
            [0.77224448, 0.14803879, 0.7306388],
            [0.37577687, 0.22774993, 0.7306388],
            [0.8519722, 0.62420512, 0.7306388],
            [0.57731818, 0.47954083, 0.0102715],
            [0.52043884, 0.09777799, 0.01028288],
            [0.90211803, 0.4226259, 0.01032677],
            [0.33335216, 0.6666734, 0.01482035],
            [0.90959766, 0.77585201, 0.30637615],
            [0.86626106, 0.09040873, 0.3063794],
            [0.22415808, 0.13374794, 0.30638265],
            [0.42268138, 0.52045928, 0.51027142],
            [0.47956114, 0.90222098, 0.5102828],
            [0.09788153, 0.57737421, 0.51032669],
            [0.6666474, 0.33332671, 0.51482027],
            [0.09040133, 0.22414696, 0.80637769],
            [0.13373793, 0.90959025, 0.80637932],
            [0.77584247, 0.86625217, 0.80638257],
            [0.0, 0.0, 0.05471142],
            [0.0, 0.0, 0.55471134],
        ],
        [
            [0.81615001, 0.75000014, 0.00116296],
            [0.52728096, 0.25000096, 0.0993275],
            [0.24582596, 0.75000014, 0.2198563],
            [0.74582658, 0.75000014, 0.28014376],
            [0.02728137, 0.25000096, 0.40067257],
            [0.31615042, 0.75000014, 0.49883711],
            [0.68384978, 0.25000096, 0.50116236],
            [0.97271884, 0.75000014, 0.59932757],
            [0.25417362, 0.25000096, 0.71985637],
            [0.75417321, 0.25000096, 0.78014383],
            [0.47271925, 0.75000014, 0.90067263],
            [0.1838502, 0.25000096, 0.99883717],
            [0.33804831, 0.75000014, 0.07120258],
            [0.83804789, 0.75000014, 0.42879749],
            [0.16195232, 0.25000096, 0.57120198],
            [0.6619519, 0.25000096, 0.92879756],
            [0.98245812, 0.25000096, 0.17113829],
            [0.48245853, 0.25000096, 0.32886177],
            [0.51754167, 0.75000014, 0.67113836],
            [0.01754209, 0.75000014, 0.82886184],
        ],
        [
            [0.00000000e00, 0.00000000e00, 0.00000000e00],
            [3.66334233e-01, 1.96300000e-07, 2.70922493e-01],
            [6.33665197e-01, 6.33666177e-01, 2.70923540e-01],
            [3.62000000e-08, 3.66333081e-01, 2.70923851e-01],
            [6.70000000e-09, 6.33664733e-01, 7.29076149e-01],
            [6.33664135e-01, 9.99998055e-01, 7.29076460e-01],
            [3.66336157e-01, 3.66334260e-01, 7.29077507e-01],
            [3.33333635e-01, 6.66667395e-01, 4.99998953e-01],
            [6.66667720e-01, 3.33333042e-01, 5.00000000e-01],
        ],
        [
            [0.3379644, 0.66203644, 0.01389048],
            [0.02316309, 0.97683587, 0.06948926],
            [0.70833843, 0.29165976, 0.12501892],
            [0.39352259, 0.60647824, 0.18056506],
            [0.74538243, 0.25461577, 0.2361509],
            [0.09722803, 0.90277092, 0.2916841],
            [0.44907919, 0.55092165, 0.34723485],
            [0.8009281, 0.1990701, 0.4027879],
            [0.15278103, 0.84721793, 0.45834308],
            [0.83797345, 0.16202475, 0.51392396],
            [0.52315813, 0.4768427, 0.56947169],
            [0.20833916, 0.7916598, 0.62501748],
            [0.89352691, 0.10647128, 0.68058436],
            [0.57870427, 0.42129656, 0.73611012],
            [0.93056329, 0.06943491, 0.79169347],
            [0.28241704, 0.71758191, 0.84725114],
            [0.63426956, 0.36573128, 0.90280596],
            [0.98611817, 0.01388002, 0.95835813],
            [0.0, 0.0, 0.0],
            [0.35185434, 0.64814649, 0.05556032],
            [0.03704151, 0.96295744, 0.11112454],
            [0.72221887, 0.27777932, 0.16666022],
            [0.40741437, 0.59258647, 0.22224039],
            [0.75926009, 0.24073811, 0.27778387],
            [0.11111195, 0.888887, 0.33333586],
            [0.46296234, 0.53703849, 0.38888431],
            [0.81480954, 0.18518866, 0.44443222],
            [0.16667233, 0.83332662, 0.500017],
            [0.85185117, 0.14814703, 0.55555711],
            [0.53704217, 0.46295866, 0.61112381],
            [0.22222196, 0.777777, 0.66666587],
            [0.90740847, 0.09258972, 0.72222903],
            [0.59259328, 0.40740756, 0.77777712],
            [0.94444213, 0.05555607, 0.83333],
            [0.29630132, 0.70369764, 0.88890396],
            [0.64815247, 0.35184836, 0.94445471],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.75000042, 0.50000027, 0.25000015],
            [0.15115386, 0.81961403, 0.33154037],
            [0.51192691, 0.18038651, 0.3315404],
            [0.08154025, 0.31961376, 0.40115401],
            [0.66846017, 0.81961403, 0.48807366],
            [0.08154025, 0.68038678, 0.76192703],
            [0.66846021, 0.18038651, 0.84884672],
            [0.23807355, 0.31961376, 0.91846033],
            [0.59884657, 0.68038678, 0.91846033],
            [0.50000031, 0.0, 0.50000031],
            [0.25000015, 0.50000027, 0.75000042],
        ],
    ]

    # UNUSED n = 5

    X = np.array(
        [
            generate_fchl18(
                nuclear_charges[i],
                fractional_coordinates[i],
                cell=cells[i],
                max_size=36,
                neighbors=200,
                cut_distance=7.0,
            )
            for i in range(5)
        ]
    )

    K = get_local_symmetric_kernels(X, **kernel_args)

    K_ref = np.array(
        [
            [530.03184304, 435.65196293, 198.61245535, 782.49428327, 263.53562172],
            [435.65196293, 371.35281119, 163.83766549, 643.99777576, 215.04338938],
            [198.61245535, 163.83766549, 76.12134823, 295.02739281, 99.89595704],
            [782.49428327, 643.99777576, 295.02739281, 1199.61736141, 389.31169487],
            [263.53562172, 215.04338938, 99.89595704, 389.31169487, 133.36920188],
        ]
    )

    assert np.allclose(K, K_ref), "Error in periodic FCHL"


def test_krr_fchl_alchemy():

    max_size = 23
    representation_options = dict(cut_distance=1e6, max_size=max_size)

    n_points = 20
    all_properties, all_representations, all_atoms = _get_training_data(
        n_points, representation_options=representation_options
    )

    shuffle_arrays(all_representations, all_atoms, all_properties, seed=666)

    np.set_printoptions(edgeitems=16, linewidth=6666)

    # TODO Put it in a numpy txt file
    overlap = np.array(
        [
            [
                1.0,
                0.00835282,
                0.90696062,
                0.82257756,
                0.61368025,
                0.37660345,
                0.19010927,
                0.07894037,
                0.02696323,
                0.00757568,
                0.67663385,
                0.61368025,
                0.45783336,
                0.28096329,
                0.14183016,
                0.05889311,
                0.02011579,
                0.0056518,
                0.41523683,
                0.37660345,
            ],
            [
                0.00835282,
                1.0,
                0.00757568,
                0.02696323,
                0.07894037,
                0.19010927,
                0.37660345,
                0.61368025,
                0.82257756,
                0.90696062,
                0.0056518,
                0.02011579,
                0.05889311,
                0.14183016,
                0.28096329,
                0.45783336,
                0.61368025,
                0.67663385,
                0.0034684,
                0.01234467,
            ],
            [
                0.90696062,
                0.00757568,
                1.0,
                0.90696062,
                0.67663385,
                0.41523683,
                0.20961139,
                0.08703837,
                0.02972922,
                0.00835282,
                0.90696062,
                0.82257756,
                0.61368025,
                0.37660345,
                0.19010927,
                0.07894037,
                0.02696323,
                0.00757568,
                0.67663385,
                0.61368025,
            ],
            [
                0.82257756,
                0.02696323,
                0.90696062,
                1.0,
                0.90696062,
                0.67663385,
                0.41523683,
                0.20961139,
                0.08703837,
                0.02972922,
                0.82257756,
                0.90696062,
                0.82257756,
                0.61368025,
                0.37660345,
                0.19010927,
                0.07894037,
                0.02696323,
                0.61368025,
                0.67663385,
            ],
            [
                0.61368025,
                0.07894037,
                0.67663385,
                0.90696062,
                1.0,
                0.90696062,
                0.67663385,
                0.41523683,
                0.20961139,
                0.08703837,
                0.61368025,
                0.82257756,
                0.90696062,
                0.82257756,
                0.61368025,
                0.37660345,
                0.19010927,
                0.07894037,
                0.45783336,
                0.61368025,
            ],
            [
                0.37660345,
                0.19010927,
                0.41523683,
                0.67663385,
                0.90696062,
                1.0,
                0.90696062,
                0.67663385,
                0.41523683,
                0.20961139,
                0.37660345,
                0.61368025,
                0.82257756,
                0.90696062,
                0.82257756,
                0.61368025,
                0.37660345,
                0.19010927,
                0.28096329,
                0.45783336,
            ],
            [
                0.19010927,
                0.37660345,
                0.20961139,
                0.41523683,
                0.67663385,
                0.90696062,
                1.0,
                0.90696062,
                0.67663385,
                0.41523683,
                0.19010927,
                0.37660345,
                0.61368025,
                0.82257756,
                0.90696062,
                0.82257756,
                0.61368025,
                0.37660345,
                0.14183016,
                0.28096329,
            ],
            [
                0.07894037,
                0.61368025,
                0.08703837,
                0.20961139,
                0.41523683,
                0.67663385,
                0.90696062,
                1.0,
                0.90696062,
                0.67663385,
                0.07894037,
                0.19010927,
                0.37660345,
                0.61368025,
                0.82257756,
                0.90696062,
                0.82257756,
                0.61368025,
                0.05889311,
                0.14183016,
            ],
            [
                0.02696323,
                0.82257756,
                0.02972922,
                0.08703837,
                0.20961139,
                0.41523683,
                0.67663385,
                0.90696062,
                1.0,
                0.90696062,
                0.02696323,
                0.07894037,
                0.19010927,
                0.37660345,
                0.61368025,
                0.82257756,
                0.90696062,
                0.82257756,
                0.02011579,
                0.05889311,
            ],
            [
                0.00757568,
                0.90696062,
                0.00835282,
                0.02972922,
                0.08703837,
                0.20961139,
                0.41523683,
                0.67663385,
                0.90696062,
                1.0,
                0.00757568,
                0.02696323,
                0.07894037,
                0.19010927,
                0.37660345,
                0.61368025,
                0.82257756,
                0.90696062,
                0.0056518,
                0.02011579,
            ],
            [
                0.67663385,
                0.0056518,
                0.90696062,
                0.82257756,
                0.61368025,
                0.37660345,
                0.19010927,
                0.07894037,
                0.02696323,
                0.00757568,
                1.0,
                0.90696062,
                0.67663385,
                0.41523683,
                0.20961139,
                0.08703837,
                0.02972922,
                0.00835282,
                0.90696062,
                0.82257756,
            ],
            [
                0.61368025,
                0.02011579,
                0.82257756,
                0.90696062,
                0.82257756,
                0.61368025,
                0.37660345,
                0.19010927,
                0.07894037,
                0.02696323,
                0.90696062,
                1.0,
                0.90696062,
                0.67663385,
                0.41523683,
                0.20961139,
                0.08703837,
                0.02972922,
                0.82257756,
                0.90696062,
            ],
            [
                0.45783336,
                0.05889311,
                0.61368025,
                0.82257756,
                0.90696062,
                0.82257756,
                0.61368025,
                0.37660345,
                0.19010927,
                0.07894037,
                0.67663385,
                0.90696062,
                1.0,
                0.90696062,
                0.67663385,
                0.41523683,
                0.20961139,
                0.08703837,
                0.61368025,
                0.82257756,
            ],
            [
                0.28096329,
                0.14183016,
                0.37660345,
                0.61368025,
                0.82257756,
                0.90696062,
                0.82257756,
                0.61368025,
                0.37660345,
                0.19010927,
                0.41523683,
                0.67663385,
                0.90696062,
                1.0,
                0.90696062,
                0.67663385,
                0.41523683,
                0.20961139,
                0.37660345,
                0.61368025,
            ],
            [
                0.14183016,
                0.28096329,
                0.19010927,
                0.37660345,
                0.61368025,
                0.82257756,
                0.90696062,
                0.82257756,
                0.61368025,
                0.37660345,
                0.20961139,
                0.41523683,
                0.67663385,
                0.90696062,
                1.0,
                0.90696062,
                0.67663385,
                0.41523683,
                0.19010927,
                0.37660345,
            ],
            [
                0.05889311,
                0.45783336,
                0.07894037,
                0.19010927,
                0.37660345,
                0.61368025,
                0.82257756,
                0.90696062,
                0.82257756,
                0.61368025,
                0.08703837,
                0.20961139,
                0.41523683,
                0.67663385,
                0.90696062,
                1.0,
                0.90696062,
                0.67663385,
                0.07894037,
                0.19010927,
            ],
            [
                0.02011579,
                0.61368025,
                0.02696323,
                0.07894037,
                0.19010927,
                0.37660345,
                0.61368025,
                0.82257756,
                0.90696062,
                0.82257756,
                0.02972922,
                0.08703837,
                0.20961139,
                0.41523683,
                0.67663385,
                0.90696062,
                1.0,
                0.90696062,
                0.02696323,
                0.07894037,
            ],
            [
                0.0056518,
                0.67663385,
                0.00757568,
                0.02696323,
                0.07894037,
                0.19010927,
                0.37660345,
                0.61368025,
                0.82257756,
                0.90696062,
                0.00835282,
                0.02972922,
                0.08703837,
                0.20961139,
                0.41523683,
                0.67663385,
                0.90696062,
                1.0,
                0.00757568,
                0.02696323,
            ],
            [
                0.41523683,
                0.0034684,
                0.67663385,
                0.61368025,
                0.45783336,
                0.28096329,
                0.14183016,
                0.05889311,
                0.02011579,
                0.0056518,
                0.90696062,
                0.82257756,
                0.61368025,
                0.37660345,
                0.19010927,
                0.07894037,
                0.02696323,
                0.00757568,
                1.0,
                0.90696062,
            ],
            [
                0.37660345,
                0.01234467,
                0.61368025,
                0.67663385,
                0.61368025,
                0.45783336,
                0.28096329,
                0.14183016,
                0.05889311,
                0.02011579,
                0.82257756,
                0.90696062,
                0.82257756,
                0.61368025,
                0.37660345,
                0.19010927,
                0.07894037,
                0.02696323,
                0.90696062,
                1.0,
            ],
        ]
    )

    kernel_args = {
        "alchemy": "periodic-table",
        "kernel": "gaussian",
        "kernel_args": {
            "sigma": [2.5],
        },
    }

    K_alchemy = get_local_symmetric_kernels(all_representations, **kernel_args)[0]

    kernel_args = {
        "alchemy": overlap,
        "kernel": "gaussian",
        "kernel_args": {
            "sigma": [2.5],
        },
    }

    K_custom = get_local_symmetric_kernels(all_representations, **kernel_args)[0]

    assert np.allclose(K_alchemy, K_custom), "Error in alchemy"

    nooverlap = np.eye(100)

    kernel_args = {
        "alchemy": "off",
        "kernel": "gaussian",
        "kernel_args": {
            "sigma": [2.5],
        },
    }

    K_noalchemy = get_local_symmetric_kernels(all_representations, **kernel_args)[0]

    kernel_args = {
        "alchemy": nooverlap,
        "kernel": "gaussian",
        "kernel_args": {
            "sigma": [2.5],
        },
    }
    K_custom = get_local_symmetric_kernels(all_representations, **kernel_args)[0]

    assert np.allclose(K_noalchemy, K_custom), "Error in no-alchemy"


def test_fchl_linear():

    n_points = 5
    _, representations, atoms = _get_training_data(n_points)

    K = get_local_symmetric_kernels(representations)[0]

    K_test = np.zeros((n_points, n_points))

    kernel_args = {
        "kernel": "linear",
        "kernel_args": {
            "c": [1.0],
        },
    }

    for i, Xi in enumerate(representations):
        Sii = get_atomic_kernels(Xi[: len(atoms[i])], Xi[: len(atoms[i])], **kernel_args)[0]
        for j, Xj in enumerate(representations):

            Sjj = get_atomic_kernels(Xj[: len(atoms[j])], Xj[: len(atoms[j])], **kernel_args)[0]

            Sij = get_atomic_kernels(Xi[: len(atoms[i])], Xj[: len(atoms[j])], **kernel_args)[0]

            for ii in range(Sii.shape[0]):
                for jj in range(Sjj.shape[0]):

                    l2 = Sii[ii, ii] + Sjj[jj, jj] - 2 * Sij[ii, jj]
                    K_test[i, j] += np.exp(-l2 / (2 * (2.5**2)))

    assert np.allclose(K, K_test), "Error in FCHL linear kernels"


def test_fchl_polynomial():

    n_points = 5
    _, representations, atoms = _get_training_data(n_points)

    polynomial_kernel_args = {
        "kernel": "polynomial",
        "kernel_args": {
            "alpha": [2.0],
            "c": [3.0],
            "d": [4.0],
        },
    }

    linear_kernel_args = {
        "kernel": "linear",
        "kernel_args": {
            "c": [0.0],
        },
    }

    K = get_local_symmetric_kernels(representations, **polynomial_kernel_args)[0]

    K_test = np.zeros((n_points, n_points))

    for i, Xi in enumerate(representations):
        for j, Xj in enumerate(representations):

            Sij = get_atomic_kernels(
                Xi[: len(atoms[i])], Xj[: len(atoms[j])], **linear_kernel_args
            )[0]

            for ii in range(Sij.shape[0]):
                for jj in range(Sij.shape[1]):

                    K_test[i, j] += (2.0 * Sij[ii, jj] + 3.0) ** 4.0

    assert np.allclose(K, K_test), "Error in FCHL polynomial kernels"


def test_fchl_sigmoid():

    n_points = 5
    _, representations, atoms = _get_training_data(n_points)

    sigmoid_kernel_args = {
        "kernel": "sigmoid",
        "kernel_args": {
            "alpha": [2.0],
            "c": [3.0],
        },
    }

    linear_kernel_args = {
        "kernel": "linear",
        "kernel_args": {
            "c": [0.0],
        },
    }

    K = get_local_symmetric_kernels(representations, **sigmoid_kernel_args)[0]

    K_test = np.zeros((n_points, n_points))

    for i, Xi in enumerate(representations):
        for j, Xj in enumerate(representations):

            Sij = get_atomic_kernels(
                Xi[: len(atoms[i])], Xj[: len(atoms[j])], **linear_kernel_args
            )[0]

            for ii in range(Sij.shape[0]):
                for jj in range(Sij.shape[1]):

                    # K_test[i,j] += (2.0 * Sij[ii,jj] + 3.0)**4.0
                    K_test[i, j] += np.tanh(2.0 * Sij[ii, jj] + 3.0)

    assert np.allclose(K, K_test), "Error in FCHL sigmoid kernels"


def test_fchl_multiquadratic():

    n_points = 5
    _, representations, atoms = _get_training_data(n_points)

    kernel_args = {
        "kernel": "multiquadratic",
        "kernel_args": {
            "c": [2.0],
        },
    }

    linear_kernel_args = {
        "kernel": "linear",
        "kernel_args": {
            "c": [0.0],
        },
    }

    K = get_local_symmetric_kernels(representations, **kernel_args)[0]

    K_test = np.zeros((n_points, n_points))

    for i, Xi in enumerate(representations):
        Sii = get_atomic_kernels(Xi[: len(atoms[i])], Xi[: len(atoms[i])], **linear_kernel_args)[0]
        for j, Xj in enumerate(representations):

            Sjj = get_atomic_kernels(
                Xj[: len(atoms[j])], Xj[: len(atoms[j])], **linear_kernel_args
            )[0]
            Sij = get_atomic_kernels(
                Xi[: len(atoms[i])], Xj[: len(atoms[j])], **linear_kernel_args
            )[0]

            for ii in range(Sii.shape[0]):
                for jj in range(Sjj.shape[0]):

                    l2 = Sii[ii, ii] + Sjj[jj, jj] - 2 * Sij[ii, jj]
                    K_test[i, j] += np.sqrt(l2 + 4.0)

    assert np.allclose(K, K_test), "Error in FCHL multiquadratic kernels"


def test_fchl_inverse_multiquadratic():

    n_points = 5
    _, representations, atoms = _get_training_data(n_points)

    kernel_args = {
        "kernel": "inverse-multiquadratic",
        "kernel_args": {
            "c": [2.0],
        },
    }

    linear_kernel_args = {
        "kernel": "linear",
        "kernel_args": {
            "c": [0.0],
        },
    }

    K = get_local_symmetric_kernels(representations, **kernel_args)[0]

    K_test = np.zeros((n_points, n_points))

    for i, Xi in enumerate(representations):
        Sii = get_atomic_kernels(Xi[: len(atoms[i])], Xi[: len(atoms[i])], **linear_kernel_args)[0]
        for j, Xj in enumerate(representations):

            Sjj = get_atomic_kernels(
                Xj[: len(atoms[j])], Xj[: len(atoms[j])], **linear_kernel_args
            )[0]
            Sij = get_atomic_kernels(
                Xi[: len(atoms[i])], Xj[: len(atoms[j])], **linear_kernel_args
            )[0]

            for ii in range(Sii.shape[0]):
                for jj in range(Sjj.shape[0]):

                    l2 = Sii[ii, ii] + Sjj[jj, jj] - 2 * Sij[ii, jj]
                    K_test[i, j] += 1.0 / np.sqrt(l2 + 4.0)
    assert np.allclose(K, K_test), "Error in FCHL inverse multiquadratic kernels"


def test_fchl_bessel():

    n_points = 5
    _, representations, atoms = _get_training_data(n_points)

    kernel_args = {
        "kernel": "bessel",
        "kernel_args": {
            "sigma": [2.0],
            "v": [3.0],
            "n": [2.0],
        },
    }

    linear_kernel_args = {
        "kernel": "linear",
        "kernel_args": {
            "c": [0.0],
        },
    }

    K = get_local_symmetric_kernels(representations, **kernel_args)[0]

    K_test = np.zeros((n_points, n_points))

    sigma = 2.0
    v = 3
    n = 2

    for i, Xi in enumerate(representations):
        Sii = get_atomic_kernels(Xi[: len(atoms[i])], Xi[: len(atoms[i])], **linear_kernel_args)[0]
        for j, Xj in enumerate(representations):

            Sjj = get_atomic_kernels(
                Xj[: len(atoms[j])], Xj[: len(atoms[j])], **linear_kernel_args
            )[0]
            Sij = get_atomic_kernels(
                Xi[: len(atoms[i])], Xj[: len(atoms[j])], **linear_kernel_args
            )[0]

            for ii in range(Sii.shape[0]):
                for jj in range(Sjj.shape[0]):

                    # UNUSED l2 = np.sqrt(Sii[ii, ii] + Sjj[jj, jj] - 2 * Sij[ii, jj])

                    K_test[i, j] += jn(v, sigma * Sij[ii, jj]) / Sij[ii, jj] ** (-n * (v + 1))

    assert np.allclose(K, K_test), "Error in FCHL inverse bessel kernels"


def test_fchl_l2():

    n_points = 5
    _, representations, atoms = _get_training_data(n_points)

    l2_kernel_args = {
        "kernel": "l2",
        "kernel_args": {
            "alpha": [1.0],
            "c": [0.0],
        },
    }

    K = get_local_symmetric_kernels(representations)[0]

    K_test = np.zeros((n_points, n_points))

    print(K)

    # UNUSED sigma = 2.0
    # UNUSED v = 3
    # UNUSED n = 2

    inv_sigma = -1.0 / (2.0 * 2.5**2)

    for i, Xi in enumerate(representations):
        for j, Xj in enumerate(representations):

            Sij = get_atomic_kernels(Xi[: len(atoms[i])], Xj[: len(atoms[j])], **l2_kernel_args)[0]

            for ii in range(Sij.shape[0]):
                for jj in range(Sij.shape[1]):

                    K_test[i, j] += np.exp(Sij[ii, jj] * inv_sigma)

    print(K_test)
    print(np.max(K - K_test))

    assert np.allclose(K, K_test), "Error in FCHL l2 kernels"


def test_fchl_matern():

    n_points = 5
    _, representations, atoms = _get_training_data(n_points)

    kernel_args = {
        "kernel": "matern",
        "kernel_args": {
            "sigma": [5.0],
            "n": [2.0],
        },
    }

    linear_kernel_args = {
        "kernel": "linear",
        "kernel_args": {
            "c": [0.0],
        },
    }

    K = get_local_symmetric_kernels(representations, **kernel_args)[0]

    K_test = np.zeros((n_points, n_points))

    sigma = 5.0
    n = 2
    v = n + 0.5

    for i, Xi in enumerate(representations):
        Sii = get_atomic_kernels(Xi[: len(atoms[i])], Xi[: len(atoms[i])], **linear_kernel_args)[0]
        for j, Xj in enumerate(representations):

            Sjj = get_atomic_kernels(
                Xj[: len(atoms[j])], Xj[: len(atoms[j])], **linear_kernel_args
            )[0]
            Sij = get_atomic_kernels(
                Xi[: len(atoms[i])], Xj[: len(atoms[j])], **linear_kernel_args
            )[0]

            for ii in range(Sii.shape[0]):
                for jj in range(Sjj.shape[0]):

                    l2 = np.sqrt(Sii[ii, ii] + Sjj[jj, jj] - 2 * Sij[ii, jj])

                    rho = 2 * np.sqrt(2 * v) * l2 / sigma

                    for k in range(0, n + 1):
                        fact = float(factorial(n + k)) / factorial(2 * n) * binom(n, k)
                        K_test[i, j] += np.exp(-0.5 * rho) * fact * rho ** (n - k)

    assert np.allclose(K, K_test), "Error in FCHL matern kernels"


def test_fchl_cauchy():

    n_points = 5
    _, representations, atoms = _get_training_data(n_points)

    kernel_args = {
        "kernel": "cauchy",
        "kernel_args": {
            "sigma": [2.0],
        },
    }

    linear_kernel_args = {
        "kernel": "linear",
        "kernel_args": {
            "c": [0.0],
        },
    }

    K = get_local_symmetric_kernels(representations, **kernel_args)[0]

    K_test = np.zeros((n_points, n_points))

    for i, Xi in enumerate(representations):
        Sii = get_atomic_kernels(Xi[: len(atoms[i])], Xi[: len(atoms[i])], **linear_kernel_args)[0]
        for j, Xj in enumerate(representations):

            Sjj = get_atomic_kernels(
                Xj[: len(atoms[j])], Xj[: len(atoms[j])], **linear_kernel_args
            )[0]
            Sij = get_atomic_kernels(
                Xi[: len(atoms[i])], Xj[: len(atoms[j])], **linear_kernel_args
            )[0]

            for ii in range(Sii.shape[0]):
                for jj in range(Sjj.shape[0]):

                    l2 = Sii[ii, ii] + Sjj[jj, jj] - 2 * Sij[ii, jj]
                    K_test[i, j] += 1.0 / (1.0 + l2 / 2.0**2)

    assert np.allclose(K, K_test), "Error in FCHL cauchy kernels"


def test_fchl_polynomial2():

    n_points = 5
    _, representations, atoms = _get_training_data(n_points)

    kernel_args = {
        "kernel": "polynomial2",
        "kernel_args": {
            "coeff": [[1.0, 2.0, 3.0]],
        },
    }

    linear_kernel_args = {
        "kernel": "linear",
        "kernel_args": {
            "c": [0.0],
        },
    }

    K = get_local_symmetric_kernels(representations, **kernel_args)[0]

    K_test = np.zeros((n_points, n_points))

    for i, Xi in enumerate(representations):
        Sii = get_atomic_kernels(Xi[: len(atoms[i])], Xi[: len(atoms[i])], **linear_kernel_args)[0]
        for j, Xj in enumerate(representations):

            Sjj = get_atomic_kernels(
                Xj[: len(atoms[j])], Xj[: len(atoms[j])], **linear_kernel_args
            )[0]
            Sij = get_atomic_kernels(
                Xi[: len(atoms[i])], Xj[: len(atoms[j])], **linear_kernel_args
            )[0]

            for ii in range(Sii.shape[0]):
                for jj in range(Sjj.shape[0]):

                    K_test[i, j] += 1.0 + 2.0 * Sij[ii, jj] + 3.0 * Sij[ii, jj] ** 2

    assert np.allclose(K, K_test), "Error in FCHL polynomial2 kernels"
