"""
Created on Sat Oct 21 18:39:20 2023

@author: dagpa
"""
import typing

import numpy as np

#FIXME =============================================================================
# FUNZIONI GENERALI
# N.B. citare e ringraziare JANKO E PYEMA!!
# (SDypy https://github.com/sdypy/sdypy)
# =============================================================================


def merge_mode_shapes(
    MSarr_list: typing.List[np.ndarray], reflist: typing.List[typing.List[int]]
) -> np.ndarray:
    """
Merges multiple mode shape arrays from different setups into a single mode shape array.

This function combines mode shape arrays from different experimental setups into a single array. 
It uses a list of reference sensor indices for each setup to align and scale the mode shapes. 
The function ensures that all mode shape arrays have the same number of modes and merges them 
by keeping the reference sensor data from the first setup and appropriately scaling and appending 
the data from roving sensors of subsequent setups.

Parameters:
- MSarr_list (List[np.ndarray]): A list of mode shape arrays. Each array in the list corresponds 
  to a different experimental setup. Each array should have dimensions [N x M], where N is the number 
  of sensors (including both reference and roving sensors) and M is the number of modes.
- reflist (List[List[int]]): A list of lists containing the indices of reference sensors. Each sublist 
  corresponds to the indices of the reference sensors used in the corresponding setup in `MSarr_list`. 
  Each sublist should contain the same number of elements.

Returns:
- np.ndarray: A merged mode shape array. The number of rows in the array equals the sum of the number 
  of unique sensors across all setups minus the number of reference sensors in each setup (except the first one). 
  The number of columns equals the number of modes.

Raises:
- ValueError: If the mode shape arrays in `MSarr_list` do not have the same number of modes.
"""
    Nsetup = len(MSarr_list)  # number of setup
    Nmodes = MSarr_list[0].shape[1]  # number of modes
    Nref = len(reflist[0])  # number of reference sensors
    M = Nref + np.sum(
        [MSarr_list[i].shape[0] - Nref for i in range(Nsetup)]
    )  # total number of nodes in a mode shape
    # Check if the input arrays have consistent dimensions
    for i in range(1, Nsetup):
        if MSarr_list[i].shape[1] != Nmodes:
            raise ValueError("All mode shape arrays must have the same number of modes.")
    # Initialize merged mode shape array
    merged_mode_shapes = np.zeros((M, Nmodes))
    # Loop through each mode
    for k in range(Nmodes):
        phi_1_k = MSarr_list[0][:, k]  # Save the mode shape from first setup
        phi_ref_1_k = phi_1_k[reflist[0]]  # Save the reference sensors
        merged_mode_k = np.concatenate(
            (phi_ref_1_k, np.delete(phi_1_k, reflist[0]))
        )  # initialise the merged mode shape
        # Loop through each setup
        for i in range(1, Nsetup):
            ref_ind = reflist[i]  # reference sensors indices for the specific setup
            phi_i_k = MSarr_list[i][:, k]  # mode shape of setup i
            phi_ref_i_k = MSarr_list[i][ref_ind, k]  # save data from reference sensors
            phi_rov_i_k = np.delete(
                phi_i_k, ref_ind, axis=0
            )  # saave data from roving sensors
            # Find scaling factor
            alpha_i_k = MSF(phi_ref_1_k, phi_ref_i_k)
            # Merge mode
            merged_mode_k = np.hstack((merged_mode_k, alpha_i_k * phi_rov_i_k))

        merged_mode_shapes[:, k] = merged_mode_k

    return merged_mode_shapes


# -----------------------------------------------------------------------------


def MSF(phi_1, phi_2):
    """Modal Scale Factor.

    If ``phi_1`` and ``phi_2`` are matrices, multiple msf are returned.

    The MAF scales ``phi_1`` to ``phi_2`` when multiplying: ``msf*phi_1``.
    Also takes care of 180 deg phase difference.

    :param phi_1: Mode shape matrix X, shape: ``(n_locations, n_modes)``
        or ``n_locations``.
    :param phi_2: Mode shape matrix A, shape: ``(n_locations, n_modes)``
        or ``n_locations``.
    :return: np.ndarray, MSF values
    """
    if phi_1.ndim == 1:
        phi_1 = phi_1[:, None]
    if phi_2.ndim == 1:
        phi_2 = phi_2[:, None]

    if phi_1.shape[0] != phi_2.shape[0] or phi_1.shape[1] != phi_2.shape[1]:
        raise Exception(
            f"`phi_1` and `phi_2` must have the same shape: {phi_1.shape} "
            f"and {phi_2.shape}"
        )

    n_modes = phi_1.shape[1]
    msf = []
    for i in range(n_modes):
        _msf = np.dot(phi_2[:, i].T, phi_1[:, i]) / np.dot(phi_1[:, i].T, phi_1[:, i])

        msf.append(_msf)

    return np.array(msf).real


# -----------------------------------------------------------------------------


def MCF(phi):
    """Modal complexity factor.

    The MCF ranges from 0 to 1. It returns 0 for real modes and 1 for complex modes.
    When ``dtype`` of ``phi`` is ``complex``, the modes can still be real, if the angles
    of all components are the same.

    Additional information on MCF:
    http://www.svibs.com/resources/ARTeMIS_Modal_Help/Generic%20Complexity%20Plot.html

    :param phi: Complex mode shape matrix, shape: ``(n_locations, n_modes)``
        or ``n_locations``.
    :return: MCF (a value between 0 and 1)
    """
    if phi.ndim == 1:
        phi = phi[:, None]
    n_modes = phi.shape[1]
    mcf = []
    for i in range(n_modes):
        S_xx = np.dot(phi[:, i].real, phi[:, i].real)
        S_yy = np.dot(phi[:, i].imag, phi[:, i].imag)
        S_xy = np.dot(phi[:, i].real, phi[:, i].imag)

        _mcf = 1 - ((S_xx - S_yy) ** 2 + 4 * S_xy**2) / (S_xx + S_yy) ** 2

        mcf.append(_mcf)
    return np.array(mcf)


# -----------------------------------------------------------------------------


def MAC(phi_X, phi_A):
    """Modal Assurance Criterion.

    The number of locations (axis 0) must be the same for ``phi_X`` and
    ``phi_A``. The nubmer of modes (axis 1) is arbitrary.

    Literature:
        [1] Maia, N. M. M., and J. M. M. Silva.
            "Modal analysis identification techniques." Philosophical
            Transactions of the Royal Society of London. Series A:
            Mathematical, Physical and Engineering Sciences 359.1778
            (2001): 29-40.

    :param phi_X: Mode shape matrix X, shape: ``(n_locations, n_modes)``
        or ``n_locations``.
    :param phi_A: Mode shape matrix A, shape: ``(n_locations, n_modes)``
        or ``n_locations``.
    :return: MAC matrix. Returns MAC value if both ``phi_X`` and ``phi_A`` are
        one-dimensional arrays.
    """
    if phi_X.ndim == 1:
        phi_X = phi_X[:, np.newaxis]

    if phi_A.ndim == 1:
        phi_A = phi_A[:, np.newaxis]

    if phi_X.ndim > 2 or phi_A.ndim > 2:
        raise Exception(
            f"Mode shape matrices must have 1 or 2 dimensions (phi_X: {phi_X.ndim}, phi_A: {phi_A.ndim})"
        )

    if phi_X.shape[0] != phi_A.shape[0]:
        raise Exception(
            f"Mode shapes must have the same first dimension (phi_X: {phi_X.shape[0]}, "
            f"phi_A: {phi_A.shape[0]})"
        )

    # mine
    MAC = np.abs(np.dot(phi_X.conj().T, phi_A)) ** 2 / (
        (np.dot(phi_X.conj().T, phi_X)) * (np.dot(phi_A.conj().T, phi_A))
    )
    # original
    # MAC = np.abs(np.conj(phi_X).T @ phi_A)**2
    # for i in range(phi_X.shape[1]):
    #     for j in range(phi_A.shape[1]):
    #         MAC[i, j] = MAC[i, j] /\
    #             (np.conj(phi_X[:, i]) @ phi_X[:, i] *
    #              np.conj(phi_A[:, j]) @ phi_A[:, j])

    if MAC.shape == (1, 1):
        MAC = MAC[0, 0]

    return MAC


# -----------------------------------------------------------------------------


def PRE_MultiSetup(
    DataList: typing.List[np.ndarray], reflist: typing.List[typing.List[int]]
) -> typing.List[typing.Dict[str, np.ndarray]]:
    """
    Preprocesses multiple setups of data by separating reference and moving
    sensor information.

    Parameters:
    - DataList (list of numpy arrays): List of input data arrays for each
        setup, where each array represents sensor data for a setup.
    - reflist (list of lists): List of lists containing indices of sensors to
        be used as references for each setup.

    Returns:
    - list of dictionaries: A list of dictionaries, each containing the
        data for a setup.
        Each dictionary has the following keys:
            - 'ref': Numpy array of reference sensor data reshaped to
                (number_of_references, number_of_data_points).
            - 'mov': Numpy array of moving sensor data reshaped to
                (number_of_moving_sensors, number_of_data_points).
    """
    n_setup = len(DataList)  # number of setup
    Y = []
    for i in range(n_setup):
        y = DataList[i]
        n_ref = len(reflist[i])
        n_sens = y.shape[1]
        ref_id = reflist[i]
        mov_id = list(range(n_sens))
        for ii in range(n_ref):
            mov_id.remove(ref_id[ii])
        ref = y[:, ref_id]
        mov = y[:, mov_id]
        # TO DO: check that len(n_ref) is the same in all setup

        # N.B. ONLY FOR TEST
        # Y.append({"ref": np.array(ref).reshape(n_ref,-1)})
        Y.append(
            {
                "ref": np.array(ref).T.reshape(n_ref, -1),
                "mov": np.array(mov).T.reshape(
                    (n_sens - n_ref),
                    -1,
                ),
            }
        )

    return Y


# -----------------------------------------------------------------------------


def invperm(p):
    """
    Compute the inverse permutation of a given array.

    Given a permutation array `p`, this function creates an array `q` such 
    that `q[p[i]] = i` for each element `i` in `p`. This effectively computes 
    the inverse of the permutation represented by `p`.

    Parameters:
    p (array-like): A permutation of integers from 0 to n-1, where n is the 
                    length of the array.

    Returns:
    ndarray: An array representing the inverse permutation of `p`.

    Example:
    >>> invperm(np.array([3, 0, 2, 1]))
    array([1, 3, 2, 0])
    """
    q = np.empty_like(p)
    q[p] = np.arange(len(p))
    return q


# -----------------------------------------------------------------------------


def find_map(arr1, arr2):
    """
    Map the elements of one array to another based on sorting order.

    This function finds a mapping from the elements of `arr1` to `arr2` such 
    that the order of elements in `arr1` is preserved in `arr2`. It first 
    sorts both arrays and then uses the inverse permutation of the sorting 
    order of `arr1` to map its elements to the corresponding elements in `arr2`.

    Parameters:
    arr1 (array-like): The first input array.
    arr2 (array-like): The second input array, which should have the same 
                       length as `arr1`.

    Returns:
    ndarray: An array of indices that maps the sorted version of `arr1` to 
             the sorted version of `arr2`.

    Example:
    >>> find_map(np.array([10, 30, 20]), np.array([3, 2, 1]))
    array([2, 0, 1])
    """
    o1 = np.argsort(arr1)
    o2 = np.argsort(arr2)
    return o2[invperm(o1)]