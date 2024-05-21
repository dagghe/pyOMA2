# -*- coding: utf-8 -*-
"""
General Utility Functions module.
Part of the pyOMA2 package.
Author:
Dag Pasca
"""

import logging
import typing

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

# =============================================================================
# FUNZIONI GENERALI
# =============================================================================


def lab_stab(
    Fn: np.ndarray,
    Sm: np.ndarray,
    Ms: np.ndarray,
    ordmin: int,
    ordmax: int,
    step: int,
    err_fn: float,
    err_xi: float,
    err_ms: float,
    max_xi: float,
    mpc_lim: typing.Optional[float] = None,
    mpd_lim: typing.Optional[float] = None,
):
    """
    Construct a Stability Chart for modal analysis.

    This function evaluates the stability of identified modes based on their behavior
    across different model orders. It generates a label matrix where each element
    represents the stability of a mode at a specific order.

    Parameters
    ----------
    Fn : numpy.ndarray
        Array of frequency poles for each model order.
    Sm : numpy.ndarray
        Array of damping ratios for each model order.
    Ms : numpy.ndarray
        3D array of mode shapes for each model order.
    ordmin : int
        Minimum model order to consider.
    ordmax : int
        Maximum model order to consider.
    step : int
        Step size when iterating through model orders.
    err_fn : float
        Threshold for relative frequency difference for stability checks.
    err_xi : float
        Threshold for relative damping ratio difference for stability checks.
    err_ms : float
        Threshold for Modal Assurance Criterion (MAC) for stability checks.
    max_xi : float
        Maximum allowed damping ratio.

    Returns
    -------
    numpy.ndarray
        Stability label matrix (Lab),
            where each element represents the stability
        category of a mode at a specific model order.

    Notes
    -----
    Stability is assessed based on the consistency of mode frequency, damping, and
    shape across successive model orders. Different stability categories are assigned
    based on how closely each mode adheres to the specified thresholds.
    """
    Lab = np.zeros(Fn.shape, dtype="int")

    # -----------------------------------------------------------------------------
    # REMOVING HARD CONDITIONS
    # Create Mask array to pick only damping xi, which are xi> 0 and xi<max_xi
    Mask = np.logical_and(Sm < max_xi, Sm > 0).astype(int)
    # Mask Damping Array
    Sm1 = Sm * Mask
    Sm1[Sm1 == 0] = np.nan
    # Mask Frequency Array
    Fn1 = Fn * Mask
    Fn1[Fn1 == 0] = np.nan
    # Mask ModeShape array (N.B. modify mask to fit the MS dimension)
    nDOF = Ms.shape[2]
    MaskMS = np.repeat(Mask[:, :, np.newaxis], nDOF, axis=2)
    Ms1 = Ms * MaskMS
    Ms1[Ms1 == 0] = np.nan
    # -----------------------------------------------------------------------------
    # Checking MPC AND MPD
    if mpc_lim is not None:
        Mask1 = []
        for o in range(Fn1.shape[0]):
            for i in range(Fn1.shape[1]):
                try:
                    Mask1.append((MPC(Ms1[o, i, :]) >= mpc_lim).astype(int))
                except Exception:
                    Mask1.append(0)
        Mask1 = np.array(Mask1).reshape(Fn1.shape)
        Fn1 = Fn1 * Mask1
        Fn1[Fn1 == 0] = np.nan
        Sm1 = Sm1 * Mask1
        Sm1[Sm1 == 0] = np.nan

    if mpd_lim is not None:
        Mask2 = []
        for o in range(Fn1.shape[0]):
            for i in range(Fn1.shape[1]):
                try:
                    Mask2.append((MPD(Ms1[o, i, :]) <= mpd_lim).astype(int))
                except Exception:
                    Mask2.append(0)
        Mask2 = np.array(Mask2).reshape(Fn1.shape)
        Fn1 = Fn1 * Mask2
        Fn1[Fn1 == 0] = np.nan
        Sm1 = Sm1 * Mask2
        Sm1[Sm1 == 0] = np.nan
    # -----------------------------------------------------------------------------
    # STABILITY BETWEEN CONSECUTIVE ORDERS
    for o in range(ordmin, ordmax, step):
        ii = int((o - ordmin) / step)

        f_n = Fn1[:, ii].reshape(-1, 1)
        xi_n = Sm1[:, ii].reshape(-1, 1)
        phi_n = Ms1[:, ii, :]

        f_n1 = Fn1[:, ii - 1].reshape(-1, 1)
        xi_n1 = Sm1[:, ii - 1].reshape(-1, 1)
        phi_n1 = Ms1[:, ii - 1, :]

        if ii != 0 and ii != 1:
            for i in range(len(f_n)):
                try:
                    idx = np.nanargmin(np.abs(f_n1 - f_n[i]))

                    cond1 = np.abs(f_n[i] - f_n1[idx]) / f_n[i]
                    cond2 = np.abs(xi_n[i] - xi_n1[idx]) / xi_n[i]
                    cond3 = 1 - MAC(phi_n[i, :], phi_n1[idx, :])
                    if cond1 < err_fn and cond2 < err_xi and cond3 < err_ms:
                        Lab[i, ii] = 7  # Stable

                    elif cond1 < err_fn and cond3 < err_ms:
                        # Stable frequency, stable mode shape
                        Lab[i, ii] = 6

                    elif cond1 < err_fn and cond2 < err_xi:
                        Lab[i, ii] = 5  # Stable frequency, stable damping

                    elif cond2 < err_xi and cond3 < err_ms:
                        Lab[i, ii] = 4  # Stable damping, stable mode shape

                    elif cond2 < err_xi:
                        Lab[i, ii] = 3  # Stable damping

                    elif cond3 < err_ms:
                        Lab[i, ii] = 2  # Stable mode shape

                    elif cond1 < err_fn:
                        Lab[i, ii] = 1  # Stable frequency

                    else:
                        Lab[i, ii] = 0  # Nuovo polo o polo instabile
                except Exception as e:
                    # If f_n[i] is nan, do nothin, n.b. the lab stays 0
                    logger.debug(e)
    return Lab


def merge_mode_shapes(
    MSarr_list: typing.List[np.ndarray], reflist: typing.List[typing.List[int]]
) -> np.ndarray:
    """
    Merges multiple mode shape arrays from different setups into a single mode shape array.

    Parameters
    ----------
    MSarr_list : List[np.ndarray]
        A list of mode shape arrays. Each array in the list corresponds
        to a different experimental setup. Each array should have dimensions [N x M], where N is the number
        of sensors (including both reference and roving sensors) and M is the number of modes.
    reflist : List[List[int]]
        A list of lists containing the indices of reference sensors. Each sublist
        corresponds to the indices of the reference sensors used in the corresponding setup in `MSarr_list`.
        Each sublist should contain the same number of elements.

    Returns
    -------
    np.ndarray
        A merged mode shape array. The number of rows in the array equals the sum of the number
        of unique sensors across all setups minus the number of reference sensors in each setup
        (except the first one). The number of columns equals the number of modes.

    Raises
    ------
    ValueError
        If the mode shape arrays in `MSarr_list` do not have the same number of modes.
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
    merged_mode_shapes = np.zeros((M, Nmodes)).astype(complex)
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


def MPC(phi: np.ndarray) -> float:
    """
    Modal phase collinearity
    """
    S = np.cov(phi.real, phi.imag)
    lambd = np.linalg.eigvals(S)
    MPC = (lambd[0] - lambd[1]) ** 2 / (lambd[0] + lambd[1]) ** 2
    return MPC


def MPD(phi: np.ndarray) -> float:
    """
    Mean phase deviation
    """

    U, s, VT = np.linalg.svd(np.c_[phi.real, phi.imag])
    V = VT.T
    w = np.abs(phi)
    num = phi.real * V[1, 1] - phi.imag * V[0, 1]
    den = np.sqrt(V[0, 1] ** 2 + V[1, 1] ** 2) * np.abs(phi)
    MPD = np.sum(w * np.arccos(np.abs(num / den))) / np.sum(w)
    return MPD


def MSF(phi_1: np.ndarray, phi_2: np.ndarray) -> np.ndarray:
    """
    Calculates the Modal Scale Factor (MSF) between two sets of mode shapes.

    Parameters
    ----------
    phi_1 : ndarray
        Mode shape matrix X, shape: (n_locations, n_modes) or n_locations.
    phi_2 : ndarray
        Mode shape matrix A, shape: (n_locations, n_modes) or n_locations.

    Returns
    -------
    ndarray
        The MSF values, real numbers that scale `phi_1` to `phi_2`.

    Raises
    ------
    Exception
        If `phi_1` and `phi_2` do not have the same shape.
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


def MCF(phi: np.ndarray) -> np.ndarray:
    """
    Calculates the Modal Complexity Factor (MCF) for mode shapes.

    Parameters
    ----------
    phi : ndarray
        Complex mode shape matrix, shape: (n_locations, n_modes) or n_locations.

    Returns
    -------
    ndarray
        MCF values, ranging from 0 (for real modes) to 1 (for complex modes).
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


def MAC(phi_X: np.ndarray, phi_A: np.ndarray) -> np.ndarray:
    """
    Calculates the Modal Assurance Criterion (MAC) between two sets of mode shapes.

    Parameters
    ----------
    phi_X : ndarray
        Mode shape matrix X, shape: (n_locations, n_modes) or n_locations.
    phi_A : ndarray
        Mode shape matrix A, shape: (n_locations, n_modes) or n_locations.

    Returns
    -------
    ndarray
        MAC matrix. Returns a single MAC value if both `phi_X` and `phi_A` are
        one-dimensional arrays.

    Raises
    ------
    Exception
        If mode shape matrices have more than 2 dimensions or if their first dimensions do not match.
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


def pre_MultiSetup(
    dataList: typing.List[np.ndarray], reflist: typing.List[typing.List[int]]
) -> typing.List[typing.Dict[str, np.ndarray]]:
    """
    Preprocesses data from multiple setups by separating reference and moving sensor data.

    Parameters
    ----------
    DataList : list of numpy arrays
        List of input data arrays for each setup, where each array represents sensor data.
    reflist : list of lists
        List of lists containing indices of sensors used as references for each setup.

    Returns
    -------
    list of dicts
        A list of dictionaries, each containing the data for a setup.
        Each dictionary has keys 'ref' and 'mov' corresponding to reference and moving sensor data.
    """
    n_setup = len(dataList)  # number of setup
    Y = []
    for i in range(n_setup):
        y = dataList[i]
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


def invperm(p: np.ndarray) -> np.ndarray:
    """
    Compute the inverse permutation of a given array.

    Parameters
    ----------
    p : array-like
        A permutation of integers from 0 to n-1, where n is the length of the array.

    Returns
    -------
    ndarray
        An array representing the inverse permutation of `p`.

    Example
    -------
    >>> invperm(np.array([3, 0, 2, 1]))
    array([1, 3, 2, 0])
    """
    q = np.empty_like(p)
    q[p] = np.arange(len(p))
    return q


# -----------------------------------------------------------------------------


def find_map(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Maps the elements of one array to another based on sorting order.

    Parameters
    ----------
    arr1 : array-like
        The first input array.
    arr2 : array-like
        The second input array, which should have the same length as `arr1`.

    Returns
    -------
    ndarray
        An array of indices that maps the sorted version of `arr1` to the sorted version of `arr2`.

    Example
    -------
    >>> find_map(np.array([10, 30, 20]), np.array([3, 2, 1]))
    array([2, 0, 1])
    """
    o1 = np.argsort(arr1)
    o2 = np.argsort(arr2)
    return o2[invperm(o1)]


# -----------------------------------------------------------------------------


def filter_data(
    data: np.ndarray,
    fs: float,
    Wn: float,
    order: int = 4,
    btype: str = "lowpass",
):
    """
    Apply a Butterworth filter to the input data.

    This function designs and applies a digital Butterworth filter to the input data array. The filter
    is applied in a forward-backward manner using the second-order sections representation to minimize
    phase distortion.

    Parameters
    ----------
    data : array_like
        The input signal to filter. If `data` is a multi-dimensional array, the filter is applied along
        the first axis.
    fs : float
        The sampling frequency of the input data.
    Wn : array_like
        The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; for
        bandpass and bandstop filters, Wn is a length-2 sequence.
    order : int, optional
        The order of the filter. Higher order means a sharper frequency cutoff, but the filter will
        also be less stable. The default is 4.
    btype : str, optional
        The type of filter to apply. Can be 'lowpass', 'highpass', 'bandpass', or 'bandstop'. The default
        is 'lowpass'.

    Returns
    -------
    filt_data : ndarray
        The filtered signal.

    Note
    ----
    This function uses `scipy.signal.butter` to design the filter and `scipy.signal.sosfiltfilt` for
    filtering to apply the filter in a zero-phase manner, which does not introduce phase delay to the
    filtered signal. For more information, see the scipy documentation for `signal.butter`
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html) and `signal.sosfiltfilt`
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).

    """
    sos = signal.butter(order, Wn, btype=btype, output="sos", fs=fs)
    filt_data = signal.sosfiltfilt(sos, data, axis=0)
    return filt_data
