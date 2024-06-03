"""
Stochastic Subspace Identification (SSI) Utility Functions module.
Part of the pyOMA2 package.
Authors:
Dag Pasca
Angelo Aloisio
"""

import logging
import typing

import numpy as np
from tqdm import tqdm, trange

np.seterr(divide="ignore", invalid="ignore")
logger = logging.getLogger(__name__)

# =============================================================================
# FUNZIONI SSI
# =============================================================================


def BuildHank(Y: np.ndarray, Yref: np.ndarray, br: int, fs: float, method: str):
    """
    Construct a Hankel matrix using various methods based on input data for System Identification.

    Parameters
    ----------
    Y : numpy.ndarray
        Time series data array, typically representing system's output.
    Yref : numpy.ndarray
        Reference data array, used in certain methods for Hankel matrix construction.
    br : int
        Number of block rows in the Hankel matrix.
    fs : float
        Sampling frequency of the data.
    method : str
        Specifies the method for Hankel matrix construction.
        One of: 'cov_mm', 'cov_unb', 'cov_bias', 'dat', and 'YfYp'.

    Returns
    -------
    numpy.ndarray
        Either a single Hankel matrix or a tuple of matrices (Yf, Yp)
        representing future and past data blocks, depending on the method.

    Raises
    ------
    ValueError
        If an invalid method is specified.

    Note
    -----
      "dat": Efficient method for assembling the Hankel matrix for data driven SSI.
      "cov_mm": Builds Hankel matrix using future and past output data with matrix multiplication.
      "cov_R": Builds Hankel matrix using correlations.
      "YfYp": Returns the future and past output data matrices Yf and Yp.
    """
    Ndat = Y.shape[1]
    p = int(br)  # number of block row (to use same notation as Dohler)
    q = int(p + 1)  # block column
    N = Ndat - p - q  # lenght of the Hankel matrix
    if method == "cov_mm":
        # Future and past Output
        Yf = np.vstack([(1 / N**0.5) * Y[:, q + 1 + i : N + q + i] for i in range(p + 1)])
        Yp = np.vstack(
            [(1 / N**0.5) * Yref[:, q + i : N + q - 1 + i] for i in range(0, -q, -1)]
        )
        Hank = np.dot(Yf, Yp.T)
        return Hank

    elif method == "cov_R":
        # Correlations
        Ri = np.array(
            [
                1 / (Ndat - k) * np.dot(Y[:, : Ndat - k], Yref[:, k:].T)
                for k in trange(p + q)
            ]
        )
        # Assembling the Toepliz matrix
        Hank = np.vstack(
            [
                np.hstack([Ri[k, :, :] for k in range(p + l_, l_ - 1, -1)])
                for l_ in range(q)
            ]
        )
        return Hank

    elif method == "dat":
        # Efficient method for assembling the Hankel matrix for data driven SSI
        # see [1]
        Yf = np.vstack([(1 / N**0.5) * Y[:, q + 1 + i : N + q + i] for i in range(p + 1)])
        n_ref = Yref.shape[0]
        Yp = np.vstack(
            [(1 / N**0.5) * Yref[:, q + i : N + q - 1 + i] for i in range(0, -q, -1)]
        )
        Ys = np.vstack((Yp, Yf))
        R3 = np.linalg.qr(Ys.T, mode="r")
        R3 = R3.T
        Hank = R3[n_ref * (p + 1) :, : n_ref * (p + 1)]
        return Hank

    elif method == "YfYp":
        Yf = np.vstack([(1 / N**0.5) * Y[:, q + 1 + i : N + q + i] for i in range(p + 1)])
        Yp = np.vstack(
            [(1 / N**0.5) * Yref[:, q + i : N + q - 1 + i] for i in range(0, -q, -1)]
        )
        return Yf, Yp

    else:
        raise ValueError(
            f'{method} is not a valid argument. "method" must be \
                         one of: "cov_mm", "cov_R", "dat", \
                         "YfYp"'
        )


# -----------------------------------------------------------------------------


def AC2MP(A: np.ndarray, C: np.ndarray, dt: float):
    """
    Convert state-space representation (A, C matrices) to modal parameters.

    Parameters
    ----------
    A : numpy.ndarray
        State matrix of the system.
    C : numpy.ndarray
        Output matrix of the system.
    dt : float
        Time step or sampling interval (1/fs, where fs is the sampling frequency).

    Returns
    -------
    tuple
        - fn : numpy.ndarray
            Natural frequencies in Hz.
        - xi : numpy.ndarray
            Damping ratios.
        - phi : numpy.ndarray
            Complex mode shapes.
    """
    Nch = C.shape[0]
    AuVal, AuVett = np.linalg.eig(A)
    Lambda = (np.log(AuVal)) * (1 / dt)
    fn = abs(Lambda) / (2 * np.pi)  # natural frequencies
    xi = -((np.real(Lambda)) / (abs(Lambda)))  # damping ratios
    # Complex mode shapes
    phi = np.dot(C, AuVett)
    # normalised (unity displacement)
    phi = np.array(
        [phi[:, ii] / phi[np.argmax(abs(phi[:, ii])), ii] for ii in range(phi.shape[1])]
    ).reshape(-1, Nch)
    return fn, xi, phi


# -----------------------------------------------------------------------------


def SSI(H: np.ndarray, br: int, ordmax: int, step: int = 1):
    """
    Perform System Identification using Stochastic Subspace Identification (SSI) method.

    Parameters
    ----------
    H : numpy.ndarray
        Hankel matrix of the system.
    br : int
        Number of block rows in the Hankel matrix.
    ordmax : int
        Maximum order to consider for system identification.
    step : int, optional
        Step size for increasing system order. Default is 1.

    Returns
    -------
    tuple
        - A : list of numpy arrays
            Estimated system matrices A for various system orders.
        - C : list of numpy arrays
            Estimated output influence matrices C for various system orders.

    Note
    -----
    Classical implementation of the SSI algorithm using the shift structure of the observability matrix.
    For more efficient implementation, consider using SSI_FAST function.
    """
    Nch = int(H.shape[0] / (br + 1))
    # SINGULAR VALUE DECOMPOSITION
    U1, S1, V1_t = np.linalg.svd(H)
    S1rad = np.sqrt(np.diag(S1))
    # initializing arrays
    # Obs = np.dot(U1[:, :ordmax], S1rad[:ordmax, :ordmax]) # Observability matrix
    # Con = np.dot(S1rad[:ordmax, :ordmax], V1_t[: ordmax, :]) # Controllability matrix
    A = []
    C = []
    # loop for increasing order of the system
    logger.info("SSI for increasing model order...")
    for ii in trange(0, ordmax + 1, step):
        Obs = np.dot(U1[:, :ii], S1rad[:ii, :ii])  # Observability matrix
        # Con = np.dot(S1rad[:ii, :ii], V1_t[: ii, :]) # Controllability matrix
        # System Matrix
        A.append(np.dot(np.linalg.pinv(Obs[: Obs.shape[0] - Nch, :]), Obs[Nch:, :]))
        # Output Influence Matrix
        C.append(Obs[:Nch, :])
        # G = Con[:, Nch:]
    logger.debug("... Done!")
    return A, C


# -----------------------------------------------------------------------------


def SSI_FAST(H: np.ndarray, br: int, ordmax: int, step: int = 1):
    """
    Perform efficient System Identification using the Stochastic Subspace Identification (SSI) method.

    Parameters
    ----------
    H : numpy.ndarray
        Hankel matrix of the system.
    br : int
        Number of block rows in the Hankel matrix.
    ordmax : int
        Maximum order to consider for system identification.
    step : int, optional
        Step size for increasing system order. Default is 1.

    Returns
    -------
    tuple
        - A : list of numpy arrays
            Estimated system matrices A for various system orders.
        - C : list of numpy arrays
            Estimated output influence matrices C for various system orders.

    Note
    -----
    This is a more efficient implementation of the SSI algorithm.
    """
    Nch = int(H.shape[0] / (br + 1))
    # SINGULAR VALUE DECOMPOSITION
    U1, S1, V1_t = np.linalg.svd(H)
    S1rad = np.sqrt(np.diag(S1))
    # initializing arrays
    Obs = np.dot(U1[:, :ordmax], S1rad[:ordmax, :ordmax])  # Observability matrix
    O_p = Obs[: Obs.shape[0] - Nch, :]
    O_m = Obs[Nch:, :]
    # QR decomposition
    Q, R = np.linalg.qr(O_p)
    S = np.dot(Q.T, O_m)
    # Con = np.dot(S1rad[:ordmax, :ordmax], V1_t[: ordmax, :]) # Controllability matrix
    A = []
    C = []
    # loop for increasing order of the system
    logger.info("SSI for increasing model order...")
    for ii in trange(0, ordmax + 1, step):
        # System Matrix
        A.append(np.dot(np.linalg.inv(R[:ii, :ii]), S[:ii, :ii]))
        # Output Influence Matrix
        C.append(Obs[:Nch, :ii])
    logger.debug("... Done!")
    return A, C


# -----------------------------------------------------------------------------


def SSI_Poles(AA: list, CC: list, ordmax: int, dt: float, step: int = 1):
    """
    Compute modal parameters from state-space models identified by Stochastic Subspace Identification (SSI).

    Parameters
    ----------
    AA : list of numpy.ndarray
        List of system matrices A for increasing model order.
    CC : list of numpy.ndarray
        List of output influence matrices C for increasing model order.
    ordmax : int
        Maximum model order considered in the system identification process.
    dt : float
        Time step or sampling interval.
    step : int, optional
        Step size for increasing model order. Default is 1.

    Returns
    -------
    tuple
        - Fn : numpy.ndarray
            Natural frequencies for each system and each order.
        - Sm : numpy.ndarray
            Damping ratios for each system and each order.
        - Ms : numpy.ndarray
            Complex mode shapes for each system and each order.

    Note
    -----
    Applies the AC2MP function to each system in AA and CC to compute modal parameters.
    The modal parameters are stored for each system and each specified order.
    """
    NAC = len(AA)
    Nch = CC[0].shape[0]
    # initialization of the matrix that contains the frequencies
    Fn = np.full((ordmax, int((ordmax) / step + 1)), np.nan)
    # initialization of the matrix that contains the damping ratios
    Sm = np.full((ordmax, int((ordmax) / step + 1)), np.nan)
    Ms = np.full(
        (ordmax, int((ordmax) / step + 1), Nch), np.nan, dtype=complex
    )  # initialization of the matrix that contains the damping ratios
    for ii in range(NAC):
        A = AA[ii]
        C = CC[ii]
        fn, xi, phi = AC2MP(A, C, dt)
        Fn[: len(fn), ii] = fn  # save the frequencies
        Sm[: len(fn), ii] = xi  # save the damping ratios
        Ms[: len(fn), ii, :] = phi
    return Fn, Sm, Ms


# -----------------------------------------------------------------------------


def SSI_MulSet(
    Y: list,
    fs: float,
    br: int,
    ordmax: int,
    methodHank: str,
    step: int = 1,
    method: str = "FAST",
):
    """
    Perform Subspace System Identification SSI for multiple setup measurements.

    Parameters
    ----------
    Y : list of dictionaries
        List of dictionaries, each representing data from a different setup.
        Each dictionary must have keys 'ref' (reference sensor data) and
        'mov' (moving sensor data), with corresponding numpy arrays.
    fs : float
        Sampling frequency of the data.
    br : int
        Number of block rows in the Hankel matrix.
    ordmax : int
        Maximum order for the system identification process.
    methodHank : str
        Method for Hankel matrix construction. Can be 'cov_mm', 'cov_unb', 'cov_bias', 'dat'.
    step : int, optional
        Step size for increasing the order in the identification process. Default is 1.
    method : str, optional
        Method for system matrix computation, either 'FAST' or 'SLOW'. Default is 'FAST'.

    Returns
    -------
    tuple
        A : list of numpy arrays
            System matrices for each model order.
        C : list of numpy arrays
            Output influence matrices for each model order.
    """
    n_setup = len(Y)  # number of setup
    n_ref = Y[0]["ref"].shape[0]  # number of reference sensor

    # N.B. ONLY FOR TEST
    n_mov = [0 for i in range(n_setup)]  # number of moving sensor
    n_mov = [Y[i]["mov"].shape[0] for i in range(n_setup)]  # number of moving sensor

    n_DOF = n_ref + np.sum(n_mov)  # total number of sensors
    dt = 1 / fs
    O_mov_s = []  # initialise the scaled moving part of the observability matrix
    for kk in trange(n_setup):
        logger.debug("Analyising setup nr.: %s...", kk)
        Y_ref = Y[kk]["ref"]
        # Ndat = Y_ref.shape[1] # number of data points

        # N.B. ONLY FOR TEST
        Y_all = Y[kk]["ref"]
        Y_all = np.vstack((Y[kk]["ref"], Y[kk]["mov"]))

        r = Y_all.shape[0]  # total sensor for the ii setup
        # Build HANKEL MATRIX
        H = BuildHank(Y_all, Y_ref, br, 1 / dt, method=methodHank)
        # SINGULAR VALUE DECOMPOSITION
        U1, S1, V1_t = np.linalg.svd(H)
        S1rad = np.sqrt(np.diag(S1))
        # Observability matrix
        Obs = np.dot(U1[:, :ordmax], S1rad[:ordmax, :ordmax])
        # get reference idexes
        ref_id = np.array([np.arange(br) * (n_ref + n_mov[kk]) + j for j in range(n_ref)])
        ref_id = ref_id.flatten(order="f")
        mov_id = np.array(
            [np.arange(br) * (n_ref + n_mov[kk]) + j for j in range(n_ref, r)]
        )
        mov_id = mov_id.flatten(order="f")

        O_ref = Obs[ref_id, :]  # reference portion
        O_mov = Obs[mov_id, :]  # moving portion

        if kk == 0:
            O1_ref = O_ref  # basis
        # scale the moving observability matrix to the reference basis
        O_movs = np.dot(np.dot(O_mov, np.linalg.pinv(O_ref)), O1_ref)
        O_mov_s.append(O_movs)
        logger.debug("... Done with setup nr.: %s!", kk)

    # global observability matrix formation via block-interleaving
    Obs_all = np.zeros((n_DOF * br, ordmax))
    for ii in range(br):
        # reference portion block rows
        id1 = (ii) * n_DOF
        id2 = id1 + n_ref
        Obs_all[id1:id2, :] = O1_ref[ii * n_ref : (ii + 1) * n_ref, :]
        for jj in range(n_setup):
            # moving sensor portion block rows
            id1 = id2
            id2 = id1 + n_mov[jj]
            Obs_all[id1:id2, :] = O_mov_s[jj][ii * n_mov[jj] : (ii + 1) * n_mov[jj], :]
    if method == "FAST":
        # Obs minus last br, minus first br, respectibely
        O_p = Obs_all[: Obs_all.shape[0] - n_DOF, :]
        O_m = Obs_all[n_DOF:, :]
        # QR decomposition
        Q, R = np.linalg.qr(O_p)
        S = np.dot(Q.T, O_m)
        # Con = np.dot(S1rad[:ordmax, :ordmax], V1_t[: ordmax, :]) # Controllability matrix
        A = []
        C = []
        # loop for increasing order of the system
        for i in trange(0, ordmax + 1, step):
            # System Matrix
            A.append(np.dot(np.linalg.inv(R[:i, :i]), S[:i, :i]))
            # Output Influence Matrix
            C.append(Obs_all[:n_DOF, :i])
    elif method == "SLOW":
        A = []
        C = []
        # loop over model orders
        for i in trange(0, ordmax + 1, step):
            A.append(np.dot(np.linalg.pinv(Obs_all[:-n_DOF, :i]), Obs_all[n_DOF:, :i]))
            C.append(Obs_all[:n_DOF, :i])

    else:
        raise ValueError("method must be either 'FAST' or 'SLOW'")

    return A, C


# -----------------------------------------------------------------------------


def SSI_MPE(
    freq_ref: list,
    Fn_pol: np.ndarray,
    Xi_pol: np.ndarray,
    Phi_pol: np.ndarray,
    order: int,
    Lab: typing.Optional[np.ndarray] = None,
    rtol: float = 5e-2,
):
    """
    Extract modal parameters using Stochastic Subspace Identification (SSI) method for selected frequencies.

    Parameters
    ----------
    freq_ref : list
        List of selected frequencies for modal parameter extraction.
    Fn_pol : numpy.ndarray
        Array of natural frequencies obtained from SSI for each model order.
    Xi_pol : numpy.ndarray
        Array of damping ratios obtained from SSI for each model order.
    Phi_pol : numpy.ndarray
        3D array of mode shapes obtained from SSI for each model order.
    order : int, list of int, or 'find_min'
        Specifies the model order(s) for which the modal parameters are to be extracted.
        If 'find_min', the function attempts to find the minimum model order that provides
        stable poles for each mode of interest.
    Lab : numpy.ndarray, optional
        Array of labels identifying stable poles. Required if order='find_min'.
    rtol : float, optional
        Relative tolerance for comparing frequencies. Default is 5e-2.

    Returns
    -------
    tuple
        Fn : numpy.ndarray
            Extracted natural frequencies.
        Xi : numpy.ndarray
            Extracted damping ratios.
        Phi : numpy.ndarray
            Extracted mode shapes.
        order_out : numpy.ndarray or int
            Output model order used for extraction for each frequency.

    Raises
    ------
    ValueError
        If 'order' is not an int, list of int, or 'find_min', or if 'order' is 'find_min'
        but 'Lab' is not provided.
    """

    # if order != "find_min" and type(order) != int and type(order) != list[int]:
    #     raise ValueError(
    #         f"The argument order must either be 'find_min' or be and integer, your input is {order}"
    #     )
    if order == "find_min" and Lab is None:
        raise ValueError(
            "When order ='find_min', one must also provide the Lab list for the poles"
        )
    sel_xi = []
    sel_phi = []
    sel_freq = []
    # Loop through the frequencies given in the input list
    logger.info("Extracting SSI modal parameters")
    # =============================================================================
    # OPZIONE order = "find_min"
    # here we find the minimum model order so to get a stable pole for every mode of interest
    # -----------------------------------------------------------------------------
    if order == "find_min":
        # Find stable poles
        stable_poles = np.where(Lab == 7, Fn_pol, np.nan)
        limits = [(f - rtol, f + rtol) for f in freq_ref]

        # Accumulate frequencies within the tolerance limits
        aggregated_poles = np.zeros_like(stable_poles)
        for lower, upper in limits:
            within_limits = np.where(
                (stable_poles >= lower) & (stable_poles <= upper), stable_poles, 0
            )
            aggregated_poles += within_limits

        aggregated_poles = np.where(aggregated_poles == 0, np.nan, aggregated_poles)

        found = False
        for i in range(aggregated_poles.shape[1]):
            current_order_poles = aggregated_poles[:, i]
            unique_poles = np.unique(current_order_poles[~np.isnan(current_order_poles)])

            if len(unique_poles) == len(freq_ref) and np.allclose(
                unique_poles, freq_ref, rtol=rtol
            ):
                found = True
                sel_freq.append(unique_poles)
                for freq in unique_poles:
                    index = np.nanargmin(np.abs(current_order_poles - freq))
                    sel_xi.append(Xi_pol[index, i])
                    sel_phi.append(Phi_pol[index, i, :])
                order_out = i
                break

        if not found:
            logger.warning("Could not find any values")
            order_out = None
    # =============================================================================
    # OPZIONE 2 order = int
    # -----------------------------------------------------------------------------
    elif isinstance(order, int):
        for fj in tqdm(freq_ref):
            sel = np.nanargmin(np.abs(Fn_pol[:, order] - fj))
            fns_at_ord_ii = Fn_pol[:, order][sel]
            check = np.isclose(fns_at_ord_ii, freq_ref, rtol=rtol)
            if not check.any():
                logger.warning("Could not find any values")
                order_out = order
            else:
                sel_freq.append(Fn_pol[:, order][sel])
                sel_xi.append(Xi_pol[:, order][sel])
                sel_phi.append(Phi_pol[:, order][sel, :])
                order_out = order
    # =============================================================================
    # OPZIONE 3 order = list[int]
    # -----------------------------------------------------------------------------
    elif isinstance(order, list):
        order_out = np.array(order)
        for ii, fj in enumerate(tqdm(freq_ref)):
            sel = np.nanargmin(np.abs(Fn_pol[:, order[ii]] - fj))
            fns_at_ord_ii = Fn_pol[:, order[ii]][sel]
            check = np.isclose(fns_at_ord_ii, freq_ref, rtol=rtol)
            if not check.any():
                logger.warning("Could not find any values")
                order_out[ii] = order[ii]
            else:
                sel_freq.append(Fn_pol[:, order[ii]][sel])
                sel_xi.append(Xi_pol[:, order[ii]][sel])
                sel_phi.append(Phi_pol[:, order[ii]][sel, :])
                order_out[ii] = order[ii]
    else:
        raise ValueError('order must be either of type(int) or "find_min"')
    logger.debug("Done!")

    Fn = np.array(sel_freq).reshape(-1)
    Phi = np.array(sel_phi).T
    Xi = np.array(sel_xi)
    return Fn, Xi, Phi, order_out
