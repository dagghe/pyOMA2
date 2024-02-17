"""
Stochastic Subspace Identification (SSI) Utility Functions module.
Part of the pyOMA2 package.
Authors:
Dag Pasca
Angelo Aloisio
"""
import logging

import numpy as np
from Gen_funct import MAC
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)

# =============================================================================
# FUNZIONI SSI
# =============================================================================


def BuildHank(Y, Yref, br, fs, method):
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
      "cov_unb": Builds Hankel matrix using correlations with unbiased estimator.
      "cov_bias": Builds Hankel matrix using correlations with biased estimator.
      "YfYp": Returns the future and past output data matrices Yf and Yp.
    """
    Ndat = Y.shape[1]
    p = int(br)  # number of block row (to use same notation as Dohler)
    q = int(p + 1)  # block column
    N = Ndat - p - q  # lenght of the Hankel matrix
    if method == "cov_mm":
        # Future and past Output
        Yf = np.vstack(
            [(1 / N**0.5) * Y[:, q + 1 + i : N + q + i] for i in range(p + 1)]
        )
        Yp = np.vstack(
            [(1 / N**0.5) * Yref[:, q + i : N + q - 1 + i] for i in range(0, -q, -1)]
        )
        Hank = np.dot(Yf, Yp.T)
        return Hank

    elif method == "cov_unb":
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

    elif method == "cov_bias":
        # Correlations
        Ri = np.array(
            [1 / (Ndat) * np.dot(Y[:, : Ndat - k], Yref[:, k:].T) for k in trange(p + q)]
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
        Yf = np.vstack(
            [(1 / N**0.5) * Y[:, q + 1 + i : N + q + i] for i in range(p + 1)]
        )
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
        Yf = np.vstack(
            [(1 / N**0.5) * Y[:, q + 1 + i : N + q + i] for i in range(p + 1)]
        )
        Yp = np.vstack(
            [(1 / N**0.5) * Yref[:, q + i : N + q - 1 + i] for i in range(0, -q, -1)]
        )
        return Yf, Yp

    else:
        raise ValueError(
            f'{method} is not a valid argument. "method" must be \
                         one of: "cov_mm", "cov_unb", "cov_bias", "dat", \
                         "YfYp"'
        )


# -----------------------------------------------------------------------------


def AC2MP(A, C, dt):
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


def SSI(H, br, ordmax, step=1):
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


def SSI_FAST(H, br, ordmax, step=1):
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


def SSI_Poles(AA, CC, ordmax, dt, step=1):
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


def SSI_MulSet(Y, fs, br, ordmax, methodHank, step=1, method="FAST"):
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

    return A, C


# -----------------------------------------------------------------------------


def Lab_stab_SSI(Fn, Sm, Ms, ordmin, ordmax, step, err_fn, err_xi, err_ms, max_xi):
    """
    Construct a Stability Chart for the Stochastic Subspace Identification (SSI) method.

    Parameters
    ----------
    Fn : numpy.ndarray
        Frequency poles, shape: (ordmax, ordmax/step+1).
    Sm : numpy.ndarray
        Damping poles, shape: (ordmax, ordmax/step+1).
    Ms : numpy.ndarray
        Mode shape array, shape: (ordmax, ordmax/step+1, nch(n_DOF)).
    ordmin : int
        Minimum order of model.
    ordmax : int
        Maximum order of model.
    step : int
        Step when iterating through model orders.
    err_fn : float
        Threshold for relative frequency difference for stability checks.
    err_xi : float
        Threshold for relative damping ratio difference for stability checks.
    err_ms : float
        Threshold for Modal Assurance Criterion (MAC) for stability checks.
    max_xi : float
        Threshold for max allowed damping.

    Returns
    -------
    numpy.ndarray
        Stability label matrix (Lab), shape: (ordmax, ordmax/step+1).

    Note
    -----
    This function categorizes modes based on their stability in terms of frequency, damping, and mode shape.
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
    # STABILITY BETWEEN CONSECUTIVE ORDERS
    for i in range(ordmin, ordmax + 1, step):
        ii = int((i - ordmin) / step)

        f_n = Fn1[:, ii].reshape(-1, 1)
        xi_n = Sm1[:, ii].reshape(-1, 1)
        phi_n = Ms1[:, ii, :]

        f_n1 = Fn1[:, ii - 1].reshape(-1, 1)
        xi_n1 = Sm1[:, ii - 1].reshape(-1, 1)
        phi_n1 = Ms1[:, ii - 1, :]

        if ii != 0 and ii != 1:
            for i in range(len(f_n)):
                if np.isnan(f_n1[i]):
                    # If at the iteration i-1 the elements are all nan, do nothing
                    # n.b the lab stays 0
                    pass
                else:
                    try:
                        idx = np.nanargmin(np.abs(f_n1 - f_n[i]))

                        cond1 = np.abs(f_n[i] - f_n1[idx]) / f_n[i]
                        cond2 = np.abs(xi_n[i] - xi_n1[idx]) / xi_n[i]
                        # cond3 = 1 - GF.MAC(phi_n[i, :], phi_n1[idx, :])
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


# -----------------------------------------------------------------------------


def SSI_MPE(sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab=None, deltaf=0.05, rtol=1e-2):
    """
    Extract modal parameters using Stochastic Subspace Identification (SSI) method for selected frequencies.

    Parameters
    ----------
    sel_freq : list
        List of selected frequencies for modal parameter extraction.
    Fn_pol : numpy.ndarray
        Array of natural frequencies obtained from SSI for each model order.
    Sm_pol : numpy.ndarray
        Array of damping ratios obtained from SSI for each model order.
    Ms_pol : numpy.ndarray
        3D array of mode shapes obtained from SSI for each model order.
    order : int, list of int, or 'find_min'
        Specifies the model order(s) for which the modal parameters are to be extracted.
        If 'find_min', the function attempts to find the minimum model order that provides
        stable poles for each mode of interest.
    Lab : numpy.ndarray, optional
        Array of labels identifying stable poles. Required if order='find_min'.
    deltaf : float, optional
        Frequency bandwidth around each selected frequency for searching poles. Default is 0.05.
    rtol : float, optional
        Relative tolerance for comparing frequencies. Default is 1e-2.

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
    sel_freq1 = []
    # Loop through the frequencies given in the input list
    logger.info("Extracting SSI modal parameters")
    order_out = np.empty(len(sel_freq))
    for ii, fj in enumerate(tqdm(sel_freq)):
        # =============================================================================
        # OPZIONE order = "find_min"
        # here we find the minimum model order so to get a stable pole for every mode of interest
        # -----------------------------------------------------------------------------
        if order == "find_min":
            # keep only Stable pole
            a = np.where(Lab == 7, Fn_pol, np.nan)
            # find the limits for the search
            limits = [(fj - deltaf, fj + deltaf) for fj in sel_freq]
            # find poles between limits and append them to list
            aas = [
                np.where(((a < limits[ii][1]) & (a > limits[ii][0])), a, np.nan)
                for ii in range(len(sel_freq))
            ]
            # =============================================================================
            # N.B if deltaf is too big and a +- limits includes also another frequency from
            # sel_freq, then the method of adding the matrices together in the next loop
            # wont work.
            # DOVREI ESCLUDERE LE FREQUENZE CHE HANNO FORME MODALI DIVERSE (MAC<0.85?)
            # RISPETTO AD UNA FORMA DI RIFERIMENTO FORNITA
            # =============================================================================
            # then loop through list
            aa = 0
            for bb in aas:
                # transform nan into 0 (so to be able to add the matrices together)
                bb = np.nan_to_num(bb, copy=True, nan=0.0)
                aa += bb
            # convert back 0s to nans
            aa = np.where(aa == 0, np.nan, aa)

            ii = 0
            check = np.array([False, False])
            while check.any() is False:
                # try:
                fn_at_ord_ii = aa[:, ii]
                fn_at_ord_ii = np.unique(fn_at_ord_ii)
                # remove nans
                fn_at_ord_ii = fn_at_ord_ii[~np.isnan(fn_at_ord_ii)]

                if fn_at_ord_ii.shape[0] == len(sel_freq):
                    check = np.isclose(fn_at_ord_ii, sel_freq, rtol=rtol)
                else:
                    pass
                if ii == aa.shape[1] - 1:
                    logger.warning("Could not find any values")
                    break
                ii += 1
                # except:
                #     pass
            ii -= 1  # remove last iteration to find the correct index

            sel_freq1 = fn_at_ord_ii
            sel_xi = []
            sel_phi = []
            b = aa[:, ii]
            c = b[~np.isnan(b)]
            if c.any():
                for fj in sel_freq1:
                    r_ind = np.nanargmin(np.abs(b - fj))
                    sel_xi.append(Sm_pol[r_ind, ii])
                    sel_phi.append(Ms_pol[r_ind, ii, :])
            order_out = ii
        # =============================================================================
        # OPZIONE 2 order = int
        # -----------------------------------------------------------------------------
        elif type(order) == int:
            sel = np.nanargmin(np.abs(Fn_pol[:, order] - fj))
            fns_at_ord_ii = Fn_pol[:, order][sel]
            check = np.isclose(fns_at_ord_ii, sel_freq, rtol=rtol)
            if not check.any():
                logger.warning("Could not find any values")
                order_out = order
            else:
                sel_freq1.append(Fn_pol[:, order][sel])
                sel_xi.append(Sm_pol[:, order][sel])
                sel_phi.append(Ms_pol[:, order][sel, :])
                order_out = order
        # =============================================================================
        # OPZIONE 3 order = list[int]
        # -----------------------------------------------------------------------------
        elif type(order) == list:
            sel = np.nanargmin(np.abs(Fn_pol[:, order[ii]] - fj))
            fns_at_ord_ii = Fn_pol[:, order[ii]][sel]
            check = np.isclose(fns_at_ord_ii, sel_freq, rtol=rtol)
            if not check.any():
                logger.warning("Could not find any values")
                order_out[ii] = order[ii]
            else:
                sel_freq1.append(Fn_pol[:, order[ii]][sel])
                sel_xi.append(Sm_pol[:, order[ii]][sel])
                sel_phi.append(Ms_pol[:, order[ii]][sel, :])
                order_out[ii] = order[ii]
        else:
            raise ValueError('order must be either of type(int) or "find_min"')
    logger.debug("Done!")

    Fn = np.array(sel_freq1)
    Phi = np.array(sel_phi).T
    Xi = np.array(sel_xi)
    return Fn, Xi, Phi, order_out
