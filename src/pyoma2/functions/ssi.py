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
from scipy import linalg
from tqdm import tqdm, trange

np.seterr(divide="ignore", invalid="ignore")
logger = logging.getLogger(__name__)

# =============================================================================
# FUNZIONI SSI
# =============================================================================


def build_hank(
    Y: np.ndarray,
    Yref: np.ndarray,
    br: int,
    method: str,
    calc_unc: bool = False,
    nb: int = 100,
):
    """
    Construct a Hankel matrix using various methods with optional uncertainty calculations.

    Parameters
    ----------
    Y : np.ndarray
        The primary data matrix with shape (number of channels, number of data points).
    Yref : np.ndarray
        The reference data matrix with shape (number of reference channels, number of data points).
    br : int
        The number of block rows to use in the Hankel matrix.
    method : str
        The method to use for constructing the Hankel matrix.
        Options are 'cov_mm', 'cov_R', 'dat', or 'YfYp'.
    calc_unc : bool, optional
        Whether to calculate uncertainties (default is False).
        Only applicable for 'cov_mm' method.
    nb : int, optional
        The number of bootstrap samples to use for uncertainty calculations (default is 100).

    Returns
    -------
    Hank : np.ndarray
        The constructed Hankel matrix.
    T : np.ndarray, optional
        The uncertainty matrix. Returned only if `calc_unc` is True and `method` is 'cov_mm'.

    Raises
    ------
    AttributeError
        If `calc_unc` is True but `method` is not 'cov_mm'.
    AttributeError
        If `method` is not one of 'cov_mm', 'cov_R', 'dat', or 'YfYp'.

    Notes
    -----
    - The 'YfYp' method constructs separate future and past data matrices without combining them into a Hankel matrix.

    """
    Ndat = Y.shape[1]  # number of data points
    l = Y.shape[0]  # number of chaiiels  # noqa E741 (ambiguous variable name 'l')
    r = Yref.shape[0]  # number of reference chaiiels
    p = int(br)  # number of block row (to use same notation as Dohler)
    q = int(p + 1)  # block column
    N = Ndat - p - q  # lenght of the Hankel matrix

    if calc_unc and method != "cov_mm":
        raise AttributeError(
            "Uncertainty calculations are only available for 'cov_mm' method"
        )

    if method == "cov_mm":
        logger.info(f"Assembling Hankel matrix method: {method}...")
        # Future and past Output (Y^+ and Y^-)
        Yf = np.vstack([(1 / N**0.5) * Y[:, q + 1 + i : N + q + i] for i in range(p + 1)])
        Yp = np.vstack(
            [(1 / N**0.5) * Yref[:, q + i : N + q - 1 + i] for i in range(0, -q, -1)]
        )

        Hank = np.dot(Yf, Yp.T)

        # Uncertainty calculations
        if calc_unc is True:
            logger.info("... uncertainty calculations...")
            Nb = N // nb  # number of samples per segment
            T = np.zeros(((p + 1) * q * l * r, nb))  # Square root of SIGMA_H
            Hvec0 = Hank.reshape(-1, 1)  # vectorialised hankel
            Hcov = np.zeros(((p + 1) * l, q * r))  # Averaged version of the Hankel matrix

            for k in range(nb):
                # Section 3.2 and 5.1 of DoMe13
                Yp_k = Yf[:, (k * Nb) : ((k + 1) * Nb)]
                Ym_k = Yp[:, (k * Nb) : ((k + 1) * Nb)]
                Hcov_k = np.dot(Yp_k, Ym_k.T) / Nb

                Hcov += Hcov_k / nb
                Hcov_vec_k = Hcov_k.reshape(-1, 1)
                T[:, k] = (Hcov_vec_k - Hvec0).flatten() / np.sqrt(nb * (nb - 1))

            logger.debug("... Hankel and SIGMA_H Done!")
            return Hank, T
        else:
            logger.debug("... Hankel Done!")
            return Hank, None

    elif method == "cov_R":
        logger.info(f"Assembling Hankel matrix method: {method}...")
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

        # TODO
        # IMPLEMENTATION OF UNCERTAINTY CALCULATIONS
        logger.debug("... Hankel Done!")
        return Hank, None

    elif method == "dat":
        logger.info(f"Assembling Hankel matrix method: {method}...")
        # Efficient method for assembling the Hankel matrix for data driven SSI
        # see [1]
        Yf = np.vstack([(1 / N**0.5) * Y[:, q + 1 + i : N + q + i] for i in range(p + 1)])
        n_ref = Yref.shape[0]
        Yp = np.vstack(
            [(1 / N**0.5) * Yref[:, q + i : N + q - 1 + i] for i in range(0, -q, -1)]
        )
        Ys = np.vstack((Yp, Yf))
        # Q, R3 = np.linalg.qr(Ys.T, mode="complete")
        R21 = np.linalg.qr(Ys.T, mode="r")
        R21 = R21.T

        Hank = R21[n_ref * (p + 1) :, : n_ref * (p + 1)]

        # TODO
        # IMPLEMENTATION OF UNCERTAINTY CALCULATIONS
        logger.debug("... Hankel Done!")
        return Hank, None

    else:
        raise AttributeError(
            f'{method} is not a valid argument. "method" must be \
                         one of: "cov_mm", "cov_R", ", "dat"'
        )


# -----------------------------------------------------------------------------


def ac2mp(A: np.ndarray, C: np.ndarray, dt: float, calc_unc: bool = False):
    """
    Convert state-space matrices to modal parameters with optional uncertainty calculations.

    Parameters
    ----------
    A : np.ndarray
        The system matrix.
    C : np.ndarray
        The output matrix.
    dt : float
        The time step for discrete-time to continuous-time conversion.
    calc_unc : bool, optional
        Whether to return additional matrices for uncertainty calculations (default is False).

    Returns
    -------
    fn : np.ndarray
        Natural frequencies.
    xi : np.ndarray
        Damping ratios.
    phi : np.ndarray
        Normalized mode shapes.
    lam_c : np.ndarray
        Continuous-time eigenvalues.
    lam_d : np.ndarray, optional
        Discrete-time eigenvalues. Returned only if `calc_unc` is True.
    l_eigvt : np.ndarray, optional
        Left eigenvectors. Returned only if `calc_unc` is True.
    r_eigvt : np.ndarray, optional
        Right eigenvectors. Returned only if `calc_unc` is True.

    """
    Nch = C.shape[0]
    lam_d, l_eigvt, r_eigvt = linalg.eig(A, left=True)  # l_eigvt=chi, r_eigvt=phi
    lam_c = (np.log(lam_d)) * (1 / dt)  # to continous time
    fn = abs(lam_c) / (2 * np.pi)  # natural frequencies
    xi = -((np.real(lam_c)) / (abs(lam_c)))  # damping ratios
    # Complex mode shapes
    phi = np.dot(C, r_eigvt)  # N.B. this is \varphi
    # normalised (unity displacement)
    phi = np.array(
        [phi[:, ii] / phi[np.argmax(abs(phi[:, ii])), ii] for ii in range(phi.shape[1])]
    ).reshape(-1, Nch)
    if calc_unc is True:
        return fn, xi, phi, lam_c, lam_d, l_eigvt, r_eigvt
    else:
        return fn, xi, phi, lam_c, None, None, None


# -----------------------------------------------------------------------------


# Legacy
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


def SSI_fast(
    H: np.ndarray,
    br: int,
    ordmax: int,
    step: int = 1,
    calc_unc: bool = False,
    T: np.ndarray = None,
    nb: int = 100,
):
    """
    Perform Subspace System Identification (SSI) with optional uncertainty calculations.

    Parameters
    ----------
    H : np.ndarray
        The Hankel matrix.
    br : int
        The number of block rows.
    ordmax : int
        The maximum model order.
    step : int, optional
        The step size for increasing model order (default is 1).
    calc_unc : bool, optional
        Whether to calculate uncertainties (default is False).
    nb : int, optional
        The number of bootstrap samples to use for uncertainty calculations (default is 100).

    Returns
    -------
    Obs : np.ndarray
        The observability matrix.
    A : list of np.ndarray
        List of system matrices for each model order.
    C : list of np.ndarray
        List of output influence matrices for each model order.
    Q1, Q2, Q3, Q4 : np.ndarray, optional
        Matrices related to uncertainty calculations.
        Returned only if `calc_unc` is True.
    """

    # Nch = int(H.shape[0] / (br + 1))
    l = int(  # Number of chaiiels  # noqa E741 (ambiguous variable name 'l')
        H.shape[0] / (br + 1)
    )
    r = int(H.shape[1] / (br + 1))  # Number of reference chaiiels
    p = int(br)  # number of block row (to use same notation as Dohler)
    q = int(p + 1)  # block column

    # SINGULAR VALUE DECOMPOSITION
    U1, SIG, V1_t = np.linalg.svd(H)
    Uom = U1[:, :ordmax]
    Vom = V1_t[:, :ordmax]
    Som = SIG[:ordmax]
    S1rad = np.sqrt(np.diag(SIG))
    # initializing arrays
    Obs = np.dot(U1[:, :ordmax], S1rad[:ordmax, :ordmax])  # Observability matrix
    O_p = Obs[: Obs.shape[0] - l, :]
    O_m = Obs[l:, :]
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
        C.append(Obs[:l, :ii])
    logger.debug("... Done!")
    if calc_unc is True:
        logger.info("Calculating uncertainty...")

        # Selection matrices S1 and S2 (S1*Obs = Op; S2*Obs = Om)
        Sel1 = np.hstack([np.eye(p * l), np.zeros((p * l, l))])
        Sel2 = np.hstack([np.zeros((p * l, l)), np.eye(p * l)])

        # Build auxiliary matrices
        Q1 = np.zeros((ordmax**2, nb))
        Q2 = np.zeros((ordmax**2, nb))
        Q3 = np.zeros((ordmax**2, nb))
        Q4 = np.zeros((l * ordmax, nb))
        for ii in trange(0, ordmax, step):
            # Eq. 28
            Ki = np.linalg.inv(
                np.eye(q * r)
                + np.vstack([np.zeros((q * r - 1, q * r)), 2 * Vom[:, ii].T])
                - np.dot(H.T, H) / Som[ii] ** 2
            )
            # Eq. 29
            Bi1 = np.hstack(
                [
                    np.eye((p + 1) * l)
                    + np.dot(
                        np.dot(H / Som[ii], Ki),
                        H.T / Som[ii]
                        - np.vstack([np.zeros((q * r - 1, (p + 1) * l)), Uom[:, ii].T]),
                    ),
                    np.dot(H / Som[ii], Ki),
                ]
            )

            # Eq. 33
            Ti1 = np.dot(np.kron(np.eye(q * r), Uom[:, ii].T), T)
            Ti2 = np.dot(np.kron(Vom[:, ii].T, np.eye((p + 1) * l)), T)

            # Eq. 34
            JOHTi = np.dot(
                0.5 * 1 / np.sqrt(Som[ii]) * Uom[:, ii].reshape(-1, 1),
                np.dot(Vom[:, ii].T, Ti1).reshape(1, -1),
            ) + 1 / np.sqrt(Som[ii]) * np.dot(
                Bi1,
                np.vstack(
                    [
                        Ti2
                        - np.dot(
                            Uom[:, ii].reshape(-1, 1),
                            np.dot(Uom[:, ii].T, Ti2).reshape(1, -1),
                        ),
                        Ti1
                        - np.dot(
                            Vom[:, ii].reshape(-1, 1),
                            np.dot(Vom[:, ii].T, Ti1).reshape(1, -1),
                        ),
                    ]
                ),
            )
            # Eq. 36-37
            Q1[ii * ordmax : (ii + 1) * ordmax, :] = np.dot(np.dot(O_p.T, Sel1), JOHTi)
            Q2[ii * ordmax : (ii + 1) * ordmax, :] = np.dot(np.dot(O_m.T, Sel1), JOHTi)
            Q3[ii * ordmax : (ii + 1) * ordmax, :] = np.dot(np.dot(O_p.T, Sel2), JOHTi)
            Q4[ii * l : (ii + 1) * l, :] = np.dot(
                np.hstack([np.eye(l), np.zeros((l, p * l))]), JOHTi
            )
            logger.debug("... uncertainty calculations done!")

        return Obs, A, C, Q1, Q2, Q3, Q4

    else:
        return Obs, A, C, None, None, None, None


# -----------------------------------------------------------------------------


def SSI_poles(
    Obs: np.ndarray,
    AA: list,
    CC: list,
    ordmax: int,
    dt: float,
    step: int = 1,
    calc_unc: bool = False,
    Q1: np.ndarray = None,
    Q2: np.ndarray = None,
    Q3: np.ndarray = None,
    Q4: np.ndarray = None,
):
    """
    Calculate modal parameters (natural frequencies, damping ratios, mode shapes) for increasing model orders
    using Subspace System Identification (SSI) with optional uncertainty calculations.

    Parameters
    ----------
    Obs : np.ndarray
        The observability matrix.
    AA : list of np.ndarray
        List of system matrices for each model order.
    CC : list of np.ndarray
        List of output matrices for each model order.
    ordmax : int
        The maximum model order.
    dt : float
        The time step for discrete-time to continuous-time conversion.
    step : int, optional
        The step size for increasing model order (default is 1).
    calc_unc : bool, optional
        Whether to calculate uncertainties (default is False).
    Q1 : np.ndarray, optional
        Auxiliary matrix for uncertainty calculations.
    Q2 : np.ndarray, optional
        Auxiliary matrix for uncertainty calculations.
    Q3 : np.ndarray, optional
        Auxiliary matrix for uncertainty calculations.
    Q4 : np.ndarray, optional
        Auxiliary matrix for uncertainty calculations.

    Returns
    -------
    Fn : np.ndarray
        Natural frequencies.
    Xi : np.ndarray
        Damping ratios.
    Phi : np.ndarray
        Normalized mode shapes.
    Lambds : np.ndarray
        Continuous-time eigenvalues.
    Fn_cov : np.ndarray, optional
        Covariances of natural frequencies. Returned only if `calc_unc` is True.
    Xi_cov : np.ndarray, optional
        Covariances of damping ratios. Returned only if `calc_unc` is True.
    Phi_cov : np.ndarray, optional
        Covariances of mode shapes. Returned only if `calc_unc` is True.
    """
    # NB Nch = l
    Nch = CC[0].shape[0]
    # initialization of the matrix that contains the frequencies
    Lambds = np.full((ordmax, int((ordmax) / step + 1)), np.nan, dtype=complex)
    # initialization of the matrix that contains the frequencies
    Fn = np.full((ordmax, int((ordmax) / step + 1)), np.nan)
    # initialization of the matrix that contains the damping ratios
    Xi = np.full((ordmax, int((ordmax) / step + 1)), np.nan)
    # initialization of the matrix that contains the mode shapes
    Phi = np.full((ordmax, int((ordmax) / step + 1), Nch), np.nan, dtype=complex)

    if calc_unc is True:
        # initialization of the matrix that contains the frequencies
        Fn_cov = np.full((ordmax, int((ordmax) / step + 1)), np.nan)
        # initialization of the matrix that contains the damping ratios
        Xi_cov = np.full((ordmax, int((ordmax) / step + 1)), np.nan)
        # initialization of the matrix that contains the mode shapes
        Phi_cov = np.full(
            (ordmax, int((ordmax) / step + 1), Nch),
            np.nan,
        )

    logger.info("Calculating modal parameters...")
    for ii in trange(1, ordmax + 1, step):
        A = AA[ii]
        C = CC[ii]

        if calc_unc is True:
            fn, xi, phi, lam_c, lam_d, l_eigvt, r_eigvt = ac2mp(A, C, dt, calc_unc=True)
        else:
            fn, xi, phi, lam_c, _, _, _ = ac2mp(A, C, dt, calc_unc=False)
        Fn[: len(fn), ii] = fn  # save the frequencies
        Xi[: len(fn), ii] = xi  # save the damping ratios
        Phi[: len(fn), ii, :] = phi
        Lambds[: len(fn), ii] = lam_c

        logger.debug("... Done!")

        if calc_unc is True:
            # logger.info("Calculating uncertainty...")

            Obs_n = Obs[:, :ii]
            O_p = Obs_n[: Obs_n.shape[0] - Nch, :]
            OO = np.linalg.inv(np.dot(O_p.T, O_p))  # Step 2 Algo2

            # Permutation matrix eq. 15
            Pnn = np.zeros((ii**2, ii**2))
            for _kk in range(1, ii + 1):
                ek = np.zeros((ii, 1))
                ek[_kk - 1] = 1
                Pnn[:, (_kk - 1) * ii : _kk * ii] = np.kron(np.eye(ii), ek)

            # Selection matrix S4n
            S4_n = np.kron(
                np.hstack([np.eye(ii), np.zeros((ii, ordmax - ii))]),
                np.hstack([np.eye(ii), np.zeros((ii, ordmax - ii))]),
            )
            # Eq. 49
            Q1_n = np.dot(S4_n, Q1)
            Q2_n = np.dot(S4_n, Q2)
            Q3_n = np.dot(S4_n, Q3)
            Q4_n = np.dot(  # noqa F841 (variable not used)
                np.hstack([np.eye(Nch * ii), np.zeros((Nch * ii, Nch * (ordmax - ii)))]),
                Q4,
            )
            # Step 2 Algo2
            PnQ1 = np.dot((Pnn + np.eye(ii**2)), Q1_n)
            PnQ2_Q3 = np.dot(Pnn, Q2_n) + Q3_n

            for jj in range(len(lam_c)):
                # Eq. 44
                Qi = np.dot(
                    np.kron(r_eigvt[:, jj], np.eye(ii)), (-lam_d[jj] * PnQ1 + PnQ2_Q3)
                )
                # Lemma 5
                Mat1 = np.array(
                    [[1 / (2 * np.pi), 0], [0, 100 / (np.abs(lam_c[jj]) ** 2)]]
                )
                Mat2 = np.array(
                    [
                        [np.real(lam_c[jj]), np.imag(lam_c[jj])],
                        [
                            -(np.imag(lam_c[jj]) ** 2),
                            np.real(lam_c[jj]) * np.imag(lam_c[jj]),
                        ],
                    ]
                )
                Mat3 = np.array(
                    [
                        [np.real(lam_d[jj]), np.imag(lam_d[jj])],
                        [-np.imag(lam_d[jj]), np.real(lam_d[jj])],
                    ]
                )
                Jfx_l = (
                    1
                    / (dt * np.abs(lam_d[jj]) ** 2 * np.abs(lam_c[jj]))
                    * (np.dot(np.dot(Mat1, Mat2), Mat3))
                )
                # Eq. 43
                JaohT = (
                    1
                    / (np.dot(np.conj(l_eigvt[:, jj]), r_eigvt[:, jj]))
                    * np.dot(np.dot(np.conj(l_eigvt[:, jj]), OO), Qi)
                )

                # Eq. 42
                Ufx = np.dot(Jfx_l, np.vstack([np.real(JaohT), np.imag(JaohT)]))

                cov_fx = np.dot(Ufx, Ufx.T)  # Eq. 40

                Fn_cov[jj, ii] = abs(cov_fx[0, 0])
                Xi_cov[jj, ii] = abs(cov_fx[1, 0])

                # # FIXME
                # # THE COVARIANGE FOR THE MODESHAPE IS WRONG!!!
                # # Lemma 5
                # Mat1_1 = np.linalg.pinv(lam_d[jj]*np.eye(ii) - A)
                # Mat2_1 = (np.eye(ii) -
                #           np.dot(r_eigvt[:, jj], np.conj(l_eigvt[:, jj]))/ \
                #           np.dot(np.conj(l_eigvt[:, jj]), r_eigvt[:, jj]))
                # JpaohT = np.dot(np.dot(np.dot(Mat1_1, Mat2_1), OO), Qi)

                # phi_i = phi[jj, :]
                # phik = np.max(abs(phi_i))
                # phik_idx = np.argmax(abs(phi_i))
                # if phik_idx == 0:
                #     Mat1_2 = np.eye(Nch) - np.hstack([phi_i.reshape(-1,1),
                #                                     np.zeros((Nch,Nch-1))])
                # else:
                #     Mat1_2 = np.eye(Nch) - np.hstack([np.zeros((Nch, phik_idx-1)),
                #                                     phi_i.reshape(-1,1),
                #                                     np.zeros((Nch,Nch-phik_idx))])

                # Mat2_2 = np.dot(C,JpaohT) + np.dot(np.kron(r_eigvt[:, jj].T, np.eye(Nch)), Q4_n)

                # JpacohT = 1/phik*np.dot(Mat1_2, Mat2_2)

                # Uph=np.vstack([np.real(JpacohT), np.imag(JpacohT)])

                # cov_phi = np.dot(Uph,Uph.T)

                # Phi_cov[jj, ii, :] = abs(cov_phi[:Nch, 0])
            # logger.debug("... uncertainty calculations done!")
    if calc_unc is True:
        return Fn, Xi, Phi, Lambds, Fn_cov, Xi_cov, Phi_cov
    else:
        return Fn, Xi, Phi, Lambds, None, None, None


# -----------------------------------------------------------------------------


def SSI_multi_setup(
    Y: list,
    fs: float,
    br: int,
    ordmax: int,
    method_hank: str,
    step: int = 1,
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
        Method for Hankel matrix construction. Can be 'cov_mm', 'cov_R', 'dat'.
    step : int, optional
        Step size for increasing the order in the identification process. Default is 1.

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
    # dt = 1 / fs
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
        H, _ = build_hank(Y_all, Y_ref, br, method=method_hank, calc_unc=False)
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

    # if method == "FAST":
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

    return Obs_all, A, C


# -----------------------------------------------------------------------------


def SSI_mpe(
    freq_ref: list,
    Fn_pol: np.ndarray,
    Xi_pol: np.ndarray,
    Phi_pol: np.ndarray,
    order: int,
    Lab: typing.Optional[np.ndarray] = None,
    rtol: float = 5e-2,
    Fn_cov: np.ndarray = None,
    Xi_cov: np.ndarray = None,
    Phi_cov: np.ndarray = None,
):
    """
    Extract modal parameters using the Stochastic Subspace Identification (SSI) method
    for selected frequencies.

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
    order : int, list of int, or str
        Specifies the model order(s) for which the modal parameters are to be extracted.
        If 'find_min', the function attempts to find the minimum model order that provides
        stable poles for each mode of interest.
    Lab : numpy.ndarray, optional
        Array of labels identifying stable poles. Required if order is 'find_min'.
    rtol : float, optional
        Relative tolerance for comparing frequencies, by default 5e-2.
    Fn_cov : numpy.ndarray, optional
        Covariance array of natural frequencies, by default None.
    Xi_cov : numpy.ndarray, optional
        Covariance array of damping ratios, by default None.
    Phi_cov : numpy.ndarray, optional
        Covariance array of mode shapes, by default None.

    Returns
    -------
    tuple
        A tuple containing:
        - Fn (numpy.ndarray): Extracted natural frequencies.
        - Xi (numpy.ndarray): Extracted damping ratios.
        - Phi (numpy.ndarray): Extracted mode shapes.
        - order_out (numpy.ndarray or int): Output model order used for extraction for each frequency.
        - Fn_cov (numpy.ndarray, optional): Covariance of extracted natural frequencies.
        - Xi_cov (numpy.ndarray, optional): Covariance of extracted damping ratios.
        - Phi_cov (numpy.ndarray, optional): Covariance of extracted mode shapes.

    Raises
    ------
    AttributeError
        If 'order' is not an int, list of int, or 'find_min', or if 'order' is 'find_min'
        but 'Lab' is not provided.
    """

    if order == "find_min" and Lab is None:
        raise AttributeError(
            "When order ='find_min', one must also provide the Lab list for the poles"
        )

    sel_xi = []
    sel_phi = []
    sel_freq = []
    if Fn_cov is not None:
        sel_xi_cov = []
        sel_phi_cov = []
        sel_freq_cov = []

    # Loop through the frequencies given in the input list
    logger.info("Extracting SSI modal parameters")
    # =============================================================================
    # OPZIONE order = "find_min"
    # here we find the minimum model order so to get a stable pole for every mode of interest
    # -----------------------------------------------------------------------------
    if order == "find_min":
        # Find stable poles
        stable_poles = np.where(Lab == 1, Fn_pol, np.nan)
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
                    if Fn_cov is not None:
                        sel_freq_cov.append(Fn_cov[index, i])
                        sel_xi_cov.append(Xi_cov[index, i])
                        sel_phi_cov.append(Phi_cov[index, i, :])
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
                if Fn_cov is not None:
                    sel_freq_cov.append(Fn_cov[:, order][sel])
                    sel_xi_cov.append(Xi_cov[:, order][sel])
                    sel_phi_cov.append(Phi_cov[:, order][sel, :])
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
                if Fn_cov is not None:
                    sel_freq_cov.append(Fn_cov[:, order[ii]][sel])
                    sel_xi_cov.append(Xi_cov[:, order[ii]][sel])
                    sel_phi_cov.append(Phi_cov[:, order[ii]][sel, :])
                order_out[ii] = order[ii]
    else:
        raise AttributeError(
            'order must be either of type(int), type(list(int)) or "find_min"'
        )
    logger.debug("Done!")

    Fn = np.array(sel_freq).reshape(-1)
    Phi = np.array(sel_phi).T
    Xi = np.array(sel_xi)

    if Fn_cov is not None:
        Fn_cov = np.array(sel_freq_cov).reshape(-1)
        Phi_cov = np.array(sel_phi_cov).T
        Xi_cov = np.array(sel_xi_cov)
        return Fn, Xi, Phi, order_out, Fn_cov, Xi_cov, Phi_cov
    else:
        return Fn, Xi, Phi, order_out, None, None, None
