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
    method: typing.Literal["cov", "cov_R", "dat", "IOcov"],
    calc_unc: bool = False,
    nb: int = 50,
    U: np.ndarray = None,
) -> typing.Tuple[np.ndarray, typing.Optional[np.ndarray]]:
    """
    Construct a Hankel matrix using specified method with optional uncertainty estimation.

    Depending on the selected method, this function assembles a Hankel matrix using covariance-based,
    data-driven, or input-output approaches, with an option to estimate uncertainty.

    Parameters
    ----------
    Y : np.ndarray
        Output data matrix with shape (number of output channels, number of data points).
    Yref : np.ndarray
        Reference output data matrix with shape (number of reference channels, number of data points).
    br : int
        Number of block rows in the Hankel matrix.
    method : {'cov', 'cov_R', 'dat', 'IOcov'}
        Method used for Hankel matrix construction:
        - 'cov': Covariance-based using future and past output data.
        - 'cov_R': Covariance-based using correlation matrices.
        - 'dat': Data-driven using QR decomposition.
        - 'IOcov': Input-output covariance-based method (requires `U`).
    calc_unc : bool, optional
        If True, compute an uncertainty matrix using data segmentation. Default is False.
    nb : int, optional
        Number of segments for uncertainty estimation. Default is 50.
    U : np.ndarray, optional
        Input data matrix with shape (number of input channels, number of data points).
        Required if `method` is 'IOcov'.

    Returns
    -------
    Hank : np.ndarray
        Assembled Hankel matrix according to the selected method.
    T : np.ndarray or None
        Matrix containing uncertainty estimates, returned only if `calc_unc` is True; otherwise, None.

    Raises
    ------
    AttributeError
        If an unsupported method is specified or if input requirements are not met
        (e.g., `U` is not provided for 'IOcov').
    AttributeError
        If the number of data points per block is insufficient for uncertainty estimation.

    Notes
    -----
    Uncertainty calculations follow the approach detailed in [DOME13]_.
    """

    Ndat = Y.shape[1]  # number of data points
    l = Y.shape[0]  # number of channels # noqa E741 (ambiguous variable name 'l')
    r = Yref.shape[0]  # number of reference channels
    p = br - 1  # number of block rows
    q = br  # block columns
    N = Ndat - p - q  # length of the Hankel matrix

    T = None
    logger.info("Assembling Hankel matrix method: %s...", method)

    if method == "cov":
        # Future and past Output (Y^+ and Y^-)
        Yf = np.vstack([Y[:, q + i : N + q + i] for i in range(p + 1)])
        Yp = np.vstack([Yref[:, (q - 1) + i : N + (q - 1) + i] for i in range(0, -q, -1)])

        Hank = np.dot(Yf, Yp.T) / N

        # Uncertainty calculations
        if calc_unc:
            logger.info("... calculating cov(H)...")
            Nb = N // nb  # number of samples per segment
            T = np.zeros(((p + 1) * q * l * r, nb))  # Square root of SIGMA_H
            Hvec0 = Hank.reshape(-1, order="F")  # vectorized Hankel

            for k in range(nb):
                # Section 3.2 and 5.1 of DoMe13
                Yf_k = Yf[:, (k * Nb) : ((k + 1) * Nb)]
                Yp_k = Yp[:, (k * Nb) : ((k + 1) * Nb)]
                Hcov_k = np.dot(Yf_k, Yp_k.T) / Nb

                Hcov_vec_k = Hcov_k.reshape(-1, order="F")
                if Hcov_vec_k.shape[0] < T.shape[0]:
                    raise AttributeError(
                        "Not enough data points per data block."
                        "Try reducing the number of data blocks, nb and/or the number of block-rows, br"
                    )
                else:
                    T[:, k] = (Hcov_vec_k - Hvec0) / np.sqrt(nb * (nb - 1))

    elif method == "cov_R":
        # Correlations
        Ri = np.array(
            [1 / (N - k) * np.dot(Y[:, k:], Yref[:, : Ndat - k].T) for k in range(p + q)]
        )
        # Assembling the Toepliz matrix
        Hank = np.vstack(
            [np.hstack([Ri[i + j, :, :] for j in range(p + 1)]) for i in range(q)]
        )

        # Uncertainty calculations
        if calc_unc is True:
            logger.info("... calculating cov(H)...")
            Nb = N // nb  # number of samples per segment
            T = np.zeros(((p + 1) * q * l * r, nb))  # Square root of SIGMA_H
            Hvec0 = Hank.reshape(-1, 1, order="f")  # vectorialised hankel

            for j in range(1, nb + 1):
                print(j, nb)
                Ri = np.array([])
                for k in range(p + q):
                    print(f"{k=}, {p=}, {q=}")
                    res = np.array(
                        [1 / (Nb - k) * np.dot(Y[:, : j * Nb - k], Yref[:, k : j * Nb].T)]
                    )
                    Ri = np.vstack([Ri, res])
                Hcov_j = np.vstack(
                    [np.hstack([Ri[i + j, :, :] for j in range(p + 1)]) for i in range(q)]
                )

                Hcov_vec_j = Hcov_j.reshape(-1, 1, order="f")
                if Hcov_vec_j.shape[0] < T.shape[0]:
                    raise AttributeError(
                        "Not enough data points per data block."
                        "Try reducing the number of data blocks, nb and/or the number of block-rows, br"
                    )
                else:
                    T[:, j - 1] = (Hcov_vec_j - Hvec0).flatten() / np.sqrt(nb * (nb - 1))

    elif method == "dat":
        # Efficient method for assembling the Hankel matrix for data-driven SSI
        Yf = np.vstack([Y[:, q + i : N + q + i] for i in range(p + 1)])
        Yp = np.vstack([Yref[:, (q - 1) + i : N + (q - 1) + i] for i in range(0, -q, -1)])
        Ys = np.vstack((Yp / np.sqrt(N), Yf / np.sqrt(N)))

        R21 = np.linalg.qr(Ys.T, mode="r").T
        Hank = R21[r * (p + 1) :, : r * (p + 1)]

        # Uncertainty calculations
        if calc_unc:
            logger.info("... calculating cov(H)...")
            Nb = N // nb  # number of samples per segment
            T = np.zeros(((p + 1) * q * l * r, nb))  # Square root of SIGMA_H
            Hvec0 = Hank.reshape(-1, order="F")  # vectorized Hankel

            for j in range(nb):
                Yf_k = Yf[:, j * Nb : (j + 1) * Nb]
                Yp_k = Yp[:, j * Nb : (j + 1) * Nb]
                Ys_k = np.vstack((Yp_k / np.sqrt(Nb), Yf_k / np.sqrt(Nb)))

                R = np.linalg.qr(Ys_k.T, mode="r").T
                Hdat_j = R[r * (p + 1) :, : r * (p + 1)]

                Hdat_vec_j = Hdat_j.reshape(-1, order="F")
                if Hdat_vec_j.shape[0] < T.shape[0]:
                    raise AttributeError(
                        "Not enough data points per data block."
                        "Try reducing the number of data blocks, nb and/or the number of block-rows, br"
                    )
                else:
                    T[:, j] = (Hdat_vec_j - Hvec0) / np.sqrt(nb * (nb - 1))

    elif method == "IOcov":
        try:
            inp = U.shape[0]  # number of input channels
        except AttributeError as e:
            raise AttributeError(
                "U must be provided when using method 'IOcov'."
                "Please provide the input data matrix U."
            ) from e
        # preallocate
        Uf = np.zeros(((p + 1) * inp, N))
        Up = np.zeros((q * inp, N))
        Yf = np.zeros(((p + 1) * l, N))
        Yp = np.zeros((q * r, N))

        # build past‐input blocks Up and past‐output blocks Yp
        for i in range(q):
            # take columns i .. i+N-1
            Up[i * inp : (i + 1) * inp, :] = U[:, i : i + N]
            Yp[i * r : (i + 1) * r, :] = Yref[:, i : i + N]

        # build future‐input blocks Uf and future‐output blocks Yf
        for i in range(p + 1):
            Uf[i * inp : (i + 1) * inp, :] = U[:, q + i : q + i + N]
            Yf[i * l : (i + 1) * l, :] = Y[:, q + i : q + i + N]

        R2 = Yf.dot(Yp.T) / N
        R4 = Yf.dot(Uf.T) / N
        R8 = Uf.dot(Uf.T) / N
        R5 = Yp.dot(Uf.T) / N

        # Finally compute hk
        Hank = R2 - R4 @ np.linalg.pinv(R8) @ R5.T

        # Uncertainty calculations
        logger.info("... calculating cov(H)...")
        if calc_unc:
            Nb = N // nb  # number of samples per segment
            T = np.zeros(((p + 1) * q * l * r, nb))  # Square root of SIGMA_H
            Hvec0 = Hank.reshape(-1, order="F")  # vectorized Hankel

            for k in trange(nb):
                Yf_k = Yf[:, (k * Nb) : ((k + 1) * Nb)]
                Yp_k = Yp[:, (k * Nb) : ((k + 1) * Nb)]

                Uf_k = Uf[:, (k * Nb) : ((k + 1) * Nb)]
                # Up_k = Up[:, (k * Nb) : ((k + 1) * Nb)]

                R2_k = np.dot(Yf_k, Yp_k.T) / Nb
                R4_k = np.dot(Yf_k, Uf_k.T) / Nb
                R5_k = np.dot(Yp_k, Uf_k.T) / Nb
                R8_k = np.dot(Uf_k, Uf_k.T) / Nb

                # Finally compute hk
                Hcov_k = R2_k - R4_k @ np.linalg.pinv(R8_k) @ R5_k.T

                Hcov_vec_k = Hcov_k.reshape(-1, order="F")

                if Hcov_vec_k.shape[0] < T.shape[0]:
                    raise AttributeError(
                        "Not enough data points per data block."
                        "Try reducing the number of data blocks, nb and/or the number of block-rows, br"
                    )
                else:
                    T[:, k] = (Hcov_vec_k - Hvec0) / np.sqrt(nb * (nb - 1))

    else:
        raise AttributeError(
            f'{method} is not a valid argument. "method" must be '
            f'one of: "cov", "cov_R", "dat", "IOcov"'
        )

    logger.debug("... Hankel and SIGMA_H Done!")
    return Hank, T


# -----------------------------------------------------------------------------


def synt_spctr(
    A: np.ndarray,
    C: np.ndarray,
    G: np.ndarray,
    R0: np.ndarray,
    omega: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Compute the synthetic output power spectral density matrix.

    This function evaluates the power spectral density (PSD) of system outputs using
    a state-space model over a range of angular frequencies.

    Parameters
    ----------
    A : np.ndarray
        State matrix of shape (n, n).
    C : np.ndarray
        Output matrix of shape (p, n).
    G : np.ndarray
        Input influence matrix of shape (n, p).
    R0 : np.ndarray
        Output noise covariance matrix of shape (p, p).
    omega : np.ndarray
        Array of angular frequencies at which the PSD is computed.
    dt : float
        Sampling time interval.

    Returns
    -------
    S_YY : np.ndarray
        Power spectral density matrix of shape (p, p, N), where N is the number of frequency points.
    """
    n = A.shape[0]
    p = C.shape[0]
    assert A.shape == (n, n)
    assert C.shape == (p, n)
    assert G.shape == (n, p)
    assert R0.shape == (p, p)

    # build z array and preallocate Sy
    z_vals = np.exp(1j * omega * dt)  # shape (N,)
    N = z_vals.size
    S_YY = np.zeros((p, p, N), dtype=complex)

    # preallocate identity matrix
    I_n = np.eye(n)

    # Fill 3D array slice by slice
    for k, z in enumerate(z_vals):
        M1 = z * I_n - A
        inv1 = np.linalg.inv(M1)  # (n×n)
        T1 = C @ inv1 @ G  # (p×p)

        M2 = (1 / z) * I_n - A.T
        inv2 = np.linalg.inv(M2)  # (n×n)
        T2 = G.T @ inv2 @ C.T  # (p×p)

        # assemble
        S_YY[:, :, k] = T1 + R0 + T2
    return S_YY


# Legacy
def SSI(
    H: np.ndarray, br: int, ordmax: int, step: int = 1
) -> typing.Tuple[np.ndarray, typing.List[np.ndarray], typing.List[np.ndarray]]:
    """
    Perform System Identification using the Stochastic Subspace Identification (SSI) method.

    The SSI algorithm estimates system matrices and output influence matrices for increasing
    model orders, based on a provided Hankel matrix.

    Parameters
    ----------
    H : np.ndarray
        The Hankel matrix of the system.
    br : int
        Number of block rows in the Hankel matrix.
    ordmax : int
        Maximum system order to consider for identification.
    step : int, optional
        Step size for increasing system order. Default is 1.

    Returns
    -------
    Obs : np.ndarray
        The observability matrix for the system.
    A : list of np.ndarray
        Estimated system matrices for various system orders.
    C : list of np.ndarray
        Estimated output influence matrices for various system orders.

    Notes
    -----
    This is a classical implementation of SSI using the shift structure of the observability
    matrix. For faster implementations, consider using `SSI_fast`.
    """

    Nch = int(H.shape[0] / (br))
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
    return Obs, A, C


# -----------------------------------------------------------------------------


def SSI_fast(
    H: np.ndarray,
    br: int,
    ordmax: int,
    step: int = 1,
) -> typing.Tuple[
    typing.List[np.ndarray],
    typing.List[np.ndarray],
    typing.List[np.ndarray],
    typing.List[np.ndarray],
]:
    """
    Perform a fast implementation of Stochastic Subspace Identification (SSI).

    This optimized SSI method uses QR decomposition to accelerate the extraction of system
    and output matrices for varying model orders.

    Parameters
    ----------
    H : np.ndarray
        The Hankel matrix.
    br : int
        Number of block rows in the Hankel matrix.
    ordmax : int
        Maximum system order for identification.
    step : int, optional
        Step size for increasing system order. Default is 1.

    Returns
    -------
    Obs : np.ndarray
        The global observability matrix of the system.
    A : list of np.ndarray
        List of estimated system matrices for each model order.
    C : list of np.ndarray
        List of estimated output influence matrices for each model order.
    G : list of np.ndarray
        List of estimated next state-output covariance matrices


    Notes
    -----
    This method is computationally more efficient than the classical SSI approach,
    leveraging QR decomposition for rapid estimation, see [DOME13]_.
    """

    l = int(H.shape[0] / (br))  # noqa E741 (ambiguous variable name 'l')  Number of channels

    # SINGULAR VALUE DECOMPOSITION
    U, SIG, VT = np.linalg.svd(H)

    S1rad = np.sqrt(np.diag(SIG))
    # initializing arrays
    Obs = np.dot(U[:, :ordmax], S1rad[:ordmax, :ordmax])  # Observability matrix
    Con = np.dot(S1rad[:ordmax, :ordmax], VT[:ordmax, :])  # Controllability matrix

    Oup = Obs[: Obs.shape[0] - l, :]
    Odw = Obs[l:, :]
    # # Extract system matrices
    # QR decomposition
    Q, R = np.linalg.qr(Oup)
    S = np.dot(Q.T, Odw)

    # Initialize A and C matrices
    A, C, G = [], [], []
    # loop for increasing order of the system
    logger.info("SSI for increasing model order...")
    for n in trange(0, ordmax + 1, step):
        A.append(np.dot(np.linalg.inv(R[:n, :n]), S[:n, :n]))
        C.append(Obs[:l, :n])
        G.append(Con[:n, -l:])
    logger.debug("... Done!")
    return Obs, A, C, G


# -----------------------------------------------------------------------------


def SSI_poles(
    Obs: np.ndarray,
    AA: typing.List[np.ndarray],
    CC: typing.List[np.ndarray],
    ordmax: int,
    dt: float,
    step: int = 1,
    HC: bool = True,
    xi_max: float = 0.1,
    calc_unc: bool = False,
    H: np.ndarray = None,
    T: np.ndarray = None,
) -> typing.Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    typing.Optional[np.ndarray],
    typing.Optional[np.ndarray],
    typing.Optional[np.ndarray],
    typing.Optional[np.ndarray],
]:
    """
    Calculate modal parameters using the Stochastic Subspace Identification (SSI) method.

    The function computes natural frequencies, damping ratios, and mode shapes for a range
    of system orders. Optional uncertainty propagation can be performed.

    Parameters
    ----------
    Obs : np.ndarray
        The global observability matrix.
    AA : list of np.ndarray
        List of estimated system matrices for each model order.
    CC : list of np.ndarray
        List of output matrices for each model order.
    ordmax : int
        The maximum model order.
    dt : float
        Sampling time step.
    step : int, optional
        Step size for increasing model order. Default is 1.
    HC : bool, optional
        Whether to apply Hard Criteria to remove unstable poles. Default is True.
    xi_max : float, optional
        Maximum allowed damping ratio. Default is 0.1.
    calc_unc : bool, optional
        Whether to calculate uncertainties for modal parameters. Default is False.
    H : np.ndarray, optional
        The Hankel matrix, required for uncertainty propagation. Default is None.
    T : np.ndarray, optional
        Auxiliary uncertainty matrix `cov(H)` used in uncertainty propagation.

    Returns
    -------
    Fn : np.ndarray
        Array of natural frequencies for each system order.
    Xi : np.ndarray
        Array of damping ratios for each system order.
    Phi : np.ndarray
        Normalised (to unity) mode shapes for each system order.
    Lambdas : np.ndarray
        Continuous-time eigenvalues for each system order.
    Fn_std : np.ndarray, optional
        Standard deviations of natural frequencies, returned if `calc_unc` is True.
    Xi_std : np.ndarray, optional
        Standard deviations of damping ratios, returned if `calc_unc` is True.
    Phi_std : np.ndarray, optional
        Standard deviations of mode shapes, returned if `calc_unc` is True.
    """

    # NB Nch = l

    l = int(CC[0].shape[0])  # noqa E741 (ambiguous variable name 'l')
    p = int(Obs.shape[0] / l - 1)
    q = int(p + 1)  # block column

    # Build selector matrices
    S1 = np.hstack([np.eye(p * l), np.zeros((p * l, l))])
    S2 = np.hstack([np.zeros((p * l, l)), np.eye(p * l)])

    # initialization of the matrix that contains the frequencies
    Lambdas = np.full((ordmax, int((ordmax) / step + 1)), np.nan, dtype=complex)
    # initialization of the matrix that contains the frequencies
    Fn = np.full((ordmax, int((ordmax) / step + 1)), np.nan)
    # initialization of the matrix that contains the damping ratios
    Xi = np.full((ordmax, int((ordmax) / step + 1)), np.nan)
    # initialization of the matrix that contains the mode shapes
    Phi = np.full((ordmax, int((ordmax) / step + 1), l), np.nan, dtype=complex)

    if calc_unc:
        nb = T.shape[1]
        r = int(H.shape[1] / q)  # Number of reference channels

        # Calculate SVD and truncate at nth order
        U, S, VT = np.linalg.svd(H)

        # initialization of the matrix that contains the frequencies
        Fn_std = np.full((ordmax, int((ordmax) / step + 1)), np.nan)
        # initialization of the matrix that contains the damping ratios
        Xi_std = np.full((ordmax, int((ordmax) / step + 1)), np.nan)
        # initialization of the matrix that contains the mode shapes
        Phi_std = np.full(
            (ordmax, int((ordmax) / step + 1), l),
            np.nan,
        )

        # SVD truncation at ordmax
        Un = U[:, :ordmax]
        Vn = VT.T[:, :ordmax]
        Sn = np.diag(S[:ordmax])

        Q1 = np.zeros(((ordmax) ** 2, nb))
        Q2 = np.zeros_like(Q1)
        Q3 = np.zeros_like(Q1)
        Q4 = np.zeros((l * ordmax, nb))
        logger.info("... propagating uncertainty...")
        for ii in trange(1, ordmax + 1, step):
            sn_k = Sn[ii - 1, ii - 1]
            Vn_k = Vn[:, ii - 1].reshape(-1, 1)
            Un_k = Un[:, ii - 1].reshape(-1, 1)
            # Eq. 28
            Kj = (
                np.eye(q * r)
                + np.vstack([np.zeros((q * r - 1, q * r)), 2 * Vn_k.T])
                - np.dot(H.T, H) / sn_k**2
            )
            Ki = np.linalg.inv(Kj)
            # Eq. 29
            Bi1 = np.eye((p + 1) * l) + (H / sn_k) @ Ki @ (
                (H.T / sn_k) - np.vstack([np.zeros((q * r - 1, (p + 1) * l)), Un_k.T])
            )
            Bi1 = np.hstack([Bi1, (H / sn_k) @ Ki])
            # Remark 9
            # Eq. 33
            Ti1 = np.kron(np.eye(q * r), Un_k.T) @ T
            Ti2 = np.kron(Vn_k.T, np.eye((p + 1) * l)) @ T
            # Eq. 34
            JOHTi = (1 / (2 * np.sqrt(sn_k))) * (Un_k @ (Vn_k.T @ Ti1)) + (
                1 / np.sqrt(sn_k)
            ) * (
                Bi1
                @ np.vstack(
                    [Ti2 - (Un_k @ (Un_k.T @ Ti2)), Ti1 - (Vn_k @ (Vn_k.T @ Ti1))]
                )
            )
            # Eq. 36-37
            Q1[(ii - 1) * ordmax : (ii) * ordmax, :] = ((S1 @ Obs).T @ S1) @ JOHTi
            Q2[(ii - 1) * ordmax : (ii) * ordmax, :] = ((S2 @ Obs).T @ S1) @ JOHTi
            Q3[(ii - 1) * ordmax : (ii) * ordmax, :] = ((S1 @ Obs).T @ S2) @ JOHTi
            Q4[(ii - 1) * l : (ii) * l, :] = (
                np.hstack([np.eye(l), np.zeros((l, p * l))]) @ JOHTi
            )

    logger.info("Calculating modal parameters for increasing model order...")
    for nn in trange(0, ordmax + 1, step):
        n = nn // step
        Oi = Obs[:, :nn]
        Oup = np.dot(S1, Oi)
        A = AA[n]
        C = CC[n]

        # Check if A is an empty array and continue if it is
        if A.size == 0:
            continue

        # Compute modal parameters
        lam_d, l_eigvt, r_eigvt = linalg.eig(A, left=True)  # l_eigvt=chi, r_eigvt=phi
        lam_c = (np.log(lam_d)) * (1 / dt)  # to continous time
        xi = -((np.real(lam_c)) / (abs(lam_c)))  # damping ratios
        phi = np.dot(C, r_eigvt)  # N.B. this is \varphi
        fn = abs(lam_c) / (2 * np.pi)  # natural frequencies

        if HC is not False:
            # REMOVING UNSTABLE POLES to speed up loop on lam_c
            # Identify elements that have their conjugate in lam_c
            mask_conj = np.isin(lam_c, np.conj(lam_c))
            # Remove conjugate duplicates by keeping only one element from each pair
            unique_mask = mask_conj.copy()
            processed = set()
            for i, elem in enumerate(lam_c):
                if unique_mask[i]:
                    conj = np.conj(elem)
                    if conj in processed:
                        unique_mask[i] = False  # Remove the conjugate duplicate
                    else:
                        processed.add(elem)  # Mark the element as processed
            # Filter lam_c based on unique_mask
            lam_c = np.where(unique_mask, lam_c, np.nan)
            # Apply damping mask
            mask_damp = (xi > 0) & (xi < xi_max)
            lam_c = np.where(mask_damp, lam_c, np.nan)
            # Filtered frequencies and damping
            fn = abs(lam_c) / (2 * np.pi)  # natural frequencies
            xi = -((np.real(lam_c)) / (abs(lam_c)))  # damping ratios
            # Expand the mask to 3D by adding a new axis (for mode shape)
            expandedmask1 = np.expand_dims(unique_mask, axis=-1)
            expandedmask2 = np.expand_dims(mask_damp, axis=-1)
            # Repeat the mask along the new dimension
            expandedmask1 = np.repeat(expandedmask1, Phi.shape[2], axis=-1)
            expandedmask2 = np.repeat(expandedmask2, Phi.shape[2], axis=-1)
            # mask the values
            phi1 = np.where(expandedmask1, phi.T, np.nan)
            phi1 = np.where(expandedmask2, phi1, np.nan)
            phi = phi1.T

        idx = np.argmax(abs(phi), axis=0)
        vmaxs = [phi[idx[k1], k1] for k1 in range(len(idx))]

        Fn[: len(fn), n] = fn  # save the frequencies
        Xi[: len(fn), n] = xi  # save the damping ratios
        # Phi[: len(fn), n, :] = phi.T
        Lambdas[: len(fn), n] = lam_c

        if calc_unc:
            In = np.eye(nn)
            Inl = np.eye(nn * l)
            Zero = np.zeros((nn, (ordmax) - nn))
            Zero1 = np.zeros((nn * l, l * ((ordmax) - nn)))
            matr = np.hstack((In, Zero))
            matr1 = np.hstack((Inl, Zero1))

            # Selection matrix
            S4_n = np.kron(matr, matr)
            # Proposition 11
            Q1n = np.dot(S4_n, Q1)
            Q2n = np.dot(S4_n, Q2)
            Q3n = np.dot(S4_n, Q3)
            Q4n = np.dot(matr1, Q4)

            Un = U[:, :nn]
            Vn = VT.T[:, :nn]
            Sn = np.diag(S[:nn])

            Pnn = np.zeros((nn**2, nn**2))
            for jj in range(nn):
                ek = np.zeros(nn).reshape(-1, 1)
                ek[jj, 0] = 1
                Pnn[:, jj * nn : (jj + 1) * nn] = np.kron(np.eye(nn), ek)

            OO = np.linalg.pinv(np.dot(Oup.T, Oup))
            PnQ1 = (Pnn + np.eye(nn**2)) @ Q1n
            PnQ23 = Pnn @ Q2n + Q3n

            # step 3 algo 2
            for jj in range(len(lam_c)):
                if not np.isnan(fn[jj]):
                    # step 4 algo 2
                    # Eq. 44
                    Qi = np.dot(
                        np.kron(r_eigvt[:, jj].T, np.eye(nn)), (-lam_d[jj] * PnQ1 + PnQ23)
                    )
                    # Lemma 5 - fn and xi
                    Mat1 = np.array(
                        [[1 / (2 * np.pi), 0], [0, 1 / (np.abs(lam_c[jj]) ** 2)]]
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
                    # Eq. 40
                    cov_fx = np.dot(Ufx, Ufx.T)
                    # standard deviation
                    Fn_std[jj, n] = cov_fx[0, 0] ** 0.5
                    Xi_std[jj, n] = cov_fx[1, 1] ** 0.5

                    # Lemma 5 - phi
                    Mat1_1 = np.linalg.pinv(lam_d[jj] * np.eye(nn) - A)
                    Mat2_1 = np.eye(nn) - np.dot(
                        r_eigvt[:, jj].reshape(-1, 1),
                        l_eigvt[:, jj].reshape(-1, 1).conj().T,
                    ) / np.dot(
                        l_eigvt[:, jj].reshape(-1, 1).conj().T,
                        r_eigvt[:, jj].reshape(-1, 1),
                    )
                    # Eq. 46
                    JpaohT = np.dot(np.dot(np.dot(Mat1_1, Mat2_1), OO), Qi)

                    if idx[jj] == 0:
                        Mat1_2 = np.eye(l) - np.hstack(
                            [phi[:, jj].reshape(-1, 1), np.zeros((l, l - 1))]
                        )
                    else:
                        Mat1_2 = np.eye(l) - np.hstack(
                            [
                                np.zeros((l, idx[jj])),
                                phi[:, jj].reshape(-1, 1),
                                np.zeros((l, l - idx[jj] - 1)),
                            ]
                        )

                    Mat2_2 = np.dot(C, JpaohT) + np.dot(
                        np.kron(r_eigvt[:, jj].T, np.eye(l)), Q4n
                    )
                    # Eq. 45
                    JpacohT = 1 / vmaxs[jj] * np.dot(Mat1_2, Mat2_2)

                    Uph = np.vstack([np.real(JpacohT), np.imag(JpacohT)])
                    # Eq. 40
                    cov_phi = np.dot(Uph, Uph.T)
                    # standard deviation
                    Phi_std[jj, n, :] = abs(np.diag(cov_phi[:l, :l])) ** 0.5
            logger.debug("... uncertainty calculations done!")

        try:
            # Normalisation to unity
            # idx = np.argmax(abs(phi), axis=0)
            phi = (
                np.array(
                    [
                        phi[:, ii] / phi[np.argmax(abs(phi[:, ii])), ii]
                        for ii in range(phi.shape[1])
                    ]
                )
                .reshape(-1, l)
                .T
            )
            # vmaxs = [phi[idx[k1], k1] for k1 in range(len(idx))]
        except Exception as e:
            logging.debug(f"Ignored exception during normalization: {e}")
            pass

        Phi[: len(fn), n, :] = phi.T
    if calc_unc is True:
        return Fn, Xi, Phi, Lambdas, Fn_std, Xi_std, Phi_std
    else:
        return Fn, Xi, Phi, Lambdas, None, None, None


# -----------------------------------------------------------------------------


def SSI_multi_setup(
    Y: list,
    fs: float,
    br: int,
    ordmax: int,
    method_hank: str,
    step: int = 1,
) -> typing.Tuple[
    np.ndarray,
    typing.List[np.ndarray],
    typing.List[np.ndarray],
]:
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
        Method for Hankel matrix construction. Can be 'cov', 'cov_R', 'dat'.
    step : int, optional
        Step size for increasing the order in the identification process. Default is 1.

    Returns
    -------
    tuple
        Obs_all : numpy array of the global observability matrix.
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
    Oup = Obs_all[: Obs_all.shape[0] - n_DOF, :]
    Odw = Obs_all[n_DOF:, :]
    # QR decomposition
    Q, R = np.linalg.qr(Oup)
    S = np.dot(Q.T, Odw)
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
    order: typing.Union[int, list, str],
    step: int,
    Lab: typing.Optional[np.ndarray] = None,
    rtol: float = 5e-2,
    Fn_std: np.ndarray = None,
    Xi_std: np.ndarray = None,
    Phi_std: np.ndarray = None,
) -> typing.Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    typing.Union[np.ndarray, int],
    typing.Optional[np.ndarray],
    typing.Optional[np.ndarray],
    typing.Optional[np.ndarray],
]:
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
    Fn_std : numpy.ndarray, optional
        Covariance array of natural frequencies, by default None.
    Xi_std : numpy.ndarray, optional
        Covariance array of damping ratios, by default None.
    Phi_std : numpy.ndarray, optional
        Covariance array of mode shapes, by default None.

    Returns
    -------
    tuple
        A tuple containing:
        - Fn (numpy.ndarray): Extracted natural frequencies.
        - Xi (numpy.ndarray): Extracted damping ratios.
        - Phi (numpy.ndarray): Extracted mode shapes.
        - order_out (numpy.ndarray or int): Output model order used for extraction for each frequency.
        - Fn_std (numpy.ndarray, optional): Covariance of extracted natural frequencies.
        - Xi_std (numpy.ndarray, optional): Covariance of extracted damping ratios.
        - Phi_std (numpy.ndarray, optional): Covariance of extracted mode shapes.

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
    if Fn_std is not None:
        sel_Xi_std = []
        sel_Phi_std = []
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
                    if Fn_std is not None:
                        sel_freq_cov.append(Fn_std[index, i])
                        sel_Xi_std.append(Xi_std[index, i])
                        sel_Phi_std.append(Phi_std[index, i, :])
                order_out = i * step
                break

        if not found:
            logger.warning("Could not find any values")
            order_out = None
    # =============================================================================
    # OPZIONE 2 order = int
    # -----------------------------------------------------------------------------
    elif isinstance(order, int):
        order = int(order / step)
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
                if Fn_std is not None:
                    sel_freq_cov.append(Fn_std[:, order][sel])
                    sel_Xi_std.append(Xi_std[:, order][sel])
                    sel_Phi_std.append(Phi_std[:, order][sel, :])
                order_out = order * step
    # =============================================================================
    # OPZIONE 3 order = list[int]
    # -----------------------------------------------------------------------------
    elif isinstance(order, list):
        # Convert each element in the order list to model orders
        order = [int(o / step) for o in order]
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
                if Fn_std is not None:
                    sel_freq_cov.append(Fn_std[:, order[ii]][sel])
                    sel_Xi_std.append(Xi_std[:, order[ii]][sel])
                    sel_Phi_std.append(Phi_std[:, order[ii]][sel, :])
                order_out[ii] = order[ii] * step
    else:
        raise AttributeError(
            'order must be either of type(int), type(list(int)) or "find_min"'
        )
    logger.debug("Done!")

    Fn = np.array(sel_freq).reshape(-1)
    Phi = np.array(sel_phi).T
    Xi = np.array(sel_xi)

    if Fn_std is not None:
        Fn_std = np.array(sel_freq_cov).reshape(-1)
        Phi_std = np.array(sel_Phi_std).T
        Xi_std = np.array(sel_Xi_std)
        return Fn, Xi, Phi, order_out, Fn_std, Xi_std, Phi_std
    else:
        return Fn, Xi, Phi, order_out, None, None, None
