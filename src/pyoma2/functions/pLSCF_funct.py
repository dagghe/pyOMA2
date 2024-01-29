"""
Created on Sat Oct 21 19:12:47 2023

@author: dagpa
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# FUNZIONI PolyMAX
# =============================================================================


def pLSCF(Sy, dt, ordmax):
    """
    Perform system identification using the p-LSCF (poly-reference Least Square
    Complex Frequency) method, also known as polyMAX.

    Parameters:
    Sy (numpy.ndarray): 3D array representing the power spectral density (PSD) matrix.
                        Shape should be (Nf, Nch, Nref), where:
                        Nf - Number of frequencies,
                        Nch - Number of channels, and
                        Nref - Number of reference channels.

    dt (float): Time resolution or sampling interval.

    ordmax (int): Maximum model order for calculating the companion matrix.

    Returns:
    list: List of companion matrices for each order up to ordmax.

    Notes:"""
    Nf = Sy.shape[2]
    freq = np.arange(0, Nf) * (1 / dt / (2 * Nf))
    # p-LSCF - METODO CON MATRICI REALI
    freq_w = 2 * np.pi * freq

    # The PSD matrix should be in the format (k, o, o) where:
    # k=1,2,...Nf; and o=1,2...Nch
    Sy = np.moveaxis(Sy, 2, 0)
    Nf = Sy.shape[0]
    Nch = Sy.shape[1]
    Nref = Sy.shape[2]
    A = []
    # Calculation of companion matrix A and modal parameters for each order
    for j in range(1, ordmax + 1):  # loop for increasing model order
        M = np.zeros(((j + 1) * Nref, (j + 1) * Nref))  # inizializzo
        X0 = np.array([np.exp(1j * freq_w * dt * jj) for jj in range(j + 1)]).T
        X0h = X0.conj().T  # Calculate complex transpose
        R0 = np.real(np.dot(X0h, X0))  # 4.163

        for o in range(0, Nch):  # loop on channels
            Y0 = np.array([np.kron(-X0[kk, :], Sy[kk, o, :]) for kk in range(Nf)])
            S0 = np.real(np.dot(X0h, Y0))  # 4.164
            T0 = np.real(np.dot(Y0.conj().T, Y0))  # 4.165
            # np.linalg.solve(R0, S0))) # 4.167
            M += 2 * (T0 - np.dot(np.dot(S0.T, np.linalg.inv(R0)), S0))
        alfa = np.linalg.solve(
            -M[: j * Nref, : j * Nref], M[: j * Nref, j * Nref : (j + 1) * Nref]
        )  # 4.169
        alfa = np.vstack((alfa, np.eye(Nref)))
        # beta0 = np.linalg.solve(-R0, np.dot(S0,alfa))
        # Companion matrix
        AA = np.zeros((j * Nref, j * Nref))
        for ii in range(j):
            Aj = alfa[ii * Nref : (ii + 1) * Nref, :]
            AA[(j - 1) * Nref :, ii * Nref : (ii + 1) * Nref] = -Aj.T
        if j == 1:
            A.append(np.zeros((0, 0)))
        AA[: (j - 1) * Nref, Nref : j * Nref] = np.eye((j - 1) * Nref)
        A.append(AA)
    return A


# -----------------------------------------------------------------------------


# def pLSCF_NEW(Sy, dt, ordmax):
#     pass
#     return


# -----------------------------------------------------------------------------


def pLSCF_Poles(A, ordmax, dt, methodSy, nxseg, Nref=None):
    Nch = int(A[1].shape[0])
    if Nref is None:
        Nref = Nch
    Fn = np.full((ordmax * Nch, ordmax + 1), np.nan)  # initialise
    Sm = np.full((ordmax * Nch, ordmax + 1), np.nan)  # initialise
    # initialise    for ii in range(NAC):
    Ls = np.full((ordmax * Nch, ordmax + 1), np.nan, dtype=complex)
    for j in range(1, ordmax + 1):  # loop for increasing model order
        # Eigenvalueproblem
        [my, My] = np.linalg.eig(A[j])
        lambd = np.log(my) / dt  # From discrete-time to continuous time 4.136
        # replace with nan every value with negative real part (should be the other way around!)
        lambd = np.where(np.real(lambd) < 0, np.nan, lambd)
        if methodSy == "cor":  # correct for exponential window
            tau = -(nxseg - 1) / np.log(0.01)
            lambd = lambd - 1 / tau

        Ls[: (j) * Nch, (j)] = lambd
        # Natural frequencies (Hz) 4.137
        Fn[: (j) * Nch, (j)] = abs(lambd) / (2 * np.pi)
        # Damping ratio initial calc 4.139
        Sm[: (j) * Nch, (j)] = (np.real(lambd)) / abs(lambd)
    return Fn, Sm, Ls


# -----------------------------------------------------------------------------


def Lab_stab_pLSCF(Fn, Sm, ordmax, err_fn, err_xi, max_xi):
    """
    Helping function for the construction of the Stability Chart when using
    poly-reference Least Square Complex Frequency (pLSCF, also known as
    Polymax) method.

    This function performs stability analysis of identified poles,
    it categorizes modes based on their stability in terms
    of frequency and damping.

    :param Fr: Frequency matrix, shape: ``(n_locations, n_modes)``
    :param Sm: Damping matrix, shape: ``(n_locations, n_modes)``
    :param ordmax: Maximum order of modes to consider (exclusive)
    :param err_fn: Threshold for relative frequency difference for stability checks
    :param err_xi: Threshold for relative damping ratio difference for stability checks
    :param nch: Number of channels (modes) in the analysis

    :return: Stability label matrix (Lab), shape: ``(n_locations, n_modes)``
        - 3: Stable Pole (frequency and damping)
        - 2: Stable damping
        - 1: Stable frequency
        - 0: New or unstable pole

    Note:

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

    # -----------------------------------------------------------------------------
    # STABILITY BETWEEN CONSECUTIVE ORDERS
    for nn in range(1, ordmax + 1):

        f_n = Fn1[:, nn].reshape(-1, 1)
        xi_n = Sm[:, nn].reshape(-1, 1)

        f_n1 = Fn1[:, nn - 1].reshape(-1, 1)
        xi_n1 = Sm[:, nn - 1].reshape(-1, 1)

        if nn != 0:

            for i in range(len(f_n)):

                if np.isnan(f_n[i]):
                    pass
                else:
                    try:
                        idx = np.nanargmin(np.abs(f_n1 - f_n[i]))

                        cond1 = np.abs(f_n[i] - f_n1[idx]) / f_n[i]
                        cond2 = np.abs(xi_n[i] - xi_n1[idx]) / xi_n[i]

                        if cond1 < err_fn and cond2 < err_xi:
                            Lab[i, nn] = 3  # Stable Pole

                        elif cond2 < err_xi:
                            Lab[i, nn] = 2  # Stable damping

                        elif cond1 < err_fn:
                            Lab[i, nn] = 1  # Stable frequency

                        else:
                            Lab[i, nn] = 0  # Nuovo polo o polo instabile
                    except Exception as e:
                        logger.debug(e)
    return Lab


# -----------------------------------------------------------------------------


def pLSCF_MPE(sel_freq, Sy, Fn_pol, Sm_pol, Ls_pol, order, dt, DF=1):
    """
    Bla bla bla
    """
    sel_freq1 = []
    sel_xi = []
    sel_lam = []
    for fj in sel_freq:
        if order == "find_min":
            # here we find the minimum model order so to get a stable pole for every mode of interest
            pass
        else:  # when the model order is provided
            # Find closest frequency index

            sel = np.nanargmin(np.abs(Fn_pol[:, order] - fj))

            sel_freq1.append(Fn_pol[:, order][sel])
            sel_xi.append(Sm_pol[:, order][sel])
            sel_lam.append(Ls_pol[:, order][sel])

    Nch, Nref, Nf = Sy.shape
    w_sel = np.array(sel_freq) * (2 * np.pi)
    Nm = len(sel_lam)  # numero modi
    Phi = np.zeros((Nch, Nm), dtype=complex)
    freq = np.arange(0, Nf) * (1 / dt / (2 * Nf))
    freq_rad = 2 * np.pi * freq
    LL = np.zeros((Nch * Nm, Nch * Nm), dtype=complex)  # inizializzo
    GL = np.zeros((Nch * Nm, Nch), dtype=complex)  # inizializzo
    Sy = np.moveaxis(Sy, 2, 0)
    for ww in w_sel:  # loop su poli selezionati

        # =============================================================================
        # QUI MI SA CHE MI SON SCORDATO UN PEZZO (i coniguati) SU LAMBDA_L e GAMMA_L
        # =============================================================================
        # idx_w = np.argmin(np.abs(freq_rad-ww)) # trovo indice
        # df = int(1/dt/Nf)
        # nn = DF*df
        # loop sulle linee di frequenza intorno al polo fisico (+nn e -nn)
        # for kk in range(idx_w-nn, idx_w+nn):
        for kk in range(Nf):
            GL += np.array(
                [Sy[kk, :, :] / (1j * freq_rad[kk] - sel_lam[jj]) for jj in range(Nm)]
            ).reshape(-1, Nch)

            LL += np.array(
                [
                    np.array(
                        [
                            np.eye(Nch)
                            / (
                                (1j * freq_rad[kk] - sel_lam[jj1])
                                * (1j * freq_rad[kk] - sel_lam[jj2])
                            )
                            for jj2 in range(Nm)
                        ]
                    )
                    .reshape((Nch * Nm, Nch), order="c")
                    .T
                    for jj1 in range(Nm)
                ]
            ).reshape((Nch * Nm, Nch * Nm))

    R = np.linalg.solve(LL, GL)  # matrice dei residui (fi@fi^T

    for jj in range(len(w_sel)):
        # SVD della matrice dei residui per ciascun modo fisico del sistema
        U, S, VT = np.linalg.svd(R[jj * Nch : (jj + 1) * Nch, :])

        phi = U[:, 0]  # la forma modale è la prima colonna di U

        idmax = np.argmax(abs(phi))
        phiN = phi / phi[idmax]  # normalised (unity displacement)

        Phi[:, jj] = phiN
    # Save results
    Fn = np.array(sel_freq1)
    Xi = np.array(sel_xi)

    return Fn, Xi, Phi



def pLSCF_MPE_New(sel_freq, Sy, Fn_pol, Sm_pol, Ls_pol, order, dt, DF=1):
    """
    Bla bla bla
    """
    sel_freq1 = []
    sel_xi = []
    sel_lam = []
    for fj in sel_freq:
        if order == "find_min":
            # here we find the minimum model order so to get a stable pole for every mode of interest
            pass
        else:  # when the model order is provided
            # Find closest frequency index

            sel = np.nanargmin(np.abs(Fn_pol[:, order] - fj))

            sel_freq1.append(Fn_pol[:, order][sel])
            sel_xi.append(Sm_pol[:, order][sel])
            sel_lam.append(Ls_pol[:, order][sel])

    Nch, Nref, Nf = Sy.shape
    w_sel = np.array(sel_freq) * (2 * np.pi)
    Nm = len(sel_lam)  # numero modi
    Phi = np.zeros((Nch, Nm), dtype=complex)
    freq = np.arange(0, Nf) * (1 / dt / (2 * Nf))
    freq_rad = 2 * np.pi * freq
    LL = np.zeros((Nch * Nm, Nch * Nm), dtype=complex)  # inizializzo
    GL = np.zeros((Nch * Nm, Nch), dtype=complex)  # inizializzo
    Sy = np.moveaxis(Sy, 2, 0)
    sel_lam_conj = np.conj(sel_lam)
    sel_lam1 = np.empty((len(sel_lam)*2), dtype=complex)
    sel_lam1[0::2] = sel_lam
    sel_lam1[1::2] = sel_lam_conj
    
    for kk in range(Nf):
        GL += np.array(
            [Sy[kk, :, :] / (1j * freq_rad[kk] - sel_lam1[jj]) for jj in range(2*Nm)]
        ).reshape(-1, Nch)

        LL += np.array(
            [
                np.array(
                    [
                        np.eye(Nch)
                        / (
                            (1j * freq_rad[kk] - sel_lam1[jj1])
                            * (1j * freq_rad[kk] - sel_lam1[jj2])
                        )
                        for jj2 in range(2*Nm)
                    ]
                )
                .reshape((Nch * Nm, Nch), order="c")
                .T
                for jj1 in range(2*Nm)
            ]
        ).reshape((Nch * 2*Nm, Nch * 2*Nm))

    R = np.linalg.solve(LL, GL)  # matrice dei residui (fi@fi^T

    for jj in range(len(w_sel)):
        # SVD della matrice dei residui per ciascun modo fisico del sistema
        U, S, VT = np.linalg.svd(R[jj * Nch : (jj + 1) * Nch, :])

        phi = U[:, 0]  # la forma modale è la prima colonna di U

        idmax = np.argmax(abs(phi))
        phiN = phi / phi[idmax]  # normalised (unity displacement)

        Phi[:, jj] = phiN
    # Save results
    Fn = np.array(sel_freq1)
    Xi = np.array(sel_xi)

    return Fn, Xi, Phi

