"""
poly-reference Least Square Complex Frequency (pLSCF) Utility Functions module.
Part of the pyOMA2 package.
Authors:
Dag Pasca
"""
import itertools
import logging

import numpy as np

# import matplotlib.pyplot as plt
from tqdm import tqdm, trange

np.seterr(divide="ignore", invalid="ignore")
logger = logging.getLogger(__name__)

# =============================================================================
# FUNZIONI PolyMAX
# =============================================================================


def pLSCF(
    Sy,
    dt,
    ordmax,
    sgn_basf=-1,
):
    if sgn_basf == -1:
        constr = "LO"
    if sgn_basf == 1:
        constr = "HI"

    Nch = Sy.shape[1]
    Nref = Sy.shape[0]
    Nf = Sy.shape[2]

    fs = 1 / dt
    freq = np.linspace(0.0, fs / 2, Nf)
    omega = 2 * np.pi * freq
    Omega = np.exp(sgn_basf * 1j * omega * dt)

    Ad = []
    Bn = []
    for n in trange(1, ordmax + 1):
        M = np.zeros(((n + 1) * Nch, (n + 1) * Nch))  # iNchzializzo
        Xo = np.array([Omega**i for i in range(n + 1)]).T
        Xoh = Xo.conj().T
        Ro = np.real(np.dot(Xoh, Xo))  # 4.163
        So_s = []
        for o in range(0, Nref):  # loop on channels
            Syo = Sy[o, :, :]
            Yo = np.array([-np.kron(xo, Hoi) for xo, Hoi in zip(Xo, Syo.T)])
            So = np.real(np.dot(Xoh, Yo))  # 4.164
            To = np.real(np.dot(Yo.conj().T, Yo))  # 4.165
            # M += To - np.dot(np.dot(So.T, np.linalg.inv(Ro)), So)
            M += To - np.dot(So.T.conj(), np.linalg.solve(Ro, So))
            So_s.append(So)

        if constr == "LO":
            alpha = np.r_[
                np.eye(Nch),
                np.linalg.solve(
                    -M[Nch : (n + 1) * Nch, Nch : (n + 1) * Nch],
                    M[Nch : (n + 1) * Nch, 0:Nch],
                ),
            ]
        elif constr == "HI":
            alpha = np.r_[
                np.linalg.solve(
                    -M[0 : n * Nch, 0 : n * Nch], M[0 : n * Nch, n * Nch : (n + 1) * Nch]
                ),
                np.eye(Nch),
            ]

        A_den = alpha.reshape((-1, Nch, Nch))
        beta = np.array(
            [np.linalg.solve(-Ro, np.dot(So_s[o], alpha)) for o in range(Nref)]
        )
        B_num = np.moveaxis(beta, 1, 0)
        Ad.append(A_den)
        Bn.append(B_num)

    return Ad, Bn


def pLSCF_Poles(Ad, Bn, dt, methodSy, nxseg):
    Fns = []
    Xis = []
    Phis = []
    for ii in range(len(Ad)):
        A_den = Ad[ii]
        B_num = Bn[ii]

        A, C = rmfd2AC(A_den, B_num)

        fn, xi, phi = AC2MP_poly(A, C, dt, methodSy, nxseg)
        fn[fn == np.inf] = np.nan
        Fns.append(fn)
        Xis.append(xi)
        Phis.append(phi)
    # Transform each array in list
    Fns = [list(c) for c in Fns]
    Xis = [list(c) for c in Xis]
    Phi1 = [list(c) for c in Phis]

    # Fill with nan values so to get same size and then convert back to array
    Fns = np.array(list(itertools.zip_longest(*Fns, fillvalue=np.nan)))
    Xis = np.array(list(itertools.zip_longest(*Xis, fillvalue=np.nan)))
    Phi1 = []
    for phi in Phis:
        phi1 = np.full((len(Phis[-1]), phi.shape[1]), np.nan).astype(complex)
        phi1[: len(phi)] = phi
        Phi1.append(phi1)

    Phi1 = np.array(Phi1)
    Phi1 = np.moveaxis(Phi1, 1, 0)
    return Fns, Xis, Phi1


def rmfd2AC(A_den, B_num):
    """


    Parameters
    ----------
    A_den : TYPE
        DESCRIPTION.
    B_num : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    n, l, m = B_num.shape
    A = np.zeros((n * m, n * m))
    A[m:, :-m] = np.eye((n - 1) * m)
    C = np.zeros((l, n * m))
    Bn_last = B_num[-1]
    Ad_last = A_den[-1]
    for i, (Adi, Bni) in enumerate(zip(A_den[:-1][::-1], B_num[:-1][::-1])):
        prod = np.linalg.solve(Ad_last, Adi)
        A[:m, i * m : (i + 1) * m] = -prod
        C[:, i * m : (i + 1) * m] = Bni - np.dot(Bn_last, prod)
    return A, C


def AC2MP_poly(A, C, dt, methodSy, nxseg):
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
    lambd1 = (np.log(AuVal)) * (1 / dt)
    # replace with nan every value with positive real part
    lambd = np.where(np.real(lambd1) > 0, np.nan, lambd1)
    # also for the part fact
    Q = np.array(
        [
            np.where(
                np.real(lambd1[ii]) > 0, np.repeat(np.nan, AuVett.shape[1]), AuVett[:, ii]
            )
            for ii in range(len(lambd))
        ]
    ).T
    # correct for exponential window
    if methodSy == "cor":
        tau = -(nxseg - 1) / np.log(0.01)
        lambd = lambd - 1 / tau
    fn = abs(lambd) / (2 * np.pi)  # natural frequencies
    xi = -((np.real(lambd)) / (abs(lambd)))  # damping ratios
    # Complex mode shapes
    phi = np.dot(C, Q)
    # normalised (unity displacement)
    phi = np.array(
        [phi[:, ii] / phi[np.argmax(abs(phi[:, ii])), ii] for ii in range(phi.shape[1])]
    ).reshape(-1, Nch)
    return fn, xi, phi


# -----------------------------------------------------------------------------


def pLSCF_MPE(sel_freq, Fn_pol, Xi_pol, Phi_pol, order, Lab=None, deltaf=0.05, rtol=1e-2):
    """
    Extract modal parameters using XXX method for selected frequencies.

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
    Phi_pol = np.moveaxis(Phi_pol, 1, 0)
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
            while check.any() == False:  # noqa: E712
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
                    sel_xi.append(Xi_pol[r_ind, ii])
                    sel_phi.append(Phi_pol[r_ind, ii, :])
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
                sel_xi.append(Xi_pol[:, order][sel])
                sel_phi.append(Phi_pol[:, order][sel, :])
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
                sel_xi.append(Xi_pol[:, order[ii]][sel])
                sel_phi.append(Phi_pol[:, order[ii]][sel, :])
                order_out[ii] = order[ii]
        else:
            raise ValueError('order must be either of type(int) or "find_min"')
    logger.debug("Done!")

    Fn = np.array(sel_freq1)
    Phi = np.array(sel_phi).T
    Xi = np.array(sel_xi)
    return Fn, Xi, Phi, order_out
