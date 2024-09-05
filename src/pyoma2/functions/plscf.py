"""
poly-reference Least Square Complex Frequency (pLSCF) Utility Functions module.
Part of the pyOMA2 package.
Authors:
Dag Pasca
"""

import itertools
import logging
import typing

import numpy as np

# import matplotlib.pyplot as plt
from tqdm import tqdm, trange

np.seterr(divide="ignore", invalid="ignore")
logger = logging.getLogger(__name__)

# =============================================================================
# FUNZIONI PolyMAX
# =============================================================================


def pLSCF(
    Sy: np.ndarray, dt: float, ordmax: int, sgn_basf: int = -1.0
) -> typing.Tuple[typing.List[np.ndarray], typing.List[np.ndarray]]:
    """
    Perform the poly-reference Least Square Complex Frequency (pLSCF) algorithm.

    Parameters
    ----------
    Sy : numpy.ndarray
        Spectral density matrix of the system.
    dt : float
        Time step of the measurement data.
    ordmax : int
        Maximum model order for the algorithm.
    sgn_basf : int, optional
        Sign of the basis function, -1 for 'LO' and 1 for 'HI', by default -1.

    Returns
    -------
    tuple of list numpy.ndarray
        - Ad : The denominator polynomial coefficients of the transfer function.
        - Bn : The numerator polynomial coefficients of the transfer function.
    """
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


def pLSCF_poles(
    Ad: np.ndarray, Bn: np.ndarray, dt: float, methodSy: str, nxseg: int
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract poles from the pLSCF algorithm results.

    Parameters
    ----------
    Ad : numpy.ndarray
        Denominator polynomial coefficients from pLSCF.
    Bn : numpy.ndarray
        Numerator polynomial coefficients from pLSCF.
    dt : float
        Time step of the measurement data.
    methodSy : str
        Method used for the PSD estimation (either "per" or "cor")
    nxseg : int
        Number of segments used in the algorithm.

    Returns
    -------
    tuple of numpy.ndarray
        - Fns : Natural frequencies from the pLSCF analysis.
        - Xis : Damping ratios from the pLSCF analysis.
        - Phis : Mode shapes from the pLSCF analysis.
        - Lambdas : Complex poles from the pLSCF analysis.
    """
    Fns = []
    Xis = []
    Phis = []
    Lambds = []
    for ii in range(len(Ad)):
        A_den = Ad[ii]
        B_num = Bn[ii]
        A, C = rmfd2ac(A_den, B_num)

        fn, xi, phi, lam_c = ac2mp_poly(A, C, dt, methodSy, nxseg)
        fn[fn == np.inf] = np.nan
        Fns.append(fn)
        Xis.append(xi)
        Phis.append(phi)
        Lambds.append(lam_c)
    # Transform each array in list
    Fns = [list(c) for c in Fns]
    Xis = [list(c) for c in Xis]
    Phi1 = [list(c) for c in Phis]
    Lambds = [list(c) for c in Lambds]

    # Fill with nan values so to get same size and then convert back to array
    Fns = np.array(list(itertools.zip_longest(*Fns, fillvalue=np.nan)))
    Xis = np.array(list(itertools.zip_longest(*Xis, fillvalue=np.nan)))
    Lambds = np.array(list(itertools.zip_longest(*Lambds, fillvalue=np.nan)))
    Phi1 = []
    for phi in Phis:
        phi1 = np.full((len(Phis[-1]), phi.shape[1]), np.nan).astype(complex)
        phi1[: len(phi)] = phi
        Phi1.append(phi1)

    Phi1 = np.array(Phi1)
    Phi1 = np.moveaxis(Phi1, 1, 0)
    return Fns, Xis, Phi1, Lambds


def rmfd2ac(A_den: np.ndarray, B_num: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Convert Right Matrix Fraction Description (RMFD) to state-space representation.

    Parameters
    ----------
    A_den : numpy.ndarray
        Denominator matrix of the RMFD.
    B_num : numpy.ndarray
        Numerator matrix of the RMFD.

    Returns
    -------
    tuple of numpy.ndarray
        - A : State matrix of the system.
        - C : Output matrix of the system.
    """
    n, l_, m = B_num.shape
    A = np.zeros((n * m, n * m))
    A[m:, :-m] = np.eye((n - 1) * m)
    C = np.zeros((l_, n * m))
    Bn_last = B_num[-1]
    Ad_last = A_den[-1]
    for i, (Adi, Bni) in enumerate(zip(A_den[:-1][::-1], B_num[:-1][::-1])):
        prod = np.linalg.solve(Ad_last, Adi)
        A[:m, i * m : (i + 1) * m] = -prod
        C[:, i * m : (i + 1) * m] = Bni - np.dot(Bn_last, prod)
    return A, C


def ac2mp_poly(
    A: np.ndarray, C: np.ndarray, dt: float, methodSy: str, nxseg: int
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert state-space representation to modal parameters.

    Parameters
    ----------
    A : numpy.ndarray
        State matrix of the system.
    C : numpy.ndarray
        Output matrix of the system.
    dt : float
        Time step or sampling interval.
    methodSy : str
        Method used for PSD estimation.
    nxseg : int
        Number of segments used in the algorithm.

    Returns
    -------
    tuple of numpy.ndarray
        - fn : Natural frequencies in Hz.
        - xi : Damping ratios.
        - phi : Complex mode shapes.
        - lam_c : Complex poles.
    """
    Nch = C.shape[0]
    lam_d, AuVett = np.linalg.eig(A)
    lambd = (np.log(lam_d)) * (1 / dt)
    # replace with nan every value with positive real part
    lam_c = np.where(np.real(lambd) > 0, np.nan, lambd)
    # also for the part fact
    Q = np.array(
        [
            np.where(
                np.real(lambd[ii]) > 0, np.repeat(np.nan, AuVett.shape[1]), AuVett[:, ii]
            )
            for ii in range(len(lam_c))
        ]
    ).T
    # correct for exponential window
    if methodSy == "cor":
        tau = -(nxseg - 1) / np.log(0.01)
        lam_c = lam_c - 1 / tau
    fn = abs(lam_c) / (2 * np.pi)  # natural frequencies
    xi = -((np.real(lam_c)) / (abs(lam_c)))  # damping ratios
    # Complex mode shapes
    phi = np.dot(C, Q)
    # normalised (unity displacement)
    phi = np.array(
        [phi[:, ii] / phi[np.argmax(abs(phi[:, ii])), ii] for ii in range(phi.shape[1])]
    ).reshape(-1, Nch)
    return fn, xi, phi, lam_c


# -----------------------------------------------------------------------------


def pLSCF_mpe(
    sel_freq: typing.List[float],
    Fn_pol: np.ndarray,
    Xi_pol: np.ndarray,
    Phi_pol: np.ndarray,
    order: typing.Union[int, typing.List[int], str] = "find_min",
    Lab: np.ndarray = None,
    deltaf: float = 0.05,
    rtol: float = 1e-2,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract modal parameters using the pLSCF method for selected frequencies.

    Parameters
    ----------
    sel_freq : list of float
        Selected frequencies for modal parameter extraction.
    Fn_pol : numpy.ndarray
        Natural frequencies obtained from the pLSCF method.
    Xi_pol : numpy.ndarray
        Damping ratios obtained from the pLSCF method.
    Phi_pol : numpy.ndarray
        Mode shapes obtained from the pLSCF method.
    order : int, list of int, or 'find_min'
        Model order for extraction.
    Lab : numpy.ndarray, optional
        Labels identifying stable poles.
    deltaf : float, optional
        Frequency bandwidth for searching poles, by default 0.05.
    rtol : float, optional
        Relative tolerance for frequency comparison, by default 1e-2.

    Returns
    -------
    tuple of numpy.ndarray
        - Fn : Extracted natural frequencies.
        - Xi : Extracted damping ratios.
        - Phi : Extracted mode shapes.
        - order_out : Model order used for extraction.
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
        elif isinstance(order, int):
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
        elif isinstance(order, list):
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
