"""
Created on Sat Oct 21 18:51:51 2023

@author: dagpa
"""
import logging

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from tqdm import tqdm, trange

from . import Gen_funct as GF

logger = logging.getLogger(__name__)

# =============================================================================
# FUNZIONI FDD
# =============================================================================


def SD_PreGER(Y, fs, nxseg=1024, pov=0.5, method="per"):
    dt = 1 / fs
    n_setup = len(Y)  # number of setup
    n_ref = Y[0]["ref"].shape[0]  # number of reference sensor
    # n_mov = [Y[i]["mov"].shape[0] for i in range(n_setup)] # number of moving sensor
    # n_DOF = n_ref+np.sum(n_mov) # total number of sensors
    Gyy = []
    for ii in trange(n_setup):
        logger.debug("Analyising setup nr.: %s...", ii)

        Y_ref = Y[ii]["ref"]
        Y_mov = Y[ii]["mov"]
        # Ndat =  Y[ii]["ref"].shape[1] # number of data points
        Y_all = np.vstack((Y[ii]["ref"], Y[ii]["mov"]))
        # r = Y_all.shape[0] # total sensor for the ii setup

        if method == "per":
            # noverlap = nxseg*pov
            freq, Sy_allref = SD_Est(Y_all, Y_ref, dt, nxseg, method)
            _, Sy_allmov = SD_Est(Y_all, Y_mov, dt, nxseg, method)
            Gyy.append(np.hstack((Sy_allref, Sy_allmov)))

        elif method == "cor":
            freq, Sy_allref = SD_Est(Y_all, Y_ref, dt, nxseg, method)
            _, Sy_allmov = SD_Est(Y_all, Y_mov, dt, nxseg, method)
            Gyy.append(np.hstack((Sy_allref, Sy_allmov)))
        logger.debug("... Done with setup nr.: %s!", ii)

    Gy_refref = (
        1 / n_setup * np.sum([Gyy[ii][:n_ref, :n_ref] for ii in range(n_setup)], axis=0)
    )

    Gg = []
    # Scale spectrum to reference spectrum
    for ff in range(len(freq)):
        G1 = [
            np.dot(
                np.dot(
                    Gyy[ii][n_ref:, :n_ref][:, :, ff],
                    np.linalg.inv(Gyy[ii][:n_ref, :n_ref][:, :, ff]),
                ),
                Gy_refref[:, :, ff],
            )
            for ii in range(n_setup)
        ]
        G2 = np.vstack(G1)
        G3 = np.vstack([Gy_refref[:, :, ff], G2])
        Gg.append(G3)

    Gy = np.array(Gg)
    Gy = np.moveaxis(Gy, 0, 2)
    return freq, Gy


# -----------------------------------------------------------------------------


def SD_Est(
    Yall,
    Yref,
    dt,
    nxseg=1024,
    method="cor",
    pov=0.5,
):
    """
    Estimate the Cross-Spectral Density (CSD) using either the correlogram
        method or the periodogram method.

    Parameters:
        Yall (ndarray): Input signal data.
        Yref (ndarray): Reference signal data.
        dt (float): Sampling interval.
        nxseg (int): Length of each segment for CSD estimation.
        method (str, optional): Method for CSD estimation, either "cor" for
            correlogram method or "per" for periodogram. Default is "cor".
        pov (float, optional): Proportion of overlap for the periodogram
            method. Default is 0.5.

    Returns:
        tuple: A tuple containing the frequency values and the estimated
            Cross-Spectral Density (CSD).
            freq (ndarray): Array of frequencies.
            Sy (ndarray): Cross-Spectral Density (CSD) estimation."""

    if method == "cor":
        Ndat = Yref.shape[1]  # number of data points
        # n_ref =  Yref.shape[0] # number of data points
        # n_all = Yall.shape[0]
        # Calculating Auto e Cross-Spectral Density (Y_all, Y_ref)
        logger.debug("Estimating spectrum...")
        R_i = np.array(
            [
                1 / (Ndat - ii) * np.dot(Yall[:, : Ndat - ii], Yref[:, ii:].T)
                for ii in trange(nxseg)
            ]
        )
        logger.debug("... Done!")

        nxseg, nr, nc = R_i.shape
        # N.B. beta = 1/tau
        tau = -(nxseg - 1) / np.log(0.01)
        W = signal.windows.exponential(nxseg, center=0, tau=tau, sym=False)
        R = np.zeros((nr, nc, nxseg))
        for ii in range(nr):
            for jj in range(nc):
                R[ii, jj, :] = R_i[:, ii, jj] * W

        Sy = np.zeros((nr, nc, nxseg), dtype=complex)
        R11 = np.zeros(nxseg)
        for r in range(nr):
            for c in range(nc):
                R11 = R[r, c, :]
                # for full spectrum, do not multiply by zero
                R1 = np.concatenate((R11, np.flipud(R11[:nxseg]) * 0))
                G = np.fft.fft(R1)
                Sy[r, c, :] = G[:nxseg]
        freq = np.arange(0, nxseg) * (1 / dt / (2 * nxseg))  # Frequency vector

    elif method == "per":
        noverlap = nxseg * pov
        Ndat = Yref.shape[1]  # number of data points
        n_ref = Yref.shape[0]  # number of data points
        n_all = Yall.shape[0]
        # Calculating Auto e Cross-Spectral Density (Y_all, Y_ref)
        freq, Sy = signal.csd(
            Yall.reshape(n_all, 1, Ndat),
            Yref.reshape(1, n_ref, Ndat),
            fs=1 / dt,
            nperseg=nxseg,
            noverlap=noverlap,
            window="hann",
        )
    return freq, Sy


# -----------------------------------------------------------------------------


def SD_svalsvec(SD):
    """
    Compute the singular values and singular vectors for a given set of
        Cross-Spectral Density (CSD) matrices.

    Parameters:
    SD (ndarray): Array of Cross-Spectral Density (CSD) matrices, with
        shape (number_of_rows, number_of_columns, number_of_frequencies).

    Returns:
    tuple: A tuple containing the singular values and the singular vectors.
        S_val (ndarray): Singular values.
        S_vec (ndarray): Singular vectors."""
    nr, nc, nf = SD.shape
    Sval = np.zeros((nf, nc))
    S_val = np.empty((nf, nc, nc))
    S_vec = np.empty((nf, nr, nr), dtype=complex)
    for k in range(nf):
        U1, S, _ = np.linalg.svd(SD[:, :, k])
        U1_1 = U1.conj().T
        Sval[k, :] = np.sqrt(S)
        S_val[k, :, :] = np.diag(np.sqrt(S))
        S_vec[k, :, :] = U1_1
    S_val = np.moveaxis(S_val, 0, 2)
    S_vec = np.moveaxis(S_vec, 0, 2)
    return S_val, S_vec


# -----------------------------------------------------------------------------


def FDD_MPE(
    Sval,
    Svec,
    freq,
    sel_freq,
    DF=0.1,
):
    # Sval, Svec = SD_svalsvec(Sy)
    Nch, Nref, Nf = Sval.shape

    Freq = []
    Fi = []
    index = []
    maxSy_diff = []
    logger.info("Extracting FDD modal parameters")
    for sel_fn in tqdm(sel_freq):
        # Frequency bandwidth where the peak is searched
        lim = (sel_fn - DF, sel_fn + DF)
        idxlim = (
            np.argmin(np.abs(freq - lim[0])),
            np.argmin(np.abs(freq - lim[1])),
        )  # Indices of the limits
        # Ratios between the first and second singular value
        diffS1S2 = Sval[0, 0, idxlim[0] : idxlim[1]] / Sval[1, 1, idxlim[0] : idxlim[1]]
        maxDiffS1S2 = np.max(diffS1S2)  # Looking for the maximum difference
        idx1 = np.argmin(np.abs(diffS1S2 - maxDiffS1S2))  # Index of the max diff
        idxfin = idxlim[0] + idx1  # Final index

        # Modal properties
        fn_FDD = freq[idxfin]  # Frequency
        phi_FDD = Svec[0, :, idxfin]  # Mode shape
        # Normalized (unity displacement)
        phi_FDDn = phi_FDD / phi_FDD[np.argmax(np.abs(phi_FDD))]

        Freq.append(fn_FDD)
        Fi.append(phi_FDDn)
        index.append(idxfin)
        maxSy_diff.append(maxDiffS1S2)
    logger.debug("Done!")

    Fn = np.array(Freq)
    Phi = np.array(Fi).T
    index = np.array(index)
    return Fn, Phi


# -----------------------------------------------------------------------------
# COMMENT
# Utility function (Hidden for users?)
def SDOF_bellandMS(Sy, dt, sel_fn, phi_FDD, method="FSDD", cm=1, MAClim=0.85, DF=1.0):
    Sval, Svec = SD_svalsvec(Sy)
    Nch = phi_FDD.shape[0]
    nxseg = Sval.shape[2]
    freq = np.arange(0, nxseg) * (1 / dt / (2 * nxseg))
    # Frequency bandwidth where the peak is searched
    lim = (sel_fn - DF, sel_fn + DF)
    idxlim = (
        np.argmin(np.abs(freq - lim[0])),
        np.argmin(np.abs(freq - lim[1])),
    )  # Indices of the limits
    # Initialise SDOF bell and Mode Shape
    SDOFbell = np.zeros(len(np.arange(idxlim[0], idxlim[1])), dtype=complex)
    SDOFms = np.zeros((len(np.arange(idxlim[0], idxlim[1])), Nch), dtype=complex)

    for csm in range(cm):  # Loop through close mode (if any, default 1)
        # Frequency Spatial Domain Decomposition variation (defaulf)
        if method == "FSDD":
            # Save values that satisfy MAC > MAClim condition
            SDOFbell += np.array(
                [
                    np.dot(
                        np.dot(phi_FDD.conj().T, Sy[:, :, el]), phi_FDD
                    )  # Enhanced PSD matrix (frequency filtered)
                    if GF.MAC(phi_FDD, Svec[csm, :, el]) > MAClim
                    else 0
                    for el in range(int(idxlim[0]), int(idxlim[1]))
                ]
            )
            # Do the same for mode shapes
            SDOFms += np.array(
                [
                    Svec[csm, :, el]
                    if GF.MAC(phi_FDD, Svec[csm, :, el]) > MAClim
                    else np.zeros(Nch)
                    for el in range(int(idxlim[0]), int(idxlim[1]))
                ]
            )
        elif method == "EFDD":
            SDOFbell += np.array(
                [
                    Sval[csm, csm, l_]
                    if GF.MAC(phi_FDD, Svec[csm, :, l_]) > MAClim
                    else 0
                    for l_ in range(int(idxlim[0]), int(idxlim[1]))
                ]
            )
            SDOFms += np.array(
                [
                    Svec[csm, :, l_]
                    if GF.MAC(phi_FDD, Svec[csm, :, l_]) > MAClim
                    else np.zeros(Nch)
                    for l_ in range(int(idxlim[0]), int(idxlim[1]))
                ]
            )

    SDOFbell1 = np.zeros((nxseg), dtype=complex)
    SDOFms1 = np.zeros((nxseg, Nch), dtype=complex)
    SDOFbell1[idxlim[0] : idxlim[1]] = SDOFbell
    SDOFms1[idxlim[0] : idxlim[1], :] = SDOFms
    return SDOFbell1, SDOFms1


# -----------------------------------------------------------------------------


def EFDD_MPE(
    Sy,
    freq,
    dt,
    sel_freq,
    methodSy,
    method="FSDD",
    DF1=0.1,
    DF2=1.0,
    cm=1,
    MAClim=0.85,
    sppk=3,
    npmax=20,
):
    Sval, Svec = SD_svalsvec(Sy)

    Nch, Nref, nxseg = Sval.shape
    # number of points for the inverse transform (zeropadding)
    nIFFT = (int(nxseg)) * 5
    Freq_FDD, Phi_FDD = FDD_MPE(Sval, Svec, freq, sel_freq, DF=DF1)

    # Initialize Results
    PerPlot = []
    Fn_E = []
    Phi_E = []
    Xi_E = []

    logger.info("Extracting EFDD modal parameters")
    for n in trange(len(sel_freq)):  # looping through all frequencies to estimate
        phi_FDD = Phi_FDD[:, n]  # Select reference mode shape (from FDD)
        sel_fn = sel_freq[n]
        SDOFbell, SDOFms = SDOF_bellandMS(
            Sy, dt, sel_fn, phi_FDD, method=method, cm=cm, MAClim=MAClim, DF=DF2
        )

        # indices of the singular values in SDOFsval
        idSV = np.array(np.where(SDOFbell)).T
        # =============================================================================
        # # Mode shapes (singular vectors) associated to each singular values
        # # and weighted with respect to the singular value itself
        # FIs = [ SDOFbell[idSV[u]] * SDOFms[idSV[u],:]
        #         for u in range(len(idSV)) ]
        # FIs = np.squeeze(np.array(FIs))
        # meanFi = np.mean(FIs,axis=0)
        # # Normalised mode shape (unity disp)
        # meanFi = meanFi/meanFi[np.argmax(abs(meanFi))]
        # =============================================================================
        # Autocorrelation function (Free Decay)
        SDOFcorr1 = np.fft.ifft(SDOFbell, n=nIFFT, axis=0, norm="ortho").real
        df = 1 / dt / nxseg
        tlag = 1 / df  # time lag
        time = np.linspace(0, tlag, len(SDOFcorr1) // 2)  # t

        # NORMALISED AUTOCORRELATION
        normSDOFcorr = SDOFcorr1[: len(SDOFcorr1) // 2] / SDOFcorr1[np.argmax(SDOFcorr1)]

        # finding where x = 0
        sgn = np.sign(normSDOFcorr).real  # finding the sign
        # finding where the sign changes (crossing x)
        sgn1 = np.diff(sgn, axis=0)
        zc1 = np.where(sgn1)[0]  # Zero crossing indices

        # finding maximums and minimums (peaks) of the autoccorelation
        maxSDOFcorr = [
            np.max(normSDOFcorr[zc1[_i] : zc1[_i + 2]])
            for _i in range(0, len(zc1) - 2, 2)
        ]
        minSDOFcorr = [
            np.min(normSDOFcorr[zc1[_i] : zc1[_i + 2]])
            for _i in range(0, len(zc1) - 2, 2)
        ]
        if len(maxSDOFcorr) > len(minSDOFcorr):
            maxSDOFcorr = maxSDOFcorr[:-1]
        elif len(maxSDOFcorr) < len(minSDOFcorr):
            minSDOFcorr = minSDOFcorr[:-1]

        minmax = np.array((minSDOFcorr, maxSDOFcorr))
        minmax = np.ravel(minmax, order="F")

        # finding the indices of the peaks
        maxSDOFcorr_idx = [np.argmin(abs(normSDOFcorr - maxx)) for maxx in maxSDOFcorr]
        minSDOFcorr_idx = [np.argmin(abs(normSDOFcorr - minn)) for minn in minSDOFcorr]
        minmax_idx = np.array((minSDOFcorr_idx, maxSDOFcorr_idx))
        minmax_idx = np.ravel(minmax_idx, order="F")

        # Peacks and indices of the peaks to be used in the fitting
        minmax_fit = np.array([minmax[_a] for _a in range(sppk, sppk + npmax)])
        minmax_fit_idx = np.array([minmax_idx[_a] for _a in range(sppk, sppk + npmax)])

        # estimating the natural frequency from the distance between the peaks
        # *2 because we use both max and min
        Td = np.diff(time[minmax_fit_idx]) * 2
        Td_EFDD = np.mean(Td)

        fd_EFDD = 1 / Td_EFDD  # damped natural frequency

        # Log decrement
        delta = np.array(
            [
                2 * np.log(np.abs(minmax[0]) / np.abs(minmax[ii]))
                for ii in range(len(minmax_fit))
            ]
        )

        # Fit
        def _fit(x, m):
            return m * x

        lam, _ = curve_fit(_fit, np.arange(len(minmax_fit)), delta)

        # damping ratio
        if methodSy == "cor":  # correct for exponential window
            tau = -(nxseg - 1) / np.log(0.01)
            lam = 2 * lam - 1 / tau  # lam*2 because we use both max and min
        elif methodSy == "per":
            lam = 2 * lam  # lam*2 because we use both max and min

        xi_EFDD = lam / np.sqrt(4 * np.pi**2 + lam**2)

        fn_EFDD = fd_EFDD / np.sqrt(1 - xi_EFDD**2)

        # Finally appending the results
        Fn_E.append(fn_EFDD)
        Xi_E.append(xi_EFDD)
        Phi_E.append(phi_FDD)

        PerPlot.append(
            [freq, time, SDOFbell, Sval, idSV, normSDOFcorr, minmax_fit_idx, lam, delta]
        )
    logger.debug("Done!")

    Fn = np.array(Fn_E)
    Xi = np.array(Xi_E)
    Phi = np.array(Phi_E).T

    return Fn, Xi, Phi, PerPlot
