"""
Frequency Domain Decomposition (FDD) Utility Functions module.
Part of the pyOMA2 package.
Author:
Dag Pasca
"""

import logging
import typing

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from tqdm import tqdm, trange

from .gen import MAC

logger = logging.getLogger(__name__)

# =============================================================================
# FUNZIONI FDD
# =============================================================================


def SD_PreGER(
    Y: typing.List[typing.Dict[str, np.ndarray]],
    fs: float,
    nxseg: int = 1024,
    pov: float = 0.5,
    method: typing.Literal["per", "cor"] = "per",
):
    """
    Estimate the PSD matrix for a multi-setup experiment using either the correlogram
    method or the periodogram method.

    Parameters
    ----------
    Y : list of dicts
        A list where each element corresponds to a different setup. Each element is a
        dictionary with keys 'ref' and 'mov' for reference and moving sensor data,
        respectively. Each should be a numpy array with dimensions [N x M], where N is the
        number of sensors and M is the number of data points.
    fs : float
        Sampling frequency of the data.
    nxseg : int, optional
        Number of data points in each segment for spectral analysis. Default is 1024.
    pov : float, optional
        Proportion of overlap between segments in spectral analysis. Default is 0.5.
    method : str, optional
        Method for spectral density estimation. 'per' for periodogram and 'cor' for
        correlogram method. Default is 'per'.

    Returns
    -------
    tuple
        freq : ndarray
            Array of frequency values at which the spectral densities are evaluated.
        Sy : ndarray
            The scaled spectral density matrices. The shape of the array is [N x N x K],
            where N is the total number of sensors (reference + moving) and K is the number
            of frequency points.

    Note
    -----
    The function uses an internal function 'SD_Est' to estimate the spectral densities.
    The logger is used for debugging purposes to track the progress of analysis.
    """
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
            freq, Sy_allref = SD_est(Y_all, Y_ref, dt, nxseg, method)
            _, Sy_allmov = SD_est(Y_all, Y_mov, dt, nxseg, method)
            Gyy.append(np.hstack((Sy_allref, Sy_allmov)))

        elif method == "cor":
            freq, Sy_allref = SD_est(Y_all, Y_ref, dt, nxseg, method)
            _, Sy_allmov = SD_est(Y_all, Y_mov, dt, nxseg, method)
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

    Sy = np.array(Gg)
    Sy = np.moveaxis(Sy, 0, 2)
    return freq, Sy


# -----------------------------------------------------------------------------


def SD_est(
    Yall,
    Yref,
    dt,
    nxseg=1024,
    method="cor",
    pov=0.5,
):
    """
    Estimate the Cross-Spectral Density (CSD) using either the correlogram method or the
    periodogram method.

    Parameters
    ----------
    Yall : ndarray
        Input signal data.
    Yref : ndarray
        Reference signal data.
    dt : float
        Sampling interval.
    nxseg : int, optional
        Length of each segment for CSD estimation. Default is 1024.
    method : str, optional
        Method for CSD estimation, either "cor" for correlogram method or "per" for
        periodogram. Default is "cor".
    pov : float, optional
        Proportion of overlap for the periodogram method. Default is 0.5.

    Returns
    -------
    tuple
        freq : ndarray
            Array of frequencies.
        Sy : ndarray
            Cross-Spectral Density (CSD) estimation.
    """
    if method == "cor":
        Ndat = Yref.shape[1]  # number of data points
        n_ref = Yref.shape[0]
        n_all = Yall.shape[0]
        # Calculating Auto e Cross-Spectral Density (Y_all, Y_ref)
        _, Pxy = signal.csd(
            Yall.reshape(n_all, 1, Ndat),
            Yref.reshape(1, n_ref, Ndat),
            nperseg=nxseg // 2,
            nfft=nxseg,
            noverlap=0,
            window="boxcar",
        )
        Rxy = np.fft.irfft(Pxy)

        tau = -Rxy.shape[2] / np.log(0.01)
        win = signal.windows.exponential(Rxy.shape[2], center=0, tau=tau, sym=False)
        Rxy *= win
        Sy = np.fft.rfft(Rxy)
        freq = np.arange(0, Sy.shape[2]) * (1 / dt / (nxseg))  # Frequency vector

    elif method == "per":
        noverlap = nxseg * pov
        Ndat = Yref.shape[1]
        n_ref = Yref.shape[0]
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
    Compute the singular values and singular vectors for a given set of Cross-Spectral
    Density (CSD) matrices.

    Parameters
    ----------
    SD : ndarray
        Array of Cross-Spectral Density (CSD) matrices, with shape
        (number_of_rows, number_of_columns, number_of_frequencies).

    Returns
    -------
    tuple
        S_val : ndarray
            Singular values.
        S_vec : ndarray
            Singular vectors.
    """
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


def FDD_mpe(
    Sval,
    Svec,
    freq,
    sel_freq,
    DF=0.1,
):
    """
    Extracts modal parameters using the Frequency Domain Decomposition (FDD) method.

    Parameters
    ----------
    Sval : ndarray
        A 3D array of singular values. Dimensions are [Nch, Nref, Nf], where Nch is the
        number of channels, Nref is the number of reference channels, and Nf is the
        number of frequency points.
    Svec : ndarray
        A 3D array of singular vectors corresponding to Sval. Dimensions are the same as Sval.
    freq : ndarray
        1D array of frequency values corresponding to the singular values and vectors.
    sel_freq : list or ndarray
        Selected frequencies around which modal parameters are to be extracted.
    DF : float, optional
        Frequency bandwidth around each selected frequency within which the function
        searches for a peak. Default is 0.1.

    Returns
    -------
    tuple
        Fn : ndarray
            Extracted modal frequencies.
        Phi : ndarray
            Corresponding normalized mode shapes (each column corresponds to a mode shape).

    Note
    -----
    The function assumes that the first singular value and vector correspond to the dominant
    mode at each frequency point.
    """
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
    """
    Computes the SDOF bell and mode shapes for a specified frequency range using FSDD or
    EFDD methods.

    Parameters
    ----------
    Sy : ndarray
        Spectral matrix of the system. Expected dimensions are [Nch, Nch, Nf], where Nch is
        the number of channels and Nf is the number of frequency points.
    dt : float
        Time interval of the data sampling.
    sel_fn : float
        Selected modal frequency around which the SDOF analysis is to be performed.
    phi_FDD : ndarray
        Mode shape corresponding to the selected modal frequency.
    method : str, optional
        Method for SDOF analysis. Supports 'FSDD' for Frequency Spatial Domain Decomposition
        and 'EFDD' for Enhanced Frequency Domain Decomposition. Default is 'FSDD'.
    cm : int, optional
        Number of close modes to consider in the analysis. Default is 1.
    MAClim : float, optional
        Threshold for the Modal Assurance Criterion (MAC) to filter modes. Default is 0.85.
    DF : float, optional
        Frequency bandwidth around the selected frequency for analysis. Default is 1.0.

    Returns
    -------
    tuple
        SDOFbell1 : ndarray
            The SDOF bell (power spectral density) of the selected mode.
        SDOFms1 : ndarray
            The mode shapes corresponding to the SDOF bell.
    """

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
                    if MAC(phi_FDD, Svec[csm, :, el]) > MAClim
                    else 0
                    for el in range(int(idxlim[0]), int(idxlim[1]))
                ]
            )
            # Do the same for mode shapes
            SDOFms += np.array(
                [
                    Svec[csm, :, el]
                    if MAC(phi_FDD, Svec[csm, :, el]) > MAClim
                    else np.zeros(Nch)
                    for el in range(int(idxlim[0]), int(idxlim[1]))
                ]
            )
        elif method == "EFDD":
            SDOFbell += np.array(
                [
                    Sval[csm, csm, l_] if MAC(phi_FDD, Svec[csm, :, l_]) > MAClim else 0
                    for l_ in range(int(idxlim[0]), int(idxlim[1]))
                ]
            )
            SDOFms += np.array(
                [
                    Svec[csm, :, l_]
                    if MAC(phi_FDD, Svec[csm, :, l_]) > MAClim
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


def EFDD_mpe(
    Sy: np.ndarray,
    freq: np.ndarray,
    dt: float,
    sel_freq: typing.Sequence[float],
    methodSy: str,
    method: str = "FSDD",
    DF1: float = 0.1,
    DF2: float = 1.0,
    cm: int = 1,
    MAClim: float = 0.85,
    sppk: int = 3,
    npmax: int = 20,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, typing.List[typing.Any]]:
    """
    Extracts modal parameters using the Enhanced Frequency Domain Decomposition (EFDD)
    and Frequency Spatial Domain Decomposition (FSDD) algorithms.

    Parameters
    ----------
    Sy : ndarray
        Spectral matrix with dimensions [Nch, Nch, Nf] where Nch is the number of channels
        and Nf is the number of frequency points.
    freq : ndarray
        Array of frequency values corresponding to the spectral matrix.
    dt : float
        Sampling interval of the data.
    sel_freq : sequence of float
        Selected modal frequencies around which parameters are to be estimated.
    methodSy : str
        Method used for spectral density estimation:
        - 'cor' for correlation
        - 'per' for periodogram
    method : str, optional
        Specifies the SDOF analysis method ('FSDD' or 'EFDD'). Default is 'FSDD'.
    DF1 : float, optional
        Frequency bandwidth for initial FDD modal parameter extraction. Default is 0.1.
    DF2 : float, optional
        Frequency bandwidth for SDOF analysis. Default is 1.0.
    cm : int, optional
        Number of close modes to consider. Default is 1.
    MAClim : float, optional
        Threshold for the Modal Assurance Criterion (MAC) to filter modes. Default is 0.85.
    sppk : int, optional
        Number of initial peaks to skip in autocorrelation analysis. Default is 3.
    npmax : int, optional
        Maximum number of peaks to consider in the curve fitting for damping ratio
        estimation. Default is 20.

    Returns
    -------
    Fn : ndarray
        Estimated natural frequencies [Hz].
    Xi : ndarray
        Estimated damping ratios.
    Phi : ndarray
        Mode shapes array of shape [Nch, Nm], where Nm = len(sel_freq).
    PerPlot : list
        List of per-mode metadata for plotting and analysis. Each element is a list:
        [freq, time, SDOF bell, Sval, idSV, normalized autocorr,
         peak indices, fitted lambda, log-decrement values].
    """
    logger = logging.getLogger(__name__)

    # 1) Spectral decomposition
    Sval, Svec = SD_svalsvec(Sy)
    Nch, Nref, nxseg = Sval.shape

    # 2) Prepare zero-padding and time axis
    pad_factor = 5
    n_fft = nxseg * pad_factor
    df = 1.0 / (dt * nxseg)
    t_record = 1.0 / df

    # 3) Initial FDD extraction
    Freq_FDD, Phi_FDD = FDD_mpe(Sval, Svec, freq, sel_freq, DF=DF1)

    # 4) Containers for results
    PerPlot: typing.List[typing.Any] = []
    Fn_list, Xi_list, Phi_list = [], [], []

    # logger.info("Starting EFDD modal extraction for %d modes", len(sel_freq))
    for idx, sel_fn in enumerate(sel_freq, start=1):
        # logger.info("Mode %d/%d: sel_fn=%.4f Hz", idx, len(sel_freq), sel_fn)
        phi_ref = Phi_FDD[:, idx - 1]

        # Single-DOF bell and mode shape
        SDOFbell, _ = SDOF_bellandMS(
            Sy, dt, sel_fn, phi_ref, method=method, cm=cm, MAClim=MAClim, DF=DF2
        )
        idSV = np.array(np.where(SDOFbell)).T

        # 5) Autocorrelation
        sdof_corr = np.fft.ifft(SDOFbell, n=n_fft, norm="ortho").real
        half = sdof_corr.size // 2
        time = np.linspace(0, t_record, half)
        norm_corr = sdof_corr[:half] / sdof_corr[0]

        # 6) Peak detection
        peaks_pos, _ = find_peaks(norm_corr)
        peaks_neg, _ = find_peaks(-norm_corr)
        all_peaks = np.sort(np.concatenate([peaks_pos, peaks_neg]))

        # 7) Bounds check
        required = sppk + npmax
        if all_peaks.size < required:
            raise ValueError(
                f"Mode {idx}/{len(sel_freq)}: sel_fn={sel_fn:.4f} Hz \n"
                f"Only {all_peaks.size} peaks found; Reduce npmax:{npmax}."
            )

        # 8) Select peaks and values
        peak_idx = all_peaks[sppk:required]
        peak_vals = norm_corr[peak_idx]

        # 9) Damped frequency estimation
        Td = 2.0 * np.diff(time[peak_idx])
        fd = 1.0 / np.mean(Td)

        # 10) Log-decrement & curve fit
        delta = np.log(np.abs(peak_vals[0]) / np.abs(peak_vals))
        lam, _ = curve_fit(lambda x, m: m * x, np.arange(delta.size), delta)

        # 11) Correct lambda based on spectral method
        if methodSy == "cor":
            tau = -(nxseg - 1) / np.log(0.01)
            lam = 2 * lam - 1 / tau
        else:
            lam = 2 * lam

        # 12) Compute damping ratio and natural frequency
        xi = lam / np.sqrt(4 * np.pi**2 + lam**2)
        fn = fd / np.sqrt(1 - xi**2)

        # 13) Store results
        Fn_list.append(fn)
        Xi_list.append(xi)
        Phi_list.append(phi_ref)
        PerPlot.append(
            [freq, time, SDOFbell, Sval, idSV, norm_corr, peak_idx, lam, delta]
        )

    # 14) Final assembly
    Fn = np.array(Fn_list)
    Xi = np.array(Xi_list)
    Phi = np.array(Phi_list).T

    logger.info("EFDD extraction complete.")
    return Fn, Xi, Phi, PerPlot
