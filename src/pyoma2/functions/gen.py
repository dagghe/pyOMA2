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
import pandas as pd
from scipy import linalg, signal

logger = logging.getLogger(__name__)


# =============================================================================
# FUNZIONI GENERALI
# =============================================================================
def dfphi_map_func(phi, sens_names, sens_map, cstrn=None):
    # create mode shape dataframe
    df_phi = pd.DataFrame(
        {"sName": sens_names, "Phi": phi},
    )

    # APPLY POINTS TO SENSOR MAPPING
    # check for costraints
    if cstrn is not None:
        cstr = cstrn.to_numpy(na_value=0)[:, :]
        val = cstr @ phi
        ctn_df = pd.DataFrame(
            {"cName": cstrn.index, "val": val},
        )
        # apply sensor mapping
        mapping_sens = dict(zip(df_phi["sName"], df_phi["Phi"]))
        # apply costraints mapping
        mapping_cstrn = dict(zip(ctn_df["cName"], ctn_df["val"]))
        mapping = dict(mapping_sens, **mapping_cstrn)
    # else apply only sensor mapping
    else:
        mapping = dict(zip(df_phi["sName"], df_phi["Phi"]))

    # mode shape mapped to points
    df_phi_map = sens_map.replace(mapping).astype(float)
    return df_phi_map


def import_excel_GEO1(path, ref_ind=None):
    file = pd.read_excel(path, sheet_name=None, engine="openpyxl", index_col=0)
    # TODO
    # dati di ["sns_lines", "BG_nodes", "BG_lines", "BG_Surf"]
    # come stringhe per mappare con label dei dataframe

    # Test1
    required_sheets = ["sns_names", "sns_crd", "sns_dir"]
    # Ensure required sheets exist
    if not all(sheet in file for sheet in required_sheets):
        raise Warning("At least [sns_names, sns_crd, sns_dir] must be defined!")

    # TODO
    # TEST SU TIPO DI DATI (str o float )
    # TEST SU DIMENSIONI

    # Process each sheet
    for sheet, df in file.items():
        if df.empty:
            file[sheet] = None

        if sheet == "sns_names" and df.shape[0] > 1:
            sens_names = [
                [item for item in row if not pd.isna(item)] for row in df.values.tolist()
            ]
            if ref_ind is None:
                raise AttributeError(
                    "You need to specify the reference indices for a Multi-setup test"
                )
            k = len(ref_ind[0])  # number of reference sensors (from the first setup)

            # Create reference strings and flatten the list
            sens_name_fl = [f"REF{i+1}" for i in range(k)]
            sens_name_fl += [
                item
                for i, row in enumerate(sens_names)
                for j, item in enumerate(row)
                if j not in ref_ind[i]
            ]
            file[sheet] = sens_name_fl
        elif sheet == "sns_names":  # and df.shape[0] == 1:
            file[sheet] = df.values.tolist()[0]

        if (
            sheet in ["sns_lines", "BG_nodes", "BG_lines", "BG_Surf"]
            and file[sheet] is not None
        ):
            file[sheet] = file[sheet].to_numpy()

    # Adjust to 0 indexed lines
    if file["BG_lines"] is not None:
        file["BG_lines"] = np.subtract(file["BG_lines"], 1)
    if file["sns_lines"] is not None:
        file["sns_lines"] = np.subtract(file["sns_lines"], 1)
    if file["BG_Surf"] is not None:
        file["BG_Surf"] = np.subtract(file["BG_Surf"], 1)

    # ---------------------------------------------------------------------
    sens_names = (file["sns_names"],)
    sens_coord = (file["sns_crd"],)
    sens_dir = (file["sns_dir"],)
    sens_lines = (file["sns_lines"],)
    bg_nodes = (file["BG_nodes"],)
    bg_lines = (file["BG_lines"],)
    bg_surf = (file["BG_Surf"],)
    # ---------------------------------------------------------------------
    # # Find the indices that rearrange sens_coord to sens_names
    index = np.array(sens_coord[0].index)
    newIDX = find_map(sens_names[0], index)
    # # reorder if necessary
    sens_coord_new = pd.DataFrame(
        sens_coord[0].values[newIDX, :],
        index=sens_names[0],
        columns=sens_coord[0].columns,
    )
    # sens_dir = sens_dir[newIDX, :]
    sens_dir_new = pd.DataFrame(
        sens_dir[0].values[newIDX, :], index=sens_names[0], columns=sens_coord[0].columns
    )

    return (
        sens_names[0],
        sens_coord_new,
        sens_dir_new,
        sens_lines[0],
        bg_nodes[0],
        bg_lines[0],
        bg_surf[0],
    )


def import_excel_GEO2(path, ref_ind=None):
    file = pd.read_excel(path, sheet_name=None, engine="openpyxl", index_col=0)
    # Test1
    required_sheets = ["sns_names", "pts_crd", "snsTOpts_map"]
    # Ensure required sheets exist
    if not all(sheet in file for sheet in required_sheets):
        raise Warning(
            "At least ['sns_names', 'pts_crd', 'snsTOpts_map'] must be defined!"
        )

    # TODO
    # TEST SU TIPO DI DATI (str o float )
    # TEST SU DIMENSIONI

    # Process each sheet
    for sheet, df in file.items():
        if sheet == "sns_sign" and df.empty:
            file["sns_sign"] = pd.DataFrame(
                np.ones(file["snsTOpts_map"].to_numpy().shape)
            )
        elif sheet != "sns_sign" and df.empty:
            file[sheet] = None

        if sheet == "sns_names" and df.shape[0] > 1:
            sens_names = [
                [item for item in row if not pd.isna(item)] for row in df.values.tolist()
            ]
            if ref_ind is None:
                raise AttributeError(
                    "You need to specify the reference indices for a Multi-setup test"
                )
            k = len(ref_ind[0])  # number of reference sensors (from the first setup)

            # Create reference strings and flatten the list
            sens_name_fl = [f"REF{i+1}" for i in range(k)]
            sens_name_fl += [
                item
                for i, row in enumerate(sens_names)
                for j, item in enumerate(row)
                if j not in ref_ind[i]
            ]
            file[sheet] = sens_name_fl
        elif sheet == "sns_names":  # and df.shape[0] == 1:
            file[sheet] = df.values.tolist()[0]

        if (
            sheet in ["sns_lines", "sns_surf", "BG_nodes", "BG_lines", "BG_Surf"]
            and file[sheet] is not None
        ):
            file[sheet] = file[sheet].to_numpy()

    # Adjust to 0 indexed lines
    if file["BG_lines"] is not None:
        file["BG_lines"] = np.subtract(file["BG_lines"], 1)
    if file["sns_lines"] is not None:
        file["sns_lines"] = np.subtract(file["sns_lines"], 1)
    if file["BG_Surf"] is not None:
        file["BG_Surf"] = np.subtract(file["BG_Surf"], 1)
    if file["sns_surf"] is not None:
        file["sns_surf"] = np.subtract(file["sns_surf"], 1)

    sens_names = (file["sns_names"],)
    pts_coord = (file["pts_crd"],)
    sens_map = (file["snsTOpts_map"],)
    cstrn = (file.get("snsTOpts_cstrn"),)
    sens_sign = (file["sns_sign"],)
    sens_lines = (file["sns_lines"],)
    sens_surf = (file["sns_surf"],)
    bg_nodes = (file["BG_nodes"],)
    bg_lines = (file["BG_lines"],)
    bg_surf = (file["BG_Surf"],)
    return (
        sens_names[0],
        pts_coord[0],
        sens_map[0],
        cstrn[0],
        sens_sign[0],
        sens_lines[0],
        sens_surf[0],
        bg_nodes[0],
        bg_lines[0],
        bg_surf[0],
    )


def flatten_sns_names(sens_names, ref_ind=None):
    # check if sens_names is a dataframe with one row and transform it to a list
    # FOR SINGLE-SETUP GEOMETRIES
    if isinstance(sens_names, pd.DataFrame) and sens_names.values.shape[0] == 1:
        sns_names_fl = sens_names.values.tolist()[0]
    # Check if sens_names is a DataFrame with more than one row or a list of lists
    # FOR MULTI-SETUP GEOMETRIES
    elif (isinstance(sens_names, pd.DataFrame) and sens_names.values.shape[0] > 1) or (
        isinstance(sens_names, list)
        and all(isinstance(elem, list) for elem in sens_names)
    ):
        # if sens_names is a dataframe, transform it to a list
        if isinstance(sens_names, pd.DataFrame):
            sens_names = [
                [item for item in row if not pd.isna(item)]
                for row in sens_names.values.tolist()
            ]
        n = len(sens_names)
        if ref_ind is None:
            raise AttributeError(
                "You need to specify the reference indices for a Multi-setup test"
            )
        k = len(ref_ind[0])  # number of reference sensor (from the first setup)
        sns_names_fl = []
        # Create the reference strings
        for i in range(k):
            sns_names_fl.append(f"REF{i+1}")
        # Flatten the list of strings and exclude the reference indices
        for i in range(n):
            for j in range(len(sens_names[i])):
                if j not in ref_ind[i]:
                    sns_names_fl.append(sens_names[i][j])

    elif isinstance(sens_names, list) and all(
        isinstance(elem, str) for elem in sens_names
    ):
        sns_names_fl = sens_names

    elif isinstance(sens_names, np.ndarray) and sens_names.ndim == 1:
        sns_names_fl = sens_names.tolist()

    else:
        raise ValueError(
            "The input must of type: [list(str), list(list(str)), pd.DataFrame, NDArray(str)]"
        )

    return sns_names_fl


def Exdata():
    """
    This function generates a time history of acceleration for a 5 DOF
    system.

    The function returns a (360001,5) array and a tuple containing: the
    natural frequencies of the system (fn = (5,) array); the unity
    displacement normalised mode shapes matrix (FI_1 = (5,5) array); and the
    damping ratios (xi = float)

    -------
    Returns
    -------
    acc : 2D array
        Time histories of the 5 DOF of the system.
    (fn, FI_1, xi) : tuple
        Tuple containing the natural frequencies (fn), the mode shape
        matrix (FI_1), and the damping ratio (xi) of the system.

    """

    rng = np.random.RandomState(12345)  # Set the seed
    fs = 200  # [Hz] Sampling freqiency
    T = 900  # [sec] Period of the time series

    dt = 1 / fs  # [sec] time resolution
    N = int(T / dt)  # number of data points

    t = np.linspace(0, T + dt, N)

    # =========================================================================
    # SYSTEM DEFINITION

    m = 25.91  # mass
    k = 10000.0  # stiffness

    # Mass matrix
    M = np.eye(5) * m
    _ndof = M.shape[0]  # number of DOF (5)

    # Stiffness matrix
    K = (
        np.array(
            [
                [2, -1, 0, 0, 0],
                [-1, 2, -1, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 0, -1, 2, -1],
                [0, 0, 0, -1, 1],
            ]
        )
        * k
    )

    lam, FI = linalg.eigh(K, b=M)  # Solving eigen value problem

    fn = np.sqrt(lam) / (2 * np.pi)  # Natural frequencies

    # Unity displacement normalised mode shapes
    FI_1 = np.array([FI[:, k] / max(abs(FI[:, k])) for k in range(_ndof)]).T
    # Ordering from smallest to largest
    FI_1 = FI_1[:, np.argsort(fn)]
    fn = np.sort(fn)

    # K_M = FI_M.T @ K @ FI_M # Modal stiffness
    M_M = FI_1.T @ M @ FI_1  # Modal mass

    xi = 0.02  # damping ratio for all modes (2%)
    # Modal damping
    C_M = np.diag(
        np.array([2 * M_M[i, i] * xi * fn[i] * (2 * np.pi) for i in range(_ndof)])
    )

    C = linalg.inv(FI_1.T) @ C_M @ linalg.inv(FI_1)  # Damping matrix

    # =========================================================================
    # STATE-SPACE FORMULATION

    a1 = np.zeros((_ndof, _ndof))  # Zeros (ndof x ndof)
    a2 = np.eye(_ndof)  # Identity (ndof x ndof)
    A1 = np.hstack((a1, a2))  # horizontal stacking (ndof x 2*ndof)
    a3 = -linalg.inv(M) @ K  # M^-1 @ K (ndof x ndof)
    a4 = -linalg.inv(M) @ C  # M^-1 @ C (ndof x ndof)
    A2 = np.hstack((a3, a4))  # horizontal stacking(ndof x 2*ndof)
    # vertical stacking of A1 e A2
    Ac = np.vstack((A1, A2))  # State Matrix A (2*ndof x 2*ndof))

    b2 = -linalg.inv(M)
    # Input Influence Matrix B (2*ndof x n°input=ndof)
    Bc = np.vstack((a1, b2))

    # N.B. number of rows = n°output*ndof
    # n°output may be 1, 2 o 3 (displacements, velocities, accelerations)
    # the Cc matrix has to be defined accordingly
    c1 = np.hstack((a2, a1))  # displacements row
    c2 = np.hstack((a1, a2))  # velocities row
    c3 = np.hstack((a3, a4))  # accelerations row
    # Output Influence Matrix C (n°output*ndof x 2*ndof)
    Cc = np.vstack((c1, c2, c3))

    # Direct Transmission Matrix D (n°output*ndof x n°input=ndof)
    Dc = np.vstack((a1, a1, b2))

    # =========================================================================
    # Using SciPy's LTI to solve the system

    # Defining the system
    sys = signal.lti(Ac, Bc, Cc, Dc)

    # Defining the amplitute of the force
    af = 1

    # Assembling the forcing vectors (N x ndof) (random white noise!)
    # N.B. N=number of data points; ndof=number of DOF
    u = np.array([rng.randn(N) * af for r in range(_ndof)]).T

    # Solving the system
    tout, yout, xout = signal.lsim(sys, U=u, T=t)

    # d = yout[:,:5] # displacement
    # v = yout[:,5:10] # velocity
    a = yout[:, 10:]  # acceleration

    # =========================================================================
    # Adding noise
    # SNR = 10*np.log10(_af/_ar)
    SNR = 10  # Signal-to-Noise ratio
    ar = af / (10 ** (SNR / 10))  # Noise amplitude

    # Initialize the arrays (copy of accelerations)
    acc = a.copy()
    for _ind in range(_ndof):
        # Measurments POLLUTED BY NOISE
        acc[:, _ind] = a[:, _ind] + ar * rng.randn(N)

    # # Subplot of the accelerations
    # fig, axs = plt.subplots(5,1,sharex=True)
    # for _nr in range(_ndof):
    #     axs[_nr].plot(t, a[:,_nr], alpha=1, linewidth=1, label=f'story{_nr+1}')
    #     axs[_nr].legend(loc=1, shadow=True, framealpha=1)
    #     axs[_nr].grid(alpha=0.3)
    #     axs[_nr].set_ylabel('$mm/s^2$')
    # axs[_nr].set_xlabel('t [sec]')
    # fig.suptitle('Accelerations plot', fontsize=12)
    # plt.show()

    return acc, (fn, FI_1, xi)


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
    for oo in range(ordmin, ordmax + 1, step):
        o = int(oo / step)

        f_n = Fn1[:, o].reshape(-1, 1)
        xi_n = Sm1[:, o].reshape(-1, 1)
        phi_n = Ms1[:, o, :]

        f_n1 = Fn1[:, o - 1].reshape(-1, 1)
        xi_n1 = Sm1[:, o - 1].reshape(-1, 1)
        phi_n1 = Ms1[:, o - 1, :]

        if o != 0 and o != 1:
            for i in range(len(f_n)):
                try:
                    idx = np.nanargmin(np.abs(f_n1 - f_n[i]))

                    cond1 = np.abs(f_n[i] - f_n1[idx]) / f_n[i]
                    cond2 = np.abs(xi_n[i] - xi_n1[idx]) / xi_n[i]
                    cond3 = 1 - MAC(phi_n[i, :], phi_n1[idx, :])
                    if cond1 < err_fn and cond2 < err_xi and cond3 < err_ms:
                        Lab[i, o] = 7  # Stable

                    elif cond1 < err_fn and cond3 < err_ms:
                        # Stable frequency, stable mode shape
                        Lab[i, o] = 6

                    elif cond1 < err_fn and cond2 < err_xi:
                        Lab[i, o] = 5  # Stable frequency, stable damping

                    elif cond2 < err_xi and cond3 < err_ms:
                        Lab[i, o] = 4  # Stable damping, stable mode shape

                    elif cond2 < err_xi:
                        Lab[i, o] = 3  # Stable damping

                    elif cond3 < err_ms:
                        Lab[i, o] = 2  # Stable mode shape

                    elif cond1 < err_fn:
                        Lab[i, o] = 1  # Stable frequency

                    else:
                        Lab[i, o] = 0  # Nuovo polo o polo instabile
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
