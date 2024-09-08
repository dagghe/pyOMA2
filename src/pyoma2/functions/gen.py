# -*- coding: utf-8 -*-
"""
General Utility Functions module.
Part of the pyOMA2 package.
Author:
Dag Pasca
"""

import logging
import pickle
import typing

import numpy as np
import pandas as pd
from scipy import linalg, signal

logger = logging.getLogger(__name__)


# =============================================================================
# FUNZIONI GENERALI
# =============================================================================
def applymask(list_arr, mask, len_phi):
    """
    Apply a mask to a list of arrays, filtering their values based on the mask.

    Parameters
    ----------
    list_arr : list of np.ndarray
        List of arrays to be filtered. Arrays can be 2D or 3D.
    mask : np.ndarray
        2D boolean array indicating which values to keep (True) or set to NaN (False).
    len_phi : int
        The length of the mode shape dimension for expanding the mask to 3D.

    Returns
    -------
    list of np.ndarray
        List of filtered arrays with the same shapes as the input arrays.

    Notes
    -----
    - If an array in `list_arr` is 3D, the mask is expanded to 3D and applied.
    - If an array in `list_arr` is 2D, the original mask is applied directly.
    - Values not matching the mask are set to NaN.
    """
    # Expand the mask to 3D by adding a new axis (for mode shape)
    expandedmask1 = np.expand_dims(mask, axis=-1)
    # Repeat the mask along the new dimension
    expandedmask1 = np.repeat(expandedmask1, len_phi, axis=-1)
    list_filt_arr = []
    for arr in list_arr:
        if arr is None:
            list_filt_arr.append(None)
        elif arr.ndim == 3:
            list_filt_arr.append(np.where(expandedmask1, arr, np.nan))
        elif arr.ndim == 2:
            list_filt_arr.append(np.where(mask, arr, np.nan))
    return list_filt_arr


# -----------------------------------------------------------------------------


def HC_conj(lambd):
    """
    Apply Hard validation Criteria (HC), retaining only those elements which have their conjugates also present in the array.

    Parameters
    ----------
    lambd : np.ndarray
        Array of complex numbers.

    Returns
    -------
    filt_lambd : np.ndarray
        Array of the same shape as `lambd` with only elements that have their conjugates also present.
        Other elements are set to NaN.
    mask : np.ndarray
        Boolean array of the same shape as `lambd`, where True indicates that the element and its conjugate are both present.
    """
    # Create a set to store elements and their conjugates
    element_set = set(lambd.flatten())

    # Create a mask to identify elements to keep
    mask = np.zeros(lambd.shape, dtype=bool)

    for i in range(lambd.shape[0]):
        for j in range(lambd.shape[1]):
            element = lambd[i, j]
            conjugate = np.conj(element)
            # Check if both element and its conjugate are in the set
            if element in element_set and conjugate in element_set:
                mask[i, j] = True

    # Create an output array filled with NaNs
    filt_lambd = np.full(lambd.shape, np.nan, dtype=lambd.dtype)

    # Copy elements that satisfy the condition to the output array
    filt_lambd[mask] = lambd[mask]

    return filt_lambd, mask


# -----------------------------------------------------------------------------


def HC_damp(damp, max_damp):
    """
    Apply Hard validation Criteria (HC), retaining only those elements which are positive and less than a specified maximum (damping).

    Parameters
    ----------
    damp : np.ndarray
        Array of damping ratios.
    max_damp : float
        Maximum allowed damping ratio.

    Returns
    -------
    filt_damp : np.ndarray
        Array of the same shape as `damp` with elements that do not satisfy the condition set to NaN.
    mask : np.ndarray
        Boolean array of the same shape as `damp`, where True indicates that the element is positive and less than `max_damp`.

    """
    mask = np.logical_and(damp < max_damp, damp > 0).astype(int)
    filt_damp = damp * mask
    filt_damp[filt_damp == 0] = np.nan
    # should be the same as
    # filt_damp = np.where(damp, np.logical_and(damp < max_damp, damp > 0), damp, np.nan)
    return filt_damp, mask


# -----------------------------------------------------------------------------


def HC_phi_comp(phi, mpc_lim, mpd_lim):
    """
    Apply Hard validation Criteria (HC), based on modal phase collinearity (MPC) and modal phase deviation (MPD) limits.

    Parameters
    ----------
    phi : np.ndarray
        Array of mode shapes with shape (number of modes, number of channels, mode shape length).
    mpc_lim : float
        Minimum allowed value for modal phase collinearity.
    mpd_lim : float
        Maximum allowed value for modal phase deviation.

    Returns
    -------
    mask_mpd : np.ndarray
        Boolean array indicating elements that satisfy the MPD condition.
    mask_mpc : np.ndarray
        Boolean array indicating elements that satisfy the MPC condition.
    """
    mask = []
    for o in range(phi.shape[0]):
        for i in range(phi.shape[1]):
            try:
                mask.append((MPD(phi[o, i, :]) <= mpd_lim).astype(int))
            except Exception:
                mask.append(0)
    mask = np.array(mask).reshape((phi.shape[0], phi.shape[1]))
    mask1 = np.expand_dims(mask, axis=-1)
    mask1 = np.repeat(mask1, phi.shape[2], axis=-1)
    Phi = phi * mask1
    Phi[Phi == 0] = np.nan

    mask2 = []
    for o in range(phi.shape[0]):
        for i in range(phi.shape[1]):
            try:
                mask2.append((MPC(phi[o, i, :]) >= mpc_lim).astype(int))
            except Exception:
                mask2.append(0)
    mask2 = np.array(mask2).reshape((phi.shape[0], phi.shape[1]))
    mask3 = np.expand_dims(mask2, axis=-1)
    mask3 = np.repeat(mask3, phi.shape[2], axis=-1)
    Phi = phi * mask3
    Phi[Phi == 0] = np.nan

    return mask1[:, :, 0], mask3[:, :, 0]


# -----------------------------------------------------------------------------


def HC_cov(Fn_cov, max_cov):
    """
    Apply Hard validation Criteria (HC), retaining only those elements which have a covariance less than a specified maximum.

    Parameters
    ----------
    Fn_cov : np.ndarray
        Array of frequency covariances.
    max_cov : float
        Maximum allowed covariance.

    Returns
    -------
    filt_cov : np.ndarray
        Array of the same shape as `Fn_cov` with elements that do not satisfy the condition set to NaN.
    mask : np.ndarray
        Boolean array of the same shape as `Fn_cov`, where True indicates that the element is less than `max_cov`.

    """
    mask = (Fn_cov < max_cov).astype(int)
    filt_cov = Fn_cov * mask
    filt_cov[filt_cov == 0] = np.nan
    # should be the same as
    # filt_damp = np.where(damp, np.logical_and(damp < max_damp, damp > 0), damp, np.nan)
    return filt_cov, mask


# -----------------------------------------------------------------------------


def SC_apply(Fn, Xi, Phi, ordmin, ordmax, step, err_fn, err_xi, err_phi):
    """
    Apply Soft validation Criteria (SC) to determine the stability of modal parameters between consecutive orders.

    Parameters
    ----------
    Fn : np.ndarray
        Array of natural frequencies.
    Xi : np.ndarray
        Array of damping ratios.
    Phi : np.ndarray
        Array of mode shapes.
    ordmin : int
        Minimum model order.
    ordmax : int
        Maximum model order.
    step : int
        Step size for increasing model order.
    err_fn : float
        Tolerance for the natural frequency error.
    err_xi : float
        Tolerance for the damping ratio error.
    err_phi : float
        Tolerance for the mode shape error.

    Returns
    -------
    Lab : np.ndarray
        Array of labels indicating stability (1 for stable, 0 for unstable).
    """
    # inirialise labels
    Lab = np.zeros(Fn.shape, dtype="int")

    # SOFT CONDITIONS
    # STABILITY BETWEEN CONSECUTIVE ORDERS
    for oo in range(ordmin, ordmax + 1, step):
        o = int(oo / step)

        f_n = Fn[:, o].reshape(-1, 1)
        xi_n = Xi[:, o].reshape(-1, 1)
        phi_n = Phi[:, o, :]

        f_n1 = Fn[:, o - 1].reshape(-1, 1)
        xi_n1 = Xi[:, o - 1].reshape(-1, 1)
        phi_n1 = Phi[:, o - 1, :]

        # Skip the first order as it has no previous order to compare with
        if o == 0:
            continue

        for i in range(len(f_n)):
            try:
                idx = np.nanargmin(np.abs(f_n1 - f_n[i]))

                cond1 = np.abs(f_n[i] - f_n1[idx]) / f_n[i]
                cond2 = np.abs(xi_n[i] - xi_n1[idx]) / xi_n[i]
                cond3 = 1 - MAC(phi_n[i, :], phi_n1[idx, :])
                if cond1 < err_fn and cond2 < err_xi and cond3 < err_phi:
                    Lab[i, o] = 1  # Stable
                else:
                    Lab[i, o] = 0  # Nuovo polo o polo instabile
            except Exception as e:
                # If f_n[i] is nan, do nothin, n.b. the lab stays 0
                logger.debug(e)
    return Lab


# -----------------------------------------------------------------------------


def dfphi_map_func(phi, sens_names, sens_map, cstrn=None):
    """
    Maps mode shapes to sensor locations and constraints, creating a dataframe.

    Parameters
    ----------
    phi : np.ndarray
        Array of mode shapes.
    sens_names : list
        List of sensor names corresponding to the mode shapes.
    sens_map : pd.DataFrame
        DataFrame containing the sensor mappings.
    cstrn : pd.DataFrame, optional
        DataFrame containing constraints, by default None.

    Returns
    -------
    pd.DataFrame
        DataFrame with mode shapes mapped to sensor points.
    """
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


# -----------------------------------------------------------------------------


def check_on_geo1(file_dict, ref_ind=None):
    """
    Validates and processes sensor and background geometry data from a dictionary of dataframes.

    Parameters
    ----------
    file_dict : dict
        Dictionary containing dataframes of sensor and background geometry data.
    ref_ind : list, optional
        List of reference indices for sensor names, by default None.

    Returns
    -------
    tuple
        A tuple containing:
        - sens_names (list): List of sensor names.
        - sens_coord (pd.DataFrame): DataFrame of sensor coordinates.
        - sens_dir (np.ndarray): Array of sensor directions.
        - sens_lines (np.ndarray or None): Array of sensor lines.
        - BG_nodes (np.ndarray or None): Array of background nodes.
        - BG_lines (np.ndarray or None): Array of background lines.
        - BG_surf (np.ndarray or None): Array of background surfaces.

    Raises
    ------
    ValueError
        If required sheets are missing or invalid.
        If shapes of 'sensors coordinates' and 'sensors directions' do not match.
        If 'sensors coordinates' or 'BG nodes' does not have 3 columns.
        If 'BG lines' does not have 2 columns.
        If 'BG surfaces' does not have 3 columns.
        If sensor names are not present in the index of 'sensors coordinates'.
    """

    # Remove INFO sheet from dict of dataframes
    if "INFO" in file_dict:
        del file_dict["INFO"]

    # -----------------------------------------------------------------------------
    required_sheets = ["sensors names", "sensors coordinates", "sensors directions"]
    all_sheets = required_sheets + [
        "sensors lines",
        "BG nodes",
        "BG lines",
        "BG surfaces",
    ]

    # Ensure required sheets exist
    if not all(sheet in file_dict for sheet in required_sheets):
        raise ValueError(f"At least the sheets {required_sheets} must be defined!")

    # Ensure all sheets are valid
    for sheet in file_dict:
        if sheet not in all_sheets:
            raise ValueError(
                f"'{sheet}' is not a valid name. Valid sheet names are: \n"
                f"{all_sheets}"
            )

    # -----------------------------------------------------------------------------
    # Check 'sensors coordinates' shape
    if file_dict["sensors coordinates"].values.shape[1] != 3:
        raise ValueError(
            "'sensors coordinates' should have 3 columns for the x,y and z coordinates."
            f"'sensors coordinates' have {file_dict['sensors coordinates'].values.shape[1]} columns"
        )

    # Check on same shape 'sensors coordinates' and 'sensors directions'
    if (
        file_dict["sensors coordinates"].values.shape
        != file_dict["sensors directions"].values.shape
    ):
        raise ValueError(
            "'sensors coordinates' and 'sensors directions' must have the same shape.\n"
            f"'sensors coordinates' shape is {file_dict['sensors coordinates'].values.shape} while 'sensors directions' shape is {file_dict['sensors directions'].values.shape}"
        )

    # Check 'BG nodes' shape
    if (
        file_dict.get("BG nodes") is not None
        and not file_dict["BG nodes"].empty
        and file_dict["BG nodes"].values.shape[1] != 3
    ):
        raise ValueError(
            "'BG nodes' should have 3 columns for the x,y and z coordinates."
            f"'BG nodess' have {file_dict['BG nodes'].values.shape[1]} columns"
        )

    # Check 'BG lines' shape
    if (
        file_dict.get("BG lines") is not None
        and not file_dict["BG lines"].empty
        and file_dict["BG lines"].values.shape[1] != 2
    ):
        raise ValueError(
            "'BG lines' should have 2 columns for the starting and ending node of the line."
            f"'BG lines' have {file_dict['BG lines'].values.shape[1]} columns"
        )

    # Check 'BG surfaces' shape
    if (
        file_dict.get("BG surfaces") is not None
        and not file_dict["BG surfaces"].empty
        and file_dict["BG surfaces"].values.shape[1] != 3
    ):
        raise ValueError(
            "'BG surfaces' should have 3 columns for the i,j and k node of the triangle."
            f"'BG surfaces' have {file_dict['BG surfaces'].values.shape[1]} columns"
        )

    # Check on same index 'sensors coordinates' and 'sensors directions'
    if (
        file_dict["sensors coordinates"].index.to_list()
        != file_dict["sensors directions"].index.to_list()
    ):
        raise ValueError(
            "'sensors coordinates' and 'sensors directions' must have the same index.\n"
            f"'sensors coordinates' index is {file_dict['sensors coordinates'].index} while 'sensors directions' index is {file_dict['sensors directions'].index}"
        )

    # Extract the relevant dataframes
    sens_names = file_dict["sensors names"]
    sens_names = flatten_sns_names(sens_names, ref_ind)

    # Check for the presence of each string in the list
    if not all(
        item in file_dict["sensors coordinates"].index.to_list() for item in sens_names
    ):
        raise ValueError(
            "All sensors names must be present as index of the sensors coordinates dataframe!"
        )

    # -----------------------------------------------------------------------------
    # Find the indices that rearrange sens_coord to sens_names
    # newIDX = find_map(sens_names, file_dict['sensors coordinates'].index.to_numpy())
    # reorder if necessary
    sens_coord = file_dict["sensors coordinates"].reindex(index=sens_names)
    sens_dir = file_dict["sensors directions"].reindex(index=sens_names).values

    # -----------------------------------------------------------------------------
    # Adjust to 0 indexing
    for key in ["sensors lines", "BG lines", "BG surfaces"]:
        if key in file_dict and not file_dict[key].empty:
            file_dict[key] = file_dict[key].sub(1)
    # -----------------------------------------------------------------------------
    # if there is no entry create an empty one
    for key in all_sheets:
        if key not in file_dict:
            file_dict[key] = pd.DataFrame()

    # Transform to None empty dataframes
    for sheet, df in file_dict.items():
        if df.empty:
            file_dict[sheet] = None

        # Transform to array relevant dataframes
        if (
            sheet in ["sensors lines", "BG nodes", "BG lines", "BG surfaces"]
            and file_dict[sheet] is not None
        ):
            file_dict[sheet] = file_dict[sheet].to_numpy()

    sens_lines = file_dict["sensors lines"]
    BG_nodes = file_dict["BG nodes"]
    BG_lines = file_dict["BG lines"]
    BG_surf = file_dict["BG surfaces"]

    return (sens_names, sens_coord, sens_dir, sens_lines, BG_nodes, BG_lines, BG_surf)


# -----------------------------------------------------------------------------


def check_on_geo2(file_dict, ref_ind=None, fill_na="zero"):
    """
    Validates and processes sensor and background geometry data from a dictionary of dataframes.

    Parameters
    ----------
    file_dict : dict
        Dictionary containing dataframes of sensor and background geometry data.
    ref_ind : list, optional
        List of reference indices for sensor names, by default None.
    fill_na : str, optional
        Method to fill missing values in the mapping dataframe, by default "zero".

    Returns
    -------
    tuple
        A tuple containing:
        - sens_names (list): List of sensor names.
        - pts_coord (pd.DataFrame): DataFrame of points coordinates.
        - sens_map (pd.DataFrame): DataFrame of sensor mappings.
        - cstr (pd.DataFrame or None): DataFrame of constraints.
        - sens_sign (pd.DataFrame): DataFrame of sensor signs.
        - sens_lines (np.ndarray or None): Array of sensor lines.
        - sens_surf (np.ndarray or None): Array of sensor surfaces.
        - BG_nodes (np.ndarray or None): Array of background nodes.
        - BG_lines (np.ndarray or None): Array of background lines.
        - BG_surf (np.ndarray or None): Array of background surfaces.

    Raises
    ------
    ValueError
        If required sheets are missing or invalid.
        If shapes of 'points coordinates' and 'mapping' do not match.
        If 'points coordinates' or 'BG nodes' does not have 3 columns.
        If 'BG lines' does not have 2 columns.
        If 'BG surfaces' does not have 3 columns.
        If sensor names are not present in the mapping dataframe.
        If constraints columns do not correspond to sensor names.
        If constraints names are not the same as those used in the mapping.
    """
    # Remove INFO sheet from dict of dataframes
    if "INFO" in file_dict:
        del file_dict["INFO"]

    # -----------------------------------------------------------------------------
    required_sheets = ["sensors names", "points coordinates", "mapping"]
    all_sheets = required_sheets + [
        "constraints",
        "sensors sign",
        "sensors lines",
        "sensors surfaces",
        "BG nodes",
        "BG lines",
        "BG surfaces",
    ]

    # Ensure required sheets exist
    if not all(sheet in file_dict for sheet in required_sheets):
        raise ValueError(f"At least the sheets {required_sheets} must be defined!")

    # Ensure all sheets are valid
    for sheet in file_dict:
        if sheet not in all_sheets:
            raise ValueError(
                f"'{sheet}' is not a valid name. Valid sheet names are: \n"
                f"{all_sheets}"
            )

    # -----------------------------------------------------------------------------
    # Check 'points coordinates' shape
    if file_dict["points coordinates"].values.shape[1] != 3:
        raise ValueError(
            "'points coordinates' should have 3 columns for the x,y and z coordinates."
            f"'points coordinates' have {file_dict['points coordinates'].values.shape[1]} columns"
        )

    # Check on same shape 'points coordinates' and 'mapping'
    if file_dict["points coordinates"].values.shape != file_dict["mapping"].values.shape:
        raise ValueError(
            "'points coordinates' and 'mapping' must have the same shape.\n"
            f"'points coordinates' shape is {file_dict['points coordinates'].values.shape} while 'mapping' shape is {file_dict['mapping'].values.shape}"
        )

    # Check on shape for 'sensors sign'
    if (
        file_dict.get("sensors sign") is not None
        and not file_dict["sensors sign"].empty
        and file_dict["points coordinates"].values.shape
        != file_dict["sensors sign"].values.shape
    ):
        raise ValueError(
            "'points coordinates' and 'sensors sign' must have the same shape.\n"
            f"'points coordinates' shape is {file_dict['points coordinates'].values.shape} while 'sensors sign' shape is {file_dict['sensors sign'].values.shape}"
        )

    # Check 'BG nodes' shape
    if (
        file_dict.get("BG nodes") is not None
        and not file_dict["BG nodes"].empty
        and file_dict["BG nodes"].values.shape[1] != 3
    ):
        raise ValueError(
            "'BG nodes' should have 3 columns for the x,y and z coordinates."
            f"'BG nodess' have {file_dict['BG nodes'].values.shape[1]} columns"
        )

    # Check 'BG lines' shape
    if (
        file_dict.get("BG lines") is not None
        and not file_dict["BG lines"].empty
        and file_dict["BG lines"].values.shape[1] != 2
    ):
        raise ValueError(
            "'BG lines' should have 2 columns for the starting and ending node of the line."
            f"'BG lines' have {file_dict['BG lines'].values.shape[1]} columns"
        )

    # Check 'BG surfaces' shape
    if (
        file_dict.get("BG surfaces") is not None
        and not file_dict["BG surfaces"].empty
        and file_dict["BG surfaces"].values.shape[1] != 3
    ):
        raise ValueError(
            "'BG surfaces' should have 3 columns for the i,j and k node of the triangle."
            f"'BG surfaces' have {file_dict['BG surfaces'].values.shape[1]} columns"
        )

    # if there is no 'sensors sign' create one
    if file_dict.get("sensors sign") is None or file_dict["sensors sign"].empty:
        sens_sign = pd.DataFrame(
            np.ones(file_dict["points coordinates"].values.shape),
            columns=file_dict["points coordinates"].columns,
        )
        file_dict["sensors sign"] = sens_sign

    # -----------------------------------------------------------------------------
    # Check that mapping contains all sensors name
    # Extract the relevant dataframes
    sens_names = file_dict["sensors names"]
    sens_names = flatten_sns_names(sens_names, ref_ind)
    df_map = file_dict["mapping"]
    constraints = file_dict["constraints"].fillna(0)

    if fill_na == "zero":
        df_map = df_map.fillna(0.0)
    # elif fill_na == "interp":
    #     df_map = df_map.fillna("interp")

    file_dict["mapping"] = df_map

    # Step 1: Flatten the DataFrame to a single list of values
    map_fl = df_map.values.flatten()
    # Step 2: Convert all values to strings
    map_str_fl = [str(value) for value in map_fl]
    # Step 3: Check for the presence of each string in the list
    if not all(item in map_str_fl for item in sens_names):
        raise ValueError("All sensors names must be present in the mapping dataframe!")

    # -----------------------------------------------------------------------------
    # Check that the constraints columns correspond to sensors names
    columns = constraints.columns.to_list()
    indices = constraints.index.to_list()
    if not all(item in sens_names for item in columns):
        raise ValueError(
            "The constraints columns names must correspond to sensors names.\n"
            f"constraints columns names: {columns}, \n"
            f"sensors names: {sens_names}"
        )

    # -----------------------------------------------------------------------------
    # Check that the constraints names are the same as those used in mapping
    list_of_possible_constraints = ["0", "0.0", "interp"]
    # remove values equal to sensors names and other possible values (should be left with only contraints)
    map_str_cstr = [
        value
        for value in map_str_fl
        if value not in sens_names and value not in list_of_possible_constraints
    ]
    if not all(item in map_str_cstr for item in indices):
        raise ValueError(
            "The constraints names (index column) must be the same as those used in mapping.\n"
            f"constraints index column: {indices}, \n"
            f"mapping : {map_str_cstr}"
        )

    # -----------------------------------------------------------------------------
    # Add missing sensor names with all zeros to the constraints DataFrame
    missing_sensors = [name for name in sens_names if name not in columns]
    for name in missing_sensors:
        constraints[name] = 0

    # Reorder columns to match the order of sens_names if necessary
    file_dict["constraints"] = constraints[sens_names]

    # -----------------------------------------------------------------------------
    # Adjust to 0 indexing
    for key in ["sensors lines", "sensors surfaces", "BG lines", "BG surfaces"]:
        if key in file_dict and not file_dict[key].empty:
            file_dict[key] = file_dict[key].sub(1)
    # -----------------------------------------------------------------------------
    # if there is no entry create an empty one
    for key in all_sheets:
        if key not in file_dict:
            file_dict[key] = pd.DataFrame()

    # Transform to None empty dataframes
    for sheet, df in file_dict.items():
        if df.empty:
            file_dict[sheet] = None

        # Transform to array relevant dataframes
        if (
            sheet
            in [
                "sensors lines",
                "sensors surfaces",
                "BG nodes",
                "BG lines",
                "BG surfaces",
            ]
            and file_dict[sheet] is not None
        ):
            file_dict[sheet] = file_dict[sheet].to_numpy()

    # sens_names = file_dict["sensors names"]
    pts_coord = file_dict["points coordinates"]
    sens_map = file_dict["mapping"]
    cstr = file_dict["constraints"]
    sens_sign = file_dict["sensors sign"]
    sens_lines = file_dict["sensors lines"]
    sens_surf = file_dict["sensors surfaces"]
    BG_nodes = file_dict["BG nodes"]
    BG_lines = file_dict["BG lines"]
    BG_surf = file_dict["BG surfaces"]

    return (
        sens_names,
        pts_coord,
        sens_map,
        cstr,
        sens_sign,
        sens_lines,
        sens_surf,
        BG_nodes,
        BG_lines,
        BG_surf,
    )


# -----------------------------------------------------------------------------


def flatten_sns_names(sens_names, ref_ind=None):
    """
    Ensures that sensors names is in the correct form (1D list of strings) for both
    single-setup or multi-setup geometries.

    Parameters
    ----------
    sens_names : list, pd.DataFrame, or np.ndarray
        Sensor names which can be a list of strings, list of lists of strings, DataFrame,
        or 1D numpy array of strings.
    ref_ind : list, optional
        List of reference indices for multi-setup geometries, by default None.

    Returns
    -------
    list
        Flattened list of sensor names.

    Raises
    ------
    AttributeError
        If `ref_ind` is not provided for multi-setup geometries.
    ValueError
        If `sens_names` is not of the expected types.
    """
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


# -----------------------------------------------------------------------------


def example_data():
    """
    This function generates a time history of acceleration for a 5 DOF
    system.

    The function returns a (360001,5) array and a tuple containing: the
    natural frequencies of the system (fn = (5,) array); the unity
    displacement normalised mode shapes matrix (FI_1 = (5,5) array); and the
    damping ratios (xi = float)

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
    Ndat = int(T / dt)  # number of data points

    t = np.linspace(0, T + dt, Ndat)

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
    u = np.array([rng.randn(Ndat) * af for r in range(_ndof)]).T

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
        acc[:, _ind] = a[:, _ind] + ar * rng.randn(Ndat)

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


# -----------------------------------------------------------------------------


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
    Calculate the Modal Phase Collinearity (MPC) of a complex mode shape.

    The MPC is a measure of the collinearity between the real and imaginary parts
    of a mode shape. A value of 1 indicates perfect collinearity, while lower values
    indicate a more complex (non-collinear) mode.

    Parameters
    ----------
    phi : ndarray
        Complex mode shape vector, shape: (n_locations, ).

    Returns
    -------
    float
        MPC value, ranging between 0 and 1, where 1 indicates perfect collinearity.
    """
    S = np.cov(phi.real, phi.imag)
    lambd = np.linalg.eigvals(S)
    MPC = (lambd[0] - lambd[1]) ** 2 / (lambd[0] + lambd[1]) ** 2
    return MPC


# -----------------------------------------------------------------------------


def MPD(phi: np.ndarray) -> float:
    """
    Calculate the Mean Phase Deviation (MPD) of a complex mode shape.

    The MPD measures the deviation of the mode shape phases from a purely
    real mode. It quantifies the phase variation along the mode shape.

    Parameters
    ----------
    phi : ndarray
        Complex mode shape vector, shape: (n_locations, ).

    Returns
    -------
    float
        MPD value, representing the average deviation of the phase from a
        purely real mode.
    """

    U, s, VT = np.linalg.svd(np.c_[phi.real, phi.imag])
    V = VT.T
    w = np.abs(phi)
    num = phi.real * V[1, 1] - phi.imag * V[0, 1]
    den = np.sqrt(V[0, 1] ** 2 + V[1, 1] ** 2) * np.abs(phi)
    MPD = np.sum(w * np.arccos(np.abs(num / den))) / np.sum(w)
    return MPD


# -----------------------------------------------------------------------------


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
    # MAC = np.abs(np.dot(phi_X.conj().T, phi_A)) ** 2 / (
    #     (np.dot(phi_X.conj().T, phi_X)) * (np.dot(phi_A.conj().T, phi_A))
    # )
    # original
    MAC = np.abs(np.conj(phi_X).T @ phi_A) ** 2
    MAC = MAC.astype(complex)
    for i in range(phi_X.shape[1]):
        for j in range(phi_A.shape[1]):
            MAC[i, j] = MAC[i, j] / (
                np.conj(phi_X[:, i]) @ phi_X[:, i] * np.conj(phi_A[:, j]) @ phi_A[:, j]
            )

    if MAC.shape == (1, 1):
        MAC = MAC[0, 0]

    return MAC.real


# -----------------------------------------------------------------------------


def pre_multisetup(
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


# -----------------------------------------------------------------------------


def save_to_file(setup: object, file_name: str):
    """
    Save the specified setup instance to a file.

    This method serializes the current instance and saves it to a file using the pickle module.

    Parameters
    ----------
    setup : obj
        The Setup class that is to be saved.
    file_name : str
        The name (path) of the file where the setup instance will be saved.
    """
    with open(file_name, "wb") as f:
        pickle.dump(setup, f)


def load_from_file(file_name: str):
    """
    Load a setup instance from a file.

    This method deserializes a saved setup instance from the specified file.

    Parameters
    ----------
    file_name : str
        The name (path) of the file from which the setup instance will be loaded.

    Returns
    -------
    Setup
        An instance of the setup loaded from the file.
    """
    with open(file_name, "rb") as f:
        instance = pickle.load(f)  # noqa S301
    return instance


def read_excel_file(
    path: str,
    sheet_name: typing.Optional[str] = None,
    engine: str = "openpyxl",
    index_col: int = 0,
    **kwargs,
) -> dict:
    """
    Read an Excel file and return its contents as a dictionary.

    Parameters:
    -----------
    path : str
        The path to the Excel file.
    sheet_name : str, optional
        The name of the sheet to read. If None, all sheets are read. Default is None.
    engine : str, optional
        The engine to use for reading the Excel file. Default is 'openpyxl'.
    index_col : int, optional
        The column to use as the index. Default is 0
    **kwargs : dict, optional
        Additional keyword arguments to pass to pd.read_excel.

    Returns:
    --------
    dict
        A dictionary containing the contents of the Excel file, with sheet names as keys.

    Raises:
    -------
    ImportError
        If the specified engine is not available.
    RuntimeError
        If an error occurs while reading the Excel file.
    """
    try:
        file_dict = pd.read_excel(
            path, sheet_name=sheet_name, engine=engine, index_col=index_col, **kwargs
        )
        return file_dict
    except ImportError as e:
        raise ImportError(
            "Optional package 'openpyxl' is not installed. "
            "Install 'openpyxl' with 'pip install openpyxl' or 'pip install pyoma_2[pyvista]'"
        ) from e
    except Exception as e:
        logger.error("An error occurred while reading the Excel file: %s", e)
        raise RuntimeError(
            f"An error occurred while reading the Excel file: {e.__class__}: {e}"
        ) from e
