# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 16:31:54 2023

@author: dpa
"""

import numpy as np

# =============================================================================
# Helping Funcitons
# =============================================================================
def MAC(phi_1, phi_2):
    """Modal Assurance Criterion.

    The number of locations (axis 0) must be the same for ``phi_1`` and
    ``phi_2``. The nubmer of modes (axis 1) is arbitrary.

    Literature:
        [1] Maia, N. M. M., and J. M. M. Silva. 
            "Modal analysis identification techniques." Philosophical
            Transactions of the Royal Society of London. Series A: 
            Mathematical, Physical and Engineering Sciences 359.1778 
            (2001): 29-40. 

    :param phi_1: Mode shape matrix X, shape: ``(n_locations, n_modes)``
        or ``n_locations``.
    :param phi_2: Mode shape matrix A, shape: ``(n_locations, n_modes)``
        or ``n_locations``.
    :return: MAC matrix. Returns MAC value if both ``phi_1`` and ``phi_2`` are
        one-dimensional arrays.
    """
    if phi_1.ndim == 1:
        phi_1 = phi_1[:, np.newaxis]
    
    if phi_2.ndim == 1:
        phi_2 = phi_2[:, np.newaxis]
    
    if phi_1.ndim > 2 or phi_2.ndim > 2:
        raise Exception(f'Mode shape matrices must have 1 or 2 dimensions (phi_1: {phi_1.ndim}, phi_2: {phi_2.ndim})')

    if phi_1.shape[0] != phi_2.shape[0]:
        raise Exception(f'Mode shapes must have the same first dimension (phi_1: {phi_1.shape[0]}, phi_2: {phi_2.shape[0]})')

    MAC = np.abs(phi_1.conj().T @ phi_2)**2 / \
        ((phi_1.conj().T @ phi_1)*(phi_2.conj().T @ phi_2))

    return MAC


def MSF(phi_1, phi_2):
    """Modal Scale Factor.

    If ``phi_1`` and ``phi_2`` are matrices, multiple msf are returned.

    The MAF scales ``phi_1`` to ``phi_2`` when multiplying: ``msf*phi_1``. 
    Also takes care of 180 deg phase difference.

    :param phi_1: Mode shape matrix X, shape: ``(n_locations, n_modes)``
        or ``n_locations``.
    :param phi_2: Mode shape matrix A, shape: ``(n_locations, n_modes)``
        or ``n_locations``.
    :return: np.ndarray, MSF values
    """
    if phi_1.ndim == 1:
        phi_1 = phi_1[:, None]
    if phi_2.ndim == 1:
        phi_2 = phi_2[:, None]
    
    if phi_1.shape[0] != phi_2.shape[0] or phi_1.shape[1] != phi_2.shape[1]:
        raise Exception(f'`phi_1` and `phi_2` must have the same shape: {phi_1.shape} and {phi_2.shape}')

    n_modes = phi_1.shape[1]
    msf = []
    for i in range(n_modes):
        _msf = (phi_2[:, i].T @ phi_1[:, i]) / \
                (phi_1[:, i].T @ phi_1[:, i])

        msf.append(_msf)

    return np.array(msf).real


def MCF(phi):
    """ Modal complexity factor.

    The MCF ranges from 0 to 1. It returns 0 for real modes and 1 for complex modes. 
    When ``dtype`` of ``phi`` is ``complex``, the modes can still be real, if the angles 
    of all components are the same.

    Additional information on MCF:
    http://www.svibs.com/resources/ARTeMIS_Modal_Help/Generic%20Complexity%20Plot.html
    
    :param phi: Complex mode shape matrix, shape: ``(n_locations, n_modes)``
        or ``n_locations``.
    :return: MCF (a value between 0 and 1)
    """
    if phi.ndim == 1:
        phi = phi[:, None]
    n_modes = phi.shape[1]
    mcf = []
    for i in range(n_modes):
        S_xx = np.dot(phi[:, i].real, phi[:, i].real)
        S_yy = np.dot(phi[:, i].imag, phi[:, i].imag)
        S_xy = np.dot(phi[:, i].real, phi[:, i].imag)
        
        _mcf = 1 - ((S_xx - S_yy)**2 + 4*S_xy**2) / (S_xx + S_yy)**2
        
        mcf.append(_mcf)
    return np.array(mcf)


def _stab_SSI(Fr, Sm, Ms, ordmin, ordmax, err_fn, err_xi, err_ms):
    """
    Helping function for the construction of the Stability Chart when using 
    Subspace Identification (SSI) method.

    This function performs stability analysis of identified poles.
    It categorizes modes based on their stabilityin terms of frequency, 
    damping, and mode shape.

    :param Fr: Frequency poles, shape: ``(ordmax, ordmax/2+1)``
    :param Sm: Damping poles, shape: ``(ordmax, ordmax/2+1)``
    :param Ms: Mode shape array, shape: ``(ordmax, ordmax/2+1, nch)``
    :param ordmin: Minimum order of model
    :param ordmax: Maximum order of model
    :param err_fn: Threshold for relative frequency difference for stability checks
    :param err_xi: Threshold for relative damping ratio difference for stability checks
    :param err_ms: Threshold for Modal Assurance Criterion (MAC) for stability checks

    :return: Stability label matrix (Lab), shape: ``(n_locations, n_modes)``
        - 7: Stable (frequency, damping, mode shape)
        - 6: Stable (frequency, mode shape)
        - 5: Stable (frequency, damping)
        - 4: Stable (damping, mode shape)
        - 3: Stable (damping)
        - 2: Stable (mode shape)
        - 1: Stable (frequency)
        - 0: New or unstable pole
    
    Note:
        nch = number of channesl (number of time series)
    """
    Lab = np.zeros(Fr.shape , dtype='int')

    for n in range(ordmin, ordmax+1, 2):
        _ind_new = int((n-ordmin)/2)

        f_n = Fr[:,_ind_new].reshape(-1,1)
        xi_n = Sm[:,_ind_new].reshape(-1,1)
        phi_n = Ms[:, _ind_new, :]

        f_n1 = Fr[:,_ind_new-1].reshape(-1,1)
        xi_n1 = Sm[:,_ind_new-1].reshape(-1,1)
        phi_n1 = Ms[:, _ind_new-1, :]

        if n != 0 and n != 2:
            
            for i in range(len(f_n)):
                
                if np.isnan(f_n[i]):
                    pass
                else:

                    idx = np.nanargmin(np.abs(f_n1 - f_n[i] ))

                    cond1 = np.abs(f_n[i] - f_n1[idx]) / f_n[i]
                    cond2 = np.abs(xi_n[i] - xi_n1[idx]) / xi_n[i]
                    cond3 = 1 - MAC(phi_n[i, :], phi_n1[idx, :])

                    if cond1 < err_fn and cond2 < err_xi and cond3 < err_ms:
                        Lab[i, _ind_new] = 7 # Stable
            
                    elif cond1 < err_fn  and cond3 < err_ms:
                        Lab[i, _ind_new] = 6 # Stable frequency, stable mode shape
            
                    elif cond1 < err_fn  and cond2 < err_xi:
                        Lab[i, _ind_new] = 5 # Stable frequency, stable damping
                        
                    elif cond2 < err_xi  and cond3 < err_ms:
                        Lab[i, _ind_new] = 4 # Stable damping, stable mode shape
    
                    elif cond2 < err_xi:
                        Lab[i, _ind_new] = 3 # Stable damping
                        
                    elif cond3 < err_ms:
                        Lab[i, _ind_new] = 2 # Stable mode shape
    
                    elif cond1 < err_fn:
                        Lab[i, _ind_new] = 1 # Stable frequency
    
                    else:
                        Lab[i, _ind_new] = 0  # Nuovo polo o polo instabile

    return Lab


def _stab_pLSCF(Fr, Sm, ordmax, err_fn, err_xi, nch):
    """
    Helping function for the construction of the Stability Chart when using 
    poly-reference Least Square Complex Frequency (pLSCF, also known as 
    Polymax) method.

    This function performs stability analysis of identified poles, it categorizes modes based on their stability in terms
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
    Lab = np.zeros(Fr.shape , dtype='int')

    for nn in range(ordmax):

        f_n = Fr[:, nn].reshape(-1,1)
        xi_n = Sm[:, nn].reshape(-1,1)

        f_n1 = Fr[:, nn-1].reshape(-1,1)
        xi_n1 = Sm[:, nn-1].reshape(-1,1)

        if nn != 0:
            
            for i in range(len(f_n)):

                if np.isnan(f_n[i]):
                    pass
                else:
                    try:
                        idx = np.nanargmin(np.abs(f_n1 - f_n[i] ))
    
                        cond1 = np.abs(f_n[i] - f_n1[idx]) / f_n[i]
                        cond2 = np.abs(xi_n[i] - xi_n1[idx]) / xi_n[i]
    
                        if cond1 < err_fn and cond2 < err_xi:
                            Lab[i, nn] = 3 # Stable Pole
    
                        elif cond2 < err_xi:
                            Lab[i, nn] = 2 # Stable damping
                            
                        elif cond1 < err_fn:
                            Lab[i, nn] = 1 # Stable frequency
                            
                        else:
                            Lab[i, nn] = 0  # Nuovo polo o polo instabile
                    except:
                        pass
    return Lab
