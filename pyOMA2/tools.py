# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 16:31:54 2023

@author: dpa
"""

import numpy as np
from scipy import signal
import scipy as sp

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

# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------

def Exdata():
    '''
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
    '''

    rng = np.random.RandomState(12345) # Set the seed
    fs = 100 # [Hz] Sampling freqiency
    T = 3600 # [sec] Period of the time series (60 minutes)
    
    dt = 1/fs # [sec] time resolution
    df = 1/T # [Hz] frequency resolution
    N = int(T/dt) # number of data points 
    fmax = fs/2 # Nyquist frequency

    t = np.arange(0, T, dt) # time instants array

    fs = np.arange(0, fmax+df, df) # spectral lines array

    #-------------------
    # SYSTEM DEFINITION

    m = 25.91 # mass
    k = 10000. # stiffness

    # Mass matrix
    M = np.eye(5)*m
    _ndof = M.shape[0] # number of DOF (5)

    # Stiffness matrix
    K = np.array([[2,-1,0,0,0],
                  [-1,2,-1,0,0],
                  [0,-1,2,-1,0],
                  [0,0,-1,2,-1],
                  [0,0,0,-1,1]])*k

    lam , FI = sp.linalg.eigh(K,b=M) # Solving eigen value problem

    fn = np.sqrt(lam)/(2*np.pi) # Natural frequencies
    
    # Unity displacement normalised mode shapes
    FI_1 = np.array([FI[:,k]/max(abs(FI[:,k])) for k in range(_ndof)]).T
    # Ordering from smallest to largest
    FI_1 = FI_1[:, np.argsort(fn)]
    fn = np.sort(fn)

    # K_M = FI_M.T @ K @ FI_M # Modal stiffness
    M_M = FI_1.T @ M @ FI_1 # Modal mass

    xi = 0.02 # damping ratio for all modes (2%)
    # Modal damping
    C_M = np.diag(np.array([2*M_M[i, i]*xi*fn[i]*(2*np.pi) for i in range(_ndof)]))

    C = sp.linalg.inv(FI_1.T) @ C_M @ sp.linalg.inv(FI_1) # Damping matrix
    # C = LA.solve(LA.solve(FI_1.T, C_M), FI_1)
    # n = _ndof*2 # order of the system
    
    #-------------------
    # STATE-SPACE FORMULATION
    
    a1 = np.zeros((_ndof,_ndof)) # Zeros (ndof x ndof)
    a2 = np.eye(_ndof) # Identity (ndof x ndof)
    A1 = np.hstack((a1,a2)) # horizontal stacking (ndof x 2*ndof)
    a3 = -sp.linalg.inv(M) @ K # M^-1 @ K (ndof x ndof)
    # a3 = -LA.solve(M, K) # M^-1 @ K (ndof x ndof)
    a4 = -sp.linalg.inv(M) @ C # M^-1 @ C (ndof x ndof)
    # a4 = -LA.solve(M, C) # M^-1 @ C (ndof x ndof)
    A2 = np.hstack((a3,a4)) # horizontal stacking(ndof x 2*ndof)
    # vertical stacking of A1 e A2
    Ac = np.vstack((A1,A2)) # State Matrix A (2*ndof x 2*ndof))
    
    b2 = -sp.linalg.inv(M)
     # Input Influence Matrix B (2*ndof x n°input=ndof)
    Bc = np.vstack((a1,b2))
    
    # N.B. number of rows = n°output*ndof 
    # n°output may be 1, 2 o 3 (displacements, velocities, accelerations)
    # the Cc matrix has to be defined accordingly
    c1 = np.hstack((a2,a1)) # displacements row
    c2 = np.hstack((a1,a2)) # velocities row
    c3 = np.hstack((a3,a4)) # accelerations row
    # Output Influence Matrix C (n°output*ndof x 2*ndof)
    Cc = np.vstack((c1,c2,c3)) 
    
    # Direct Transmission Matrix D (n°output*ndof x n°input=ndof)
    Dc = np.vstack((a1,a1, b2)) 
    
    #-------------------
    # Using SciPy's LTI to solve the system
    
    # Defining the system
    sys = signal.lti(Ac, Bc, Cc, Dc) 
    
    # Defining the amplitute of the force
    af = 1
    
    # Assembling the forcing vectors (N x ndof) (random white noise!)
    # N.B. N=number of data points; ndof=number of DOF
    u = np.array([rng.randn(N)*af for r in range(_ndof)]).T
    
    # Solving the system
    tout, yout, xout = signal.lsim(sys, U=u, T=t)
    
    # d = yout[:,:5] # displacement
    # v = yout[:,5:10] # velocity
    a = yout[:,10:] # acceleration
    
    #-------------------
    # Adding noise
    # SNR = 10*np.log10(_af/_ar)
    SNR = 10 # Signal-to-Noise ratio
    ar = af/(10**(SNR/10)) # Noise amplitude
    
    # Initialize the arrays (copy of accelerations)
    acc = a.copy()
    for _ind in range(_ndof):
        # Measurments POLLUTED BY NOISE
        acc[:,_ind] = a[:,_ind] + ar*rng.randn(N)

    #-------------------
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

    return acc, (fn,FI_1,xi)

# -----------------------------------------------------------------------------

def merge_mode_shapes(MSarr_list, refsens_idx_list):
    Nmodes = MSarr_list[0].shape[1]
    Nsetup = len(MSarr_list)
    Nref = len(refsens_idx_list[0])
    M = Nref + np.sum([ MSarr_list[i].shape[0]- Nref for i in range(Nsetup)])

    # Check if the input arrays have consistent dimensions

    for i in range(1, Nsetup):
        if MSarr_list[i].shape[1] != Nmodes:
            raise ValueError("All mode shape arrays must have the same number of modes.")

    # Initialize merged mode shape array
    merged_mode_shapes = np.zeros((M, Nmodes))

    # Loop through each mode
    for k in range(Nmodes):
        phi_1_k = MSarr_list[0][:, k] # Save the mode shape from first setup
        phi_ref_1_k = phi_1_k[refsens_idx_list[0]] # Save the reference sensors

        merged_mode_k = phi_1_k.copy() # initialise the merged mode shape 
        # Loop through each setup
        for i in range(1, Nsetup):
            ref_indices = refsens_idx_list[i] # reference sensors indices for the specific setup
            phi_i_k = MSarr_list[i][:, k] # mode shape of setup i
            phi_ref_i_k = MSarr_list[i][ref_indices, k] # save data from reference sensors
            phi_rov_i_k = np.delete(phi_i_k, ref_indices, axis=0) # saave data from roving sensors
            # Find scaling factor
            alpha_i_k = MSF(phi_ref_1_k, phi_ref_i_k)
            # Merge mode
            merged_mode_k = np.hstack((merged_mode_k, alpha_i_k * phi_rov_i_k ))

        merged_mode_shapes[:, k]  = merged_mode_k

    return merged_mode_shapes

