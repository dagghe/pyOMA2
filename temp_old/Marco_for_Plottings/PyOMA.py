# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:49:48 2021

@author: dagpa
"""

import numpy as np
from scipy import linalg as LA
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
import matplotlib.patches as patches
import seaborn as sns
import mplcursors

# =============================================================================
# FUNZIONI PRONTE
# =============================================================================

def MaC(Fi1,Fi2):
    '''
    This function returns the Modal Assurance Criterion (MAC) for two mode 
    shape vectors.
    
    If the input arrays are in the form (n,) (1D arrays) the output is a 
    scalar, if the input are in the form (n,m) the output is a (m,m) matrix
    (MAC matrix).
    
    ----------
    Parameters
    ----------
    Fi1 : array (1D or 2D)
        First mode shape vector (or matrix).
    Fi2 : array (1D or 2D)
        Second mode shape vector (or matrix). 
        
    -------
    Returns
    -------
    MAC : float or (2D array)
        Modal Assurance Criterion.
    '''
    
    MAC = np.abs(Fi1.conj().T @ Fi2)**2 / \
        ((Fi1.conj().T @ Fi1)*(Fi2.conj().T @ Fi2))
        
    return MAC


#------------------------------------------------------------------------------

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
    
    t = np.arange(0, T+dt, dt) # time instants array
    
    fs = np.arange(0, fmax+df, df) # spectral lines array
    
    
    # =========================================================================
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
    
    lam , FI = LA.eigh(K,b=M) # Solving eigen value problem
    
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
    
    C = LA.inv(FI_1.T) @ C_M @ LA.inv(FI_1) # Damping matrix
    
    # n = _ndof*2 # order of the system
    
    # =========================================================================
    # STATE-SPACE FORMULATION
    
    a1 = np.zeros((_ndof,_ndof)) # Zeros (ndof x ndof)
    a2 = np.eye(_ndof) # Identity (ndof x ndof)
    A1 = np.hstack((a1,a2)) # horizontal stacking (ndof x 2*ndof)
    a3 = -LA.inv(M) @ K # M^-1 @ K (ndof x ndof)
    a4 = -LA.inv(M) @ C # M^-1 @ C (ndof x ndof)
    A2 = np.hstack((a3,a4)) # horizontal stacking(ndof x 2*ndof)
    # vertical stacking of A1 e A2
    Ac = np.vstack((A1,A2)) # State Matrix A (2*ndof x 2*ndof))
    
    b2 = -LA.inv(M)
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
    
    # =========================================================================
    # Using SciPy's LTI to solve the system
    
    # Defining the system
    sys = signal.lti(Ac, Bc, Cc, Dc) 
    
    # Defining the amplitute of the force
    af = 1
    
    # Assembling the forcing vectors (N x ndof) (random white noise!)
    # N.B. N=number of data points; ndof=number of DOF
    u = np.array([rng.randn(N+1)*af for r in range(_ndof)]).T
    
    # Solving the system
    tout, yout, xout = signal.lsim(sys, U=u, T=t)
    
    # d = yout[:,:5] # displacement
    # v = yout[:,5:10] # velocity
    a = yout[:,10:] # acceleration
    
    # =========================================================================
    # Adding noise
    # SNR = 10*np.log10(_af/_ar)
    SNR = 10 # Signal-to-Noise ratio
    ar = af/(10**(SNR/10)) # Noise amplitude
    
    # Initialize the arrays (copy of accelerations)
    acc = a.copy()
    for _ind in range(_ndof):
        # Measurments POLLUTED BY NOISE
        acc[:,_ind] = a[:,_ind] + ar*rng.randn(N+1)
        
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

#------------------------------------------------------------------------------
    
def SSIdatStaDiag(data, fs, br, ordmax=None, lim=(0.01,0.05,0.02,0.1), 
                  method='1'):
    '''
    This function perform the Data-driven Stochastic sub-Space Identification 
    algorithm.
    
    The function returns the Stabilization Diagram (Plot) for the given
    data. Furthermore it returns a dictionary that contains the results needed
    by the function SSImodEX().
    
    ----------
    Parameters
    ----------
    data : 2D array
        The time history records (N°data points x N°channels).
    fs : float
        The sampling frequency.
    br : integer
        The number of block rows (time shifts).
    ordmax : None or integer
        The maximum model order to use in the construction of the 
        stabilisation diagram. None (default) is equivalent to the maximum 
        allowable model order equal to br*data.shape[1].
    lim : tuple
        Limit values to use for the stability requirements of the poles. The 
        first three values are used to check the stability of the poles.
            - Frequency: (f(n)-f(n+1))/f(n) < lim[0] (default to 0.01)
            - Damping: (xi(n)-xi(n+1))/xi(n) < lim[1] (default to 0.05)
            - Mode shape: 1-MAC((phi(n),phi(n+1)) < lim[2] (default to 0.02)
        The last value (lim[3]) is used to remove all the poles that have a
        higher damping ratio (default to 0.1).

    method : "1" or "2"
        Method to use in the estimation of the state matrix A:
            - method "1" (default) : the first method uses the kalman state 
                                     sequence S_(i+1) 
            - method "2" : the second method takes advantages of the shift of
                           the observability matrix
    -------
    Returns
    -------
    fig1 : matplotlib figure
        Stabilisation diagram. 
        Take advantage of the mplcursors module to identify the stable poles.
    Results : dictionary
        Dictionary of results.
        This dictionary will be passed as argument to the SSImodEX() function.
    '''
    
    ndat=data.shape[0] # Number of data points
    nch=data.shape[1] # Number of channel
    
    # If the maximum order is not given (default) it is set as the maximum
    # allowable model order which is: number of block rows * number of channels
    if ordmax == None:
        ordmax = br*nch
        
    freq_max = fs/2 # Nyquist Frequency
    
    # unpack the limits used for the construction of the Stab Diag
    lim_f, lim_s, lim_ms, lim_s1 = lim[0], lim[1], lim[2], lim[3]

    Yy=data.T # 
# =============================================================================
    j=ndat-2*br+1; # Dimension of the Hankel matrix

    H=np.zeros((nch*2*br,j)) # Initialization of the Hankel matrix
    for k in range(0,2*br):
     	H[k*nch:((k+1)*nch),:]=(1/j**0.5)*Yy[:,k:k+j] # calculating Hankel matrix
    
    # LQ factorization of the Hankel matrix
    Q , L = np.linalg.qr(H.T)
    L = L.T
    Q = Q.T
    
    a = nch*br
    b = nch
    
    L21 = L[a:a+b,:a]
    L22 = L[a:a+b,a:a+b]
    L31 = L[a+b:,:a]
    L32 = L[a+b:,a:a+b]
    
    Q1 = Q[:a,:]
    Q2 = Q[a:a+b,:]
    
    P_i = np.vstack((L21,L31)) @ Q1 # Projection Matrix P_i
    P_im1 = np.hstack((L31,L32)) @ np.vstack((Q1, Q2)) # Projection P_(i-1)
    Y_i = np.hstack((L21,L22)) @ np.vstack((Q1, Q2)) # Output sequence
    
    # SINGULAR VALUE DECOMPOSITION
    U1, S1, V1_t = np.linalg.svd(P_i,full_matrices=False)
    S1 = np.diag(S1)
    S1rad=np.sqrt(S1)
    
# =============================================================================
    # initializing arrays
    Fr=np.full((ordmax, int((ordmax)/2+1)), np.nan) # initialization of the matrix that contains the frequencies
    Fr_lab=np.full((ordmax, int((ordmax)/2+1)), np.nan)  # initialization of the matrix that contains the labels of the poles
    Sm=np.full((ordmax, int((ordmax)/2+1)), np.nan) # initialization of the matrix that contains the damping ratios
    Ms = []  # initialization of the matrix (list of arrays) that contains the mode shapes
    for z in range(0, int((ordmax)/2+1)):
        Ms.append(np.zeros((nch, z*(2))))

    # loop for increasing order of the system
    for _ind in range(0, ordmax+1, 2):

        S11 = np.zeros((_ind, _ind)) # Inizializzo
        U11 = np.zeros((br*nch, _ind)) # Inizializzo
        V11 = np.zeros((_ind, br*nch)) # Inizializzo
        O_1 = np.zeros((br*nch - nch, _ind)) # Inizializzo
        O_2 = np.zeros((br*nch - nch, _ind)) # Inizializzo
        
        # Extraction of the submatrices for the increasing order of the system
        S11[:_ind, :_ind] = S1rad[:_ind, :_ind] # 
        U11[:br*nch, :_ind] = U1[:br*nch, :_ind] # 
        V11[:_ind, :br*nch] = V1_t[:_ind, :br*nch] # 

        O = U11 @ S11 # Observability matrix
        S = np.linalg.pinv(O) @ P_i # Kalman filter state sequence

        O_1[:,:] = O[:O.shape[0] - nch,:]
        O_2[:,:] = O[nch:,:]

        # Estimate of the discrete Matrices A and C
        if method == '2': # Method 2 
            A = np.linalg.pinv(O_1) @ O_2 
            C = O[:nch,:]     
            # Ci sarebbero da calcolare le matrici G e R0 

        else:  # Method 1
            Sp1 = np.linalg.pinv(O_1) @ P_im1 # kalman state sequence S_(i+1)
        
            AC = np.vstack((Sp1,Y_i)) @ np.linalg.pinv(S) 
            A = AC[:Sp1.shape[0]]
            C = AC[Sp1.shape[0]:]
            # Ci sarebbero da calcolare le matrici G e R0 

      
        [_AuVal, _AuVett] = np.linalg.eig(A) 
        Lambda =(np.log(_AuVal))*fs 
        fr = abs(Lambda)/(2*np.pi) # Natural frequencies of the system
        smorz = -((np.real(Lambda))/(abs(Lambda))) # damping ratios
# =============================================================================
        # This is a fix for a bug. We make shure that there are not nans
        # (it has, seldom, happened that at the first iteration the first
        # eigenvalue was negative, yielding the log to return a nan that
        # messed up with the plot of the stabilisation diagram)
        for j in range(len(fr)):
            if np.isnan(fr[j]) == True:
                fr[j] = 0
# =============================================================================
        # Output Influence Matrix
        C = O[:nch,:]
        
        # Complex mode shapes
        Mcomp = C@_AuVett
        # Mreal = np.real(C@_AuVett)
        
        # we are increasing 2 orders at each step
        _ind_new = int(_ind/2) 
    
        Fr[:len(fr),_ind_new] = fr # save the frequencies   
        Sm[:len(fr),_ind_new] = smorz # save the damping ratios
        Ms[_ind_new] = Mcomp # save the mode shapes
        
# =============================================================================
        # Check stability of poles
        # 0 = Unstable pole 
        # 1 = Stable for frequency
        # 2 = Stable for frequency and damping
        # 3 = Stable for frequency and mode shape
        # 4 = Stable pole
        
        for idx, (_freq, _smor) in enumerate(zip(fr,smorz)):
            if _ind_new == 0 or _ind_new == 1: # at the first iteration every pole is new
                Fr_lab[:len(fr),_ind_new] = 0 # 
    
            else:
                # Find the index of the pole that minimize the difference with iteration(order) n-1
                ind2 = np.nanargmin(abs(_freq - Fr[:,_ind_new - 1]) 
                                    - min(abs(_freq - Fr[:,_ind_new - 1])))
                        
                Fi_n = Mcomp[:, idx] # Modal shape iteration n
                Fi_nmeno1 = Ms[int(_ind_new-1)][:,ind2] # Modal shape iteration n-1
                
                # aMAC = np.abs(Fi_n@Fi_nmeno1)**2 / ((Fi_n@Fi_n)*(Fi_nmeno1@Fi_nmeno1)) # autoMAC
                aMAC =  MaC(Fi_n,Fi_nmeno1)
                
                cond1 = abs(_freq - Fr[ind2, _ind_new-1])/_freq
                cond2 = abs(_smor - Sm[ind2, _ind_new-1])/_smor
                cond3 = 1 - aMAC
                
                if cond1 < lim_f and cond2 < lim_s and cond3 < lim_ms:
                    Fr_lab[idx,_ind_new] = 4 # STABLE POLE
    
                elif cond1 < lim_f  and cond3 < lim_ms:
                    Fr_lab[idx,_ind_new] = 3 # Stable for freq. and m.shape
    
                elif cond1 < lim_f  and cond2 < lim_s:
                    Fr_lab[idx,_ind_new] = 2 # Stable for freq. and damp.
    
                elif cond1 < lim_f:
                    Fr_lab[idx,_ind_new] = 1 # Stable for freq.
                else:
                    Fr_lab[idx,_ind_new] = 0  # New or unstable pole
# ============================================================================= 
# Stabilisation Diagram
# =============================================================================
# Flatten everything
    _x = Fr.flatten(order='f')
    _y = np.array([_i//len(Fr) for _i in range(len(_x))])
    _l = Fr_lab.flatten(order='f')
    _d = Sm.flatten(order='f')
    # Creating a dataframe out of the flattened results
    df = pd.DataFrame(dict(Frequency=_x, Order=_y, Label=_l, Damp=_d))
    
# =============================================================================
    # Reduced dataframe (without nans) where the modal info is saved
    df1 = df.copy()
    df1 = df1.dropna()
    emme = []
    # here I look for the index of the shape associated to a given pole
    for effe,order in zip(df1.Frequency,df1.Order):
        emme.append(np.nanargmin(abs(effe - Fr[:, order]))) # trovo l'indice 
    # append the list of indexes to the dataframe
    emme = np.array(emme)
    df1['Emme'] = emme
# =============================================================================
    df2 = df1.copy()
    # removing the poles that have damping exceding the limit value
    df2.Frequency = df2.Frequency.where(df2.Damp < lim_s1) 
    # removing the poles that have negative damping 
    df2.Frequency = df2.Frequency.where(df2.Damp > 0)
    
    
    # Physical poles compare in pairs (complex + conjugate) 
    # I look for the poles that DO NOT have a pair and I remove them from the dataframe
    df3 = df2.Frequency.drop_duplicates(keep=False) 
    df2 = df2.where(~(df2.isin(df3))) # 
    df2 = df2.dropna()# Dropping nans
    df2 = df2.drop_duplicates(subset='Frequency') # removing conjugates
    
    
    # df4 = df4.where(df2.Order > ordmin).dropna() # Tengo solo i poli sopra ordmin
    # assigning colours to the labels
    _colors = {0:'Red', 1:'darkorange', 2:'gold', 3:'yellow', 4:'Green'} 
    
    fig1, ax1 = plt.subplots()
    ax1 = sns.scatterplot(x=df2['Frequency'], y=df2['Order']*2, hue=df2['Label'], palette=_colors)
    
    ax1.set_xlim(left=0, right=freq_max)
    ax1.set_ylim(bottom=0, top=ordmax)
    ax1.xaxis.set_major_locator(MultipleLocator(freq_max/10))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax1.xaxis.set_minor_locator(MultipleLocator(freq_max/100))
    ax1.set_title('''{0} - shift: {1}'''.format('Stabilization Diagram', br))
    ax1.set_xlabel('Frequency [Hz]')
    mplcursors.cursor()
    # plt.show()
    
    Results={}
    # if ordmin == None:
    #     ordmin = 0
    Results['Data'] = {'Data': data}
    Results['Data']['Samp. Freq.'] = fs
    Results['Data']['Ord min, max'] = (0, ordmax)
    Results['Data']['Block rows'] = br
    
    Results['All Poles'] = df1
    Results['Reduced Poles'] = df2
    Results['Modes'] = Ms
   
    return fig1, Results


#------------------------------------------------------------------------------


def SSIcovStaDiag(data, fs, br, ordmax=None, lim=(0.01,0.05,0.02,0.1), 
                  method='1'):
    '''
    This function perform the covariance-driven Stochastic sub-Space 
    Identification algorithm.
    
    The function returns the Stabilization Diagram (Plot) for the given
    data. Furthermore it returns a dictionary that contains the results needed
    by the function SSImodEX().
    
    ----------
    Parameters
    ----------
    data : 2D array
        The time history records (N°data points x N°channels).
    fs : float
        The sampling frequency.
    br : integer
        The number of block rows (time shifts).
    ordmax : None or integer
        The maximum model order to use in the construction of the 
        stabilisation diagram. None (default) is equivalent to the maximum 
        allowable model order equal to br*data.shape[1].
    lim : tuple
        Limit values to use for the stability requirements of the poles. The 
        first three values are used to check the stability of the poles.
            - Frequency: (f(n)-f(n+1))/f(n) < lim[0] (default to 0.01)
            - Damping: (xi(n)-xi(n+1))/xi(n) < lim[1] (default to 0.05)
            - Mode shape: 1-MAC((phi(n),phi(n+1)) < lim[2] (default to 0.02)
        The last value (lim[3]) is used to remove all the poles that have a
        higher damping ratio (default to 0.1, N.B. in structural dynamics
        we usually deal with underdamped system)

    method : "1" or "2"
        Method to use in the estimation of the state matrix A:
            - method "1" (default) : the first method takes advantages of the 
                                     shift structrure of the observability
                                     matrix.
            - method "2" : the second method is based on the decomposition 
                           property of the one-lag shifted Toeplitz matrix.
    -------
    Returns
    -------
    fig1 : matplotlib figure
        Stabilisation diagram. 
        Take advantage of the mplcursors module to identify the stable poles.
    Results : dictionary
        Dictionary of results.
        This dictionary will be passed as argument to the SSImodEX() function.
    '''
    
    ndat=data.shape[0] # Number of data points
    nch=data.shape[1] # Number of channel
    
    # If the maximum order is not given (default) it is set as the maximum
    # allowable model order which is: number of block rows * number of channels
    if ordmax == None:
        ordmax = br*nch
        
    freq_max = fs/2 # Nyquist Frequency
    
    # unpack the limits used for the construction of the Stab Diag
    lim_f, lim_s, lim_ms, lim_s1 = lim[0], lim[1], lim[2], lim[3]

    Yy=data.T # 
        
# =============================================================================
    # Calculating R[i] (with i from 0 to 2*br)
    R_is = np.array([1/(ndat - _s)*(Yy[:, : ndat - _s]@Yy[:, _s:].T) for _s in range(br*2+1)]) 
    
    # Assembling the Toepliz matrix
    Tb = np.vstack([np.hstack([R_is[_o,:,:] for _o in range(br+_l, _l, -1)]) for _l in range(br)])
    
    # One-lag shifted Toeplitz matrix (used in "NExT-ERA" method)
    Tb2 = np.vstack([np.hstack([R_is[_o,:,:] for _o in range(br+_l, _l,-1)]) for _l in range(1,br+1)])
    

    # SINGULAR VALUE DECOMPOSITION
    U1, S1, V1_t = np.linalg.svd(Tb)
    S1 = np.diag(S1)
    S1rad=np.sqrt(S1)
    
# =============================================================================
    # initializing arrays
    Fr=np.full((ordmax, int((ordmax)/2+1)), np.nan) # initialization of the matrix that contains the frequencies
    Fr_lab=np.full((ordmax, int((ordmax)/2+1)), np.nan)  # initialization of the matrix that contains the labels of the poles
    Sm=np.full((ordmax, int((ordmax)/2+1)), np.nan) # initialization of the matrix that contains the damping ratios
    Ms = []  # initialization of the matrix (list of arrays) that contains the mode shapes
    for z in range(0, int((ordmax)/2+1)):
        Ms.append(np.zeros((nch, z*(2))))

    # loop for increasing order of the system
    for _ind in range(0, ordmax+1, 2):

        S11 = np.zeros((_ind, _ind)) # Inizializzo
        U11 = np.zeros((br*nch, _ind)) # Inizializzo
        V11 = np.zeros((_ind, br*nch)) # Inizializzo
        O_1 = np.zeros((br*nch - nch, _ind)) # Inizializzo
        O_2 = np.zeros((br*nch - nch, _ind)) # Inizializzo

        # Extraction of the submatrices for the increasing order of the system
        S11[:_ind, :_ind] = S1rad[:_ind, :_ind] # 
        U11[:br*nch, :_ind] = U1[:br*nch, :_ind] # 
        V11[:_ind, :br*nch] = V1_t[:_ind, :br*nch] # 

        O = U11 @ S11 # Observability matrix
        # _GAM = S11 @ V11 # Controllability matrix
        
        O_1[:,:] = O[:O.shape[0] - nch,:]
        O_2[:,:] = O[nch:,:]

        # Estimating matrix A
        if method == '2':
            A = np.linalg.inv(S11)@U11.T@Tb2@V11.T@np.linalg.inv(S11) # Method 2 "NExT-ERA"
        else:
            A = np.linalg.pinv(O_1)@O_2 # Method 1 (BALANCED_REALIZATION)
        
        [_AuVal, _AuVett] = np.linalg.eig(A)
        Lambda =(np.log(_AuVal))*fs 
        fr = abs(Lambda)/(2*np.pi) # natural frequencies
        smorz = -((np.real(Lambda))/(abs(Lambda))) # damping ratios
# =============================================================================
        # This is a fix for a bug. We make shure that there are not nans
        # (it has, seldom, happened that at the first iteration the first
        # eigenvalue was negative, yielding the log to return a nan that
        # messed up with the plot of the stabilisation diagram)
        for _j in range(len(fr)):
            if np.isnan(fr[_j]) == True:
                fr[_j] = 0
# =============================================================================
        # Output Influence Matrix
        C = O[:nch,:]
        
        # Complex mode shapes
        Mcomp = C@_AuVett
        # Mreal = np.real(C@_AuVett)
        
        # we are increasing 2 orders at each step
        _ind_new = int(_ind/2) 
    
        Fr[:len(fr),_ind_new] = fr # save the frequencies   
        Sm[:len(fr),_ind_new] = smorz # save the damping ratios
        Ms[_ind_new] = Mcomp # save the mode shapes
        
# =============================================================================
        # Check stability of poles
        # 0 = Unstable pole labe
        # 1 = Stable for frequency
        # 2 = Stable for frequency and damping
        # 3 = Stable for frequency and mode shape
        # 4 = Stable pole
        
        for idx, (_freq, _smor) in enumerate(zip(fr,smorz)):
            if _ind_new == 0 or _ind_new == 1: # at the first iteration every pole is new
                Fr_lab[:len(fr),_ind_new] = 0 # 
    
            else:
                # Find the index of the pole that minimize the difference with iteration(order) n-1
                ind2 = np.nanargmin(abs(_freq - Fr[:,_ind_new - 1]) 
                                    - min(abs(_freq - Fr[:,_ind_new - 1])))
                        
                Fi_n = Mcomp[:, idx] # Modal shape iteration n
                Fi_nmeno1 = Ms[int(_ind_new-1)][:,ind2] # Modal shape iteration n-1
                
                # aMAC = np.abs(Fi_n@Fi_nmeno1)**2 / ((Fi_n@Fi_n)*(Fi_nmeno1@Fi_nmeno1)) # autoMAC
                aMAC =  MaC(Fi_n,Fi_nmeno1)
                
                cond1 = abs(_freq - Fr[ind2, _ind_new-1])/_freq
                cond2 = abs(_smor - Sm[ind2, _ind_new-1])/_smor
                cond3 = 1 - aMAC
                
                if cond1 < lim_f and cond2 < lim_s and cond3 < lim_ms:
                    Fr_lab[idx,_ind_new] = 4 # 
    
                elif cond1 < lim_f  and cond3 < lim_ms:
                    Fr_lab[idx,_ind_new] = 3 # 
    
                elif cond1 < lim_f  and cond2 < lim_s:
                    Fr_lab[idx,_ind_new] = 2 # 
    
                elif cond1 < lim_f:
                    Fr_lab[idx,_ind_new] = 1 #
                else:
                    Fr_lab[idx,_ind_new] = 0  # Nuovo polo o polo instabile
# ============================================================================= 
# Stabilisation Diagram
# =============================================================================
# Flatten everything
    _x = Fr.flatten(order='f')
    _y = np.array([_i//len(Fr) for _i in range(len(_x))])
    _l = Fr_lab.flatten(order='f')
    _d = Sm.flatten(order='f')
    # Creating a dataframe out of the flattened results
    df = pd.DataFrame(dict(Frequency=_x, Order=_y, Label=_l, Damp=_d))
    
# =============================================================================
    # Reduced dataframe (without nans) where the modal info is saved
    df1 = df.copy()
    df1 = df1.dropna()
    emme = []
    # here I look for the index of the shape associated to a given pole
    for effe,order in zip(df1.Frequency,df1.Order):
        emme.append(np.nanargmin(abs(effe - Fr[:, order]))) # trovo l'indice 
    # append the list of indexes to the dataframe
    emme = np.array(emme)
    df1['Emme'] = emme
# =============================================================================
    df2 = df1.copy()
    # removing the poles that have damping exceding the limit value
    df2.Frequency = df2.Frequency.where(df2.Damp < lim_s1) 
    # removing the poles that have negative damping 
    df2.Frequency = df2.Frequency.where(df2.Damp > 0)
    
    
    # Physical poles compare in pairs (complex + conjugate) 
    # I look for the poles that DO NOT have a pair and I remove them from the dataframe
    df3 = df2.Frequency.drop_duplicates(keep=False) 
    df2 = df2.where(~(df2.isin(df3))) # 
    df2 = df2.dropna()# Dropping nans
    df2 = df2.drop_duplicates(subset='Frequency') # removing conjugates
    
    
    # df4 = df4.where(df2.Order > ordmin).dropna() # Tengo solo i poli sopra ordmin
    # assigning colours to the labels
    _colors = {0:'Red', 1:'darkorange', 2:'gold', 3:'yellow', 4:'Green'} 
    
    fig1, ax1 = plt.subplots()
    ax1 = sns.scatterplot(x=df2['Frequency'], y=df2['Order']*2, hue=df2['Label'], palette=_colors)
    
    ax1.set_xlim(left=0, right=freq_max)
    ax1.set_ylim(bottom=0, top=ordmax)
    ax1.xaxis.set_major_locator(MultipleLocator(freq_max/10))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax1.xaxis.set_minor_locator(MultipleLocator(freq_max/100))
    ax1.set_title('''{0} - shift: {1}'''.format('Stabilization Diagram', br))
    ax1.set_xlabel('Frequency [Hz]')
    mplcursors.cursor()
    # plt.show()
    
    Results={}
    # if ordmin == None:
    #     ordmin = 0
    Results['Data'] = {'Data': data}
    Results['Data']['Samp. Freq.'] = fs
    Results['Data']['Ord min max'] = (0, ordmax)
    Results['Data']['Block rows'] = br
    
    Results['All Poles'] = df1
    Results['Reduced Poles'] = df2
    Results['Modes'] = Ms
   
    return fig1, Results


#------------------------------------------------------------------------------


def SSIModEX(FreQ, Results, deltaf=0.05, aMaClim=0.95):
    '''
    This function extracts the modal properties (frequencies, damping ratios, 
    mode shapes) and returns the results organised in a dictionary.
    
    This function takes as second argument the results from either
    SSIdatStaDia() or SSIcovStaDia() functions.
    
    ----------
    Parameters
    ----------
    FreQ : array (or list)
        Array containing the frequencies, identified from the stabilisation
        diagram, which we want to extract.
    Results : dictionary
        Dictionary of results obtained either from SSIdatStaDia() or from
        SSIcovStaDia().
    deltaf : float
        Tolerance to use when searching for FreQ[i] in the results. 
        Default to 0.05.
    aMaClim : float
        Modal Assurance Criterion limit value. The poles that have a MAC 
        value less than aMaClim are excluded from the calculation of the 
        modal properties.

    -------
    Returns
    -------
    Results : dictionary
        Dictionary containing the modal properties (frequencies, damping
        ratios, mode shapes) of the system.
    '''
    
    df2 = Results['Reduced Poles']
    Ms = Results['Modes'] 
    
    Freq = []
    Damp = []
    Fi = []
    for _x in FreQ:
        xmeno1, xpiu1 = _x-deltaf, _x+deltaf # tolerance limit 
        # saving only the poles whithin the limts
        df3 = df2.copy
        df3 = df2.where((df2.Frequency < xpiu1) & (df2.Frequency > xmeno1))
        df3 = df3.dropna()
        
        npoli = len(df3['Frequency'].values) # number of poles
        
        AutoMacche = np.zeros((npoli, npoli),dtype=complex) # initialization
        # Looping throug the extracted poles to calculate the autoMAC
        # matrix (between the poles)
        for b in range(npoli): # first loop
            zuno = int(df3['Order'].values[b]) # index 1 of the mode shape
            fiuno = Ms[zuno][:,int(df3['Emme'].values[b])] # shape 1
            for k in range(npoli): # secondo loop
                zdue = int(df3['Order'].values[k]) # index 1 of the mode shape 
                fidue = Ms[zdue][:,int(df3['Emme'].values[k])] # shape 2
                    
                AutoMacche[b, k] = MaC(fiuno,fidue) # MaC between every pole
        # I look for the pole that have the highest sum of macs
        SAmaC = np.sum(AutoMacche, axis=1) #adding up every value on a column
        idxmax = np.argmax(SAmaC) # 
        
        # Index 1 of reference shape
        MSrefidx1 = int(df3['Order'].values[idxmax])
        # Index 2 of reference shape
        MSrefidx2 = int(df3['Emme'].values[idxmax])
        firef = Ms[MSrefidx1][:,MSrefidx2] # Reference shape
        
        idmax = np.argmax(abs(firef))
        firef = firef/firef[idmax] # normalised (unity displacement)


        # keeping only the poles that have MAC > MAClim value
        AMaC = AutoMacche[idxmax]
        df3['AMaC'] = AMaC
        df3 = df3.where(df3.AMaC > aMaClim)
        df3 = df3.dropna()
        
        FrMean = df3.Frequency.mean() # Mean frequency
        # FrStd = df3.Frequency.std() # dev.std.
        DampMean = df3.Damp.mean() # Mean damping
        # DampStd = df3.Damp.std() # dev.std. 
        
        Freq.append(FrMean)
        Damp.append(DampMean)
        Fi.append(firef)
    
    Freq = np.array(Freq)
    Damp = np.array(Damp)
    Fi = np.array(Fi)
    
    Results={}
    Results['Frequencies'] = Freq
    Results['Damping'] = Damp
    Results['Mode Shapes'] = Fi.T
        
    return Results


#------------------------------------------------------------------------------
    

def FDDsvp(data, fs, df=0.01, pov=0.5, window='hann'):
    '''
    This function perform the Frequency Domain Decomposition algorithm.
    
    The function return the plot of the singular values of the power spectral
    density. The cross power spectral density is estimated using 
    scipy.signal.csd() function, which in turn is based on Welch's method.
    Furthermore it returns a dictionary that contains the results needed
    by the function FDDmodEX().
    ----------
    Parameters
    ----------
    data : array
        The time history records (Ndata x Nchannels).
    fs : float
        The sampling frequency.
    df : float
        Desired frequency resolution. Default to 0.01 (Hz).
    pov : float
        Percentage of overlap between segments. Default to 50%.
    window : str or tuple or array_like
        Desired window to use. Window is passed to scipy.signal's get_window
        function (see SciPy.org for more info). Default to "hann" which stands
        for a “Hanning” window.

    -------
    Returns
    -------
    fig1 : matplotlib figure
        Plot of the singular values of the power spectral matrix.
    Results : dictionary
        Dictionary of results to be passed to FDDmodEX()
    '''  
    
    # ndat=data.shape[0] # Number of data points
    nch=data.shape[1] # Number of channels
    freq_max = fs/2 # Nyquist frequency
    nxseg = fs/df # number of point per segments
#    nseg = ndat // nxseg # number of segments
    noverlap = nxseg // (1/pov) # Number of overlapping points
    
    # Initialization
    PSD_matr = np.zeros((nch, nch, int((nxseg)/2+1)), dtype=complex) 
    S_val = np.zeros((nch, nch, int((nxseg)/2+1))) 
    S_vec = np.zeros((nch, nch, int((nxseg)/2+1)), dtype=complex) 
    
    # Calculating Auto e Cross-Spectral Density
    for _i in range(0, nch):
        for _j in range(0, nch):
            _f, _Pxy = signal.csd(data[:, _i],data[:, _j], fs=fs, nperseg=nxseg, noverlap=noverlap, window=window)
            PSD_matr[_i, _j, :] = _Pxy
            
    # Singular value decomposition     
    for _i in range(np.shape(PSD_matr)[2]):
        U1, S1, _V1_t = np.linalg.svd(PSD_matr[:,:,_i])
        U1_1=np.transpose(U1) 
        S1 = np.diag(S1)
        S_val[:,:,_i] = S1
        S_vec[:,:,_i] = U1_1
    
    # Plot of the singular values in log scale
    fig, ax = plt.subplots()
    for _i in range(nch):
    #    ax.semilogy(_f, S_val[_i, _i]) # scala log
        ax.plot(_f[:], 10*np.log10(S_val[_i, _i])) # decibel
    ax.grid()
    ax.set_xlim(left=0, right=freq_max)
    ax.xaxis.set_major_locator(MultipleLocator(freq_max/10))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.xaxis.set_minor_locator(MultipleLocator(freq_max/100))
    ax.set_title("Singular values plot - (Freq. res. ={0})".format(df))
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'dB $[g^2/Hz]$')    
    # ax.set_ylabel(r'dB $\left[\frac{\left(\frac{m}{s^2}\right)^2}{Hz}\right]$')    
    mplcursors.cursor()
    
    Results={}
    Results['Data'] = {'Data': data}
    Results['Data']['Samp. Freq.'] = fs
    Results['Data']['Freq. Resol.'] = df
    Results['Singular Values'] = S_val
    Results['Singular Vectors'] = S_vec
    Results['PSD Matrix'] = PSD_matr
    Results['f'] = _f
    
    return fig, Results


#------------------------------------------------------------------------------
    

def FDDmodEX(FreQ, Results, ndf=2):
    '''
    This function returns the modal parameters estimated according to the
    Frequency Domain Decomposition method.
    
    ----------
    Parameters
    ----------
    FreQ : array (or list)
        Array containing the frequencies, identified from the singular values
        plot, which we want to extract.
    Results : dictionary
        Dictionary of results obtained from FDDsvp().
    ndf : float
        Number of spectral lines in the proximity of FreQ[i] where the peak
        is searched.

    -------
    Returns
    -------
    fig1 : matplotlib figure
        Stabilisation diagram ...
    Results : dictionary
        Dictionary of results ...
    '''
    
#    data = Results['Data']['Data']
    fs = Results['Data']['Samp. Freq.']
    df = Results['Data']['Freq. Resol.']
    S_val = Results['Singular Values']
    S_vec = Results['Singular Vectors']
    deltaf=ndf*df
#    ndat=data.shape[0] #
#    nch=data.shape[1] #
    freq_max = fs/2 # Nyquist
#    nxseg = fs/df # 

    f = np.linspace(0, int(freq_max), int(freq_max*(1/df)+1)) # spectral lines
 
    Freq = []
    index = []
    Fi = []

    for _x in FreQ:
#        idx = np.argmin(abs(f-_x)) 
        lim = (_x - deltaf, _x + deltaf) # frequency bandwidth where the peak is searched
        idxlim = (np.argmin(abs(f-lim[0])), np.argmin(abs(f-lim[1])))
        # ratios between the first and second singular value 
        diffS1S2 = S_val[0,0,idxlim[0]:idxlim[1]]/S_val[1,1,idxlim[0]:idxlim[1]]
        maxDiffS1S2 = np.max(diffS1S2) # looking for the maximum difference
        idx1 = np.argmin(abs(diffS1S2 - maxDiffS1S2))
        idxfin = idxlim[0] + idx1 
# =============================================================================
        # Modal properties
        fr_FDD = f[idxfin] # Frequency
        fi_FDD = S_vec[0,:,idxfin] # Mode shape
        idx3 = np.argmax(abs(fi_FDD))
        fi_FDDn = fi_FDD/fi_FDD[idx3] # normalised (unity displacement)
        fiFDDn = np.array(fi_FDDn)
        
        Freq.append(fr_FDD)
        Fi.append(fiFDDn)
        index.append(idxfin)
        
    Freq = np.array(Freq)
    Fi = np.array(Fi)
    index = np.array(index)   
        
    Results={}
    Results['Frequencies'] = Freq
    Results['Mode Shapes'] = Fi.T
    Results['Freq. index'] = index

    return Results


#------------------------------------------------------------------------------


def EFDDmodEX(FreQ, Results, ndf=2, MAClim=0.85, sppk=3, npmax=20,
              method='FSDD', plot=False):
    '''
    This function returns the modal parameters estimated according to the
    enhanced version of the Frequency Domain Decomposition method.
    
    ----------
    Parameters
    ----------
    FreQ : array (or list)
        Array containing the frequencies, identified from the singular values
        plot, which we want to extract.
    Results : dictionary
        Dictionary of results obtained from FDDsvp().
    ndf : float
        Number of spectral lines in the proximity of FreQ[i] where the peak
        is searched.
    MAClim : float
        MAC rejection level for the extraction of the SDOF bell function.
    sppk : integer
        Number of peaks to skip at the beginning of the autocorrelation 
        function when calculating the damping ratio (through the fit on the 
        log decrement)
    npmax : integer
        Number of (consecutive) points to use in the autocorrelation function
        calculating the damping ratio (through the fit on the log decrement)
    method : "EFDD" or "FSDD"
        Method used to extract the SDOF bell function. Default to "FSDD", uses 
        the Frequency Spatial Domain Decomposition algorithm. Method "EFDD" 
        uses the classical Enhanced Frequency Domain Decomposition algorithm.
    plot : True or False
        Wether to plot or not the results. Defalut to False.
    -------
    Returns
    -------
    fig1 : matplotlib figure
        Stabilisation diagram ...
    Results : dictionary
        Dictionary of results ...
    '''
    
    data = Results['Data']['Data']
    fs = Results['Data']['Samp. Freq.']
    df = Results['Data']['Freq. Resol.']
    S_val = Results['Singular Values']
    S_vec = Results['Singular Vectors']
    PSD_matr = Results['PSD Matrix']*2
    
    Res = FDDmodEX(FreQ, Results, ndf=ndf)
    Freq, Fi, index = Res['Frequencies'], Res['Mode Shapes'], Res['Freq. index']
    
#    ndat=data.shape[0] #
    nch=data.shape[1] # Number of channels
    freq_max = fs/2 # Nyquist frequency
#    nxseg = fs/df # number of points per segment
#    nseg = ndat // nxseg # number of segments
#    dur=ndat/fs # [sec] Period of the acquisition
    tlag = 1/df # time lag
    Nf = freq_max/df+1 # number of spectral lines
    f = np.linspace(0, int(freq_max), int(Nf)) # all spectral lines
    
    nIFFT = (int(Nf))*20 # number of points for the inverse transform (zeropadding)
    
    # Normlization (unity disp) of al shapes
    S_vec_n = S_vec.copy()
    for _n in range(nch):
        Fis = S_vec[_n,:,:]
        idxFimax = np.array([np.argmax(abs(Fis[:,_i])) for _i in range(int(Nf))])
        S_vec_n[_n,:,:] = np.array([Fis[:,_j]/Fis[idxFimax[_j],_j] for _j in range(int(Nf))]).T
    
    Freq_E = []
    Fi_E = []
    Damp_E = []
    
    # looping through all frequencies to estimate
    for _l,_f in enumerate(Freq):
        # Initialization of the arrays
        _fi=Fi[:,_l]
        SDOFsval = np.zeros(int(Nf)) # 
        SDOFsval1 = np.zeros(int(Nf),dtype=complex)
        SDOFsvec = np.zeros((nch,int(Nf)),dtype=complex)
        
        # collecting the singular values (and associated singular vectors) that
        # constitute the representative SDOF bell function
        _p = 0
        # I'm looping through the spectral lines ABOVE the reference freq. and
        # checking if the MAC (with respect to the reference) satisfy 
        # MAC > MAClim
        # (N.B. this part of code should probably be improved!)
        while MaC(_fi, S_vec_n[0,:, index[_l] + _p]) > MAClim: # here I check the FIRST singular vector
                 # collecting the singular values for EFDD
                SDOFsval[(index[_l] + _p)] = S_val[0,0, index[_l] + _p]
                # collecting singular values and applying FSDD
                SDOFsval1[(index[_l] + _p)] = _fi.conj().T@PSD_matr[:,:, index[_l] + _p]@_fi 
                # and the singular vectors
                SDOFsvec[:,(index[_l] + _p)] = S_vec_n[0,:, index[_l] + _p] 
                _p +=1
        while MaC(_fi, S_vec_n[1,:, index[_l] + _p]) > MAClim: # here I check the SECOND singular vector
                SDOFsval[(index[_l] + _p)] = S_val[1,1, index[_l] + _p] # si aggiungono i relativi Singular values alla SFOF bell
                SDOFsval1[(index[_l] + _p)] = _fi.conj().T@PSD_matr[:,:, index[_l] + _p]@_fi # si aggiungono i relativi Singular values alla SFOF bell
                SDOFsvec[:,(index[_l] + _p)] = S_vec_n[1,:, index[_l] + _p] # e anche i relativi Singular vectors alla SFOF bell
                _p +=1
                
        _p = 0
        # Here I do the same as before only for the value BELOW the reference freq.
        while MaC(_fi, S_vec_n[0,:, index[_l] - _p]) > MAClim:
                SDOFsval[(index[_l] - _p)] = S_val[0,0, index[_l] - _p] # si aggiungono i relativi Singular values alla SFOF bell
                SDOFsval1[(index[_l] - _p)] = _fi.conj().T@PSD_matr[:,:, index[_l] - _p]@_fi # si aggiungono i relativi Singular values alla SFOF bell
                SDOFsvec[:,(index[_l] - _p)] = S_vec_n[0,:, index[_l] - _p] # e anche i relativi Singular vectors alla SFOF bell
                _p +=1     
        while MaC(_fi, S_vec_n[1,:, index[_l] - _p]) > MAClim:
                SDOFsval[(index[_l] - _p)] = S_val[1,1, index[_l] - _p] # si aggiungono i relativi Singular values alla SFOF bell
                SDOFsval1[(index[_l] - _p)] = _fi.conj().T@PSD_matr[:,:, index[_l] - _p]@_fi # si aggiungono i relativi Singular values alla SFOF bell
                SDOFsvec[:,(index[_l] - _p)] = S_vec_n[1,:, index[_l] - _p] # e anche i relativi Singular vectors alla SFOF bell
                _p +=1  
        
        # Checking which method to use
        if method == 'EFDD':
            SDOFsvalF = SDOFsval.copy()
        elif method == 'FSDD':
            SDOFsvalF = SDOFsval1.copy()
        
        # indices of the singular values in SDOFsval       
        idSV = np.array(np.where(SDOFsvalF)).T
        fsval = f[idSV]
        
        # Mode shapes (singular vectors) associated to each singular values
        # and weighted with respect to the singular value itself
        FIs = [SDOFsvalF[idSV[_u]]*SDOFsvec[:,idSV[_u]] for _u in range(len(idSV))]
        FIs = np.squeeze(np.array(FIs))
        
        # mean and std
        # stdFIs = np.std(FIs.real,axis=0)
        meanFi = np.mean(FIs,axis=0)
 
        _idx6 = np.argmax(abs(meanFi))
        # Normalised mode shape (unity disp)
#        meanFireal = meanFi.real/meanFi[_idx6].real 
        meanFi = meanFi/meanFi[_idx6] 
        
        
        # Autocorrelation function (Free Decay)
        SDOFcorr = np.fft.ifft(SDOFsvalF,n=nIFFT,axis=0,norm='ortho').real 
        timeLag = np.linspace(0,tlag,len(SDOFcorr)) # t
        
        # NORMALISED AUTOCORRELATION
        idxmax = np.argmax(SDOFcorr)
        normSDOFcorr = SDOFcorr[:len(SDOFcorr)//2]/SDOFcorr[idxmax]
   
        # finding where x = 0
        sgn = np.sign(normSDOFcorr).real # finding the sign
        sgn1 = np.diff(sgn,axis=0) # finding where the sign changes (intersept with x=0)
        zc1 = np.where(sgn1)[0] # Zero crossing indices
    
        # finding maximums and minimums (peacks) of the autoccorelation
        maxSDOFcorr = [np.max(normSDOFcorr[zc1[_i]:zc1[_i+2]]) for _i in range(0,len(zc1)-2,2)]
        minSDOFcorr = [np.min(normSDOFcorr[zc1[_i]:zc1[_i+2]]) for _i in range(0,len(zc1)-2,2)]
        if len(maxSDOFcorr) > len(minSDOFcorr):
            maxSDOFcorr = maxSDOFcorr[:-1]
        elif len(maxSDOFcorr) < len(minSDOFcorr):
            minSDOFcorr = minSDOFcorr[:-1]
        minmax = np.array((minSDOFcorr, maxSDOFcorr))
        minmax = np.ravel(minmax, order='F')
        
        # finding the indices of the peacks
        maxSDOFcorr_idx = [np.argmin(abs(normSDOFcorr-maxx)) for maxx in maxSDOFcorr]
        minSDOFcorr_idx = [np.argmin(abs(normSDOFcorr-minn)) for minn in minSDOFcorr]
        minmax_idx = np.array((minSDOFcorr_idx, maxSDOFcorr_idx))
        minmax_idx = np.ravel(minmax_idx, order='F')
        
        # Peacks and indices of the peacks to be used in the fitting
        minmax_fit = np.array([minmax[_a] for _a in range(sppk,sppk+npmax)])
        minmax_fit_idx = np.array([minmax_idx[_a] for _a in range(sppk,sppk+npmax)])
        
        # estimating the natural frequency from the distance between the peaks
        Td = np.diff(timeLag[minmax_fit_idx])*2 # *2 because we use both max and min
        Td_EFDD = np.mean(Td)
        
        fd_EFDD = 1/Td_EFDD # damped natural frequency
        
        # Log decrement 
        delta = np.array([2*np.log(np.abs(minmax[0])/np.abs(minmax[_i])) for _i in range(len(minmax_fit))])
            
        # Fit
        _fit = lambda x,m:m*x
        m, _ = curve_fit(_fit, np.arange(len(minmax_fit)), delta)
        
        # damping ratio
        xi_EFDD = m/np.sqrt(4*np.pi**2 + m**2)
        fn_EFDD = fd_EFDD/np.sqrt(1-xi_EFDD**2)
    
    
        # Finally appending the results to the returned dictionary
        Freq_E.append(fn_EFDD)
        Damp_E.append(xi_EFDD)
        Fi_E.append(meanFi)
        
#------------------------------------------------------------------------------
# If the plot option is activated we return the following plots    
        # build a rectangle in axes coords
        left, width = .25, .5
        bottom, height = .25, .5
        right = left + width
        top = bottom + height
        # axes coordinates are 0,0 is bottom left and 1,1 is upper right

        if plot == True:
            # PLOT 1 - Plotting the SDOF bell function extracted
            _fig, ((_ax1,_ax2),(_ax3,_ax4)) = plt.subplots(nrows=2,ncols=2)
            _ax1.plot(f, 10*np.log10(S_val[0,0]), c='b')
            _ax1.plot(fsval, 10*np.log10(SDOFsvalF[idSV].real), c='r',label='SDOF bell')
            _ax1.set_title("SDOF Bell function")
            _ax1.set_xlabel('Frequency [Hz]')
            _ax1.set_ylabel(r'dB $[V^2/Hz]$')
            _ax1.legend()
            
            # Plot 2
            _ax2.plot(timeLag[:len(SDOFcorr)//2], normSDOFcorr)
            _ax2.set_title("Auto-correlation Function")
            _ax2.set_xlabel('Time lag[s]')
            _ax2.set_ylabel('Normalized correlation') 

            # PLOT 3 (PORTION for FIT)
            _ax3.plot(timeLag[:minmax_fit_idx[-1]], normSDOFcorr[:minmax_fit_idx[-1]])
            _ax3.scatter(timeLag[minmax_fit_idx], normSDOFcorr[minmax_fit_idx])
            _ax3.set_title("Portion for fit")
            _ax3.set_xlabel('Time lag[s]')
            _ax3.set_ylabel('Normalized correlation')  
            
            # PLOT 4 (FIT)
            _ax4.scatter(np.arange(len(minmax_fit)), delta)
            _ax4.plot(np.arange(len(minmax_fit)), m*np.arange(len(minmax_fit)))
     
            _ax4.text(left, top, r'''$f_n$ = %.3f
$\xi$ = %.2f%s'''% (fn_EFDD, float(xi_EFDD)*100,"%"),transform=_ax4.transAxes)
     
            _ax4.set_title("Fit - Frequency and Damping")
            _ax4.set_xlabel(r'counter $k^{th}$ extreme')
            _ax4.set_ylabel(r'$2ln\left(r_0/|r_k|\right)$')    
    
            plt.tight_layout()
#------------------------------------------------------------------------------   
    
    Freq = np.array(Freq_E)
    Damp = np.array(Damp_E)
    Fi = np.array(Fi_E)

    Results={}
    Results['Frequencies'] = Freq
    Results['Damping'] = Damp
    Results['Mode Shapes'] = Fi.T

    return Results


