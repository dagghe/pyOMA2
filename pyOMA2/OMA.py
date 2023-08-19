import os
import glob
import numpy as np
from numpy.linalg import pinv
from scipy import signal
from scipy.optimize import curve_fit
from scipy.linalg import eig
import matplotlib.pyplot as plt


from . import tools
from . import Sel_from_plot

# =============================================================================
# pyOMA classes
# =============================================================================
class Model():
    """
    Model.
    """

    def __init__(self, data, fs):
        
            """
            Bla bla bla

            """

            try:
                self.data = np.asarray(data)
            except:
                raise Exception("Cannot convert data into a numpy array")

            try:
                self.fs = float(fs)
            except:
                raise Exception("Cannot convert sampling frequency into a float")

            self.Ndat = data.shape[0]
            self.Nch = data.shape[1]
            self.data = data
            self.samp_freq = fs
            self.freq_max = self.samp_freq / 2  # Nyquist frequency
            self.Results = {}

# -----------------------------------------------------------------------------
# Algorithms / Methods
# -----------------------------------------------------------------------------

    def FDDsvp(self, df=0.01, pov=0.5, window="hann"):
        """
        Bla bla bla

        """
        nxseg = self.samp_freq / df  # number of points per segment
        noverlap = int(nxseg * pov)  # Number of overlapping points
        Nf = int(nxseg / 2) + 1  # Number of frequency lines

        # Calculating Auto e Cross-Spectral Density
        _f, PSD_matr = signal.csd(
            self.data.T.reshape(self.Nch, 1, self.Ndat),
            self.data.T.reshape(1, self.Nch, self.Ndat),
            fs=self.samp_freq,
            nperseg=nxseg,
            noverlap=noverlap,
            window=window,
        )

        # Singular Value loop
        S_val = np.empty((self.Nch, self.Nch, Nf))
        S_vec = np.empty((self.Nch, self.Nch, Nf), dtype=complex)
        for i in range(Nf):
            U1, S1, V1_t = np.linalg.svd(PSD_matr[:, :, i])
            U1_1 = U1.T
            S_val[:, :, i] = np.sqrt(np.diag(S1))
            S_vec[:, :, i] = U1_1


        self.Results["FDD"] = {
            "df": df,
            "freqs": _f,
            "S_val": S_val,
            "S_vec": S_vec,
            "PSD_matr": PSD_matr,
        }

#------------------------------------------------------------------------------

    def FDDmodEX(self, ndf=5):
        """
        Bla bla bla

        """
        df = self.Results["FDD"]["df"]
        S_val = self.Results["FDD"]["S_val"]
        S_vec = self.Results["FDD"]["S_vec"]
        deltaf = ndf * df
        freqs = self.Results["FDD"]["freqs"] 
        sel_freq = self.sel_freq
        
        Freq = []
        Fi = []
        index = []
        
        for freq in sel_freq:
            lim = (freq - deltaf, freq + deltaf)  # Frequency bandwidth where the peak is searched
            idxlim = (np.argmin(np.abs(freqs - lim[0])), 
                      np.argmin(np.abs(freqs - lim[1])))  # Indices of the limits
            
            # Ratios between the first and second singular value 
            diffS1S2 = S_val[0, 0, idxlim[0]:idxlim[1]] / \
                       S_val[1, 1, idxlim[0]:idxlim[1]]
            maxDiffS1S2 = np.max(diffS1S2)  # Looking for the maximum difference
            idx1 = np.argmin(np.abs(diffS1S2 - maxDiffS1S2))  # Index of the max diff
            idxfin = idxlim[0] + idx1  # Final index
            
            # Modal properties
            fr_FDD = freqs[idxfin]  # Frequency
            fi_FDD = S_vec[0, :, idxfin]  # Mode shape
            fi_FDDn = fi_FDD / fi_FDD[np.argmax(np.abs(fi_FDD))]  # Normalized (unity displacement)
            
            Freq.append(fr_FDD)
            Fi.append(fi_FDDn)
            index.append(idxfin)
            
        Freq = np.array(Freq)
        Fi = np.array(Fi)
        index = np.array(index)
        
        self.Results["FDD"]["Fn"] = Freq
        self.Results["FDD"]["Phi"] = Fi.T
        self.Results["FDD"]["Freq ind"] = index

#------------------------------------------------------------------------------

    def EFDDmodEX(self, ndf=5, cm=1 , MAClim=0.85, sppk=3, npmax=30,
                  method='FSDD', plot=False):
        '''
        Bla bla bla
        '''
        sel_freq = self.sel_freq
        df = self.Results["FDD"]["df"]
        S_val = self.Results["FDD"]["S_val"]
        S_vec = self.Results["FDD"]["S_vec"]
        PSD_matr = self.Results["FDD"]["PSD_matr"]
        freqs = self.Results["FDD"]["freqs"]
        # Run FDD to get a first estimate of the modal properties
        self.FDDmodEX(ndf=ndf)
        Freq, Fi = self.Results["FDD"]["Fn"], self.Results["FDD"]["Phi"]

        freq_max = self.freq_max
        tlag = 1/df # time lag
        Nf = freq_max/df+1 # number of spectral lines
        # f = np.linspace(0, int(freq_max), int(Nf)) # all spectral lines

        nIFFT = (int(Nf))*20 # number of points for the inverse transform (zeropadding)

        # Initialize Results
        Freq_E = []
        Fi_E = []
        Damp_E = []
        Figs = []

        for n in range(len(sel_freq)): # looping through all frequencies to estimate
            _fi = Fi[: , n] # Select reference mode shape (from FDD)
            # Initialise SDOF bell and Mode Shape
            SDOFbell = np.zeros(int(Nf), dtype=complex) # 
            SDOFms = np.zeros((int(Nf), self.Nch), dtype=complex)

            for csm in range(cm):# Loop through close mode (if any, default 1)
                # Frequency Spatial Domain Decomposition variation (defaulf)
                if method == "FSDD": 
                    # Save values that satisfy MAC > MAClim condition
                    SDOFbell += np.array([_fi.conj().T@PSD_matr[:,:, _l]@_fi # Enhanced PSD matrix (frequency filtered)
                                        if tools.MAC(_fi, S_vec[csm,:,_l]) > MAClim 
                                        else 0 
                                        for _l in range(int(Nf))])
                    # Do the same for mode shapes
                    SDOFms += np.array([ S_vec[csm,:,_l]
                                        if tools.MAC(_fi, S_vec[csm,:,_l]) > MAClim 
                                        else np.zeros(self.Nch) 
                                        for _l in range(int(Nf))]) 
                # Classical Enhanced Frequency Domain Decomposition method
                else:
                    SDOFbell += np.array([S_val[csm, csm, _l]
                                        if tools.MAC(_fi, S_vec[csm,:,_l]) > MAClim 
                                        else 0 
                                        for _l in range(int(Nf) )])
                    SDOFms += np.array([ S_vec[csm,:,_l]
                                        if tools.MAC(_fi, S_vec[csm,:,_l]) > MAClim 
                                        else np.zeros(self.Nch) 
                                        for _l in range(int(Nf))])

            # indices of the singular values in SDOFsval
            idSV = np.array(np.where(SDOFbell)).T
            fsval = freqs[idSV]

            # Mode shapes (singular vectors) associated to each singular values
            # and weighted with respect to the singular value itself
            FIs = [ SDOFbell[idSV[u]] * SDOFms[idSV[u],:] 
                   for u in range(len(idSV)) ]
            FIs = np.squeeze(np.array(FIs))

            meanFi = np.mean(FIs,axis=0)

            # Normalised mode shape (unity disp)
            meanFi = meanFi/meanFi[np.argmax(abs(meanFi))] 

            # Autocorrelation function (Free Decay)
            SDOFcorr1 = np.fft.ifft(SDOFbell,n=nIFFT,axis=0,norm='ortho').real 
            timeLag = np.linspace(0,tlag,len(SDOFcorr1)) # t

            # NORMALISED AUTOCORRELATION
            idxmax = np.argmax(SDOFcorr1)
            normSDOFcorr = SDOFcorr1[:len(SDOFcorr1)//2]/SDOFcorr1[idxmax]

            # finding where x = 0
            sgn = np.sign(normSDOFcorr).real # finding the sign
            sgn1 = np.diff(sgn,axis=0) # finding where the sign changes (intersept with x=0)
            zc1 = np.where(sgn1)[0] # Zero crossing indices

            # finding maximums and minimums (peaks) of the autoccorelation
            maxSDOFcorr = [np.max(normSDOFcorr[zc1[_i]:zc1[_i+2]]) 
                           for _i in range(0,len(zc1)-2,2)]
            minSDOFcorr = [np.min(normSDOFcorr[zc1[_i]:zc1[_i+2]]) 
                           for _i in range(0,len(zc1)-2,2)]
            if len(maxSDOFcorr) > len(minSDOFcorr):
                maxSDOFcorr = maxSDOFcorr[:-1]
            elif len(maxSDOFcorr) < len(minSDOFcorr):
                minSDOFcorr = minSDOFcorr[:-1]
            minmax = np.array((minSDOFcorr, maxSDOFcorr))
            minmax = np.ravel(minmax, order='F')

            # finding the indices of the peaks
            maxSDOFcorr_idx = [np.argmin(abs(normSDOFcorr-maxx)) 
                               for maxx in maxSDOFcorr]
            minSDOFcorr_idx = [np.argmin(abs(normSDOFcorr-minn)) 
                               for minn in minSDOFcorr]
            minmax_idx = np.array((minSDOFcorr_idx, maxSDOFcorr_idx))
            minmax_idx = np.ravel(minmax_idx, order='F')

            # Peacks and indices of the peaks to be used in the fitting
            minmax_fit = np.array([minmax[_a] 
                                   for _a in range(sppk,sppk+npmax)])
            minmax_fit_idx = np.array([minmax_idx[_a] 
                                       for _a in range(sppk,sppk+npmax)])

            # estimating the natural frequency from the distance between the peaks
            Td = np.diff(timeLag[minmax_fit_idx])*2 # *2 because we use both max and min
            Td_EFDD = np.mean(Td)

            fd_EFDD = 1/Td_EFDD # damped natural frequency

            # Log decrement 
            delta = np.array([2*np.log(np.abs(minmax[0])/np.abs(minmax[ii])) 
                              for ii in range(len(minmax_fit))])

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


            # If the plot option is activated we return the following plots
            # build a rectangle in axes coords
            left, width = .25, .5
            bottom, height = .25, .5
            right = left + width
            top = bottom + height
            # axes coordinates are 0,0 is bottom left and 1,1 is upper right

            if plot:
                # PLOT 1 - Plotting the SDOF bell function extracted
                _fig, ((_ax1,_ax2),(_ax3,_ax4)) = plt.subplots(nrows=2,ncols=2)
                _ax1.plot(freqs, 10*np.log10(S_val[0,0]), c='b')
                _ax1.plot(fsval, 10*np.log10(SDOFbell[idSV].real), c='r',
                          label='SDOF bell')
                _ax1.set_title("SDOF Bell function")
                _ax1.set_xlabel('Frequency [Hz]')
                _ax1.set_ylabel(r'dB $[V^2/Hz]$')
                _ax1.legend()

                # Plot 2
                _ax2.plot(timeLag[:len(SDOFcorr1)//2], normSDOFcorr)
                _ax2.set_title("Auto-correlation Function")
                _ax2.set_xlabel('Time lag[s]')
                _ax2.set_ylabel('Normalized correlation') 

                # PLOT 3 (PORTION for FIT)
                _ax3.plot(timeLag[:minmax_fit_idx[-1]], 
                          normSDOFcorr[:minmax_fit_idx[-1]])
                _ax3.scatter(timeLag[minmax_fit_idx],
                             normSDOFcorr[minmax_fit_idx])
                _ax3.set_title("Portion for fit")
                _ax3.set_xlabel('Time lag[s]')
                _ax3.set_ylabel('Normalized correlation')  

                # PLOT 4 (FIT)
                _ax4.scatter(np.arange(len(minmax_fit)), delta)
                _ax4.plot(np.arange(len(minmax_fit)),
                          m*np.arange(len(minmax_fit)))

                _ax4.text(left, top, r'''$f_n$ = %.3f
                $\xi$ = %.2f%s'''% (fn_EFDD, float(xi_EFDD)*100,"%"),
                transform=_ax4.transAxes)

                _ax4.set_title("Fit - Frequency and Damping")
                _ax4.set_xlabel(r'counter $k^{th}$ extreme')
                _ax4.set_ylabel(r'$2ln\left(r_0/|r_k|\right)$')

                plt.tight_layout()
                Figs.append(_fig)

        Freq = np.array(Freq_E)
        Damp = np.array(Damp_E)
        Fi = np.array(Fi_E)

        self.Results["EFDD"] = {"Method": method}
        self.Results["EFDD"]['Fn'] = Freq
        self.Results["EFDD"]['Phi'] = Fi.T
        self.Results["EFDD"]['xi'] = Damp
        self.Results["EFDD"]['Figs'] = Figs

#------------------------------------------------------------------------------

    def SSIcov(self, br, ordmin=0, ordmax=None, method='1'):
        '''
        Bla bla bla
        '''
        Nch = self.Nch
        
        try:
            self.br = int(br)
        except:
            pass

        # If the maximum order is not given (default) it is set as the maximum
        # allowable model order which is: number of block rows * number of channels
        if ordmax == None:
            self.SSI_ordmax = self.br*Nch
        else:
            self.SSI_ordmax = ordmax
        self.SSI_ordmin = ordmin

        Yy=self.data.T # 

        # Calculating R[i] (with i from 0 to 2*br)
        R_is = np.array( [ 1/(self.Ndat - _s) * (Yy[:, : self.Ndat - _s] @ \
                           Yy[:, _s:].T) for _s in range(br*2+1)]) 

        # Assembling the Toepliz matrix
        Tb = np.vstack([ np.hstack(
                                    [ R_is[_o,:,:] 
                                     for _o in range(br+_l, _l,-1)]
                                    )
                        for _l in range(br)])

        if method == "2":
            # One-lag shifted Toeplitz matrix (used in "NExT-ERA" method)
            Tb2 = np.vstack([ np.hstack(
                                        [ R_is[_o,:,:] 
                                         for _o in range(br + _l, _l, -1) ]
                                        ) 
                             for _l in range(1, br+1)])

        # SINGULAR VALUE DECOMPOSITION
        U1, S1, V1_t = np.linalg.svd(Tb)
        S1 = np.diag(S1)
        S1rad=np.sqrt(S1)

        # initializing arrays
        Fr=np.full((self.SSI_ordmax, int((self.SSI_ordmax)/2+1)), np.nan) # initialization of the matrix that contains the frequencies
        Sm=np.full((self.SSI_ordmax, int((self.SSI_ordmax)/2+1)), np.nan) # initialization of the matrix that contains the damping ratios
        Ms=np.full((self.SSI_ordmax, int((self.SSI_ordmax)/2+1), Nch), np.nan, 
                   dtype=complex) # initialization of the matrix that contains the damping ratios

        # loop for increasing order of the system
        for ii in range(self.SSI_ordmin, self.SSI_ordmax+1, 2):
            O = U1[:br*Nch, :ii] @ S1rad[:ii, :ii] # Observability matrix
            # _GAM = S11 @ V11 # Controllability matrix

            # Estimating matrix A
            if method == '2':# Method 2 "NExT-ERA"
                A = np.linalg.inv(S1rad[:ii, :ii]) @ U1[:br*Nch, :ii].T @ \
                Tb2 @ V1_t[:ii, :br*Nch].T @ np.linalg.inv(S1rad[:ii, :ii]) 
            else:# Method 1 (BALANCED_REALIZATION)
                A = pinv(O[:O.shape[0] - Nch,:]) @ O[Nch:,:] 

            # Output Influence Matrix
            C = O[:Nch,:]

            [_AuVal, _AuVett] = np.linalg.eig(A)
            Lambda =(np.log(_AuVal))*self.samp_freq
            fr = abs(Lambda)/(2*np.pi) # natural frequencies
            smorz = -((np.real(Lambda))/(abs(Lambda))) # damping ratios
            # --------------
            # This is a fix for a bug. We make shure that there are not nans
            # (it has, seldom, happened that at the first iteration the first
            # eigenvalue was negative, yielding the log to return a nan that
            # messed up with the plot of the stabilisation diagram)
            for _j in range(len(fr)):
                if np.isnan(fr[_j]) == True:
                    fr[_j] = 0
            # --------------

            # Complex mode shapes
            Mcomp = C @ _AuVett
            # normalised (unity displacement)
            Mcomp = np.array([ Mcomp[:, ii]/Mcomp[np.argmax(abs(Mcomp[:, ii])), ii] 
                     for ii in range(Mcomp.shape[1])]).reshape(-1, Nch)

            # we are increasing 2 orders at each step
            _ind_new = int((ii-self.SSI_ordmin)/2) 

            Fr[:len(fr),_ind_new] = fr # save the frequencies
            Sm[:len(fr),_ind_new] = smorz # save the damping ratios
            Ms[:len(fr), _ind_new, :] = Mcomp


        self.Results["SSIcov"] = {"Method": method}
        self.Results["SSIcov"]['Fn_poles'] = Fr
        self.Results["SSIcov"]['xi_poles'] = Sm
        self.Results["SSIcov"]['Phi_poles'] = Ms
        
#------------------------------------------------------------------------------

    def SSImodEX(self):
        """
        Bla bla bla
        """
        FreQ = self.sel_freq
        XI = self.sel_xi
        Fi = np.array(self.sel_phi)
        
        # Save in dictionary of results
        self.Results["SSIcov"]['Fn'] = FreQ
        self.Results["SSIcov"]['Phi'] = Fi.T
        self.Results["SSIcov"]['xi'] = XI

#------------------------------------------------------------------------------

    def pLSCF(self, ordmax, df=0.01, pov=0.5, window="hann"):
        '''
        Bla bla bla
        '''
        Nch = self.Nch
        # PSD
        nxseg = self.samp_freq / df  # number of point per segments
        #    nseg = self.Ndat // nxseg # number of segments
        noverlap = nxseg // (1 / pov)  # Number of overlapping points
        Nf = int((nxseg) / 2 + 1) # Number of frequency lines

        # Calculating Auto e Cross-Spectral Density
        _f, PSD_matr = signal.csd(
            self.data.T.reshape(Nch, 1, self.Ndat),
            self.data.T.reshape(1, Nch, self.Ndat),
            fs=self.samp_freq,
            nperseg=nxseg,
            noverlap=noverlap,
            window=window,
        )

        # p-LSCF - METODO CON MATRICI REALI
        self.pLSCF_ordmax = ordmax
        freq = 2*np.pi*_f

        # The PSD matrix should be in the format (k, o, o) where:
        # k=1,2,...Nf; and o=1,2...l
        Sy = np.copy(PSD_matr.T)

        dt = 1/self.samp_freq
        Fr = np.full((ordmax*Nch, ordmax), np.nan) # initialise
        Sm = np.full((ordmax*Nch, ordmax), np.nan) # initialise
        Ws = np.full((ordmax*Nch, ordmax), np.nan) # initialise

        # Calculation of companion matrix A and modal parameters for each order
        for j in range(1, ordmax+1): # loop for increasing model order
            M = np.zeros(((j+1)*Nch, (j+1)*Nch)) # inizializzo
            I0 = np.array([np.exp(1j*freq*dt*jj) for jj in range(j+1)]).T
            I0h = I0.conj().T # Calculate complex transpose
            R0 = np.real(I0h @ I0) # 4.163

            for o in range(0, Nch): # loop on channels
                Y0 = np.array([np.kron(-I0[kk, :], Sy[kk, o, :]) for kk in range(Nf)])
                S0 = np.real(I0h @ Y0) # 4.164
                T0 = np.real(Y0.conj().T @ Y0) # 4.165
                M += 2*(T0 - (S0.T @ np.linalg.solve(R0, S0))) # 4.167

            alfa = np.linalg.solve(-M[: j*Nch, : j*Nch], M[: j*Nch, j*Nch: (j+1)*Nch]) # 4.169
            alfa = np.vstack((alfa, np.eye(Nch)))

            # beta0 = np.linalg.solve(-R0, S0@alfa)

            # Companion matrix 
            A = np.zeros((j*Nch, j*Nch));
            for ii in range(j):
                Aj = alfa[ii*Nch: (ii+1)*Nch, :]
                A[(j-1)*Nch: , ii*Nch: (ii+1)*Nch] = -Aj.T
            A[: (j-1)*Nch, Nch: j*Nch] = np.eye(((j-1)*Nch));

            # Eigenvalueproblem
            [my, My] = eig(A);
            lambd = np.log(my)/dt # From discrete-time to continuous time 4.136

            # replace with nan every value with negative real part (should be the other way around!)
            lambd = np.where(np.real(lambd)<0, np.nan, lambd)

            Fr[:j*Nch, j-1] = abs(lambd)/(2*np.pi) # Natural frequencies (Hz) 4.137
            Sm[:j*Nch, j-1] = ((np.real(lambd))/abs(lambd)) # Damping ratio initial calc 4.139
            
        self.ordmax = ordmax
        self.lambd = lambd
        self.Sy = Sy
        self.freqs = _f
        self.Results["pLSCF"] = {"ordmax": ordmax}
        self.Results["pLSCF"]['Fn_poles'] = Fr
        self.Results["pLSCF"]['xi_poles'] = Sm


#------------------------------------------------------------------------------

    def pLSCFmodEx(self):
        '''
        Bla bla bla
        '''
        Fr = self.Results["pLSCF"]['Fn_poles']
        Sm = self.Results["pLSCF"]['xi_poles']
        Lab = tools._stab_pLSCF(Fr, Sm, self.ordmax, 0.01, 0.05, self.Nch)
        
        # Frstab = np.where(Lab == 3, Fr, np.nan)
        lambd = self.lambd
        Nch = self.Nch
        w = self.Results["pLSCF"]['Fn_poles']*(2*np.pi)
        
        for j in range(1, self.ordmax+1): # loop for increasing model order
            # FORME MODALI
            lambd = lambd[~np.isnan(lambd)] # elimino da lambda tutti i nan
            w = w[~np.isnan(w)] # stessa cosa per omega

            Nm = len(lambd) # numero modi 
            LL = np.zeros((Nch*Nm, Nch*Nm), dtype=complex) # inizializzo
            GL = np.zeros((Nch*Nm, Nch), dtype=complex) # inizializzo
            # trovo l'indice della linea spettrale piu vicina alla frequenza del polo selezionato
            for ww in w: # loop su poli fisici del sistema
                idx_w = np.argmin(np.abs(self.freqs-ww)) # trovo indice

                # loop sulle linee di frequenza intorno al polo fisico (+20 e -20)
                for kk in range(idx_w-20, idx_w+20):

                    GL += np.array(
                        [ self.Sy[kk, :, :]/(1j*self.freqs[kk]-lambd[jj]) for jj in range(Nm)]
                                    ).reshape(-1, Nch)

                    LL += np.array([
                          np.array([np.eye(Nch)/((1j*self.freqs[kk]-lambd[jj1])*(1j*self.freqs[kk]-lambd[jj2])) 
                                    for jj2 in range(Nm)]).reshape((Nch*Nm, Nch),order="c").T
                                    for jj1 in range(Nm)]).reshape((Nch*Nm, Nch*Nm))

                R = np.linalg.solve(LL, GL) # matrice dei residui (fi@fi^T)

                for jj in range(Nm): 
                    # SVD della matrice dei residui per ciascun modo fisico del sistema
                    U, S, VT = np.linalg.svd(R[jj*Nch: (jj+1)*Nch, :])

                    phi = U[: , 0] # la forma modale è la prima colonna di U

                    idmax = np.argmax(abs(phi))
                    phiN = phi/phi[idmax] # normalised (unity displacement)

                    # Fi[jj*l: (jj+1)*l, j-1]= np.real(phiN)




#------------------------------------------------------------------------------

    def sel_peak_FDD(self, freqlim=None, ndf=5):
        """
        Bla bla bla
        """
        _ = Sel_from_plot.SelFromPlot(self, freqlim=freqlim, plot="FDD")
        self.FDDmodEX(ndf)


    def sel_peak_EFDD(self, freqlim=None, ndf=5, cm=1 , MAClim=0.85, sppk=3, 
                      npmax=30, method='FSDD', plot=False):
        """
        Bla bla bla
        """
        _ = Sel_from_plot.SelFromPlot(self, freqlim=freqlim, plot="FDD")
        self.EFDDmodEX(ndf, cm, MAClim, sppk, npmax, method, plot)


    def sel_pole_SSIcov(self, freqlim=None, ordmin=0, ordmax=None, method='1'):
        """
        Bla bla bla
        """
        self.FDDsvp()
        _ = Sel_from_plot.SelFromPlot(self, freqlim=freqlim, plot="SSI")
        self.SSImodEX()


    def sel_pole_pLSCF(self, freqlim=None):
        """
        Bla bla bla
        """
        self.FDDsvp()
        _ = Sel_from_plot.SelFromPlot(self, freqlim=freqlim, plot="pLSCF")
        # self.pLSCFmodEx()


# =============================================================================
# DA TESTARE
    def get_mod_FDD(self, sel_freq, ndf=5):
        """
        Bla bla bla
        """
        self.sel_freq = sel_freq
        self.FDDmodEX(ndf)


    def get_mod_EFDD(self, sel_freq, ndf=5, cm=1 , MAClim=0.85, sppk=3, 
                     npmax=30, method='FSDD', plot=False):
        """
        Bla bla bla
        """
        self.sel_freq = sel_freq
        self.EFDDmodEX(ndf, cm, MAClim, sppk, npmax, method, plot)

