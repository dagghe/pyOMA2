import os
import glob
import numpy as np
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

    def __init__(self, Nsetup = 1):
        
            """
            :param Nsetup: number of setup (default to 1) 

            """
            try:
                self.Nsetup = int(Nsetup)
            except:
                raise Exception('Nsetup must be integer')
            if self.Nsetup < 1:
                raise Exception('Nsetup must be more than 1')
            elif self.Nsetup > 1:
                # self.data_all = np.zeros((self.Ndat, self.Nch, self.Nsetup))
                self.data_all = []

                self.Ndat_all = np.zeros(self.Nsetup)
                self.Nch_all = np.zeros(self.Nsetup)
                self.samp_freq_all = np.zeros(self.Nsetup)
                self.freq_max_all = np.zeros(self.Nsetup)
                self.sup_num = []
                self.ref_ch_all = []

            self.Results = {}


    def add_data(self, exp_data, fs, sup_num=1, ref_ch=None):
        """
        Bla bla bla

        """
        # if isinstance(data, self):
            


        if self.Nsetup > 1:
            try:
                sup_num <= self.Nsetup
            except:
                raise Exception(f"sup_number cannot be greater than {self.Nsetup}")
            
            try:
                ref_ch != None
            except:
                raise Exception("You must specify the reference channels in a multisetup (list of index)")
            
            # self.data_all[:,:, sup_num]
            self.data_all.append(exp_data)
            
            self.Ndat_all[sup_num] = exp_data.shape[0]
            self.Nch_all[sup_num] = exp_data.shape[1]
            self.samp_freq_all[sup_num] = fs
            self.freq_max_all[sup_num] = fs / 2  # Nyquist frequency
            self.ref_ch_all.append(ref_ch) 
            self.sup_num.append(sup_num)
            
            # if len(self.sup_num) == self.Nsetup:
            #     pass

        else:
            self.Ndat = exp_data.shape[0]
            self.Nch = exp_data.shape[1]
            self.data = exp_data
            self.samp_freq = fs
            self.freq_max = self.samp_freq / 2  # Nyquist frequency

# -----------------------------------------------------------------------------
# Algorithms / Methods
# -----------------------------------------------------------------------------

    def FDDsvp(self, df=0.01, pov=0.5, window="hann", simnum=0):
        """
        Bla bla bla

        """
        self.sim_num = simnum

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


        self.Results[f"FDD_{simnum}"] = {
            "df": df,
            "freqs": _f,
            "S_val": S_val,
            "S_vec": S_vec,
            "PSD_matr": PSD_matr,
        }


    def FDDsvp_multi(self, df=0.01, pov=0.5, window="hann"):
        """
        Bla bla bla

        """
        
        for ii in self.sup_num:

            self.data = self.data_all[ii]
            self.Ndat = self.data.shape[0]
            self.Nch = self.data.shape[1]
            self.samp_freq = self.samp_freq_all[ii]
            self.freq_max = self.freq_max_all[ii]
            
            # run sim on data
            self.FDDsvp(df=df, pov=pov, window=window, simnum=ii)


#------------------------------------------------------------------------------

    def FDDmodEX(self, ndf=5):
        """
        Bla bla bla

        """
        simnum = self.sim_num
        df = self.Results[f"FDD_{simnum}"]["df"]
        S_val = self.Results[f"FDD_{simnum}"]["S_val"]
        S_vec = self.Results[f"FDD_{simnum}"]["S_vec"]
        deltaf = ndf * df
        freqs = self.Results[f"FDD_{simnum}"]["freqs"] 
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
        
        self.Results[f"FDD_{simnum}"]["Fn"] = Freq
        self.Results[f"FDD_{simnum}"]["Phi"] = Fi.T
        self.Results[f"FDD_{simnum}"]["Freq ind"] = index

#------------------------------------------------------------------------------

    def EFDDmodEX(self, ndf=5, cm=1 , MAClim=0.85, sppk=3, npmax=30,
                  method='FSDD', plot=False):
        '''
        Bla bla bla
        '''
        simnum = self.sim_num
        df = self.Results[f"FDD_{simnum}"]["df"]
        S_val = self.Results[f"FDD_{simnum}"]["S_val"]
        S_vec = self.Results[f"FDD_{simnum}"]["S_vec"]
        PSD_matr = self.Results[f"FDD_{simnum}"]["PSD_matr"]
        freqs = self.Results[f"FDD_{simnum}"]["freqs"]
        # Run FDD to get a first estimate of the modal properties
        self.FDDmodEX(ndf=ndf)
        Freq, Fi = self.Results[f"FDD_{simnum}"]["Fn"], self.Results[f"FDD_{simnum}"]["Phi"]

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

        for n in range(len(Freq)): # looping through all frequencies to estimate
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
                                        else np.zeros(len(Freq)) 
                                        for _l in range(int(Nf))]) 
                # Classical Enhanced Frequency Domain Decomposition method
                else:
                    SDOFbell += np.array([S_val[csm, csm, _l]
                                        if tools.MAC(_fi, S_vec[csm,:,_l]) > MAClim 
                                        else 0 
                                        for _l in range(int(Nf) )])
                    SDOFms += np.array([ S_vec[csm,:,_l]
                                        if tools.MAC(_fi, S_vec[csm,:,_l]) > MAClim 
                                        else np.zeros(len(Freq)) 
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

        self.Results[f"EFDD_{simnum}"] = {"Method": method}
        self.Results[f"EFDD_{simnum}"]['Fn'] = Freq
        self.Results[f"EFDD_{simnum}"]['Phi'] = Fi.T
        self.Results[f"EFDD_{simnum}"]['xi'] = Damp
        self.Results[f"EFDD_{simnum}"]['Figs'] = Figs

#------------------------------------------------------------------------------

    def SSIcov(self, br, ordmin=0, ordmax=None, method='1', simnum=0):
        '''
        Bla bla bla
        '''
        self.sim_num = simnum
        
        try:
            self.br = int(br)
        except:
            pass

        # If the maximum order is not given (default) it is set as the maximum
        # allowable model order which is: number of block rows * number of channels
        if ordmax == None:
            self.SSI_ordmax = self.br*self.Nch
        else:
            self.SSI_ordmax = ordmax
        self.SSI_ordmin = ordmin
        
        Yy=self.data.T # 

        # Calculating R[i] (with i from 0 to 2*br)
        R_is = np.array( [ 1/(self.Ndat - _s) * (Yy[:, : self.Ndat - _s] @ \
                           Yy[:, _s:].T) for _s in range(br*2+1)]) 
        
        # Assembling the Toepliz matrix
        Tb = np.vstack([np.hstack([R_is[_o,:,:] for _o in range(br+_l, _l,-1)]) for _l in range(br)])
        
        # One-lag shifted Toeplitz matrix (used in "NExT-ERA" method)
        Tb2 = np.vstack([np.hstack([R_is[_o,:,:] for _o in range(br+_l,_l,-1)]) for _l in range(1,br+1)])

        # SINGULAR VALUE DECOMPOSITION
        U1, S1, V1_t = np.linalg.svd(Tb)
        S1 = np.diag(S1)
        S1rad=np.sqrt(S1)


        # initializing arrays
        Fr=np.full((self.SSI_ordmax, int((self.SSI_ordmax)/2+1)), np.nan) # initialization of the matrix that contains the frequencies
        Sm=np.full((self.SSI_ordmax, int((self.SSI_ordmax)/2+1)), np.nan) # initialization of the matrix that contains the damping ratios
        Ms=np.full((self.SSI_ordmax, int((self.SSI_ordmax)/2+1), self.Nch), np.nan, dtype=complex) # initialization of the matrix that contains the damping ratios

        # loop for increasing order of the system
        for _ind in range(self.SSI_ordmin, self.SSI_ordmax+1, 2):

            S11 = np.zeros((_ind, _ind)) # Inizializzo
            U11 = np.zeros((br*self.Nch, _ind)) # Inizializzo
            V11 = np.zeros((_ind, br*self.Nch)) # Inizializzo
            O_1 = np.zeros((br*self.Nch - self.Nch, _ind)) # Inizializzo
            O_2 = np.zeros((br*self.Nch - self.Nch, _ind)) # Inizializzo

            # Extraction of the submatrices for the increasing order of the system
            S11[:_ind, :_ind] = S1rad[:_ind, :_ind] # 
            U11[:br*self.Nch, :_ind] = U1[:br*self.Nch, :_ind] # 
            V11[:_ind, :br*self.Nch] = V1_t[:_ind, :br*self.Nch] # 

            O = U11 @ S11 # Observability matrix
            # _GAM = S11 @ V11 # Controllability matrix
            
            O_1[:,:] = O[:O.shape[0] - self.Nch,:]
            O_2[:,:] = O[self.Nch:,:]

            # Estimating matrix A
            if method == '2':
                A = np.linalg.inv(S11)@U11.T@Tb2@V11.T@np.linalg.inv(S11) # Method 2 "NExT-ERA"
            else:
                A = np.linalg.pinv(O_1)@O_2 # Method 1 (BALANCED_REALIZATION)

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
            # Output Influence Matrix
            C = O[:self.Nch,:]

            # Complex mode shapes
            Mcomp = C @ _AuVett
            Mcomp = np.array([ Mcomp[:, ii]/Mcomp[np.argmax(abs(Mcomp[:, ii])), ii] 
                     for ii in range(Mcomp.shape[1])]).reshape(-1, self.Nch)

            Mreal = np.real(C@_AuVett)
            for ii in range(Mreal.shape[1]):
                idmax = np.argmax(abs(Mreal[:, ii]))
                Mreal[:, ii] = Mreal[:, ii]/Mreal[idmax, ii] # normalised (unity displacement)

            # we are increasing 2 orders at each step
            _ind_new = int((_ind-self.SSI_ordmin)/2) 

            Fr[:len(fr),_ind_new] = fr # save the frequencies
            Sm[:len(fr),_ind_new] = smorz # save the damping ratios
            Ms[:len(fr), _ind_new, :] = Mcomp


        self.Results[f"SSIcov_{simnum}"] = {"Method": method}
        self.Results[f"SSIcov_{simnum}"]['Fn_poles'] = Fr
        self.Results[f"SSIcov_{simnum}"]['xi_poles'] = Sm
        self.Results[f"SSIcov_{simnum}"]['Phi_poles'] = Ms
#------------------------------------------------------------------------------

    def pLSCF(self, ordmax, df=0.01, pov=0.5, window="hann", simnum=0):
        '''
        Bla bla bla
        '''
        
        self.sim_num = simnum
        
        # PSD
        nxseg = self.samp_freq / df  # number of point per segments
        #    nseg = self.Ndat // nxseg # number of segments
        noverlap = nxseg // (1 / pov)  # Number of overlapping points
        Nf = int((nxseg) / 2 + 1) # Number of frequency lines

        # Calculating Auto e Cross-Spectral Density
        _f, PSD_matr = signal.csd(
            self.data.T.reshape(self.Nch, 1, self.Ndat),
            self.data.T.reshape(1, self.Nch, self.Ndat),
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
        Fr = np.full((ordmax*self.Nch, ordmax), np.nan) # initialise
        Sm = np.full((ordmax*self.Nch, ordmax), np.nan) # initialise
        Ws = np.full((ordmax*self.Nch, ordmax), np.nan) # initialise

        # Calculation of companion matrix A and modal parameters for each order
        for j in range(1, ordmax+1): # loop for increasing model order

            M = np.zeros(((j+1)*self.Nch, (j+1)*self.Nch)) # inizializzo

            I0 = np.array([np.exp(1j*freq*dt*jj) for jj in range(j+1)]).T

            I0h = I0.conj().T # Calculate complex transpose

            R0 = np.real(I0h @ I0) # 4.163

            for o in range(0, self.Nch): # loop on channels

                Y0 = np.array([np.kron(-I0[kk, :], Sy[kk, o, :]) for kk in range(Nf)])

                S0 = np.real(I0h @ Y0) # 4.164
                T0 = np.real(Y0.conj().T @ Y0) # 4.165

                M += 2*(T0 - (S0.T @ np.linalg.solve(R0, S0))) # 4.167

            alfa = np.linalg.solve(-M[: j*self.Nch, : j*self.Nch], M[: j*self.Nch, j*self.Nch: (j+1)*self.Nch]) # 4.169
            alfa = np.vstack((alfa, np.eye(self.Nch)))

            # beta0 = np.linalg.solve(-R0, S0@alfa)

            # Companion matrix 
            A = np.zeros((j*self.Nch, j*self.Nch));
            for ii in range(j):
                Aj = alfa[ii*self.Nch: (ii+1)*self.Nch, :]
                A[(j-1)*self.Nch: , ii*self.Nch: (ii+1)*self.Nch] = -Aj.T
            A[: (j-1)*self.Nch, self.Nch: j*self.Nch] = np.eye(((j-1)*self.Nch));

            # Eigenvalueproblem
            [my, My] = eig(A);
            # my = np.diag(My);
            lambd = np.log(my)/dt # From discrete-time to continuous time 4.136
            # lambd1 = lambd.copy()
            # replace with nan every value with negative real part (should be the other way around!)
            lambd = np.where(np.real(lambd)<0, np.nan, lambd)

            Fr[:j*self.Nch, j-1] = abs(lambd)/(2*np.pi) # Natural frequencies (Hz) 4.137
            Sm[:j*self.Nch, j-1] = ((np.real(lambd))/abs(lambd)) # Damping ratio initial calc 4.139
            Ws[:j*self.Nch, j-1] = abs(lambd) # Natural frequencies (rad/s)

        self.Results[f"pLSCF_{simnum}"] = {"ordmax": ordmax}
        self.Results[f"pLSCF_{simnum}"]['Fn_poles'] = Fr
        self.Results[f"pLSCF_{simnum}"]['xi_poles'] = Sm
        self.Results[f"pLSCF_{simnum}"]['w_pole'] = Ws

#------------------------------------------------------------------------------

    def SSImodEX(self):
        """
        Bla bla bla
        """
        simnum = self.sim_num
        
        FreQ = self.sel_freq
        XI = self.sel_xi
        Fi = np.array(self.sel_phi)
        
        # Save in dictionary of results
        self.Results[f"SSIcov_{simnum}"]['Fn'] = FreQ
        self.Results[f"SSIcov_{simnum}"]['Phi'] = Fi.T
        self.Results[f"SSIcov_{simnum}"]['xi'] = XI
    

#------------------------------------------------------------------------------

    def sel_peak_FDD(self, freqlim=None, ndf=5):
        """
        Bla bla bla
        """
        _ = Sel_from_plot(self, freqlim=freqlim, plot="FDD")
        self.FDDmodEX(ndf)


    def sel_peak_EFDD(self, freqlim=None, ndf=5, cm=1 , MAClim=0.85, sppk=3, 
                      npmax=30, method='FSDD', plot=False):
        """
        Bla bla bla
        """
        _ = Sel_from_plot(self, freqlim=freqlim, plot="FDD")
        self.EFDDmodEX(ndf, cm, MAClim, sppk, npmax, method, plot)


    def sel_pole_SSIcov(self, freqlim=None, ordmin=0, ordmax=None, method='1'):
        """
        Bla bla bla
        """
        self.FDDsvp()
        _ = Sel_from_plot(self, freqlim=freqlim, plot="SSI")
        self.SSImodEX()


    def sel_pole_pLSCF(self, freqlim=None):
        """
        Bla bla bla
        """
        self.FDDsvp()
        _ = Sel_from_plot(self, freqlim=freqlim, plot="pLSCF")
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

