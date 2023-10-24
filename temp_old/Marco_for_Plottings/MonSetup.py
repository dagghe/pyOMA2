import numpy as np
import pandas as pd
import re
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import PyOMA as oma
import pickle
import mplcursors

from utils import *

class MonSetup:
    """
    This object type stores data for a certain monitored setup and contains all the methods to provide the operational modal analysis from PyOMA package ver 1.4 (see https://github.com/dagghe/PyOMA).

    When instantiated, every column of the monitored vibration data table in input is associated to the indicated node and degree of freedom (dof), following the wireframe model node numbering.
    """
    def __init__(self, data , setup_layout: list, fs: float) -> None:
        """
        ----------
        Parameters
        ----------
        data : array (1D or 2D)
            Monitored vibration data table.
        setup_layout : list of strings
            Monitored dofs and node numbering indication.
            Example:
            setup_layout = ['37x','37y','38x','38y','30x']
        fs : float
            Sampling frequency in Hz.
        -------
        """
        self.fs = fs
        self._original_fs = fs
        self.q = 0 # initialize decimation factor. 0 or 1 means no decimation at all

        if isinstance(data, str):
            try:
                self.data = import_data(data).astype('float64')
            except:
                raise TypeError("Data argument type not recognized. Please provide data path as string to data file to be imported (Supported file format .txt, .csv, .pkl), or directly the numpy data array.")
        else:
            try:
                self.data = data
            except:
                raise TypeError("Data argument type not recognized. Please provide data path as string to data file to be imported (Supported file format .txt, .csv, .pkl), or directly the numpy data array.")
        self._original_imported_data = np.copy(self.data)


        self.mon_nodes_dofs = np.zeros((4,len(setup_layout)))
        for ii, tmp in enumerate(setup_layout):
            self.mon_nodes_dofs[0,ii] = int( re.findall(r'[0-9]+', tmp )[0] )
            dof = re.findall(r'[a-zA-Z]+', tmp )[0].lower()
            if dof == 'x':
                self.mon_nodes_dofs[1,ii] = 1
            elif dof == 'y':
                self.mon_nodes_dofs[2,ii] = 1
            elif dof == 'z':
                self.mon_nodes_dofs[3,ii] = 1
            else:
                raise TypeError("Unrecognized setup_layout. Please provide setup_layout as a list of string of the form NodeNumber-MonitoredDOF, e.g. ['1x','1y','1z'] means a triaxial sensor placed in node 1.")

    def detrend(self):
        '''
        This method performs a detrending procedure for every column of the stored data table.
        '''
        self.data = signal.detrend(self.data, axis=0) # Trend removal
    
    def decimate(self, q : int):
        '''
        This method performs a decimation procedure for every column of the stored data table according to scipy.signal.decimate command (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html).

        ----------
        Parameters
        ----------
        q : integer
            Decimation factor. If q<1 no decimation is performed.
        '''
        if q < 1: # q : decimation factor
            print('If you want to decimate data, please provide a decimation factor greater than 1.')
        else:
            try:
                self.data = signal.decimate(self.data,  q, ftype='fir', axis=0)
                self.fs = self.fs/q # [Hz] Decimated sampling frequency
                self.q = q
            except:
                raise ValueError(f"Error raised with this decimation factor {q}. Please, provide an integer factor greater than 1 if you want to decimate.")


    def plot_signals(self, save_to_file_path : str = '', axis_labels : list = ['Time [s]','Amplitude [m/s]'], **kwargs):
        """
        This method generates three plots stored in a list for the monitored vibration data signals according to the x, y, and z dofs separately.

        ----------
        Parameters
        ----------
        save_to_file_path : string
            If provided, the plots are stored to a file in the indicated path both as a raster (png) and vectorial (pdf) format.
        axis_labels : list
            Define labels of x and y axis respectively.
            Example: axis_labels = ['Time [s]','Amplitude [m/s]']
        **kwargs : dictionary
            Dictionary of keyword arguments to customize the plot style
        -------
        Returns
        -------
        figures : list of three matplotlib figures
        """
        palette = sns.color_palette("Spectral", self.mon_nodes_dofs.shape[1]).as_hex()
        figures =[]
        for ii,dof in enumerate(['x','y','z']):
            if any(self.mon_nodes_dofs[ii+1,:]):
                fig, ax = plt.subplots()
                custom_legend_elements = []
                for jj in range(self.mon_nodes_dofs.shape[1]):
                    if self.mon_nodes_dofs[ii+1,jj] == 1:
                        custom_legend_elements.append( Line2D([0], [0], color=palette[jj], label=f'Sensor {self.mon_nodes_dofs[0,jj]:.0f} {dof}', **kwargs) )
                        ax.plot( np.linspace(0, self.data.shape[0]/self.fs, self.data.shape[0]) , self.data[:,jj], color=palette[jj], **kwargs)
                ax.legend(handles=custom_legend_elements, loc='best')
                ax.set_xlabel(axis_labels[0]); ax.set_ylabel(axis_labels[1])
                plt.tight_layout()
                figures.append(fig)
                if len(save_to_file_path):
                    plt.savefig(save_to_file_path + os.sep + f'Monitored_signals_{dof}'+'.pdf')
                    plt.savefig(save_to_file_path + os.sep + f'Monitored_signals_{dof}'+'.png')
        return figures

    def svd_psd(self, save_to_file_path : str = '', **kwargs):
        """
        This method computes the singular value decomposition (SVD) of the power spectral density (PSD) according to PyOMA package ver 1.4.

        ----------
        Parameters
        ----------
        save_to_file_path : string
            If provided, the plots are stored to a file in the indicated path both as a raster (png) and vectorial (pdf) format.
        **kwargs : dictionary
            Dictionary of keyword arguments to set different setting for the FDDsvp function of PyOMA package.
        -------
        Returns
        -------
        fig : matplotlib figure
        """
        fig , self.fdd = oma.FDDsvp(self.data,  self.fs, **kwargs)
        if len(save_to_file_path):
            fig.tight_layout()
            pickle.dump(fig, open(save_to_file_path + os.sep + f'SVD_PSD_decimation_{self.q:d}.pickle', 'wb') )
            fig.savefig(save_to_file_path + os.sep + f'SVD_PSD_decimation_{self.q:d}'+'.pdf')
            fig.savefig(save_to_file_path + os.sep + f'SVD_PSD_decimation_{self.q:d}'+'.png')
            with open(save_to_file_path + os.sep + f'FDD_decimation_{self.q:d}.pkl', 'wb') as file:
                pickle.dump(self.fdd, file)
        return fig

    def runssicov(self, save_to_file_path : str = '', br : int = 10 ,  **kwargs):
        """
        This method computes the covariance-based variant of the stochastic subspace identification (SSIcov) according to PyOMA package ver 1.4.

        ----------
        Parameters
        ----------
        save_to_file_path : string
            If provided, the plots are stored to a file in the indicated path both as a raster (png) and vectorial (pdf) format.
        br : int
            Block row parameters, aka time shift parameter of SSI.
        **kwargs : dictionary
            Dictionary of keyword arguments to set different setting for the SSIcovStaDiag function of PyOMA package.
        -------
        Returns
        -------
        fig : matplotlib figure
            Stabilisation diagram. 
            Take advantage of the mplcursors module to identify the stable poles.
        """
        self.br = br
        fig , self.ssicov = oma.SSIcovStaDiag(self.data,  self.fs, self.br, **kwargs)
        if len(save_to_file_path):
            fig.tight_layout()
            pickle.dump(fig, open(save_to_file_path + os.sep + f'StabDiag_decimation_{self.q:d}_br_{self.br:d}.pickle', 'wb') )
            fig.savefig(save_to_file_path + os.sep + f'StabDiag_decimation_{self.q:d}_br_{self.br:d}'+'.pdf')
            fig.savefig(save_to_file_path + os.sep + f'StabDiag_decimation_{self.q:d}_br_{self.br:d}'+'.png')
            with open(save_to_file_path + os.sep + f'SSIcov_decimation_{self.q:d}_br_{self.br:d}.pkl', 'wb') as file:
                pickle.dump(self.ssicov, file)
        return fig

    def overlap_stabdiag_svpsd(self, save_to_file_path : str = '', selection_criteria:int = 4, num_sv : int = 3 ):
        """
        This method returns an overlapped graph of the selected poles from the stabilization diagram and the SVD of the PSD to aid the user for the peak-picking procedure.

        ----------
        Parameters
        ----------
        save_to_file_path : string
            If provided, the plots are stored to a file in the indicated path both as a raster (png) and vectorial (pdf) format.
        selection_criteria : int
            Select poles according to their stability checks. Admissible values are 0,1,2,3,4 :
            0 = Unstable pole label
            1 = Stable for frequency
            2 = Stable for frequency and damping
            3 = Stable for frequency and mode shape
            4 = Stable pole
        num_sv : int
            Number of SV the user want to overlap to the stabilization diagram. By default num_sv=3, meaning that the first three SV are plotted.
        -------
        Returns
        -------
        fig : matplotlib figure
            Stabilisation diagram with SVD of PSD overlapped. 
            Take advantage of the mplcursors module to identify the stable poles.
        """
        _colors = {0:'Red', 1:'darkorange', 2:'gold', 3:'yellow', 4:'Green'} 
        palette = list(sns.color_palette("Greys", num_sv).as_hex())[::-1]

        AllPoles = self.ssicov['Reduced Poles']
        Selected_poles = AllPoles[AllPoles['Label']>=selection_criteria]

        fig1, ax1 = plt.subplots()
        ax1 = sns.scatterplot(x=Selected_poles['Frequency'], y=Selected_poles['Order']*2, hue=Selected_poles['Label'], palette=_colors)
        ax1.set_xlabel('Frequency [Hz]')
        mplcursors.cursor()

        ax2 = ax1.twinx()
        for _i in range(num_sv):
        #    ax.semilogy(_f, S_val[_i, _i]) # scala log
            ax2.plot(self.fdd['f'][:], 10*np.log10(self.fdd['Singular Values'][_i, _i]), zorder=1, color=palette[_i]) # decibel
        
        ax2.set_ylabel(r'SVD of PSD in dB $[g^2/Hz]$') 
        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.patch.set_visible(False)
        plt.tight_layout()
        if len(save_to_file_path):
            plt.savefig(save_to_file_path + os.sep + f'StabDiag_overlap_SVD_decimation_{self.q:d}_BR_{self.br:d}_selection_{selection_criteria:d}'+'.pdf')
            plt.savefig(save_to_file_path + os.sep + f'StabDiag_overlap_SVD_decimation_{self.q:d}_BR_{self.br:d}_selection_{selection_criteria:d}'+'.png')
            pickle.dump(fig1, open(save_to_file_path + os.sep + f'StabDiag_overlap_SVD_decimation_{self.q:d}_BR_{self.br:d}_selection_{selection_criteria:d}.pickle', 'wb') )
        return fig1

    def get_modal_prop(self, method : str, FreQ : list , save_to_file_path : str = '', **kwargs):
        """
        This method extracts the modal properties according to the indicated method and for the list of candidate natural frequencies of interested retrieved by the peak-picking procedure.

        ----------
        Parameters
        ----------
        method : str
            Indicate the method used to extract the modal properties.
            Admissible inputs are: 'EFDD', 'FSDD', 'SSICOV', 'SSIDAT'
        FreQ : array (or list)
            Array containing the frequencies, identified from the stabilisation
            diagram, which we want to extract.
        """
        if method.upper() == 'EFDD' or method.upper() == 'FSDD':
            try :
                if hasattr(self, 'fdd'):
                    num_plots_so_Far = plt.get_fignums()

                    self.res_efdd = oma.EFDDmodEX(FreQ, self.fdd, method=method.upper(), **kwargs) 

                    self.ms_efdd = self.res_efdd['Mode Shapes'].real

                    self._n_modes_efdd = len(FreQ)
                    self.mac_efdd = np.reshape(
                            [oma.MaC(self.ms_efdd[:,l], self.ms_efdd[:,k]).real for k in range(self._n_modes_efdd) for l in range(self._n_modes_efdd)], # (self._n_modes_efdd*self._n_modes_efdd) list of MAC values 
                            (self._n_modes_efdd, self._n_modes_efdd)) # new (real) shape (self._n_modes_efdd x self._n_modes_efdd) of the MAC matrix
                    MACnames = [f'{ii[0]:.2f} Hz' for ii in self.res_efdd['Frequencies']]
                    self.mac_efdd = pd.DataFrame(self.mac_efdd, columns=MACnames, index=MACnames)
                    fig, ax = plt.subplots()
                    sns.heatmap(self.mac_efdd,cmap="jet",ax=ax,annot=True, fmt='.3f',)
                    fig.tight_layout()

                    if len(save_to_file_path):
                        with open(save_to_file_path + os.sep + f'Res_EFDD_decimation_{self.q:d}.pkl', 'wb') as file:
                            pickle.dump(self.res_efdd, file)

                        np.savetxt(save_to_file_path + os.sep + f'Res_EFDD_decimation_{self.q:d}_FREQ.txt', self.res_efdd['Frequencies'], fmt='%.6f') 
                        np.savetxt(save_to_file_path + os.sep + f'Res_EFDD_decimation_{self.q:d}_DAMPING.txt', self.res_efdd['Damping'], fmt='%.6f') 
                        np.savetxt(save_to_file_path + os.sep + f'Res_EFDD_decimation_{self.q:d}_MODESHAPES_complex.txt', self.res_efdd['Mode Shapes'], fmt='%.6f') 
                        np.savetxt(save_to_file_path + os.sep + f'Res_EFDD_decimation_{self.q:d}_MODESHAPES_real.txt', self.ms_efdd, fmt='%.6f') 

                        
                        list_of_plots = plt.get_fignums()
                        [list_of_plots.remove(item) for item in num_plots_so_Far if item in list_of_plots]

                        for ii,_i in enumerate(list_of_plots):
                            if _i == list_of_plots[-1]:
                                plt.figure(_i).savefig(save_to_file_path + os.sep + f'MAC_EFDD_decimation_{self.q:d}.pdf')
                                plt.figure(_i).savefig(save_to_file_path + os.sep + f'MAC_EFDD_decimation_{self.q:d}.png')
                            else:
                                plt.figure(_i).savefig(save_to_file_path + os.sep + f'EFDD_SDOF_bell_{self.res_efdd["Frequencies"][ii][0]:.1f}.pdf')
                                plt.figure(_i).savefig(save_to_file_path + os.sep + f'EFDD_SDOF_bell_{self.res_efdd["Frequencies"][ii][0]:.1f}.png')
                    else:
                        raise NameError(f'The attribute fdd does not exist yet! Please run the method svd_psd() prior to get modal results with method {method.upper()}.')

            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise

        elif method.upper() == 'SSICOV':
            try :
                if hasattr(self, 'ssicov'):
                    self.res_ssicov = oma.SSIModEX(FreQ, self.ssicov, **kwargs) # extracting modal properties
                    self.ms_ssicov = self.res_ssicov['Mode Shapes'].real

                    self._n_modes_ssicov = len(FreQ)
                    self.mac_ssicov = np.reshape(
                            [oma.MaC(self.ms_ssicov[:,l], self.ms_ssicov[:,k]).real for k in range(self._n_modes_ssicov) for l in range(self._n_modes_ssicov)], # (self._n_modes_efdd*self._n_modes_efdd) list of MAC values 
                            (self._n_modes_ssicov, self._n_modes_ssicov)) # new (real) shape (self._n_modes_efdd x self._n_modes_efdd) of the MAC matrix
                    MACnames = [f'{ii:.2f} Hz' for ii in self.res_ssicov['Frequencies']]
                    self.mac_ssicov = pd.DataFrame(self.mac_ssicov, columns=MACnames, index=MACnames)
                    fig, ax = plt.subplots()
                    sns.heatmap(self.mac_ssicov,cmap="jet",ax=ax,annot=True, fmt='.3f',)
                    fig.tight_layout()

                    if len(save_to_file_path):
                        np.savetxt(save_to_file_path + os.sep + f'Res_SSIcov_decimation_{self.q:d}_BR_{self.br:d}_FREQ.txt', self.res_ssicov['Frequencies'], fmt='%.6f') 
                        np.savetxt(save_to_file_path + os.sep + f'Res_SSIcov_decimation_{self.q:d}_BR_{self.br:d}_DAMPING.txt', self.res_ssicov['Damping'], fmt='%.6f') 
                        np.savetxt(save_to_file_path + os.sep + f'Res_SSIcov_decimation_{self.q:d}_BR_{self.br:d}_MODESHAPES_complex.txt', self.res_ssicov['Mode Shapes'], fmt='%.6f') 
                        np.savetxt(save_to_file_path + os.sep + f'Res_SSIcov_decimation_{self.q:d}_BR_{self.br:d}_MODESHAPES_real.txt', self.ms_ssicov, fmt='%.6f') 

                        plt.savefig(save_to_file_path + os.sep + f'MAC_SSIcov_decimation_{self.q:d}_BR_{self.br:d}'+'.pdf')
                        plt.savefig(save_to_file_path + os.sep + f'MAC_SSIcov_decimation_{self.q:d}_BR_{self.br:d}'+'.png')
                    else:
                        raise NameError(f'The attribute ssicov does not exist yet! Please run the method runssicov() prior to get modal results with method {method.upper()}.')

            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise

        elif method.upper() == 'SSIDAT':
            # runssidat() ancora da implementare
            pass
            # try :
            #     if hasattr(self, 'ssidat'):
            #         self.res_ssidat = oma.SSIModEX(FreQ, self.ssidat, **kwargs) # extracting modal properties
            #         self.ms_ssidat = self.res_ssidat['Mode Shapes'].real

            #         self._n_modes_ssidat = len(FreQ)
            #         self.mac_ssidat = np.reshape(
            #                 [oma.MaC(self.ms_ssidat[:,l], self.ms_ssidat[:,k]).real for k in range(self._n_modes_ssidat) for l in range(self._n_modes_ssidat)], # (self._n_modes_efdd*self._n_modes_efdd) list of MAC values 
            #                 (self._n_modes_ssidat, self._n_modes_ssidat)) # new (real) shape (self._n_modes_efdd x self._n_modes_efdd) of the MAC matrix
            #         MACnames = [f'{ii:.2f} Hz' for ii in self.res_ssidat['Frequencies']]
            #         self.mac_ssidat = pd.DataFrame(self.mac_ssidat, columns=MACnames, index=MACnames)
            #         fig, ax = plt.subplots()
            #         sns.heatmap(self.mac_ssidat,cmap="jet",ax=ax,annot=True, fmt='.3f',)
            #         fig.tight_layout()

            #         if len(save_to_file_path):
            #             np.savetxt(save_to_file_path + os.sep + f'Res_SSIdat_decimation_{self.q:d}_BR_{self.br:d}_FREQ.txt', self.res_ssidat['Frequencies'], fmt='%.6f') 
            #             np.savetxt(save_to_file_path + os.sep + f'Res_SSIdat_decimation_{self.q:d}_BR_{self.br:d}_DAMPING.txt', self.res_ssidat['Damping'], fmt='%.6f') 
            #             np.savetxt(save_to_file_path + os.sep + f'Res_SSIdat_decimation_{self.q:d}_BR_{self.br:d}_MODESHAPES_complex.txt', self.res_ssidat['Mode Shapes'], fmt='%.6f') 
            #             np.savetxt(save_to_file_path + os.sep + f'Res_SSIdat_decimation_{self.q:d}_BR_{self.br:d}_MODESHAPES_real.txt', self.ms_ssidat, fmt='%.6f') 

            #             plt.savefig(save_to_file_path + os.sep + f'MAC_SSIdat_decimation_{self.q:d}_BR_{self.br:d}'+'.pdf')
            #             plt.savefig(save_to_file_path + os.sep + f'MAC_SSIdat_decimation_{self.q:d}_BR_{self.br:d}'+'.png')
            #         else:
            #             raise NameError(f'The attribute ssidat does not exist yet! Please run the method runssidat() prior to get modal results with method {method.upper()}.')

            # except Exception as err:
            #     print(f"Unexpected {err=}, {type(err)=}")
            #     raise
        else:
            raise NameError(f'Unrecognized method {method}.')


    def crossmac_ssi_fdd(self, save_to_file_path : str = '',):
        """
        This method computes the crossmac of the identied mode shapes comparing the ssi and the fdd methods. To run the present method, both the method get_modal_prop() must have been run in advance both for ssi and fdd methods.

        ----------
        Parameters
        ----------
        save_to_file_path : string
            If provided, the plots are stored to a file in the indicated path both as a raster (png) and vectorial (pdf) format.
        """
        _n_modes = self.ms_ssicov.shape[1]
        self.crossmac_efdd_vs_ssi = np.reshape(
                [oma.MaC(self.ms_efdd[:,l], self.ms_ssicov[:,k]).real for k in range(_n_modes) for l in range(_n_modes)], # (self._n_modes_efdd*self._n_modes_efdd) list of MAC values 
                (_n_modes, _n_modes)) # new (real) shape (self._n_modes_efdd x self._n_modes_efdd) of the MAC matrix
        MACnames = [f'{ii:.2f} Hz' for ii in self.res_ssicov['Frequencies']]
        self.crossmac_efdd_vs_ssi = pd.DataFrame(self.crossmac_efdd_vs_ssi, columns=MACnames, index=MACnames)
        fig, ax = plt.subplots()
        sns.heatmap(self.crossmac_efdd_vs_ssi,cmap="jet",ax=ax,annot=True, fmt='.3f',)
        fig.tight_layout()
        if len(save_to_file_path):
            plt.savefig(save_to_file_path + os.sep + f'CrossMAC_efdd_vs_ssi'+'.pdf')
            plt.savefig(save_to_file_path + os.sep + f'CrossMAC_efdd_vs_ssi'+'.png')



    def get_mode_shape(self, method : str , num : int = 1 ): # mode_num is numbered from 1 for the user
        """
        This method gets the mode shape given the number of mode of interest (e.g. mode 1, mode 2, etc.) defined according to the adopted method.

        ----------
        Parameters
        ----------
        method : str
            Indicate the method used to extract the modal properties.
            Admissible inputs are: 'EFDD', 'FSDD', 'SSICOV', 'SSIDAT'
        num : int
            Indicate the number of the mode of interest. Modes numbering starts from 1.
        -------
        Returns
        -------
        [ mode , mon_nodes_dofs, freq , method] : list
            mode : is the mode shape column array of interest.
            mon_nodes_dofs : is the array containing information to properly connect each mode shape component to the proper node number and dof. 
            freq : the natural frequency associated to this mode.
            method : reminder of the method used to extract the mode shape of interest.
        """
        if method.upper() == 'EFDD' or method.upper() == 'FSDD':
            try:
                mode = self.ms_efdd[:,num-1]
                freq = self.res_efdd['Frequencies'][num-1][0]
            except Exception as err:
                    print(f"Error {err=}, {type(err)=}",'Possible cause: non-valid mode number.')
                    raise
        elif method.upper() == 'SSICOV':
            try:
                mode = self.ms_ssicov[:,num-1]
                freq = self.res_ssicov['Frequencies'][num-1]
            except Exception as err:
                    print(f"Error {err=}, {type(err)=}",'Possible cause: non-valid mode number.')
                    raise
        elif method.upper() == 'SSIDAT':
            # runssidat() ancora da implementare
            pass
        else:
            raise NameError(f'Unrecognized method {method}.')
        return [ mode , self.mon_nodes_dofs, freq , method]

