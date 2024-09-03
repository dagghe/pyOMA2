# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 01:10:49 2024

@author: dpa
"""

import numpy as np

# from scipy import signal
import openseespy.opensees as ops
import pandas as pd

# import vfo.vfo as vfo


def Model_Eig():
    # Units
    _cm = 1e-2
    _MPa = 1e6
    # Materiale - CLS
    Emod = 25000 * _MPa  # EModulo cls
    Gmod = 10000 * _MPa  # GModulo cls
    rho = 2500  # kg/m^3
    # Pilastri
    b1 = 30 * _cm
    h1 = 45 * _cm
    A1 = b1 * h1
    I1 = b1 * h1**3 / 12
    _alfa = 3 + 1.8 * b1 / h1
    I1tor = (h1 * b1**2) / _alfa
    Mcol = A1 * rho
    # Travi
    b2 = 30 * _cm
    h2 = 55 * _cm
    A2 = b2 * h2
    I2yy = b2 * h2**3 / 12
    I2zz = h2 * b2**3 / 12
    _alfa = 3 + 1.8 * b2 / h2
    I2tor = (h2 * b2**2) / _alfa
    Mbeam = A2 * rho
    # Massa solai
    Asol = 4.5 * 6  # area solaio
    tsol = 20 * _cm  # spessore solaio
    Msol = Asol * tsol * rho

    # -----------------------------------------------------------------------------
    # GEOMETRY
    # Coordinates of building's perimetral nodes
    coords = [
        [1, 1],
        [7, 1],
        [13, 1],
        [19, 1],
        [1, 5.5],
        [7, 5.5],
        [13, 5.5],
        [19, 5.5],
        [13, 10],
        [19, 10],
    ]  # coordinates of the nodes

    zs = [0, 3, 6, 9]  # floors' height

    # sensor nodes (and floor's center of mass)
    coords1 = [[4, 3.25], [10, 3.25], [16, 3.25], [16, 7.25]]

    # floors mass
    mi = np.ones(len(coords1)) * Msol
    # center of mass mass
    xs = np.array(coords1)[:, 0]
    xcm = np.sum(mi * xs) / np.sum(mi)  # x center of mass
    ys = np.array(coords1)[:, 1]
    ycm = np.sum(mi * ys) / np.sum(mi)  # y center of mass

    # =============================================================================
    # INIZIO MODELLO
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    # Add nodes in coords
    jj = 0
    for zz in zs:
        for ii, coord in zip(range(jj, jj + len(coords)), coords):
            ops.node(ii + 1, coord[0], coord[1], zz)  # node tag and location
        nLabs = ops.getNodeTags()
        jj = nLabs[-1]
    nLabsDOF = ops.getNodeTags()  # nodes labels

    # add extra nodes (coords1)
    for zz in zs[1:]:
        for ii, coord in zip(range(jj, jj + len(coords1)), coords1):
            ops.node(ii + 1, coord[0], coord[1], zz)  # node tag and location
            ops.mass(ii + 1, Msol, Msol, 0, 0, 0, 0)  # node tag and location
        nLabs = ops.getNodeTags()
        jj = nLabs[-1]
    nLabsEx = ops.getNodeTags()[-len(coords1) * 3 :]  # nodes labels

    # add Center of Mass nodes
    for zz in zs[1:]:
        ops.node(jj + 1, xcm, ycm, zz)  # node tag and location
        jj += 1
    nLabs = ops.getNodeTags()
    nLabsCM = ops.getNodeTags()[-3:]

    # list that contains the x-coords
    _xxs = [ops.nodeCoord(ii, 1) for ii in nLabs]
    # list that contains the y-coords
    _yys = [ops.nodeCoord(ii, 2) for ii in nLabs]
    # list that contains the z-coords
    _zzs = [ops.nodeCoord(ii, 3) for ii in nLabs]
    # Create dataframe with the labels and coordinates info
    _all = np.vstack((nLabs, _xxs, _yys, _zzs))
    dfNodes = pd.DataFrame(_all.T)
    dfNodes.columns = ["Label", "X", "Y", "Z"]

    # Fix nodes at z=0
    labs1 = (
        dfNodes[(dfNodes["Z"] < 0 + 1e-3) & (dfNodes["Z"] > 0 - 1e-3)]
        .Label.to_numpy()
        .astype(int)
    )
    for ii in labs1:
        # label, Ux, Uy, Uz, Rx, Ry, Rz
        ops.fix(int(ii), 1, 1, 1, 1, 1, 1)
    nLabsDOF = [x for x in nLabsDOF if x not in labs1]

    # add columns
    ops.geomTransf("Linear", 1, 1, 0, 0)
    jj = 1
    for xx, yy in zip(np.array(coords)[:, 0], np.array(coords)[:, 1]):
        labs1 = dfNodes[
            (dfNodes["X"] < xx + 1e-3)
            & (dfNodes["X"] > xx - 1e-3)
            & (dfNodes["Y"] < yy + 1e-3)
            & (dfNodes["Y"] > yy - 1e-3)
        ].Label.to_numpy()
        for ii in range(len(labs1) - 1):
            ops.element(
                "elasticBeamColumn",
                jj,
                int(labs1[ii]),
                int(labs1[ii + 1]),
                A1,
                Emod,
                Gmod,
                I1tor,
                I1,
                I1,
                1,
                "-mass",
                Mcol,
            )
            jj += 1

    # add beams direction x
    ops.geomTransf("Linear", 2, 0, 0, 1)
    jj = ops.getEleTags()[-1] + 1
    for zz in zs[1:]:
        for yy in np.unique(np.array(coords)[:, 1]):
            labs1 = dfNodes[
                (dfNodes["Y"] < yy + 1e-3)
                & (dfNodes["Y"] > yy - 1e-3)
                & (dfNodes["Z"] < zz + 1e-3)
                & (dfNodes["Z"] > zz - 1e-3)
            ].Label.to_numpy()
            for ii in range(len(labs1) - 1):
                ops.element(
                    "elasticBeamColumn",
                    jj,
                    int(labs1[ii]),
                    int(labs1[ii + 1]),
                    A2,
                    Emod,
                    Gmod,
                    I2tor,
                    I2yy,
                    I2zz,
                    2,
                    "-mass",
                    Mbeam,
                )
                jj += 1

    # add beams direction y
    jj = ops.getEleTags()[-1] + 1
    for zz in zs[1:]:
        for xx in np.unique(np.array(coords)[:, 0]):
            labs1 = dfNodes[
                (dfNodes["X"] < xx + 1e-3)
                & (dfNodes["X"] > xx - 1e-3)
                & (dfNodes["Z"] < zz + 1e-3)
                & (dfNodes["Z"] > zz - 1e-3)
            ].Label.to_numpy()
            for ii in range(len(labs1) - 1):
                ops.element(
                    "elasticBeamColumn",
                    jj,
                    int(labs1[ii]),
                    int(labs1[ii + 1]),
                    A2,
                    Emod,
                    Gmod,
                    I2tor,
                    I2yy,
                    I2zz,
                    2,
                    "-mass",
                    Mbeam,
                )
                jj += 1

    # Add rigid diaphragm constraint to each floor
    ii = 0
    for zz in zs[1:]:
        labs1 = (
            dfNodes[(dfNodes["Z"] < zz + 1e-3) & (dfNodes["Z"] > zz - 1e-3)]
            .Label.to_numpy()
            .tolist()
        )
        labsSlave = [int(x) for x in labs1 if x not in nLabsCM]
        labMaster = nLabsCM[ii]
        labExtra = labsSlave[-4:]
        ops.rigidDiaphragm(3, labMaster, *labsSlave)
        ops.fix(labMaster, 0, 0, 1, 1, 1, 0)
        for kk in labExtra:
            ops.fix(kk, 0, 0, 1, 1, 1, 0)
        ii += 1

    # save nodes of lines
    _alleleTags = ops.getEleTags()
    lines = [ops.eleNodes(tag) for tag in _alleleTags]
    lines = np.array(lines)

    # -----------------------------------------------------------------------------
    # SOLVE EIGENVALUE PROBLEM
    # eigensolvers = ['-genBandArpack', '-symmBandLapack', '-fullGenLapack']
    numEigen = 9
    eigenValues = np.array(ops.eigen("-genBandArpack", numEigen))
    fns = np.sqrt(eigenValues) / (2 * 3.1415)  # save the frequencies

    for modnumb in range(1, len(fns) + 1):
        # Mode shapes
        Mod1DispX = []
        Mod1DispY = []
        for ii in dfNodes.Label:
            ii = int(ii)
            Mod1DispX.append(ops.nodeEigenvector(ii, modnumb, 1))
            Mod1DispY.append(ops.nodeEigenvector(ii, modnumb, 2))

        dfNodes[f"ModalDispX{modnumb}"] = Mod1DispX
        dfNodes[f"ModalDispY{modnumb}"] = Mod1DispY

    _wi = eigenValues[0] ** 0.5
    _wj = eigenValues[-1] ** 0.5
    xi = 0.02

    beta = 2 * xi / (_wi + _wj)
    alpha = _wi * _wj * beta

    xis = 1 / 2 * (alpha / eigenValues**0.5 + beta * eigenValues**0.5)
    # -----------------------------------------------------------------------------
    # set damping
    ops.rayleigh(alpha, 0.0, 0.0, beta)
    # -----------------------------------------------------------------------------
    return dfNodes, fns, xis, nLabsEx, nLabsDOF


def get_TH(
    rng,
    fs=100,
    T=900,
):
    dt = 1 / fs  # [sec] time resolution
    Ndat = int(T / dt) + 1  # number of data points
    # # =============================================================================
    # # NODES EXCITATION 1
    # # chosing 3*nn nodes to excite at random
    #     nn = 1
    #     dofsx = np.random.choice(nLabsDOF, nn, False).tolist()
    #     dofsy = np.random.choice(nLabsDOF, nn, False).tolist()
    #     dofsRz = np.random.choice(nLabsDOF, nn, False).tolist()

    #     # dofs = nLabsDOF
    #     w_noisesx = np.array([np.random.randn(Ndat) for r in range(len(dofsx))]).T*1e-4
    #     w_noisesy = np.array([np.random.randn(Ndat) for r in range(len(dofsy))]).T*1e-4
    #     w_noisesRz = np.array([np.random.randn(Ndat) for r in range(len(dofsRz))]).T*1e-8

    #     kk = 0
    #     for ii, dof in zip(range(len(dofsx)),dofsx):
    #         ops.timeSeries("Path", kk, "-dt", dt, "-values", *w_noisesx[:,ii],'-prependZero')
    #         ops.pattern('MultipleSupport', kk)
    #         ops.groundMotion(kk,"Plain", '-disp', kk)
    #         ops.imposedMotion(dof, 1, kk)
    #         kk += 1
    #     for ii, dof in zip(range(len(dofsy)),dofsy):
    #         ops.timeSeries("Path", kk, "-dt", dt, "-values", *w_noisesy[:,ii],'-prependZero')
    #         ops.pattern('MultipleSupport', kk)
    #         ops.groundMotion(kk,"Plain", '-disp', kk)
    #         ops.imposedMotion(dof, 2, kk)
    #         kk += 1
    #     for ii, dof in zip(range(len(dofsRz)),dofsRz):
    #         ops.timeSeries("Path", kk, "-dt", dt, "-values", *w_noisesRz[:,ii],'-prependZero')
    #         ops.pattern('MultipleSupport', kk)
    #         ops.groundMotion(kk,"Plain", '-disp', kk)
    #         ops.imposedMotion(dof, 6, kk)
    #         kk += 1
    # # -----------------------------------------------------------------------------
    # # NODES EXCITATION 2
    # # chosing nn nodes to excite at random
    # nn = 1
    # dofs = np.random.choice(nLabsDOF, nn, False).tolist()

    # # dofs = nLabsDOF
    # w_noisesx = np.array([np.random.randn(Ndat) for r in range(len(dofs))]).T*1e-4
    # w_noisesy = np.array([np.random.randn(Ndat) for r in range(len(dofs))]).T*1e-4
    # w_noisesRz = np.array([np.random.randn(Ndat) for r in range(len(dofs))]).T*1e-8

    # kk = 0
    # for ii, dof in zip(range(len(dofs)),dofs):
    #     ops.timeSeries("Path", kk, "-dt", dt, "-values", *w_noisesx[:,ii],'-prependZero')
    #     ops.pattern('MultipleSupport', kk)
    #     ops.groundMotion(kk,"Plain", '-disp', kk)
    #     ops.imposedMotion(dof, 1, kk)
    #     kk += 1
    #     ops.timeSeries("Path", kk, "-dt", dt, "-values", *w_noisesy[:,ii],'-prependZero')
    #     ops.pattern('MultipleSupport', kk)
    #     ops.groundMotion(kk,"Plain", '-disp', kk)
    #     ops.imposedMotion(dof, 2, kk)
    #     kk += 1
    #     ops.timeSeries("Path", kk, "-dt", dt, "-values", *w_noisesRz[:,ii],'-prependZero')
    #     ops.pattern('MultipleSupport', kk)
    #     ops.groundMotion(kk,"Plain", '-disp', kk)
    #     ops.imposedMotion(dof, 6, kk)
    #     kk += 1

    # =============================================================================
    # GROUND MOTION
    # dofs = nLabsDOF
    w_noisesx = rng.randn(Ndat) * 1e-4
    w_noisesy = rng.randn(Ndat) * 1e-4
    w_noisesRz = rng.randn(Ndat) * 1e-8

    rand = rng.randint(1, 11)
    ops.pattern("MultipleSupport", 1)
    ops.timeSeries("Path", 1, "-dt", dt, "-values", *w_noisesx, "-prependZero")
    ops.groundMotion(1, "Plain", "-disp", 1)
    ops.imposedMotion(rand, 1, 1)

    rand1 = rng.randint(1, 11)
    ops.pattern("MultipleSupport", 2)
    ops.timeSeries("Path", 2, "-dt", dt, "-values", *w_noisesy, "-prependZero")
    ops.groundMotion(2, "Plain", "-disp", 2)
    ops.imposedMotion(rand1, 2, 2)

    rand2 = rng.randint(1, 11)
    ops.pattern("MultipleSupport", 3)
    ops.timeSeries("Path", 3, "-dt", dt, "-values", *w_noisesRz, "-prependZero")
    ops.groundMotion(3, "Plain", "-disp", 3)
    ops.imposedMotion(rand2, 6, 3)

    # =============================================================================
    # ANALYSIS
    # =============================================================================
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGen")
    ops.algorithm("KrylovNewton")
    ops.test("NormDispIncr", 1e-12, 50)

    ok = 0

    tCurrent = ops.getTime()
    time = [tCurrent]

    accx = []
    accy = []
    dispx = []
    dispy = []
    while tCurrent < T:
        ops.algorithm("KrylovNewton")
        while ok == 0 and tCurrent < T:
            ops.test("NormDispIncr", 1e-12, 50)

            ops.integrator("Newmark", 0.5, 0.25)
            ops.analysis("Transient")
            ok = ops.analyze(1, dt)

            if ok == 0:
                tCurrent = ops.getTime()
                time.append(tCurrent)
                print("tCurrent=", tCurrent)

                dispx.append([ops.nodeDisp(ll, 1) for ll in nLabsEx])
                dispy.append([ops.nodeDisp(ll, 2) for ll in nLabsEx])

                accx.append([ops.nodeAccel(ll, 1) for ll in nLabsEx])
                accy.append([ops.nodeAccel(ll, 2) for ll in nLabsEx])

    accx = np.array(accx)
    accy = np.array(accy)
    dispx = np.array(dispx)
    dispy = np.array(dispy)

    # -----------------------------------------------------------------------------
    # data = np.hstack((accx,accy))
    # data = np.hstack((dispx,dispy))
    data = np.empty(
        (
            dispx.shape[0],
            dispx.shape[1] + dispy.shape[1],
        ),
        dtype=dispx.dtype,
    )
    data[:, 0::2] = dispx
    data[:, 1::2] = dispy

    # data[:,0::2] = accx
    # data[:,1::2] = accy
    # -----------------------------------------------------------------------------
    # Adding noise
    SNR = 5  # Signal-to-Noise ratio
    ar = np.max(data) / (10 ** (SNR / 10))  # Noise amplitude
    Ndat = data.shape[0]
    # Initialize the arrays (copy of accelerations)
    data1 = data.copy()
    for _ind in range(data1.shape[1]):
        # Measurments POLLUTED BY NOISE
        data1[:, _ind] = data[:, _ind] + ar * rng.randn(Ndat)
    # -----------------------------------------------------------------------------
    return data1


if __name__ == "__main__":
    dfNodesy, fns, xis, nLabsEx, nLabsDOF = Model_Eig()
    _rng = np.random.RandomState(123)  # Set the seed run 1
    _rng1 = np.random.RandomState(1234)  # Set the seed run 2
    _rng2 = np.random.RandomState(12344)  # Set the seed run 3

    data = get_TH(_rng, fs=200, T=600)

    dfNodesy, fns, xis, nLabsEx, nLabsDOF = Model_Eig()
    data1 = get_TH(_rng1, fs=200, T=600)

    dfNodesy, fns, xis, nLabsEx, nLabsDOF = Model_Eig()
    data2 = get_TH(_rng2, fs=200, T=600)

    # MULTISETUP
    s_1 = data[:, [22, 21, 20, 23, 18, 19, 16, 17, 8, 9]]
    s_2 = data1[:, [22, 21, 20, 10, 11, 12, 13, 14, 15, 6]]
    s_3 = data2[:, [22, 21, 20, 7, 4, 5, 2, 3, 0, 1]]
