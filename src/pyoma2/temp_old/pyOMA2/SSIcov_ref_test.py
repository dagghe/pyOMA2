"""
Created on Sat Aug 19 09:12:54 2023

@author: dpa
"""

import numpy as np


def blockhankel(Y_ref, Y_all, s):
    r = Y_all.shape[0]
    c = Y_ref.shape[0]
    R = np.zeros((r, c, 2 * s))

    for i in range(r):
        for j in range(c):
            temp = np.correlate(Y_all[i, :], Y_ref[j, :], mode="full")
            R[i, j, :] = temp[2 * s + 2 :]  # Only the positive time lags

    H = np.zeros((s * r, c * s))
    for i in range(s):
        temp = R[:, :, i : i + s].reshape(r, c * s)
        H[i * r : (i + 1) * r, :] = temp

    return H


def ssicovref(Y, order, s):
    print("SSI-cov/ref status:")
    n_setups = len(Y)  # number of different sensor arrangements/setups
    n_ref = Y[0]["ref"].shape[0]  # number of reference sensors
    n_mov = np.array(
        [y["mov"].shape[0] for y in Y]
    )  # number of moving sensors in each setup
    n_s = n_ref + np.sum(n_mov)  # total number of sensors
    Obs = [None] * n_setups

    for i in range(n_setups):
        print(f"  processing setup {i+1} of {n_setups}...")
        Y_ref = Y[i]["ref"]
        Y_mov = Y[i]["mov"]
        H = blockhankel(Y_ref, np.vstack((Y_ref, Y_mov)), s)
        U, S, _ = np.linalg.svd(H, full_matrices=False)
        S = S[:order, :order]
        obs = U[:, :order] @ np.sqrt(S)

        id = np.zeros((n_ref, s), dtype=int)
        for j in range(n_ref):
            id[j, :] = np.arange(s) * (n_ref + n_mov[i]) + j

        obs_ref = obs[id.flatten(), :].reshape(n_ref, s, order)
        obs[id.flatten(), :] = 0  # Delete reference portion from obs_mov

        if i == 0:
            obs1_ref = obs_ref.copy()

        obs = obs @ np.linalg.pinv(obs_ref) @ obs1_ref
        Obs[i] = obs

    print("  generating global observability matrix...")
    Obs_all = np.zeros((n_s * s, order))
    for i in range(s):
        id1 = i * n_s
        id2 = id1 + n_ref
        Obs_all[id1:id2, :] = obs1_ref[(i * n_ref) : (i + 1) * n_ref, :]

        for j in range(n_setups):
            id1 = id2
            id2 = id1 + n_mov[j]
            Obs_all[id1:id2, :] = Obs[j][(i * n_mov[j]) : (i + 1) * n_mov[j], :]

    A = [None] * order
    C = [None] * order

    print(f"  generating system matrices A,C for {order} model orders...")
    for i in range(order):
        A[i] = np.linalg.pinv(Obs_all[:-n_s, :i]) @ Obs_all[n_s:, :i]
        C[i] = Obs_all[:n_s, :i]

    print("SSI-cov/ref finished.")
    return A, C
