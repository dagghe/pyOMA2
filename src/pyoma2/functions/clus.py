"""
Created on Sun Nov 24 07:07:42 2024

@author: dagghe
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.cluster import (
    HDBSCAN,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    KMeans,
    SpectralClustering,
)
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import power_transform

from pyoma2.functions import gen

# =============================================================================
# CLUSTERING
# =============================================================================


class FuzzyCMeansClustering:
    """
    Fuzzy C-Means clustering algorithm class.

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to form.
    m : float, default=2.0
        Fuzziness parameter. Must be > 1.
    max_iter : int, default=100
        Maximum number of iterations of the algorithm.
    tol : float, default=1e-5
        Tolerance for convergence. If improvement is less than tol, stop.
    random_state : int, default=None
        Seed for membership matrix initialization.
    """

    def __init__(self, n_clusters=2, m=2.0, max_iter=100, tol=1e-5, random_state=None):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.membership_ = None
        self.labels_ = None

    def _initialize_membership(self, n_samples):
        rng = np.random.RandomState(self.random_state)
        U = rng.rand(n_samples, self.n_clusters)
        U = U / np.sum(U, axis=1, keepdims=True)
        return U

    def _update_centers(self, X, U):
        # U raised to power m
        Um = U**self.m
        # compute cluster centers
        centers = (Um.T @ X) / np.sum(Um.T, axis=1, keepdims=True)
        return centers

    def _update_membership(self, X, centers):
        n_samples = X.shape[0]
        U_new = np.zeros((n_samples, self.n_clusters))
        # exponent for distance ratio
        exp = 2.0 / (self.m - 1)
        for i in range(n_samples):
            distances = np.linalg.norm(X[i] - centers, axis=1)
            # avoid division by zero
            distances = np.fmax(distances, np.finfo(np.float64).eps)
            inv = distances ** (-exp)
            U_new[i] = inv / np.sum(inv)
        return U_new

    def fit(self, X, y=None):  # y ignored
        """
        Compute fuzzy c-means clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]

        # initialize membership matrix
        U = self._initialize_membership(n_samples)

        self.n_iter_ = 0
        for iteration in range(1, self.max_iter + 1):
            centers = self._update_centers(X, U)
            U_new = self._update_membership(X, centers)

            # check convergence
            if np.linalg.norm(U_new - U) < self.tol:
                self.n_iter_ = iteration
                U = U_new
                break

            U = U_new

        else:
            # loop finished without break
            self.n_iter_ = self.max_iter

        self.membership_ = U
        self.cluster_centers_ = centers
        # assign crisp labels
        self.labels_ = np.argmax(U, axis=1)
        return self

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        X = np.asarray(X, dtype=float)
        distances = np.linalg.norm(
            X[:, None, :] - self.cluster_centers_[None, :, :], axis=2
        )
        return np.argmin(distances, axis=1)

    def fit_predict(self, X, y=None):  # y ignored
        """
        Compute cluster centers and predict cluster index for each sample.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        self.fit(X)
        return self.labels_


# -----------------------------------------------------------------------------


def FCMeans(feat_arr):
    """
    Perform Fuzzy C-Means clustering on the given feature array.

    Parameters
    ----------
    feat_arr : ndarray of shape (n_samples, n_features)
        Input feature array for clustering.

    Returns
    -------
    labels_all : ndarray of shape (n_samples,)
        Cluster labels for each sample. Labels are adjusted such that the first cluster
        corresponds to the smaller centroid (stable modes).
    """
    fcm = FuzzyCMeansClustering(n_clusters=2, max_iter=500)
    fcm.fit(feat_arr)
    labels_all = fcm.labels_
    # check the centroids to establish stable and spurious modes
    centroids = fcm.cluster_centers_
    # if the first centroid is larger than the second invert the labels
    if centroids[0, 0] > centroids[1, 0]:
        labels_all = 1 - labels_all
    return labels_all


# -----------------------------------------------------------------------------


def kmeans(feat_arr):
    """
    Perform k-means clustering on the given feature array.

    Parameters
    ----------
    feat_arr : ndarray of shape (n_samples, n_features)
        Input feature array for clustering.

    Returns
    -------
    labels_all : ndarray of shape (n_samples,)
        Cluster labels for each sample. Labels are adjusted such that the first cluster
        corresponds to the smaller centroid (stable modes).
    """
    kmean = KMeans(n_clusters=2)
    kmean.fit(feat_arr)
    labels_all = kmean.labels_
    # check the centroids to establish stable and spurious modes
    centroids = kmean.cluster_centers_
    # if the first centroid is larger than the second invert the labels
    if centroids[0, 0] > centroids[1, 0]:
        labels_all = 1 - labels_all
    return labels_all


# -----------------------------------------------------------------------------


def GMM(feat_arr, dist=False):
    """
    Perform Gaussian Mixture Model (GMM) clustering on the given feature array.

    Parameters
    ----------
    feat_arr : ndarray of shape (n_samples, n_features)
        Input feature array for clustering.

    Returns
    -------
    labels_all : ndarray of shape (n_samples,)
        Cluster labels for each sample. Labels are adjusted such that the first cluster
        corresponds to the smaller mean (stable modes).
    """
    GMM = GaussianMixture(n_components=2)
    GMM.fit(feat_arr)
    labels_all = GMM.predict(feat_arr)
    # check the means to establish stable and spurious modes
    centroids = GMM.means_
    # if the first centroid is larger than the second invert the labels
    if centroids[0, 0] > centroids[1, 0]:
        labels_all = 1 - labels_all
    if dist:
        # calculate limit distance
        pi0, pi1 = GMM.weights_
        mu = GMM.means_
        # sig = GMM.covariances_
        dlim = pi0 * mu[0, 0] + pi1 * mu[1, 1]
        return labels_all, dlim
    else:
        return labels_all


# -----------------------------------------------------------------------------


def hierarc(dtot, dc, linkage, n_clusters, ordmax, step, Fns, Phis):
    """
    Perform hierarchical clustering with specified parameters.

    Parameters
    ----------
    dtot : ndarray of shape (n_samples, n_samples)
        Pairwise distance matrix for clustering.
    dc : float or str, optional
        Distance threshold for clustering. Special string options include:
        - "mu+2sig": Mean plus two standard deviations of distances.
        - "95weib": 95th percentile of a Weibull distribution fit to the distances.
        - "auto": Automatic threshold estimation based on KDE.
    n_clusters : int or str, optional
        Number of clusters. If "auto", it is calculated as 25% of the maximum order.
    linkage : {'complete', 'average', 'single'}, optional
        Linkage criterion for hierarchical clustering.
    ordmax : int
        Maximum order of clustering.
    step : float
        Step size for computing distances.
    Fns : ndarray
        Frequencies for distance calculation.
    Phis : ndarray
        Mode shapes for distance calculation.

    Returns
    -------
    labels_clus : ndarray of shape (n_samples,)
        Cluster labels for each sample.
    """
    if dc is None:
        if n_clusters is None or n_clusters == "auto":
            n_clusters = int(ordmax * 0.25)

    elif dc == "mu+2sig":
        dfn1 = dist_n_n1_f(Fns, 0, ordmax, step)
        dphin1 = dist_n_n1_phi(Phis, 0, ordmax, step)
        # calculate distance threshold
        dc = np.mean(dfn1 + dphin1) + 2 * np.std(dfn1 + dphin1)
        # set linkage to average
        linkage = "average"
        n_clusters = None

    elif dc == "95weib":
        dfn1 = dist_n_n1_f(Fns, 0, ordmax, step)
        dphin1 = dist_n_n1_phi(Phis, 0, ordmax, step)
        # calculate distance threshold
        shape, loc, scale = stats.weibull_min.fit(
            dfn1 + dphin1, floc=0
        )  # N.B. force loc to zero
        dc = stats.weibull_min.ppf(0.95, shape, loc=loc, scale=scale)
        # set linkage to average
        linkage = "average"
        n_clusters = None

    elif dc == "auto":
        x = np.triu_indices_from(dtot, k=0)
        x = dtot[x]
        xs = np.linspace(dtot.min(), dtot.max(), 500)
        kde = stats.gaussian_kde(x)
        pdf = kde(xs)
        minima_in = signal.argrelmin(pdf)[0]
        minima = pdf[minima_in]
        min_abs = minima.argmin()
        min_abs_ind = minima_in[min_abs]
        dc = xs[min_abs_ind]
        n_clusters = None

    cluster_algo = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        distance_threshold=dc,
        linkage=linkage,
    )
    dtot_c = np.copy(dtot)
    cluster_algo.fit(dtot_c)
    labels_clus = cluster_algo.labels_
    return labels_clus


# -----------------------------------------------------------------------------


def spectral(dsim, n_clusters, ordmax):
    """
    Perform spectral clustering with the given similarity matrix.

    Parameters
    ----------
    dsim : ndarray of shape (n_samples, n_samples)
        Similarity matrix for clustering.
    n_clusters : int or str, optional
        Number of clusters. If "auto", it is calculated as 25% of the maximum order.
    ordmax : int
        Maximum order.

    Returns
    -------
    labels_clus : ndarray of shape (n_samples,)
        Cluster labels for each sample.
    """
    if n_clusters is None or n_clusters == "auto":
        n_clusters = int(0.25 * ordmax)
    # try:
    cluster_algo = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
    )
    cluster_algo.fit(dsim)
    # Establish all labels
    labels_clus = cluster_algo.labels_
    return labels_clus


# -----------------------------------------------------------------------------


def affinity(dsim):
    """
    Perform affinity propagation clustering on the given similarity matrix.

    Parameters
    ----------
    dsim : ndarray of shape (n_samples, n_samples)
        Precomputed similarity matrix for clustering.

    Returns
    -------
    labels_clus : ndarray of shape (n_samples,)
        Cluster labels for each sample.
    """
    cluster_algo = AffinityPropagation(
        affinity="precomputed",
        damping=0.75,
    )
    cluster_algo.fit(dsim)
    # Establish all labels
    labels_clus = cluster_algo.labels_
    return labels_clus


# -----------------------------------------------------------------------------


def optics(dtot, min_size):
    """
    Perform OPTICS clustering on the given pairwise distance matrix.

    Parameters
    ----------
    dtot : ndarray of shape (n_samples, n_samples)
        Pairwise distance matrix for clustering.
    min_size : int
        Minimum cluster size and minimum number of samples for clustering.

    Returns
    -------
    labels_clus : ndarray of shape (n_samples,)
        Cluster labels for each sample.
    """
    cluster_algo = OPTICS(
        metric="precomputed",
        min_samples=min_size,
        min_cluster_size=min_size,
    )
    # Define small threshold
    threshold = 1e-10
    # Use np.where to ensure that small negative elements are set to their abs value
    dtot_c = np.where((dtot < 0) & (np.abs(dtot) < threshold), np.abs(dtot), dtot)
    cluster_algo.fit(dtot_c)
    # Establish all labels
    labels_clus = cluster_algo.labels_
    return labels_clus


# -----------------------------------------------------------------------------


def hdbscan(dtot, min_size):
    """
    Perform HDBSCAN clustering on the given pairwise distance matrix.

    Parameters
    ----------
    dtot : ndarray of shape (n_samples, n_samples)
        Pairwise distance matrix for clustering.
    min_size : int
        Minimum cluster size and minimum number of samples for clustering.

    Returns
    -------
    labels_clus : ndarray of shape (n_samples,)
        Cluster labels for each sample.
    """
    cluster_algo = HDBSCAN(
        metric="precomputed",
        min_samples=min_size,
        min_cluster_size=min_size,
    )
    dtot_c = np.copy(dtot)
    cluster_algo.fit(dtot_c)
    # Establish all labels
    labels_clus = cluster_algo.labels_
    return labels_clus


# -----------------------------------------------------------------------------


def reorder_clusters(clusters, labels, Fn_fl):
    """
    Reorder cluster labels based on ascending frequencies values.

    Parameters
    ----------
    clusters : dict
        Dictionary of clusters where keys are cluster labels and values are arrays of indices.
    labels : ndarray of shape (n_samples,)
        Array of cluster labels for each sample.
    Fn_fl : ndarray of shape (n_samples,)
        Frequencies corresponding to each sample.

    Returns
    -------
    new_clusters : dict
        Reordered dictionary of clusters with updated labels.
    new_labels : ndarray of shape (n_samples,)
        Array of updated cluster labels.
    """
    # Compute a representative frequency (e.g., mean) for each cluster
    cluster_stats = []
    for label, indices in clusters.items():
        cluster_freqs = Fn_fl[indices].squeeze()
        # Representative statistic: mean frequency of the cluster
        mean_freq = np.mean(cluster_freqs)
        cluster_stats.append((label, mean_freq))

    # Sort by mean frequency
    cluster_stats.sort(key=lambda x: x[1])

    # Create a mapping from old label to new label
    old_to_new = {
        old_label: new_label for new_label, (old_label, _) in enumerate(cluster_stats)
    }

    # Create the new cluster dictionary with updated labels
    new_clusters = {}
    for old_label, indices in clusters.items():
        new_label = old_to_new[old_label]
        new_clusters[new_label] = indices

    # Update labels all at once to avoid overwriting issues
    # Make a copy to avoid confusion during iteration
    new_labels = labels.copy()
    for old_label, new_label in old_to_new.items():
        new_labels[labels == old_label] = new_label

    return new_clusters, new_labels


# -----------------------------------------------------------------------------


def post_freq_lim(clusters, labels, freq_lim, Fn_fl):
    """
    Filter clusters based on specified frequency range.

    Parameters
    ----------
    clusters : dict
        Dictionary of clusters where keys are cluster labels and values are arrays of indices.
    labels : ndarray of shape (n_samples,)
        Array of cluster labels for each sample.
    freq_lim : tuple of float
        Minimum and maximum allowable frequencies (inclusive).
    Fn_fl : ndarray of shape (n_samples,)
        Frequencies corresponding to each sample.

    Returns
    -------
    clusters : dict
        Updated dictionary of clusters with outliers removed.
    labels : ndarray of shape (n_samples,)
        Updated cluster labels with outliers assigned -1.
    """
    # Frequency-based filtering
    for label, indices in clusters.items():
        # Calculate the mean frequency of the current cluster
        mean_freq = np.mean(Fn_fl[indices])
        # Find indices of the current cluster
        cluster_indices = np.where(labels == label)[0]
        # Check if the cluster size is below the threshold
        if mean_freq < freq_lim[0] or mean_freq > freq_lim[1]:
            # Assign label -1 to these indices
            labels[cluster_indices] = -1
    unique_labels = set(labels)
    unique_labels.discard(-1)
    clusters = {label: np.where(labels == label)[0] for label in unique_labels}
    return clusters, labels


# -----------------------------------------------------------------------------


def post_fn_med(clusters, labels, flattened_results):
    """
    Filter clusters based on a median frequency threshold.

    Parameters
    ----------
    clusters : dict
        Dictionary of clusters where keys are cluster labels and values are arrays of indices.
    labels : ndarray of shape (n_samples,)
        Array of cluster labels for each sample.
    flattened_results : tuple of ndarray
        Tuple containing:
        - Fn_fl : ndarray of shape (n_samples,)
            Frequencies corresponding to each sample.
        - Fn_std_fl : ndarray of shape (n_samples,)
            Standard deviations corresponding to frequencies.

    Returns
    -------
    clusters : dict
        Updated dictionary of clusters with outliers removed.
    labels : ndarray of shape (n_samples,)
        Updated cluster labels with outliers assigned -1.
    """
    Fn_fl, Fn_std_fl = flattened_results
    # Frequency-based filtering
    for _, indices in clusters.items():
        # Calculate the median frequency of the current cluster
        median_freq = np.median(Fn_fl[indices])
        # Extract frequencies and standard deviations for the current cluster
        freq = Fn_fl[indices].squeeze()
        std = Fn_std_fl[indices].squeeze()
        # Determine which poles have their (freq - std) <= median <= (freq + std)
        condition = (freq - std <= median_freq) & (freq + std >= median_freq)
        # Identify outliers that do not satisfy the condition
        outliers = indices[~condition]
        # Assign label -1 to outliers
        labels[outliers] = -1
    unique_labels = set(labels)
    unique_labels.discard(-1)
    clusters = {label: np.where(labels == label)[0] for label in unique_labels}
    return clusters, labels


# -----------------------------------------------------------------------------


def post_fn_IQR(clusters, labels, Fn_fl):
    """
    Filter clusters based on the interquartile range (IQR) of frequencies.

    Parameters
    ----------
    clusters : dict
        Dictionary of clusters where keys are cluster labels and values are arrays of indices.
    labels : ndarray of shape (n_samples,)
        Array of cluster labels for each sample.
    Fn_fl : ndarray of shape (n_samples,)
        Frequencies corresponding to each sample.

    Returns
    -------
    clusters : dict
        Updated dictionary of clusters with outliers removed.
    labels : ndarray of shape (n_samples,)
        Updated cluster labels with outliers assigned -1.
    """
    # Frequency-based filtering
    for _, indices in clusters.items():
        # Extract damping values for the current cluster
        frequencies = Fn_fl[indices].squeeze()
        # Calculate Q1 and Q3
        Q1 = np.percentile(frequencies, 25)
        Q3 = np.percentile(frequencies, 75)
        IQR = Q3 - Q1
        # Define the bounds for acceptable frequency values
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Determine which frequency values are within the bounds
        frequencies_condition = (frequencies >= lower_bound) & (
            frequencies <= upper_bound
        )
        # Identify damping outliers that do not satisfy the condition
        frequencies_outliers = indices[~frequencies_condition]
        # Assign label -1 to damping outliers
        labels[frequencies_outliers] = -1
    unique_labels = set(labels)
    unique_labels.discard(-1)
    clusters = {label: np.where(labels == label)[0] for label in unique_labels}
    return clusters, labels


# -----------------------------------------------------------------------------


def post_xi_IQR(clusters, labels, Xi_fl):
    """
    Filter clusters based on the interquartile range (IQR) of damping values.

    Parameters
    ----------
    clusters : dict
        Dictionary of clusters where keys are cluster labels and values are arrays of indices.
    labels : ndarray of shape (n_samples,)
        Array of cluster labels for each sample.
    Xi_fl : ndarray of shape (n_samples,)
        Damping values corresponding to each sample.

    Returns
    -------
    clusters : dict
        Updated dictionary of clusters with outliers removed.
    labels : ndarray of shape (n_samples,)
        Updated cluster labels with outliers assigned -1.
    """
    # Damping-based filtering
    for _, indices in clusters.items():
        # Extract damping values for the current cluster
        damping = Xi_fl[indices].squeeze()
        # Calculate Q1 and Q3
        Q1 = np.percentile(damping, 25)
        Q3 = np.percentile(damping, 75)
        IQR = Q3 - Q1
        # Define the bounds for acceptable damping values
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Determine which damping values are within the bounds
        damping_condition = (damping >= lower_bound) & (damping <= upper_bound)
        # Identify damping outliers that do not satisfy the condition
        damping_outliers = indices[~damping_condition]
        # Assign label -1 to damping outliers
        labels[damping_outliers] = -1
    unique_labels = set(labels)
    unique_labels.discard(-1)
    clusters = {label: np.where(labels == label)[0] for label in unique_labels}
    return clusters, labels


# -----------------------------------------------------------------------------


def post_min_size(clusters, labels, min_size):
    """
    Filter clusters based on a minimum cluster size.

    Parameters
    ----------
    clusters : dict
        Dictionary of clusters where keys are cluster labels and values are arrays of indices.
    labels : ndarray of shape (n_samples,)
        Array of cluster labels for each sample.
    min_size : int
        Minimum allowable cluster size.

    Returns
    -------
    clusters : dict
        Updated dictionary of clusters with small clusters removed.
    labels : ndarray of shape (n_samples,)
        Updated cluster labels with small clusters assigned -1.
    """
    unique_labels = set(labels)
    unique_labels.discard(-1)
    # Minimum size based filtering
    for label in unique_labels:
        # Find indices of the current cluster
        cluster_indices = np.where(labels == label)[0]
        # Check if the cluster size is below the threshold
        if len(cluster_indices) < min_size:
            # Assign label -1 to these indices
            labels[cluster_indices] = -1
    unique_labels = set(labels)
    unique_labels.discard(-1)
    clusters = {label: np.where(labels == label)[0] for label in unique_labels}
    return clusters, labels


# -----------------------------------------------------------------------------


def post_min_size_pctg(clusters, labels, min_pctg):
    """
    Filter clusters based on a percentage of the largest cluster size.

    Parameters
    ----------
    clusters : dict
        Dictionary of clusters where keys are cluster labels and values are arrays of indices.
    labels : ndarray of shape (n_samples,)
        Array of cluster labels for each sample.
    min_pctg : float
        Minimum allowable cluster size as a percentage of the largest cluster.

    Returns
    -------
    clusters : dict
        Updated dictionary of clusters with small clusters removed.
    labels : ndarray of shape (n_samples,)
        Updated cluster labels with small clusters assigned -1.
    """
    # Identify unique cluster labels (excluding noise/outliers labeled as -1)
    unique_labels = set(labels)
    unique_labels.discard(-1)

    # Determine the size of the largest cluster
    clusters_sizes = np.array(
        [len(np.where(labels == label)[0]) for label in unique_labels]
    )
    largest_cluster_size = np.max(clusters_sizes)

    # Calculate the minimum cluster size based on min_pctg and largest cluster size
    threshold = min_pctg * largest_cluster_size

    # Filter out clusters that do not meet the threshold
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        if len(cluster_indices) < threshold:
            labels[cluster_indices] = -1

    # Update the clusters dictionary and labels
    unique_labels = set(labels)
    unique_labels.discard(-1)
    clusters = {label: np.where(labels == label)[0] for label in unique_labels}

    return clusters, labels


# -----------------------------------------------------------------------------


def post_min_size_kmeans(labels):
    """
    Filter clusters based on size using k-means clustering.

    Parameters
    ----------
    labels : ndarray of shape (n_samples,)
        Array of cluster labels for each sample.

    Returns
    -------
    clusters : dict
        Updated dictionary of clusters with smaller clusters removed.
    labels : ndarray of shape (n_samples,)
        Updated cluster labels with small clusters assigned -1.
    """
    # Identify unique cluster labels (excluding noise/outliers labeled as -1)
    unique_labels = set(labels)
    unique_labels.discard(-1)
    clusters_sizes = np.array(
        [len(np.where(labels == label)[0]) for label in unique_labels]
    )
    unique_labels = sorted(unique_labels)  # Convert to a list for indexing
    # labels from kmeans
    labels_kmeans = kmeans(clusters_sizes.reshape(-1, 1))

    # Retain only clusters that are assigned label 0 by kmeans
    keep_clusters = [unique_labels[i] for i, lm in enumerate(labels_kmeans) if lm == 1]

    # Assign -1 to all clusters not in keep_clusters
    for label_ in unique_labels:
        if label_ not in keep_clusters:
            labels[np.where(labels == label_)[0]] = -1
    unique_labels = set(labels)
    unique_labels.discard(-1)
    clusters = {label: np.where(labels == label)[0] for label in unique_labels}
    return clusters, labels


# -----------------------------------------------------------------------------


def post_min_size_gmm(labels):
    """
    Filter clusters based on size using Gaussian Mixture Model (GMM).

    Parameters
    ----------
    labels : ndarray of shape (n_samples,)
        Array of cluster labels for each sample.

    Returns
    -------
    clusters : dict
        Updated dictionary of clusters with smaller clusters removed.
    labels : ndarray of shape (n_samples,)
        Updated cluster labels with small clusters assigned -1.
    """
    # Identify unique cluster labels (excluding noise/outliers labeled as -1)
    unique_labels = set(labels)
    unique_labels.discard(-1)
    clusters_sizes = np.array(
        [len(np.where(labels == label)[0]) for label in unique_labels]
    )
    unique_labels = sorted(unique_labels)  # Convert to a list for indexing
    # labels from gaussian mixture
    labels_gmm = GMM(clusters_sizes.reshape(-1, 1))

    # Retain only clusters that are assigned label 0 by GMM
    keep_clusters = [unique_labels[i] for i, lm in enumerate(labels_gmm) if lm == 1]

    # Assign -1 to all clusters not in keep_clusters
    for label_ in unique_labels:
        if label_ not in keep_clusters:
            labels[np.where(labels == label_)[0]] = -1
    unique_labels = set(labels)
    unique_labels.discard(-1)
    clusters = {label: np.where(labels == label)[0] for label in unique_labels}
    return clusters, labels


# -----------------------------------------------------------------------------


def post_merge_similar(clusters, labels, dtot, merge_dist):
    """
    Merge clusters that are similar based on inter-medoid distances.

    Parameters
    ----------
    clusters : dict
        Dictionary of clusters where keys are cluster labels and values are arrays of indices.
    labels : ndarray of shape (n_samples,)
        Array of cluster labels for each sample.
    dtot : ndarray of shape (n_samples, n_samples)
        Pairwise distance matrix.
    merge_dist : float
        Maximum distance threshold for merging clusters.

    Returns
    -------
    clusters : dict
        Updated dictionary of clusters after merging.
    labels : ndarray of shape (n_samples,)
        Updated cluster labels reflecting merged clusters.
    """
    from itertools import combinations

    # Merge similar clusters
    # Compute initial medoids for each cluster
    medoids = {}
    for label, indices in clusters.items():
        submatrix = dtot[np.ix_(indices, indices)]
        total_distances = submatrix.sum(axis=1)
        medoid_index = indices[np.argmin(total_distances)]
        medoids[label] = medoid_index

    # Compute inter-medoid distances
    medoid_indices = list(medoids.values())
    medoid_distances = dtot[np.ix_(medoid_indices, medoid_indices)]

    # Extract medoid labels
    medoid_labels = list(medoids.keys())

    # Find pairs of clusters to merge based on inter-medoid distances
    pairs_to_merge = [
        (i, j)
        for i, j in combinations(range(len(medoid_indices)), 2)
        if medoid_distances[i, j] < merge_dist
    ]

    # Initialize Union-Find with current cluster labels
    uf = UnionFind(medoid_labels)

    # Perform unions for clusters to merge
    for i, j in pairs_to_merge:
        label_i = medoid_labels[i]
        label_j = medoid_labels[j]
        uf.union(label_i, label_j)

    # Create a mapping from old labels to merged labels
    label_mapping = {}
    for label in medoid_labels:
        root = uf.find(label)
        label_mapping[label] = root

    # Update labels using label_mapping
    for old_label, new_label in label_mapping.items():
        labels[labels == old_label] = new_label

    # Ensure that any labels not involved in merging remain unchanged
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Exclude noise label

    clusters = {label: np.where(labels == label)[0] for label in unique_labels}
    return clusters, labels


# -----------------------------------------------------------------------------


def post_1xorder(clusters, labels, dtot, order_fl):
    """
    Ensure only one sample per order in each cluster.

    Parameters
    ----------
    clusters : dict
        Dictionary of clusters where keys are cluster labels and values are arrays of indices.
    labels : ndarray of shape (n_samples,)
        Array of cluster labels for each sample.
    dtot : ndarray of shape (n_samples, n_samples)
        Pairwise distance matrix.
    order_fl : ndarray of shape (n_samples,)
        Order values corresponding to each sample.

    Returns
    -------
    clusters : dict
        Updated dictionary of clusters with only one sample per order.
    labels : ndarray of shape (n_samples,)
        Updated cluster labels reflecting refined clusters.
    """
    medoids = {}
    for label, indices in clusters.items():
        submatrix = dtot[np.ix_(indices, indices)]
        total_distances = submatrix.sum(axis=1)
        medoid_index = indices[np.argmin(total_distances)]
        medoids[label] = medoid_index
    # Keep only one pole per cluster per order
    # Remove duplicates within each cluster based on 'order', keeping the closest to the medoid
    updated_clusters = {}
    for label, indices in clusters.items():
        if len(indices) == 0:
            continue  # Skip empty clusters

        # Compute new medoid for the merged cluster
        submatrix = dtot[np.ix_(indices, indices)]
        total_distances = submatrix.sum(axis=1)
        medoid_index = indices[np.argmin(total_distances)]
        medoids[label] = medoid_index  # Update medoid

        # Get orders and distances to new medoid for elements in the cluster
        orders_in_cluster = order_fl[indices]
        distances_to_medoid = dtot[indices, medoid_index]

        # Create a DataFrame for easy manipulation
        data = pd.DataFrame(
            {
                "index": indices,
                "order": orders_in_cluster.squeeze(),
                "distance_to_medoid": distances_to_medoid,
            }
        )

        # Sort by distance to medoid and drop duplicates based on 'order'
        data_unique = data.sort_values("distance_to_medoid").drop_duplicates(
            subset=["order"], keep="first"
        )

        # Update the cluster with unique elements
        updated_clusters[label] = data_unique["index"].values

    # Update labels to reflect the refined clusters
    labels[:] = -1  # Reset labels
    for label, indices in updated_clusters.items():
        labels[indices] = label

    unique_labels = set(labels)
    unique_labels.discard(-1)
    clusters = {label: np.where(labels == label)[0] for label in unique_labels}
    return clusters, labels


# -----------------------------------------------------------------------------


def post_MTT(clusters, labels, flattened_results):
    """
    Remove outliers using the Modified Thompson Tau technique.

    Parameters
    ----------
    clusters : dict
        Dictionary of clusters where keys are cluster labels and values are arrays of indices.
    labels : ndarray of shape (n_samples,)
        Array of cluster labels for each sample.
    flattened_results : tuple of ndarray
        Tuple containing:
        - Fn_fl : ndarray of shape (n_samples,)
            Frequencies corresponding to each sample.
        - Xi_fl : ndarray of shape (n_samples,)
            Damping values corresponding to each sample.

    Returns
    -------
    clusters : dict
        Updated dictionary of clusters with outliers removed.
    labels : ndarray of shape (n_samples,)
        Updated cluster labels with outliers assigned -1.
    """
    Fn_fl, Xi_fl = flattened_results
    # Removing outliers with the modified Thompson Tau Techinique (Neu 2017)
    for _, indices in clusters.items():
        # print(indices)
        freqs = Fn_fl[indices]
        xis = Xi_fl[indices]

        # Apply to both frequency and damping
        new_indices_fn = MTT(freqs, indices)
        new_indices_xi = MTT(xis, indices)
        # intersect the indices
        indices1 = np.intersect1d(new_indices_fn, new_indices_xi)
        # mask out the outliers
        mask = np.isin(indices, indices1, invert=True)
        labels[indices[mask]] = -1
        unique_labels = set(labels)
        unique_labels.discard(-1)
        clusters = {label: np.where(labels == label)[0] for label in unique_labels}
    return clusters, labels


# -----------------------------------------------------------------------------


def post_adjusted_boxplot(clusters, labels, flattened_results):
    """
    Remove outliers using the adjusted boxplot method.

    For each cluster, the function computes the adjusted boxplot boundaries for both frequency and damping,
    then marks as outliers those observations that do not lie within the respective inlier intervals for both
    measures. Outliers are assigned a label of -1. The clusters dictionary is updated to include only the remaining
    (non-outlier) indices.

    Parameters
    ----------
    clusters : dict
        Dictionary of clusters where keys are cluster labels and values are arrays of indices.
    labels : ndarray of shape (n_samples,)
        Array of cluster labels for each sample.
    flattened_results : tuple of ndarray
        Tuple containing:
          - Fn_fl : ndarray of shape (n_samples,)
              Frequencies corresponding to each sample.
          - Xi_fl : ndarray of shape (n_samples,)
              Damping values corresponding to each sample.

    Returns
    -------
    clusters : dict
        Updated dictionary of clusters with outliers removed.
    labels : ndarray of shape (n_samples,)
        Updated cluster labels with outliers assigned -1.
    """
    Fn_fl, Xi_fl = flattened_results

    # Process each cluster separately
    for _, indices in clusters.items():
        # Select the values corresponding to the current cluster.
        freqs = Fn_fl[indices].squeeze()
        xis = Xi_fl[indices].squeeze()

        # Compute the adjusted boxplot bounds for frequencies.
        lower_fn, upper_fn = adjusted_boxplot_bounds(freqs)
        # Determine which frequency values are inliers.
        inliers_fn = indices[(freqs >= lower_fn) & (freqs <= upper_fn)]

        # Compute the adjusted boxplot bounds for damping.
        lower_xi, upper_xi = adjusted_boxplot_bounds(xis)
        # Determine which damping values are inliers.
        inliers_xi = indices[(xis >= lower_xi) & (xis <= upper_xi)]

        # Keep only the indices that are inliers for both criteria.
        inliers_common = np.intersect1d(inliers_fn, inliers_xi)

        # Mark as outliers those indices in the cluster that are not in the intersection.
        mask_outliers = np.isin(indices, inliers_common, invert=True)
        labels[indices[mask_outliers]] = -1

    # Rebuild the clusters dictionary excluding the outliers (labeled -1).
    unique_labels = set(labels)
    unique_labels.discard(-1)
    clusters = {label: np.where(labels == label)[0] for label in unique_labels}

    return clusters, labels


# -----------------------------------------------------------------------------


def output_selection(select, clusters, flattened_results, medoid_indices):
    """
    Select output results based on the specified selection method.

    Parameters
    ----------
    select : str
        Selection method. Options include:
        - "medoid": Select medoids of clusters.
        - "avg": Select average values of clusters.
        - "fn_mean_close": Select samples with frequency closest to cluster mean.
        - "xi_med_close": Select samples with damping closest to cluster median.
    clusters : dict
        Dictionary of clusters where keys are cluster labels and values are arrays of indices.
    flattened_results : tuple of ndarray
        Tuple containing:
        - Fn_fl : ndarray of shape (n_samples,)
            Frequencies corresponding to each sample.
        - Xi_fl : ndarray of shape (n_samples,)
            Damping values corresponding to each sample.
        - Phi_fl : ndarray (n_samples, n_channels)
            Mode shape corresponding to each sample.
        - order_fl : ndarray of shape (n_samples,)
            Order values corresponding to each sample.
    medoid_indices : ndarray, optional
        Indices of medoids for each cluster.

    Returns
    -------
    Fn_out : ndarray
        Selected frequency values based on the chosen method.
    Xi_out : ndarray
        Selected damping values based on the chosen method.
    Phi_out : ndarray
        Selected additional feature values based on the chosen method.
    order_out : ndarray
        Selected order values based on the chosen method.
    """
    Fn_fl, Xi_fl, Phi_fl, order_fl = flattened_results
    # SELECTION OF RESULTS
    if select == "medoid":
        # Extract final Fn and order values for the medoids
        features = [Fn_fl, Xi_fl, Phi_fl, order_fl]
        Fn_out, Xi_out, Phi_out, order_out = filter_fl_list(features, medoid_indices)

    elif select == "avg":
        Fn_out, Xi_out, Phi_out = [], [], []
        for _, indices in clusters.items():
            Fn_out.append(np.mean(Fn_fl[indices]))
            Xi_out.append(np.mean(Xi_fl[indices]))
            Phi_out.append(np.mean(Phi_fl[indices, :], axis=0))

        Fn_out, Xi_out, Phi_out = (
            np.array(Fn_out),
            np.array(Xi_out),
            np.array(Phi_out),
        )
        order_out = None

    elif select == "fn_mean_close":
        final_indices = []
        for _, indices in clusters.items():
            mean_f = np.mean(Fn_fl[indices])
            final_indices.append(indices[np.abs(Fn_fl[indices] - mean_f).argmin()])
        final_indices = np.array(final_indices)

        final_indices = final_indices[~np.isnan(final_indices)].astype(int).reshape(-1, 1)
        features = [Fn_fl, Xi_fl, Phi_fl, order_fl]
        Fn_out, Xi_out, Phi_out, order_out = filter_fl_list(features, final_indices)

    elif select == "xi_med_close":
        final_indices = []

        for _, indices in clusters.items():
            median_Xi = np.median(Xi_fl[indices])
            final_indices.append(indices[np.abs(Xi_fl[indices] - median_Xi).argmin()])
        final_indices = np.array(final_indices)

        final_indices = final_indices[~np.isnan(final_indices)].astype(int).reshape(-1, 1)
        features = [Fn_fl, Xi_fl, Phi_fl, order_fl]
        Fn_out, Xi_out, Phi_out, order_out = filter_fl_list(features, final_indices)
    return Fn_out, Xi_out, Phi_out, order_out


# -----------------------------------------------------------------------------


def MTT(arr, indices, alpha=0.01):
    """
    Apply the Modified Thompson Tau technique to remove outliers.

    Parameters
    ----------
    arr : ndarray
        Array of values to filter.
    indices : ndarray
        Indices of the values in the original dataset.
    alpha : float, optional
        Significance level for outlier detection. Default is 0.01.

    Returns
    -------
    ind : ndarray
        Indices of values that are not outliers.
    """
    ind = indices.copy()
    arr = arr.copy()
    while True:
        n = len(arr)
        if n < 3:
            break  # Not enough data to perform the test

        mean = arr.mean()
        std = arr.std(ddof=1)
        deviations = np.abs(arr - mean)
        max_dev = deviations.max()
        max_dev_index = deviations.argmax()

        # Adjust significance level
        t = stats.t.ppf(1 - alpha / (2), n - 2)
        tau = (t * (n - 1)) / (np.sqrt(n) * np.sqrt(n - 2 + t**2))

        if max_dev / std > tau:
            # Remove outlier
            arr = np.delete(arr, max_dev_index)
            ind = np.delete(ind, max_dev_index)
        else:
            break  # No more outliers
    return ind


# -----------------------------------------------------------------------------


def adjusted_boxplot_bounds(data):
    """
    Compute the lower and upper fences of the adjusted boxplot for skewed distributions.

    For MC >= 0:
      lower_bound = Q1 - 1.5 * exp(-4 * MC) * IQR
      upper_bound = Q3 + 1.5 * exp(3 * MC) * IQR
    For MC < 0:
      lower_bound = Q1 - 1.5 * exp(-3 * MC) * IQR
      upper_bound = Q3 + 1.5 * exp(4 * MC) * IQR

    Parameters
    ----------
    data : array-like
        1D numeric data.

    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    """
    data = np.asarray(data)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # Compute medcouple (MC) using a simple O(n^2) implementation.
    # For large datasets an optimized implementation is recommended.
    def medcouple(x):
        x = np.sort(np.asarray(x))
        median = np.median(x)
        left = x[x <= median]
        right = x[x >= median]
        L = left[:, np.newaxis]
        R = right[np.newaxis, :]
        diff = R - L
        with np.errstate(divide="ignore", invalid="ignore"):
            h = (R + L - 2 * median) / diff
        h[diff == 0] = 0.0
        return np.median(h)

    MC = medcouple(data)

    if MC >= 0:
        lower_bound = Q1 - 1.5 * np.exp(-4 * MC) * IQR
        upper_bound = Q3 + 1.5 * np.exp(3 * MC) * IQR
    else:
        lower_bound = Q1 - 1.5 * np.exp(-3 * MC) * IQR
        upper_bound = Q3 + 1.5 * np.exp(4 * MC) * IQR

    return lower_bound, upper_bound


# -----------------------------------------------------------------------------


def filter_fl_list(fl_list, stab_lab):
    """
    Filter and extract stable elements from a list of feature arrays.

    Parameters
    ----------
    fl_list : list of ndarray
        List of feature arrays, where each array represents a specific feature.
    stab_lab : ndarray
        Indices of stable elements in the feature arrays.

    Returns
    -------
    list of ndarray
        List of extracted feature arrays, where only stable elements are retained.
    """
    return [fl[stab_lab].squeeze() if fl is not None else None for fl in fl_list]


# -----------------------------------------------------------------------------


def vectorize_features(features, non_nan_index):
    """
    Vectorize features by flattening them and indexing valid (non-NaN) entries.

    Parameters
    ----------
    features : list of np.ndarray
        A list of 2D or 3D arrays where each array represents a feature.
    non_nan_index : np.ndarray
        Indices of non-NaN entries in a flattened array.

    Returns
    -------
    list of np.ndarray
        List where each feature is vectorized and removed of the nan entries.
        If a feature is None, its corresponding output will be None.
        For 2D features, the output is a 1D array.
        For 3D features, the output is a 2D array with shape (len(non_nan_index), feature.shape[2]).
    """
    output = []
    for feature in features:
        if feature is None:
            output.append(None)
            continue
        if feature.ndim == 2:
            # For 2D features, flatten and index
            vec_feature = feature.flatten(order="f")[non_nan_index]
        elif feature.ndim == 3:
            # For 3D features, reshape and index
            reshaped_feature = feature.reshape(-1, feature.shape[2], order="f").real
            vec_feature = reshaped_feature[non_nan_index]
        else:
            raise ValueError("The input must be either a 2D or 3D array.")
        output.append(vec_feature)
    return output


# -----------------------------------------------------------------------------


def build_tot_simil(distances, data_dict, len_fl, weights=None):
    """
    Compute a total similarity matrix by combining multiple distance matrices with weights.

    Parameters
    ----------
    distances : list of str
        A list of distance metrics (e.g., 'dfn', 'dxi', 'dlambda', 'dMAC', 'dMPC', 'dMPD').
    data_dict : dict
        Dictionary containing data arrays corresponding to each distance metric.
        Expected keys include 'Fn_fl', 'Xi_fl', 'Lambda_fl', 'Phi_fl', 'MPC_fl', and 'MPD_fl'.
    len_fl : int
        The size of the resulting similarity matrix (len_fl x len_fl).
    weights : np.ndarray, optional
        Weights for each distance metric. Must sum to 1 if specified. By default None.

    Returns
    -------
    np.ndarray
        A total similarity matrix (square, of shape (len_fl, len_fl)).
        Values are scaled between 0 and 1.

    Raises
    ------
    AttributeError
        If the weights do not sum to 1 or if the lengths of `distances` and `weights` do not match.
    """
    if weights is None or weights == "tot_one":
        weights = np.ones(len(distances)) / len(distances)
    elif sum(weights) != 1.0:
        raise AttributeError(
            "the sum of the weights must be one, when using similarity matrices"
        )
    elif len(weights) != len(distances):
        raise AttributeError(
            f"distances and weigths must have the same length. \
distances length: {len(distances)} != weights length: {len(weights)}"
        )
    dtot = np.zeros((len_fl, len_fl))
    for i, dist in enumerate(distances):
        if dist == "dfn":
            dtot += dist_all_f(data_dict["Fn_fl"]) * weights[i]
        elif dist == "dxi":
            dtot += dist_all_f(data_dict["Xi_fl"]) * weights[i]
        elif dist == "dlambda":
            dtot += dist_all_complex(data_dict["Lambda_fl"]) * weights[i]
        elif dist == "dMAC":
            dtot += dist_all_phi(data_dict["Phi_fl"]) * weights[i]
        elif dist == "dMPC":
            dtot += dist_all_f(data_dict["MPC_fl"]) * weights[i]
        elif dist == "dMPD":
            dtot += dist_all_f(data_dict["MPD_fl"]) * weights[i]
    return 1 - dtot


# -----------------------------------------------------------------------------


def build_tot_dist(distances, data_dict, len_fl, weights=None, sqrtsqr=False):
    """
    Compute a total distance matrix by combining multiple distance matrices with weights.

    Parameters
    ----------
    distances : list of str
        A list of distance metrics (e.g., 'dfn', 'dxi', 'dlambda', 'dMAC', 'dMPC', 'dMPD').
    data_dict : dict
        Dictionary containing data arrays corresponding to each distance metric.
        Expected keys include 'Fn_fl', 'Xi_fl', 'Lambda_fl', 'Phi_fl', 'MPC_fl', and 'MPD_fl'.
    len_fl : int
        The size of the resulting distance matrix (len_fl x len_fl).
    weights : np.ndarray, optional
        Weights for each distance metric. By default, equal weights are assigned to all metrics.
    sqrtsqr : bool, optional
        Whether to apply a squared-sum approach for combining distances, by default False.

    Returns
    -------
    np.ndarray
        A total distance matrix (square, of shape (len_fl, len_fl)).

    Raises
    ------
    AttributeError
        If the lengths of `distances` and `weights` do not match.
    """
    if weights is None:
        weights = np.ones(len(distances))
    elif weights == "tot_one":
        weights = np.ones(len(distances)) / len(distances)
    elif len(weights) != len(distances):
        raise AttributeError(
            f"distances and weigths must have the same length. \
distances length: {len(distances)} != weights length: {len(weights)}"
        )
    dtot = np.zeros((len_fl, len_fl))
    for i, dist in enumerate(distances):
        if dist == "dfn":
            if sqrtsqr is not False:
                dtot += (dist_all_f(data_dict["Fn_fl"])) ** 2 * weights[i]
            else:
                dtot += dist_all_f(data_dict["Fn_fl"]) * weights[i]
        elif dist == "dxi":
            if sqrtsqr is not False:
                dtot += (dist_all_f(data_dict["Xi_fl"])) ** 2 * weights[i]
            else:
                dtot += dist_all_f(data_dict["Xi_fl"]) * weights[i]
        elif dist == "dlambda":
            if sqrtsqr is not False:
                dtot += (dist_all_complex(data_dict["Lambda_fl"])) ** 2 * weights[i]
            else:
                dtot += dist_all_complex(data_dict["Lambda_fl"]) * weights[i]
        elif dist == "dMAC":
            if sqrtsqr is not False:
                dtot += (dist_all_phi(data_dict["Phi_fl"])) ** 2 * weights[i]
            else:
                dtot += dist_all_phi(data_dict["Phi_fl"]) * weights[i]
        elif dist == "dMPC":
            if sqrtsqr is not False:
                dtot += (dist_all_f(data_dict["MPC_fl"])) ** 2 * weights[i]
            else:
                dtot += dist_all_f(data_dict["MPC_fl"]) * weights[i]
        elif dist == "dMPD":
            if sqrtsqr is not False:
                dtot += (dist_all_f(data_dict["MPD_fl"])) ** 2 * weights[i]
            else:
                dtot += dist_all_f(data_dict["MPD_fl"]) * weights[i]
    if sqrtsqr is not False:
        return np.sqrt(dtot)
    else:
        return dtot


# -----------------------------------------------------------------------------


def build_feature_array(distances, data_dict, ordmax, step, transform=None):
    """
    Build a feature array from multiple distance metrics with optional transformations.

    Parameters
    ----------
    distances : list of str
        A list of distance metrics to compute features (e.g., 'dfn', 'dxi', 'dlambda', 'dMAC', 'dMPC', 'dMPD').
    data_dict : dict
        Dictionary containing data arrays corresponding to each distance metric.
        Expected keys include 'Fns', 'Xis', 'Lambdas', 'Phis', 'MPC', and 'MPD'.
    ordmax : int
        Maximum order to consider for feature computation.
    step : int
        Step size for iterating through model orders.
    transform : str, optional
        Transformation method for features, such as 'box-cox', by default None.

    Returns
    -------
    np.ndarray
        A 2D feature array, where each column corresponds to a specific distance metric.

    Raises
    ------
    AttributeError
        If the `transform` is not 'box-cox' or None.
    """
    if transform is not None and transform != "box-cox":
        raise AttributeError(
            f"{transform} is not a valid attribute. Supported transform are `None` and `box-cox`."
        )
    feat_list = []
    for dist in distances:
        if dist == "dfn":
            feat = dist_n_n1_f(data_dict["Fns"], 0, ordmax, step).reshape(-1, 1)
            if transform is not None:
                feat = power_transform(feat, method="box-cox")
            feat_list.append(feat)
        elif dist == "dxi":
            feat = dist_n_n1_f(data_dict["Xis"], 0, ordmax, step).reshape(-1, 1)
            if transform is not None:
                feat = power_transform(feat, method="box-cox")
            feat_list.append(feat)
        elif dist == "dlambda":
            feat = dist_n_n1_f_complex(data_dict["Lambdas"], 0, ordmax, step).reshape(
                -1, 1
            )
            if transform is not None:
                feat = power_transform(feat, method="box-cox")
            feat_list.append(feat)
        elif dist == "dMAC":
            feat = dist_n_n1_phi(data_dict["Phis"], 0, ordmax, step).reshape(-1, 1)
            if transform is not None:
                feat = power_transform(feat, method="box-cox")
            feat_list.append(feat)
        elif dist == "dMPC":
            feat = dist_n_n1_f(data_dict["MPC"], 0, ordmax, step).reshape(-1, 1)
            if transform is not None:
                feat = power_transform(feat, method="box-cox")
            feat_list.append(feat)
        elif dist == "dMPD":
            feat = dist_n_n1_f(data_dict["MPD"], 0, ordmax, step).reshape(-1, 1)
            if transform is not None:
                feat = power_transform(feat, method="box-cox")
            feat_list.append(feat)
        elif dist == "MPC":
            MPC = data_dict["MPC"]
            non_nan_index = np.argwhere(~np.isnan(MPC.flatten(order="f")))
            MPC_fl = vectorize_features([MPC], non_nan_index)
            feat = MPC_fl[0].reshape(-1, 1)
            if transform is not None:
                feat = power_transform(feat, method="box-cox")
            feat_list.append(feat)
        elif dist == "MPD":
            MPD = data_dict["MPD"]
            non_nan_index = np.argwhere(~np.isnan(MPD.flatten(order="f")))
            MPD_fl = vectorize_features([MPD], non_nan_index)
            feat = MPD_fl[0].reshape(-1, 1)
            if transform is not None:
                feat = power_transform(feat, method="box-cox")
            feat_list.append(feat)
    return np.hstack(feat_list)


# -----------------------------------------------------------------------------


def oned_to_2d(list_array1d, order, shape, step):
    """
    Convert a 1D array to a 2D array based on order and shape.

    Parameters
    ----------
    array1d : np.ndarray or None
        The input 1D array to reshape.
    order : np.ndarray
        Model order array corresponding to the data points in `array1d`.
    shape : tuple of int
        The desired shape of the output 2D array.
    step : int
        Step size for iterating through model orders.

    Returns
    -------
    np.ndarray
        A 2D array reshaped from `array1d`, with NaNs where no data is present.
    """
    list_array2d = []
    for array1d in list_array1d:
        # Check if the input array is None
        if array1d is None:
            return None
        # Initialize the 2D array with NaNs
        Full = np.full(shape, np.nan, dtype=array1d.dtype)
        # Iterate over each column index
        for i in range(shape[1]):
            # Find indices where order equals the current column index
            indices = np.where(order / step == i)[0]
            values = array1d[indices]
            if len(values) == 0:
                # No values to insert for this column
                continue
            # Assign the values to the column in the 2D array
            Full[: len(values), i] = values.squeeze()
        list_array2d.append(Full)
    return list_array2d


# -----------------------------------------------------------------------------


class UnionFind:
    """
    A Union-Find data structure for efficient disjoint set operations.

    Attributes
    ----------
    parent : dict
        Maps each element to its parent in the disjoint-set forest.

    Methods
    -------
    find(elem):
        Find the root of the set containing `elem` with path compression.
    union(elem1, elem2):
        Merge the sets containing `elem1` and `elem2`.
    """

    def __init__(self, elements):
        self.parent = {elem: elem for elem in elements}

    def find(self, elem):
        if self.parent[elem] != elem:
            self.parent[elem] = self.find(self.parent[elem])  # Path compression
        return self.parent[elem]

    def union(self, elem1, elem2):
        root1 = self.find(elem1)
        root2 = self.find(elem2)
        if root1 != root2:
            # Union by label (you can choose other heuristics)
            if root1 < root2:
                self.parent[root2] = root1
            else:
                self.parent[root1] = root2


# -----------------------------------------------------------------------------


# Matrice distanze con scikit + funzione
def relative_difference_abs(x, y):
    """
    Compute the relative absolute difference between two values.

    Parameters
    ----------
    x : ndarray
        First input value, expected to be a single-element array.
    y : ndarray
        Second input value, expected to be a single-element array.

    Returns
    -------
    float
        Relative absolute difference between `x` and `y`. Returns infinity if `x` is zero.
    """
    xi = x[0]
    xj = y[0]
    if xi == 0:
        return np.inf
    return abs((xi - xj) / np.max((xi, xj)))


def MAC_difference(x, y):
    """
    Compute the Modal Assurance Criterion (MAC) difference between two mode shapes.

    Parameters
    ----------
    x : ndarray
        First mode shape vector.
    y : ndarray
        Second mode shape vector.

    Returns
    -------
    float
        The MAC difference between `x` and `y`, defined as `1 - MAC(x, y)`.
    """
    return 1 - gen.MAC(x, y)


# -----------------------------------------------------------------------------


def dist_all_f(array):
    """
    Compute a pairwise distance matrix for a flattened 1D array using relative absolute difference.

    Parameters
    ----------
    array : np.ndarray
        Input array of shape (M,) or (M, N) to compute pairwise distances.
        If 2D, the array is flattened column-wise (Fortran order).

    Returns
    -------
    np.ndarray
        Pairwise distance matrix of shape (P, P), where P is the number of non-NaN entries in `array`.
    """
    if array.ndim == 2:
        # Vettorializzazione
        array = array.flatten(order="f")
    # Rimuovo indici non nan
    non_nan_index = np.argwhere(~np.isnan(array))
    array = array[non_nan_index]
    # calculate distance matrix
    dist = pairwise_distances(array.reshape(-1, 1), metric=relative_difference_abs)
    return dist


# -----------------------------------------------------------------------------


def dist_all_phi(array):
    """
    Compute a pairwise distance matrix for 3D mode shape data using the MAC difference.

    Parameters
    ----------
    array : np.ndarray
        Input 3D array of mode shapes with shape (M, N, K).
        Each slice `array[i, :, :]` represents the mode shape data for one observation.

    Returns
    -------
    np.ndarray
        Pairwise distance matrix of shape (P, P), where P is the number of non-NaN rows in the reshaped array.
    """
    if array.ndim == 3:
        # Vettorializzazione e prendo solo i valori reali
        array = array.reshape(-1, array.shape[2], order="f").real
    # Rimuovo righe(=forme modali) nan
    array = array[~np.isnan(array).any(axis=1)]
    dist = pairwise_distances(array, metric=MAC_difference)
    return dist


# -----------------------------------------------------------------------------


def dist_all_complex(complex_array):
    """
    Compute pairwise relative distances for a 1D array of complex numbers.

    Parameters
    ----------
    complex_array : np.ndarray
        Input array of complex numbers of shape (M,) or (M, N), flattened to 1D if 2D.

    Returns
    -------
    np.ndarray
        Pairwise relative distance matrix of shape (P, P), where P is the number of valid (non-NaN) complex entries.

    Notes
    -----
    - Relative distance is computed as the modulus of the difference divided by the maximum modulus.
    - Invalid values (NaNs or infinite values) are handled gracefully and set to 0.
    """
    if complex_array.ndim == 2:
        # Vettorializzazione
        complex_array = complex_array.flatten(order="f")
    # Remove NaN entries (if any)
    valid_mask = ~np.isnan(complex_array.real) & ~np.isnan(complex_array.imag)
    valid_complex = complex_array[valid_mask]

    # Number of valid complex numbers
    n = len(valid_complex)

    if n == 0:
        return np.array([])

    # Compute pairwise differences
    diff_matrix = valid_complex[:, np.newaxis] - valid_complex[np.newaxis, :]

    # Compute pairwise distances (modulus of differences)
    distance_matrix = np.abs(diff_matrix)

    # Compute pairwise maximum modulus
    modulus_matrix = np.maximum(
        np.abs(valid_complex)[:, np.newaxis], np.abs(valid_complex)[np.newaxis, :]
    )

    # Compute relative distances, handling division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.divide(distance_matrix, modulus_matrix)
        dist[~np.isfinite(dist)] = 0.0  # Set infinities and NaNs to 0

    return dist


# -----------------------------------------------------------------------------


def dist_n_n1_f(array, ordmin, ordmax, step):
    """
    Compute distances between successive columns of a 2D array using relative differences.

    Parameters
    ----------
    array : np.ndarray
        Input 2D array of shape (M, N), where M is the number of rows and N is the number of columns.
    ordmin : int
        Minimum order for computing distances.
    ordmax : int
        Maximum order for computing distances.
    step : int
        Step size for iterating through model orders.

    Returns
    -------
    np.ndarray
        A 1D array of distances between successive columns, with NaN entries handled appropriately.
    """
    dist = np.full(array.shape, np.nan, dtype=float)
    for oo in range(ordmax, ordmin, -step):
        o = oo // step
        A_n = array[:, o].reshape(-1, 1)
        A_n1 = array[:, o - 1].reshape(-1, 1)
        for i in range(A_n.shape[0]):
            diff = np.abs(A_n1 - A_n[i])
            if np.all(np.isnan(diff)):
                continue
            idx = np.nanargmin(diff)
            max_val = np.nanmax([A_n[i], A_n1[idx]])
            # Calculate relative distances
            di = np.abs(A_n[i] - A_n1[idx]) / max_val
            dist[i, o] = di[0]
    # Flatten the distance array and create a mask for non-NaN values
    dist_fl = dist.flatten(order="f")
    non_nan_mask = ~np.isnan(dist_fl)
    dist_non_nan = dist_fl[non_nan_mask]
    # Identify the first column that does not contain all NaNs
    non_all_nan = ~np.isnan(array).all(axis=0)
    first_non_all_nan_index = np.argmax(non_all_nan)
    column = array[:, first_non_all_nan_index]
    # Count the number of non-NaN elements in this column
    count_non_nan = np.sum(~np.isnan(column))

    # Prepend ones for the initial elements
    dist_non_nan = np.insert(dist_non_nan, 0, np.repeat(1.0, count_non_nan))
    return dist_non_nan


# -----------------------------------------------------------------------------


def dist_n_n1_phi(array, ordmin, ordmax, step):
    """
    Compute distances between successive columns of a 3D mode shape array using MAC differences.

    Parameters
    ----------
    array : np.ndarray
        Input 3D array of mode shapes with shape (M, N, K).
        Each slice `array[:, :, k]` represents the mode shape data for one observation.
    ordmin : int
        Minimum order for computing distances.
    ordmax : int
        Maximum order for computing distances.
    step : int
        Step size for iterating through model orders.

    Returns
    -------
    np.ndarray
        A 1D array of distances between successive columns, with NaN entries handled appropriately.
    """
    dist = np.full(array.shape[:2], np.nan, dtype=float)

    for oo in range(ordmax, ordmin, -step):
        o = oo // step
        A_n = array[:, o, :]
        A_n1 = array[:, o - 1, :]
        for i in range(A_n.shape[0]):
            # Compute the MAC distances
            distances = 1 - gen.MAC(A_n1.T, A_n[i])
            if np.all(np.isnan(distances)):
                continue
            idx = np.nanargmin(distances)
            di = distances[idx]
            dist[i, o] = di[0].real
    # Flatten the distance array and create a mask for non-NaN values
    dist_fl = dist.flatten(order="f")
    non_nan_mask = ~np.isnan(dist_fl)
    dist_non_nan = dist_fl[non_nan_mask]
    # Aggiungo tanti 1 quanti valori ci sono nella prima colonna che non contiene solo nan
    non_all_nan = ~np.isnan(array).all(axis=0)
    first_non_all_nan_index = np.argmax(non_all_nan, axis=0)[0]
    column = array[:, first_non_all_nan_index]
    count_non_nan = np.sum(~np.isnan(column), axis=0)[0]
    dist_non_nan = np.insert(dist_non_nan, 0, np.repeat(1.0, count_non_nan))
    return dist_non_nan


# -----------------------------------------------------------------------------


def dist_n_n1_f_complex(array, ordmin, ordmax, step):
    """
    Compute distances between successive columns of a 2D complex array using relative differences.

    Parameters
    ----------
    array : np.ndarray
        Input 2D array of complex numbers with shape (M, N).
        Each column represents a different order.
    ordmin : int
        Minimum order for computing distances.
    ordmax : int
        Maximum order for computing distances.
    step : int
        Step size for iterating through model orders.

    Returns
    -------
    np.ndarray
        A 1D array of relative distances between successive columns, with NaN entries handled appropriately.
    """
    # Initialize the distance array with NaNs
    dist = np.full(array.shape, np.nan, dtype=float)

    # Iterate from ordmax down to ordmin in steps of -step
    for oo in range(ordmax, ordmin, -step):
        o = oo // step
        # Get the current and previous columns
        A_n = array[:, o]  # Shape: (M,)
        A_n1 = array[:, o - 1]  # Shape: (M,)

        # Iterate over each element in the current column
        for i in range(len(A_n)):
            # Compute the absolute differences between A_n[i] and all elements in A_n1
            diff = np.abs(A_n1 - A_n[i])  # Shape: (M,)

            # Handle NaNs in diff
            if np.all(np.isnan(diff)):
                continue  # Skip if all differences are NaN

            # Find the index of the minimum difference
            idx = np.nanargmin(diff)
            # Get the closest value from the previous column
            A_n1_closest = A_n1[idx]

            # Compute the maximum magnitude for relative distance
            max_val = np.maximum(np.abs(A_n[i]), np.abs(A_n1_closest))

            # Avoid division by zero
            di = 0.0 if max_val == 0 else np.abs(A_n[i] - A_n1_closest) / max_val

            # Assign the computed distance
            dist[i, o] = di

    # Flatten the distance array in Fortran order (column-major)
    dist_fl = dist.flatten(order="f")
    # Create a mask for non-NaN values
    non_nan_mask = ~np.isnan(dist_fl)
    # Extract non-NaN distances
    dist_non_nan = dist_fl[non_nan_mask]

    # Identify the first column that does not contain all NaNs
    non_all_nan = ~np.isnan(array).all(axis=0)
    first_non_all_nan_index = np.argmax(non_all_nan)
    column = array[:, first_non_all_nan_index]
    # Count the number of non-NaN elements in this column
    count_non_nan = np.sum(~np.isnan(column))

    # Prepend ones for the initial elements
    dist_non_nan = np.insert(dist_non_nan, 0, np.repeat(1.0, count_non_nan))

    return dist_non_nan
