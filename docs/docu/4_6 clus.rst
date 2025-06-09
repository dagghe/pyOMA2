The ``clus`` module
-------------------

This module is a part of the pyOMA2 package and provides utility functions to support the implementation
of clustering algorithms.

Functions:
    - :func:`.kmeans`: Perform k-means clustering on the given feature array.
    - :func:`.GMM`: Perform Gaussian Mixture Model (GMM) clustering on the given feature array.
    - :func:`.hierarc`: Perform hierarchical clustering with specified parameters.
    - :func:`.spectral`: Perform spectral clustering with the given similarity matrix.
    - :func:`.affinity`: Perform affinity propagation clustering on the given similarity matrix.
    - :func:`.optics`: Perform OPTICS clustering on the given pairwise distance matrix.
    - :func:`.hdbscan`: Perform HDBSCAN clustering on the given pairwise distance matrix.
    - :func:`.reorder_clusters`: Reorder cluster labels based on ascending frequencies values.
    - :func:`.post_freq_lim`: Filter clusters based on specified frequency range.
    - :func:`.post_fn_med`: Filter clusters based on a median frequency threshold.
    - :func:`.post_fn_IQR`: Filter clusters based on the interquartile range (IQR) of frequencies.
    - :func:`.post_xi_IQR`: Filter clusters based on the interquartile range (IQR) of damping values.
    - :func:`.post_min_size`: Filter clusters based on a minimum cluster size.
    - :func:`.post_min_size_pctg`: Filter clusters based on a percentage of the largest cluster size.
    - :func:`.post_min_size_kmeans`: Filter clusters based on size using k-means clustering.
    - :func:`.post_min_size_gmm`: Filter clusters based on size using Gaussian Mixture Model (GMM).
    - :func:`.post_merge_similar`: Merge clusters that are similar based on inter-medoid distances.
    - :func:`.post_1xorder`: Ensure only one sample per order in each cluster.
    - :func:`.post_MTT`: Ensure only one sample per order in each cluster.
    - :func:`.output_selection`: Select output results based on the specified selection method.
    - :func:`.MTT`: Apply the Modified Thompson Tau technique to remove outliers.
    - :func:`.filter_fl_list`: Filter and extract stable elements from a list of feature arrays.
    - :func:`.vectorize_features`: Vectorize features by flattening them and indexing valid (non-NaN) entries.
    - :func:`.build_tot_simil`: Compute a total similarity matrix by combining multiple distance matrices with weights.
    - :func:`.build_tot_dist`: Compute a total distance matrix by combining multiple distance matrices with weights.
    - :func:`.build_feature_array`: Build a feature array from multiple distance metrics with optional transformations.
    - :func:`.oned_to_2d`: Convert a 1D array to a 2D array based on order and shape.
    - :class:`.UnionFind`: A Union-Find data structure for efficient disjoint set operations.
    - :func:`.relative_difference_abs`: Compute the relative absolute difference between two values.
    - :func:`.MAC_difference`: Compute the Modal Assurance Criterion (MAC) difference between two mode shapes.
    - :func:`.dist_all_f`: Compute a pairwise distance matrix for a flattened 1D array using relative absolute difference.
    - :func:`.dist_all_phi`: Compute a pairwise distance matrix for 3D mode shape data using the MAC difference.
    - :func:`.dist_all_complex`: Compute pairwise relative distances for a 1D array of complex numbers.
    - :func:`.dist_n_n1_f`: Compute distances between successive columns of a 2D array using relative differences.
    - :func:`.dist_n_n1_phi`: Compute distances between successive columns of a 3D mode shape array using MAC differences.
    - :func:`.dist_n_n1_f_complex`: Compute distances between successive columns of a 2D complex array using relative differences.
    - :func:`.dist_all_complex`: Compute pairwise relative distances for a 1D array of complex numbers.
    - :class:`.FuzzyCMeansClustering`: Fuzzy C-Means clustering class implementation.
    - :func:`.FCMeans`: Perform Fuzzy C-Means clustering on the given feature array.
    - :func:`.post_adjusted_boxplot`: Filter clusters using the adjusted boxplot method.
    - :func:`.adjusted_boxplot_bounds`: Compute adjusted boxplot bounds (used in outlier detection).

.. automodule:: pyoma2.functions.clus
   :members:
