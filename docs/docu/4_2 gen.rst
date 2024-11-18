The ``gen`` module
------------------

Part of the pyOMA2 package, this module provides general utility functions crucial for
implementational aspects of Operational Modal Analysis (OMA). These functions support
data preprocessing, mode shape merging, and key calculations such as the Modal Assurance
Criterion (MAC), Modal Scale Factor (MSF), and Modal Complexity Factor (MCF).

Functions:
    - :func:`.applymask`: Apply a mask to a list of arrays, filtering their values based on the mask.
    - :func:`.HC_conj`: Apply Hard validation Criteria, complex conjugates.
    - :func:`.HC_damp`: Apply Hard validation Criteria, damping.
    - :func:`.HC_phi_comp`: Apply Hard validation Criteria, mode shapes complexity.
    - :func:`.HC_CoV`: Apply Hard validation Criteria, covariance.
    - :func:`.SC_apply`: Apply Soft validation Criteria.
    - :func:`.dfphi_map_func`: Maps mode shapes to sensor locations and constraints.
    - :func:`.check_on_geo1`: Validates geometry1 data.
    - :func:`.check_on_geo2`: Validates geometry2 data.
    - :func:`.flatten_sns_names`: Ensures that sensors names is in the correct form.
    - :func:`.example_data`: Generates the example dataset.
    - :func:`.merge_mode_shapes`: Merges mode shapes from different setups into a unified mode shape array.
    - :func:`.MPC`: Calculate the Modal Phase Collinearity of a complex mode shape.
    - :func:`.MPD`: Calculate the Mean Phase Deviation of a complex mode shape.
    - :func:`.MSF`: Computes the Modal Scale Factor between two mode shape sets.
    - :func:`.MCF`: Determines the complexity of mode shapes.
    - :func:`.MAC`: Calculates the correlation between two sets of mode shapes.
    - :func:`.pre_multisetup`: Preprocesses data from multiple setups, distinguishing between reference and moving sensors.
    - :func:`.invperm`: Computes the inverse permutation of an array.
    - :func:`.find_map`: Establishes a mapping between two arrays based on sorting order.
    - :func:`.filter_data`: Apply a Butterworth filter to the input data.
    - :func:`.save_to_file`: Save the specified setup instance to a file.
    - :func:`.load_from_file`: Load a setup instance from a file.


.. automodule:: pyoma2.functions.gen
   :members:
