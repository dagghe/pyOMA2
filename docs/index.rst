.. pyoma2 documentation master file, created by
   sphinx-quickstart on Fri Feb  9 03:03:59 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyOMA2's documentation!
=======================

.. image:: https://github.com/dagghe/pyOMA2/assets/64746269/aa19bc05-d452-4749-a404-b702e6fe685d

|Python| |pre-commit| |Code style: black| |Downloads| |docs|

This is the new and updated version of **pyOMA** module, a Python module designed for conducting
operational modal analysis. With this update, we've transformed **pyOMA** from a basic collection
of functions into a more sophisticated module that fully leverages the capabilities of Python classes.

Key Features & Enhancements:

- Support for single and multi-setup measurements, which includes handling multiple acquisitions with mixed reference and roving sensors.
- Interactive plots for intuitive mode selection, users can now extract desired modes directly from algorithm-generated plots.
- Structure geometry definition, enabling 3D visualization of mode shapes once modal results are obtained.
- Uncertainty estimation for modal properties in Stochastic Subspace Identification (SSI) algorithms.
- Specialized clustering classes for Automatic OMA using SSI, streamlining modal parameter extraction.
- New OMAX (OMA with Exogenous Input) functionality for SSI, expanding the module’s capabilities to handle forced excitation scenarios.

We provide five :doc:`examples` to show the modules capabilities:


Check out the project source_.


.. important::

   We have introduced some clustering specialised classes that allow the users to perform Automatic OMA.
   This update enables users to implement and compare a large number of the most popular algorithms
   introduced over the last 15 years, all within the same analysis framework. Furthermore, users have the
   flexibility to mix specific strategies from different algorithms to tailor a specific clustering process
   to their needs.


.. note::

   Please note that the project is still under active development.



Schematic organisation of the module showing inheritance between classes
========================================================================

.. image:: /img/info.png



.. Hidden TOCs

.. toctree::
   :caption: Quick start
   :maxdepth: 2
   :hidden:

   Home <self>
   Installation
   Contributing


.. toctree::
   :caption: Documentation
   :maxdepth: 2
   :hidden:

   Index <modules>


.. toctree::
   :caption: Examples
   :maxdepth: 2
   :hidden:

   examples

=============================================================


Index
=====

* :ref:`genindex`
* :ref:`modindex`


References
==========

.. [CRGF14] Rainieri, C., & Fabbrocino, G. (2014). Operational modal analysis of civil
   engineering structures. Springer, New York, 142, 143.
.. [RBCV15] Brincker, R., & Ventura, C. (2015). Introduction to operational modal analysis.
    John Wiley & Sons.
.. [BZA01] Brincker, R., Zhang, L., & Andersen, P. (2001). Modal identification of output-only
   systems using frequency domain decomposition. Smart Materials and Structures, 10(3), 441.
.. [BVA01] Brincker, R., Ventura, C. E., & Andersen, P. (2001). Damping estimation by frequency
   domain decomposition. In Proceedings of IMAC 19: A Conference on Structural Dynamics.
.. [ZWT10] Zhang, L., Wang, T., & Tamura, Y. (2010). A frequency–spatial domain decomposition
   (FSDD) method for operational modal analysis. Mechanical Systems and Signal Processing,
   24(5), 1227-1239.
.. [ZBGS20] Zaletelj, K., Bregar, T., Gorjup, D., Slavič, J. (2020) sdypy-pyEMA,
   10.5281/zenodo.4016670, https://github.com/sdypy/sdypy
.. [BPDG99] Peeters, B., & De Roeck, G. (1999). Reference-based stochastic subspace
   identification for output-only modal analysis. Mechanical Systems and
   Signal Processing, 13(6), 855-878.
.. [MiDo11] Döhler, M. (2011). Subspace-based system identification and fault detection:
   Algorithms for large systems and application to structural vibration analysis.
   Diss. Université Rennes 1.
.. [DOME13] Döhler, M., & Mevel, L. (2013). Efficient multi-order uncertainty computation
   for stochastic subspace identification. Mechanical Systems and Signal Processing, 38(2), 346-366.
.. [DLM13] Döhler, M., Lam, X. B., & Mevel, L. (2013). Uncertainty quantification for modal
   parameters from stochastic subspace identification on multi-setup measurements.
   Mechanical Systems and Signal Processing, 36(2), 562-581.
.. [SARB21] Amador, S. D., & Brincker, R. (2021). Robust multi-dataset identification with
   frequency domain decomposition. Journal of Sound and Vibration, 508, 116207.
.. [PAGL04] Peeters, B., Van der Auweraer, H., Guillaume, P., & Leuridan, J. (2004).
   The PolyMAX frequency-domain method: a new standard for modal parameter estimation?.
   Shock and Vibration, 11(3-4), 395-409.
.. [MDM16] Mellinger, P., Döhler, M., & Mevel, L. (2016).
   Variance estimation of modal parameters from output-only and input/output subspace-based system identification.
   Journal of Sound and Vibration, 379, 1–27.

.. _source: https://github.com/dagghe/pyOMA2

.. |Python| image:: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg?style=flat&logo=python&logoColor=white
    :alt: Python
    :target: https://www.python.org

.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :alt: pre-commit
   :target: https://github.com/pre-commit/pre-commit

.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :alt: Code style: black
   :target: https://img.shields.io/badge/code%20style-black-000000.svg

.. |Downloads| image:: https://img.shields.io/pepy/dt/pyOMA-2
   :alt: Downloads
   :target: https://img.shields.io/pepy/dt/pyOMA-2

.. |docs| image:: https://readthedocs.org/projects/pyoma/badge/?version=main
    :target: https://pyoma.readthedocs.io/en/main/?badge=main
    :alt: Documentation Status
