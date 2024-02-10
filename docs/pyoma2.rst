.. _code-doc-Model:
User instantiated classes
=========================

These are the classes that the users need to instantiate to perform the analysis.

.. autosummary::
   :toctree: generated

Setup classes
-------------

The ``SingleSetup`` class
#########################

.. autoclass:: pyoma2.OMA.SingleSetup
    :members:
    :show-inheritance:


The ``MultiSetup_PoSER`` class
##############################

.. autoclass:: pyoma2.OMA.MultiSetup_PoSER
    :members:
    :show-inheritance:


The ``MultiSetup_PreGER`` class
###############################

.. autoclass:: pyoma2.OMA.MultiSetup_PreGER
    :members:
    :show-inheritance:


Algorithm classes
------------------

The ``FDD_algo`` class
######################

.. autoclass:: pyoma2.algorithm.fdd.FDD_algo
    :members:
    :show-inheritance:


The ``FDD_algo_MS`` class
#########################

.. autoclass:: pyoma2.algorithm.fdd.FDD_algo_MS
    :members:
    :show-inheritance:


The ``EFDD_algo`` class
#######################

.. autoclass:: pyoma2.algorithm.fdd.EFDD_algo
    :members:
    :show-inheritance:


The ``EFDD_algo_MS`` class
##########################

.. autoclass:: pyoma2.algorithm.fdd.EFDD_algo_MS
    :members:
    :show-inheritance:


The ``FSDD_algo`` class
#######################

.. autoclass:: pyoma2.algorithm.fdd.FSDD_algo
    :members:
    :show-inheritance:


The ``SSIdat_algo`` class
#########################

.. autoclass:: pyoma2.algorithm.ssi.SSIdat_algo
    :members:
    :show-inheritance:


The ``SSIdat_algo_MS`` class
############################

.. autoclass:: pyoma2.algorithm.ssi.SSIdat_algo_MS
    :members:
    :show-inheritance:


The ``SSIcov_algo`` class
#########################

.. autoclass:: pyoma2.algorithm.ssi.SSIcov_algo
    :members:
    :show-inheritance:


The ``SSIcov_algo_MS`` class
############################

.. autoclass:: pyoma2.algorithm.ssi.SSIcov_algo_MS
    :members:
    :show-inheritance:


Support classes and modules
===========================

These classes (and functions) are not instantiated or called directly by the users, but internally by the "main" classes.


The ``BaseSetup`` class
-----------------------

.. autoclass:: pyoma2.OMA.BaseSetup
    :members:


The ``BaseAlgorithm`` class
---------------------------

.. autoclass:: pyoma2.algorithm.base.BaseAlgorithm
    :members:


The ``geometry`` module
-----------------------

.. automodule:: pyoma2.algorithm.data.geometry
   :members:
   :undoc-members:
   :show-inheritance:


The ``result`` module
---------------------

.. automodule:: pyoma2.algorithm.data.result
   :members:
   :undoc-members:
   :show-inheritance:

The ``run_params`` module
-------------------------

.. automodule:: pyoma2.algorithm.data.run_params
   :members:
   :undoc-members:
   :show-inheritance:


The ``SelFromPlot`` class
-------------------------

.. autoclass:: pyoma2.plot.Sel_from_plot.SelFromPlot
    :members:


The ``anim_mode`` module
------------------------

.. automodule:: pyoma2.plot.anim_mode
   :members:
   :undoc-members:
   :show-inheritance:


The ``utils`` module
--------------------

.. automodule:: pyoma2.utils
   :members:
   :undoc-members:
   :show-inheritance:


Functions module
================

The following functions are the "foundamental bricks" to perform the analyses.

The ``FDD_funct`` module
------------------------

.. automodule:: pyoma2.functions.FDD_funct
   :members:


The ``Gen_funct`` module
------------------------

.. automodule:: pyoma2.functions.Gen_funct
   :members:


The ``SSI_funct`` module
------------------------

.. automodule:: pyoma2.functions.SSI_funct
   :members:


The ``plot_funct`` module
-------------------------

.. automodule:: pyoma2.functions.plot_funct
   :members:
