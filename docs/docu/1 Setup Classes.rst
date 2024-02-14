Setup Classes
=============

This module offers classes specifically designed for Operational Modal Analysis (OMA),
suitable for both single and multiple setup scenarios. The classes includes methods for
data management and processing, executing algorithms, visualizing outcomes, and setting
up the geometry of structures. The module utilises two methods when dealing with data
from multiple experimental setups: Post Separate Estimation Re-scaling (PoSER) and Pre
Global Estimation Re-scaling (PreGER).

Classes:
   :class:`.SingleSetup`
      Manages and processes single-setup data for OMA.
   :class:`.MultiSetup_PoSER`
      Conducts OMA for multi-setup experiments using the PoSER approach.
   :class:`.MultiSetup_PreGER`
      Conducts OMA for multi-setup experiments with the PreGER approach.

.. Note::
   The classes are designed to be used as part of the ``pyOMA2`` package and rely on its internal data structures and algorithms.


The ``SingleSetup`` class
-------------------------

.. autoclass:: pyoma2.OMA.SingleSetup
   :members:
   :inherited-members:
   :show-inheritance:

The ``MultiSetup_PoSER`` class
------------------------------

.. autoclass:: pyoma2.OMA.MultiSetup_PoSER
   :members:
   :show-inheritance:

The ``MultiSetup_PreGER`` class
-------------------------------

.. autoclass:: pyoma2.OMA.MultiSetup_PreGER
   :members:
   :inherited-members:
   :show-inheritance:
