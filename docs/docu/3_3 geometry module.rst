The ``geometry`` module
-----------------------

This module provides classes for handling geometry-related data, specifically designed
to store and manipulate sensor and background geometry information.

Classes:
   :class:`.BaseGeometry`
      Base class for storing and managing sensor and background geometry data.
   :class:`.Geometry1`
      Class for storing and managing sensor and background geometry data for Geometry 1.
   :class:`.Geometry2`
      Class for storing and managing sensor and background geometry data for Geometry 2.
   :class:`.GeometryMixin`
      Mixin that gives the ability to define the geometry the instance of the setup class.

.. Warning::
    The module is designed to be used as part of the pyOMA2 package and relies on its
    internal data structures and algorithms.

.. automodule:: pyoma2.support.geometry.data
   :members:
   :show-inheritance:

.. automodule:: pyoma2.support.geometry.mixin
   :members:
   :show-inheritance:
