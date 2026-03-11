"""Custom types for pydantic models.

i.e. allow serialization of numpy arrays.
https://github.com/pydantic/pydantic/issues/7017
"""

from __future__ import annotations

import typing

import numpy as np
from pydantic import BeforeValidator, PlainSerializer
from typing_extensions import Annotated


def nd_array_custom_before_validator(x: typing.Any) -> np.ndarray:
    """Custom before validation logic for numpy arrays."""
    return x


def nd_array_custom_serializer(x: np.ndarray) -> str:
    """Custom serialization logic for numpy arrays."""
    return str(x)


NdArray = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_custom_before_validator),
    PlainSerializer(nd_array_custom_serializer, return_type=str),
]
