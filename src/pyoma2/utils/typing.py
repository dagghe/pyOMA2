"""Custom types for pydantic models.

i.e. allow serialization of numpy arrays.
https://github.com/pydantic/pydantic/issues/7017
"""

import numpy as np
from pydantic import BeforeValidator, PlainSerializer
from typing_extensions import Annotated


def nd_array_custom_before_validator(x):
    # custome before validation logic
    return x


def nd_array_custom_serializer(x):
    # custome serialization logic
    return str(x)


NdArray = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_custom_before_validator),
    PlainSerializer(nd_array_custom_serializer, return_type=str),
]
