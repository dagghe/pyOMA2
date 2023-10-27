import typing

from pydantic import BaseModel, ConfigDict, Field


class BaseRunParams(BaseModel):
    """Base class for input parameters."""

    model_config = ConfigDict(from_attributes=True)


class FDDRunParams(BaseRunParams):
    df: float = 0.01
    pov: float = 0.5
    window: str = "hann"


class SSIcovRunParams(BaseRunParams):
    br: int
    ordmin: int = 0
    ordmax: typing.Optional[int] = None
    method: str = "1"
