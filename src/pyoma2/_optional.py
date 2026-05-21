"""Helpers for importing optional (extra) dependencies."""

from __future__ import annotations

import importlib
from types import ModuleType


def require(module: str, extra: str) -> ModuleType:
    """Import ``module``, or raise a friendly error naming the pip extra.

    Parameters
    ----------
    module : str
        Dotted path of the module to import (e.g. ``"pyoma2.functions.plot"``).
    extra : str
        Name of the optional-dependency extra that ships it — one of
        ``"plot"``, ``"3d"`` or ``"all"``.

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    ImportError
        If the module — and therefore the optional extra — is not installed.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"This feature requires the optional '{extra}' dependencies. "
            f"Install them with:  pip install pyOMA_2[{extra}]"
        ) from exc
