"""Run mypy on positive/negative typing fixtures for algorithm and setup generics."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from pyoma2.algorithms import (
    EFDD,
    EFDD_MS,
    FDD,
    FDD_MS,
    FSDD,
    SSI,
    SSI_MS,
    FDDAlgorithm,
    pLSCF,
    pLSCF_MS,
)
from pyoma2.algorithms.data.run_params import EFDDRunParams, FDDRunParams, SSIRunParams

FIXTURES = Path(__file__).resolve().parents[1] / "typing_fixtures"
POSITIVE = FIXTURES / "positive.py"
NEGATIVE = FIXTURES / "negative.py"

_MYPY_ERROR = re.compile(
    r"^.+:(\d+): error: .+ \[([^\]]+)\]\s*$",
)

# Per-function mypy error codes in the negative fixture. Counts are exact so
# one attachment direction cannot cover for the other.
EXPECTED_NEGATIVE_CODES = {
    "invalid_type_argument_on_fdd": ["misc"],
    "forged_fdd_specialization": ["misc"],
    "invalid_type_argument_on_ssi": ["misc"],
    "wrong_run_params_type": ["arg-type"],
    "unannotated_wrong_run_params": ["arg-type", "arg-type", "arg-type"],
    "single_setup_rejects_multi_algorithm": ["arg-type", "arg-type"],
    "multi_setup_rejects_single_algorithm": ["arg-type", "arg-type", "arg-type"],
    "assign_wrong_result_type": ["assignment"],
    "efdd_is_not_a_typed_fdd": ["arg-type"],
}


def _run_mypy(path: Path) -> tuple[str, str, int]:
    pytest.importorskip("mypy")
    from mypy import api as mypy_api

    return mypy_api.run(
        [
            "--show-error-codes",
            "--no-error-summary",
            "--no-incremental",
            str(path),
        ]
    )


def _function_line_ranges(path: Path) -> dict[str, tuple[int, int]]:
    tree = ast.parse(path.read_text())
    ranges: dict[str, tuple[int, int]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            ranges[node.name] = (node.lineno, node.end_lineno or node.lineno)
    return ranges


def _parse_mypy_errors(output: str) -> tuple[list[tuple[int, str]], list[str]]:
    errors: list[tuple[int, str]] = []
    unparsed: list[str] = []
    for raw in output.splitlines():
        if ": error:" not in raw:
            continue
        match = _MYPY_ERROR.match(raw)
        if match:
            errors.append((int(match.group(1)), match.group(2)))
        else:
            unparsed.append(raw)
    return errors, unparsed


def _codes_by_function(
    path: Path, output: str
) -> tuple[dict[str, list[str]], list[tuple[int, str]]]:
    ranges = _function_line_ranges(path)
    grouped: dict[str, list[str]] = {name: [] for name in ranges}
    unassigned: list[tuple[int, str]] = []
    parsed, _unparsed = _parse_mypy_errors(output)
    for line, code in parsed:
        for name, (start, end) in ranges.items():
            if start <= line <= end:
                grouped[name].append(code)
                break
        else:
            unassigned.append((line, code))
    return grouped, unassigned


def test_ms_algorithms_are_runtime_subclasses_of_single_setup_counterparts():
    """Public multi-setup classes remain subclasses of the single-setup API."""
    assert issubclass(EFDD, FDD)
    assert issubclass(FSDD, EFDD)
    assert issubclass(FSDD, FDD)
    assert issubclass(FDD_MS, FDD)
    assert issubclass(EFDD_MS, EFDD)
    assert issubclass(EFDD_MS, FDD)
    assert issubclass(SSI_MS, SSI)
    assert issubclass(pLSCF_MS, pLSCF)


def test_run_param_factories_return_specialized_models():
    """``FDD.RunParamCls(...)`` is the documented runtime factory pattern."""
    fdd_params = FDD.RunParamCls(nxseg=512)
    assert isinstance(fdd_params, FDDRunParams)
    assert fdd_params.nxseg == 512
    assert isinstance(EFDD.RunParamCls(), EFDDRunParams)
    assert isinstance(SSI.RunParamCls(br=5, ordmax=20), SSIRunParams)
    assert issubclass(EFDDRunParams, FDDRunParams)
    assert isinstance(FDD(), FDDAlgorithm)
    assert isinstance(EFDD(), FDDAlgorithm)
    assert isinstance(FSDD(), FDDAlgorithm)


def test_positive_typing_fixtures_pass_mypy():
    """Valid generic usage (matching data types, constructors, inheritance) is clean."""
    stdout, stderr, status = _run_mypy(POSITIVE)
    assert status == 0, (
        "Expected positive typing fixtures to be mypy-clean.\n"
        f"stdout:\n{stdout}\nstderr:\n{stderr}"
    )


def test_negative_typing_fixtures_fail_mypy():
    """Mismatched data types and invalid specializations are static errors."""
    stdout, stderr, status = _run_mypy(NEGATIVE)
    assert status != 0, (
        "Expected negative typing fixtures to produce mypy errors.\n"
        f"stdout:\n{stdout}\nstderr:\n{stderr}"
    )

    out = stdout + stderr
    parsed, unparsed = _parse_mypy_errors(out)
    by_fn, unassigned = _codes_by_function(NEGATIVE, out)

    assert unparsed == [], f"unparsed mypy error lines:\n{unparsed}\n{out}"
    assert unassigned == [], (
        f"mypy diagnostics outside fixture functions:\n{unassigned}\n{out}"
    )
    assert set(by_fn) == set(EXPECTED_NEGATIVE_CODES), (
        f"Fixture functions changed.\nobserved={sorted(by_fn)}\n"
        f"expected={sorted(EXPECTED_NEGATIVE_CODES)}"
    )
    for name, expected in EXPECTED_NEGATIVE_CODES.items():
        assert by_fn[name] == expected, (
            f"{name}: expected {expected}, got {by_fn[name]}\n{out}"
        )
    expected_total = sum(len(codes) for codes in EXPECTED_NEGATIVE_CODES.values())
    observed_total = sum(len(codes) for codes in by_fn.values())
    assert len(parsed) == expected_total == observed_total, (
        f"diagnostic count mismatch: parsed={len(parsed)} "
        f"assigned={observed_total} expected={expected_total}\n{out}"
    )
