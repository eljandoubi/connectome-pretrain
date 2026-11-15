"""Run the tests against the package of this checkout, never another installed copy."""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def pytest_configure(config):
    if str(SRC) not in sys.path[:1]:
        sys.path.insert(0, str(SRC))
    import connpre

    where = Path(connpre.__file__).resolve()
    if SRC not in where.parents:
        raise pytest.UsageError(
            f"connpre is imported from {where}, not from {SRC}: "
            "run `pip install -e .` in this checkout or `PYTHONPATH=src pytest`"
        )
