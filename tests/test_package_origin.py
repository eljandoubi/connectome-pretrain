import os
import subprocess
import sys
from pathlib import Path

import connpre

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
NODE = f"tests/{Path(__file__).name}::test_tests_import_this_checkout"


def test_tests_import_this_checkout():
    assert SRC in Path(connpre.__file__).resolve().parents


def pytest_with_a_shadow_copy(tmp_path, early):
    """Run the test above with another connpre on PYTHONPATH."""
    shadow = tmp_path / "shadow"
    (shadow / "connpre").mkdir(parents=True)
    (shadow / "connpre" / "__init__.py").write_text("")
    (shadow / "early_import.py").write_text("import connpre\n")
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", NODE]
    if early:
        cmd[4:4] = ["-p", "early_import"]
    env = {**os.environ, "PYTHONPATH": str(shadow)}
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)


def test_a_shadow_copy_on_the_path_does_not_replace_the_checkout(tmp_path):
    result = pytest_with_a_shadow_copy(tmp_path, early=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_shadow_copy_imported_first_stops_the_run(tmp_path):
    result = pytest_with_a_shadow_copy(tmp_path, early=True)
    assert result.returncode != 0
    assert "not from" in result.stdout + result.stderr
