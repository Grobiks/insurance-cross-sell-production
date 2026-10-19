"""Windows launcher fix and `python -m insurance_cross_sell`."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "fix_windows_launcher.py"
spec = importlib.util.spec_from_file_location("fix_launcher", SCRIPT)
fix_launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fix_launcher)


def test_launcher_has_no_absolute_path(tmp_path):
    folder = tmp_path / "Учеба"
    folder.mkdir()
    (folder / "insurance").write_text("# script", encoding="utf-8")
    (folder / "insurance.cmd").write_text(
        f'@echo off\r\n"{folder}\\python.exe" "%~dp0\\insurance" %*\r\n',
        encoding="utf-8",
    )

    assert fix_launcher.fix(folder) is True
    content = (folder / "insurance.cmd").read_bytes()
    content.decode("ascii")  # only ASCII left: independent of code page
    assert b"%~dp0python.exe" in content
    assert fix_launcher.fix(folder) is False  # idempotent


def test_missing_launcher_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="poetry install"):
        fix_launcher.fix(tmp_path)


def test_python_m_runs_cli():
    result = subprocess.run(
        [sys.executable, "-m", "insurance_cross_sell", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "retrain" in result.stdout
