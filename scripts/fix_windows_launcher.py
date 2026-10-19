r"""Make the `insurance` command work in any Windows console encoding.

On Windows Poetry creates `.venv\Scripts\insurance.cmd` with the full
path to python.exe written in UTF-8. When that path contains non-Latin
letters (for example a Russian folder name) and the console uses another
code page (866 by default), cmd.exe misreads it and fails with "The
system cannot find the path specified".

The script rewrites the launcher so that it finds python.exe next to
itself (`%~dp0`) and contains no absolute path at all. Run it once after
`poetry install`:

    poetry run python scripts/fix_windows_launcher.py
"""

from __future__ import annotations

import sys
from pathlib import Path

LAUNCHER = '@echo off\r\n"%~dp0python.exe" "%~dp0{name}" %*\r\n'


def fix(scripts_dir: Path, name: str = "insurance") -> bool:
    """Rewrite `<name>.cmd` in `scripts_dir`; return True if changed."""
    cmd = scripts_dir / f"{name}.cmd"
    if not cmd.exists() or not (scripts_dir / name).exists():
        raise FileNotFoundError(f"{cmd} not found: run `poetry install` first")
    content = LAUNCHER.format(name=name).encode("ascii")
    if cmd.read_bytes() == content:
        return False
    cmd.write_bytes(content)
    return True


def main() -> int:
    """Fix the launcher of the active virtual environment."""
    if sys.platform != "win32":
        print("Only needed on Windows, nothing to do.")
        return 0
    scripts_dir = Path(sys.executable).parent
    changed = fix(scripts_dir)
    status = "fixed" if changed else "already fixed"
    print(f"{status}: {scripts_dir / 'insurance.cmd'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
