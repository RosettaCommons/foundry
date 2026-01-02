"""
Best-effort PyRosetta installation helper.

This mirrors the approach used in:
  https://github.com/Arielbs/rosetta-mcp-server

Usage:
  python install_pyrosetta.py
"""

from __future__ import annotations


def main() -> int:
    try:
        import pyrosetta_installer as I  # type: ignore
    except Exception as e:  # noqa: BLE001
        print("ERROR: pyrosetta-installer is not installed in this environment.")
        print("Install it first with: uv pip install pyrosetta-installer")
        print(f"Import error: {e}")
        return 1

    try:
        # Keep defaults; this will download the appropriate build if available.
        I.install_pyrosetta(silent=False, skip_if_installed=True)
    except Exception as e:  # noqa: BLE001
        print("ERROR: PyRosetta installation failed (often due to platform/Python availability).")
        print(f"Install error: {e}")
        return 2

    print("PyRosetta installation step completed. Verify with:")
    print("  python -c \"import pyrosetta; pyrosetta.init('-mute all'); print('PyRosetta OK')\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


