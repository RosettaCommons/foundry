"""Release checks for rc-foundry.

Usage:
    python .github/release.py verify [TAG]

`verify` fails unless the version in pyproject.toml is a canonical
PEP 440 version and, when a tag is given, the tag is exactly ``v<version>``.
PyPI versions are immutable, so a tag/version mismatch (e.g. tagging v1.0.1
while pyproject.toml still says 1.0.0) would burn a version number we did not
mean to publish. With no TAG argument only the version itself is checked, which
is what PR runs and manual runs on a branch do.
"""

import sys
import tomllib
from pathlib import Path

from packaging.version import InvalidVersion, Version

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def project_version() -> str:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def verify(tag: str | None) -> list[str]:
    errors = []
    version = project_version()
    try:
        canonical = str(Version(version))
    except InvalidVersion:
        return [f"pyproject.toml version {version!r} is not a valid PEP 440 version"]
    if canonical != version:
        errors.append(
            f"pyproject.toml version {version!r} is not canonical; use {canonical!r}"
        )
    if tag is not None and tag != f"v{version}":
        errors.append(
            f"tag {tag!r} does not match pyproject.toml version {version!r}; "
            f"expected tag 'v{version}'"
        )
    return errors


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3) or argv[1] != "verify":
        print(__doc__, file=sys.stderr)
        return 2
    tag = argv[2] if len(argv) == 3 and argv[2] else None
    errors = verify(tag)
    for err in errors:
        print(f"error: {err}", file=sys.stderr)
    if errors:
        return 1
    print(f"ok: version {project_version()}" + (f", tag {tag}" if tag else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
