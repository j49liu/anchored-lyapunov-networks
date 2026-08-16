#!/usr/bin/env python3
"""Regenerate SHA256SUMS from the same inventory used by the validator."""

from validate_release import HERE, released_files, sha256


def main():
    lines = [
        f"{sha256(path)}  {path.relative_to(HERE).as_posix()}"
        for path in released_files()
    ]
    (HERE / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} entries to SHA256SUMS")


if __name__ == "__main__":
    main()
