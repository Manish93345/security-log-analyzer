"""``python -m slrag`` — same entry point as the ``slrag`` console script."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
