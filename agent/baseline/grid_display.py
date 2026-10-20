"""Local display adapter for the archived animation helper's missing dependency.

Hexadecimal display is a port convention; it is not a recovered upstream palette.
"""
from collections.abc import Sequence

ARC_COLOR_CHARS = "0123456789ABCDEF"


def format_grid_ascii(grid: Sequence[Sequence[int]]) -> str:
    return "\n".join("".join(ARC_COLOR_CHARS[max(0, min(15, int(c)))] for c in row) for row in grid)
