from __future__ import annotations

import ast
import os
from typing import TYPE_CHECKING

from ..generate import dump_font_colors

if TYPE_CHECKING:
    ColorMapping = dict[str, int]

DEFAULT_FONT_NAME = "NotoSansCJKjp-Regular"
FONT_PATH = os.path.join("unibitmap", "mappings")

def get_mapping(font: str | None = None, generate_if_missing: bool = False) -> ColorMapping:
    '''Obtains the color mapping for a given font.
    
    If `generate_if_missing` is True and the mapping doesn't exist, this will generate
    & store the mapping in the appropriate directory. Otherwise, raises FileNotFoundError.
    '''
    if font is None:
        font = DEFAULT_FONT_NAME
    font += ".py"
    try:
        with open(os.path.join(FONT_PATH, font)) as fp:
            # This can still segfault/oom for malicious input.
            return ast.literal_eval(fp.read())
    except ImportError:
        if generate_if_missing:
            return dump_font_colors(font, FONT_PATH)
        else:
            raise FileNotFoundError(f"The font file {font} was not found in {FONT_PATH}")

def closest_pixel(mapping: ColorMapping, color: int) -> str:
    '''Finds the closest unicode character for the given color value in [0, 256).'''
    return min(
        mapping.items(),
        key=lambda k: abs(k[1] - color)
    )[0]