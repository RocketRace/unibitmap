'''Tools to generate custom color mappings seen in `mappings/`.'''

from __future__ import annotations

import os
import sys
import unicodedata
from datetime import datetime
from typing import TYPE_CHECKING

from freetype import Face, ft_errors, FT_LOAD_FLAGS

if TYPE_CHECKING:
    from .mappings import ColorMapping

FILE_HEADER = """# Pixel color mapping, auto-generated using {0}
# Font name: {{0}}
# Timestamp generated: {{1}}
# Demo gradient: {{2}}
""".format(os.path.basename(__file__))

def normalize_dedup_sort(colors: ColorMapping) -> ColorMapping:
    '''Rescales the colors within [0, 256).
    The maximum color is scaled to 255, and the minimum to 0.

    Deduplicates the mapping based on the colors,
    so only one character maps to a particular color.

    Sorts the mapping based on color, in increasing order.
    
    Returns the updated color mapping.
    '''
    peak = max(colors.values())
    offset = min(colors.values())
    scale = 255 / (peak - offset)

    out = []
    visited = set()

    for c, color in colors.items():
        norm = round((color - offset) * scale)
        if norm in visited:
            continue
        out.append((c, norm))
        visited.add(norm)

    out.sort(key=lambda x: x[1])
    
    return dict(out)

def resolve_font_face(path: str) -> Face:
    '''Finds the `FontTools.ttLib.TTFont` with the given name,
    from the OS supported paths. Supports most major platforms.
    '''
    # here we are back in cross-platform compatibility purgatory
    try:
        return Face(path)
    except ft_errors.FT_Exception as exc:
        # "cannot open resource"
        if exc.errcode == 1: 
            options = []
            if sys.platform == "darwin":
                options.extend([
                    "/Library/Fonts",
                    "/System/Library/Fonts",
                    os.path.expanduser("~/Library/Fonts")
                ])
            elif sys.platform == "win32":
                options.append(os.path.join(os.environ["WINDIR"], "fonts"))
            elif "linux" in sys.platform:
                linux = os.environ.get("XDG_DATA_DIRS", "/usr/share")
                options.extend([
                    os.path.join(p, "fonts")
                    for p in linux.split(":")
                ])
            for directory in options:
                for root, _, names in os.walk(directory):
                    for name in names:
                        if name == path:
                            return Face(os.path.join(root, name))
        raise FileNotFoundError
            
def is_fullwidth_identifier(char: str) -> bool:
    '''Determines whether the given character is suitable for a Python identifier.'''
    categories = ("Lu", "Li", "Lt", "Lm", "Lo", "Nl")
    widths = ("F", "W")
    return (
        unicodedata.category(char) in categories and
        unicodedata.east_asian_width(char) in widths and
        unicodedata.is_normalized("NFKC", char)
    )

def generate_colors(path: str, scale: int = 16) -> ColorMapping:
    '''Generates a mapping of fullwidth identifier characters and colors
    using font at the given path. Resulting colors are normalized to the [0, 256) range,
    and deduplicated + sorted by color.

    The greater `scale` is, the more accurate the results will be.
    However, higher values are much slower to compute. Furthermore, the relative
    increase in quality tends to plateau quickly, as anti-aliasing has diminishing returns.
    '''
    face = resolve_font_face(path)
    face.set_pixel_sizes(width=scale, height=scale)
    colors = {}
    for (c, i) in face.get_chars():
        char = chr(c)
        if is_fullwidth_identifier(char):
            # Load the glyph enough to fetch its horizontal advance
            # [in 16.16 fixed point units]
            advance = face.get_advance(i, FT_LOAD_FLAGS["FT_LOAD_DEFAULT"]) // (2**16)
            # Require "true" fullwidth. This excludes most Korean characters
            if advance == scale:
                # Render the glyph's bitmap
                face.load_glyph(i, FT_LOAD_FLAGS["FT_LOAD_RENDER"])
                bitmap = face.glyph.bitmap
                if bitmap.width <= scale and bitmap.rows <= scale:
                    # TODO: Arithmetic mean of pixels may not always be appropriate?
                    colors[char] = sum(bitmap.buffer) // (scale ** 2)
    return normalize_dedup_sort(colors)

def dump_font_colors(font_path: str, out_prefix: str) -> ColorMapping:
    '''Generates a color mapping for a font and dumps it into the specified directory.'''
    colors = generate_colors(font_path)
    font_name, _ = os.path.splitext(os.path.basename(font_path))
    out_path = f"{font_name}.py"
    if out_prefix:
        out_path = os.path.join(out_prefix, out_path)
    with open(out_path, "w") as fp:
        # TODO sanitize quotes / newlines within strings!
        fp.write(FILE_HEADER.format(
            font_name,
            datetime.utcnow().isoformat(),
            ''.join(colors)
        ))
        fp.write(str(colors))
        fp.write("\n")
    return colors

# dump_font_colors("NotoSansCJKjp-Regular.otf", "unibitmap/mappings")
# dump_font_colors("/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc", "unibitmap/mappings")