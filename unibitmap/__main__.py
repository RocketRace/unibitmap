'''Command-line interface for converting images to Unibitmap python files'''

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from . import Bitmap, Pixels
from .mappings import get_mapping

if TYPE_CHECKING:
    from . import UnicodeGrid

DEFAULT_NAME = "My" + Bitmap.__name__

FONT_KWARG = ', font="{0}"'

CODE_FORMAT = """from {0} import {1}

class {{0}}({1}{{1}}):
{{2}}
""".format(__package__, Bitmap.__name__)

def generate_code(rows: UnicodeGrid, *, name: str, tabs: bool = False, font: str | None = None) -> str:
    indent = np.full((rows.shape[0], 1), "\t") if tabs else np.full((rows.shape[0], 4), " ")
    indented = np.hstack((indent, rows))
    indent = "\t" if tabs else "    "
    font_str = "" if font is None else FONT_KWARG.format(font)
    formatted_rows = "\n".join(''.join(row) for row in indented)
    return CODE_FORMAT.format(name, font_str, formatted_rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="unibitmap",
        description="Converts images to Python programs containing unicode bitmap art which then evaluates to the original image",
        add_help=False,
    )
    parser.add_argument("file", type=argparse.FileType("rb"), help="File to read image data from")
    parser.add_argument("--help", "-H", action="help", help="Show this help message and exit")
    parser.add_argument("--out", "-o", default=None, type=str, help="Path of the resulting Python file")
    parser.add_argument("--height", "-h", default=None, type=int, help="Rescaled height of the generated bitmap")
    parser.add_argument("--width", "-w", default=None, type=int, help="Rescaled width of generated bitmap")
    parser.add_argument("--font", "-f", default=None, type=str, help="Specify the font used for bitmap generation")
    parser.add_argument("--name", "-n", default=None, type=str, help="Specify the class name in resulting file")
    parser.add_argument("--tabs", "-t", action="store_true", help="Use tabs instead of 4 spaces in resulting file")
    parser.add_argument("--generate-mapping", action="store_true", help="Generate or regenerate a color mapping for the current font, if one is missing (this operation can take several minutes!)")

    args = parser.parse_args()

    if args.out is None:
        name = args.name or DEFAULT_NAME
        out = sys.stdout
    else:
        if args.name is None:
            root, _ = os.path.splitext(args.out)
            _, name = os.path.split(root)
        else:
            name = args.name
        out = open(args.out, "w")
    
    for i in range(len(name)):
        if not name[:i + 1].isidentifier():
            name = name[:i] + "_" + name[i + 1:]
    name = name.title() or DEFAULT_NAME

    mapping = get_mapping(args.font, generate_if_missing=args.generate_mapping)
    px = Pixels.from_image(Image.open(args.file), width=args.width, height=args.height)
    out.write(generate_code(px.to_unicode(mapping), name=name, tabs=args.tabs, font=args.font))
