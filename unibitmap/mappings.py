'''Tools to manage & generate custom color mappings.'''
from __future__ import annotations

import os
import sys
import unicodedata
from typing import TYPE_CHECKING, BinaryIO

import numpy as np
from freetype import FT_LOAD_FLAGS, Face, ft_errors
from scipy import spatial

from .colorspace import RGB_TO_YCBCR_TRANSFORM, optimize_points

if TYPE_CHECKING:
    Lightness = int
    Color = tuple[int, int, int]
    LightnesssMapping = dict[str, Lightness]
    CharRgbMapping = dict[str, Color]
    YcbcrCharMapping = dict[Color, str]

DEFAULT_FONT_NAME = "NotoSansCJKsc-Regular"
FONT_PATH = os.path.join("unibitmap", "mappings")

class Colors:
    def __init__(self, char_rgb: CharRgbMapping, ycbcr_chars: YcbcrCharMapping) -> None:
        self.char_rgb = char_rgb
        self.ycbcr_chars = ycbcr_chars
        ycbcr_points = np.array(list(ycbcr_chars))
        # lightness is twice as significant as chrominances
        ycbcr_points[:, 0] *= 2
        self.tree = spatial.cKDTree(ycbcr_points) # type: ignore
    
    @classmethod
    def read_from(cls, data: BinaryIO) -> Colors:
        '''Reads the data from a buffer into a Colors.
        
        Data format (repeating 9-byte chunks):

        * [3 bytes for a utf-32 codepoint (big-endian)]
        * [3 bytes for RGB values]
        * [1 byte for luma value]
        * [2 signed bytes for CbCr values]
        '''
        read = data.read(9)
        char_rgb = {}
        ycbcr_chars = {}
        while read:
            char = chr(int.from_bytes(read[0:3], 'big'))
            rgb = tuple(read[3:6])
            y = read[6]
            cb = read[7] - 128
            cr = read[8] - 128
            ycbcr = (y, cb, cr)
            char_rgb[char] = rgb
            ycbcr_chars[ycbcr] = char
            read = data.read(9)
        return Colors(char_rgb, ycbcr_chars)
    
    def write_to(self, file: BinaryIO) -> None:
        '''Writes the Color data to a file.
        
        Data format (repeating 9-byte chunks):

        * [3 bytes for a utf-32 codepoint (big-endian)]
        * [3 bytes for RGB values]
        * [1 byte for luma value]
        * [2 signed bytes for CbCr values]
        '''
        for ycbcr, char in self.ycbcr_chars.items():
            file.write(ord(char).to_bytes(3, 'big'))
            rgb = self.char_rgb[char]
            file.write(bytes(rgb))
            y = ycbcr[0]
            cb = ycbcr[1] + 128
            cr = ycbcr[2] + 128
            file.write(bytes((y, cb, cr)))
        file.flush()

    def get_colors(self, char_grid: np.ndarray, *, ignore_unknown: bool = False) -> np.ndarray:
        '''Return an array of colors associated with `char_grid`. 
        
        If `ignore_unknown` is True, unknown chars are replaced with `(0, 0, 0)`.
        Otherwise, this raises `ValueError`.
        '''
        if ignore_unknown:
            return np.array([[self.char_rgb.get(char, (0, 0, 0)) for char in char_grid.flat]], dtype='uint8')
        else:
            try:
                return np.array([[self.char_rgb[char] for char in char_grid.flatten()]], dtype='uint8')
            except KeyError as e:
                raise ValueError(f"Unrecognized character `{e.args[0]}`")

    def get_chars(self, rgb_points: np.ndarray) -> np.ndarray:
        '''Return an array of characters that are closest to the colors specified in `rgb_points`'''
        ycbcr_points = RGB_TO_YCBCR_TRANSFORM.dot(rgb_points.T).T
        # A change in lightness is twice as significant as chrominances
        ycbcr_points[:, 0] *= 2
        _, data_indices = self.tree.query(ycbcr_points) # type: ignore
        nearest_points = self.tree.data[data_indices] # type: ignore
        nearest_points[:, 0] //= 2
        return np.array([self.ycbcr_chars[tuple(i)] for i in nearest_points]) # type: ignore

    def is_empty(self) -> bool:
        '''Number of elements is zero'''
        return len(self.char_rgb) == 0

def get_mapping(font: str | None = None, generate_if_missing: bool = False) -> Colors:
    '''Obtains the color mapping for a given font.
    
    If `generate_if_missing` is True and the mapping doesn't exist, this will generate
    & store the mapping in the appropriate directory. Otherwise, raises FileNotFoundError.
    '''
    if font is None:
        font = DEFAULT_FONT_NAME
    mapping_path = f"{font}.unimapping"
    try:
        with open(os.path.join(FONT_PATH, mapping_path), "rb") as fp:
            return Colors.read_from(fp)
    except FileNotFoundError as e:
        if generate_if_missing:
            return dump_font_mapping(font, FONT_PATH)
        else:
            raise FileNotFoundError(f"The mapping file {mapping_path} was not found in {FONT_PATH}") from e

def colorize(lightnesses: LightnesssMapping) -> Colors:
    '''Normalizes the lightnesses to [0, 256).
    The maximum color is scaled to 255, and the minimum to 0.
    
    Adds color data to the resulting lightnesses according to `colorspace.optimize_points`
    '''
    if len(lightnesses) == 0:
        return Colors({}, {})

    peak = max(lightnesses.values())
    offset = min(lightnesses.values())
    scale = 255 / (peak - offset)

    same_lightness: dict[int, list[str]] = {}

    for c, lightness in lightnesses.items():
        norm = round((lightness - offset) * scale)
        same_lightness.setdefault(norm, []).append(c)
    
    char_rgb = []
    ycbcr_chars = []
    for lightness, chars in same_lightness.items():
        for char, (rgb, ycbcr) in zip(chars, optimize_points(lightness, len(chars))):
            char_rgb.append((char, rgb))
            ycbcr_chars.append((ycbcr, char))

    char_rgb.sort()
    ycbcr_chars.sort()
    return Colors(dict(char_rgb), dict(ycbcr_chars))

def resolve_font_face(path: str, *, name_only: bool = False) -> Face:
    '''Finds the `FontTools.ttLib.TTFont` with the given name,
    from the OS supported paths. Supports most major platforms.
    '''
    # here we are back in cross-platform compatibility purgatory
    try:
        if name_only:
            for root, _, names in os.walk(os.path.curdir):
                for name in names:
                    if name_only and os.path.splitext(name)[0] == path:
                        return Face(os.path.join(root, name))
                    elif name == path:
                        return Face(os.path.join(root, name))
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
                        print(os.path.splitext(name)[0])
                        if name_only and os.path.splitext(name)[0] == path:
                            return Face(os.path.join(root, name))
                        elif name == path:
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

def generate_colors(path: str, *, scale: int = 16, name_only: bool = False) -> Colors:
    '''Generates a mapping of fullwidth identifier characters and colors
    using font at the given path. Resulting colors are normalized by lightness,
    and sorted by color.

    The greater `scale` is, the more accurate the results will be.
    However, higher values are much slower to compute. Furthermore, the relative
    increase in quality tends to plateau quickly.
    '''
    face = resolve_font_face(path, name_only=name_only)
    face.set_pixel_sizes(width=scale, height=scale)
    colors: LightnesssMapping = {}
    for (c, i) in face.get_chars():
        char = chr(c)
        if is_fullwidth_identifier(char):
            # Load the glyph enough to fetch its horizontal advance
            # [in 16.16 fixed point units]
            advance = face.get_advance(i, FT_LOAD_FLAGS["FT_LOAD_DEFAULT"]) // (2**16)
            # Require "true" fullwidth. This excludes most Korean characters, in exchange for consistent widths
            if advance == scale:
                # Render the glyph's bitmap
                face.load_glyph(i, FT_LOAD_FLAGS["FT_LOAD_RENDER"])
                bitmap = face.glyph.bitmap
                if bitmap.width <= scale and bitmap.rows <= scale:
                    # TODO: Arithmetic mean of pixels may not always be appropriate?
                    colors[char] = sum(bitmap.buffer) // (scale ** 2)
    
    return colorize(colors)

def dump_font_mapping(font: str, out_prefix: str, scale: int = 16) -> Colors:
    '''Generates a color mapping for a font and dumps it into the specified directory.'''
    root, end = os.path.split(font)
    font_name, ext = os.path.splitext(end)
    colors = generate_colors(font, scale=scale, name_only=root == ext == "")
    if colors.is_empty():
        raise ValueError("Font has no valid characters")
    out_path = f"{font_name}.unimapping"
    if out_prefix:
        out_path = os.path.join(out_prefix, out_path)
    with open(out_path, "wb") as fp:
        colors.write_to(fp)
    return colors
