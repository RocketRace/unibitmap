'''Unibitmap

TODO description
'''

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
from PIL import Image

from .mappings import Colors, get_mapping, is_fullwidth_identifier

if TYPE_CHECKING:
    UnicodeGrid = np.ndarray[tuple[Any, Any], np.dtype[np.unicode_]]

__all__ = ("Bitmap", "Pixels")

class Pixels:
    '''The internal representation of pixel bitmaps, wrapping `PIL.Image.Image`.'''
    def __init__(self, img: Image.Image) -> None:
        '''Wraps a `PIL.Image.Image`.
        
        For more flexible constructors, consider the `Pixels.from_image` and `Pixels.from_unicode` classmethods.
        '''
        self.img = img

    @classmethod
    def from_image(cls, img: Image.Image, *, width: int | None = None, height: int | None = None) -> Pixels:
        '''Constructs a `Pixels` instance from a `PIL.Image.Image`, and optional width/height scaling factors.
        
        If one of `width` and `height` are provided, the image is scaled to fit this dimension, preserving ratio.
        
        If both are provided, the image is resized as provided.
        
        Otherwise, the image is passed as-is.
        '''
        if width is not None and height is not None:
            size = width, height
        elif width is None and height is not None:
            size = img.width * height // img.height, height
        elif width is not None and height is None:
            size = width, img.height * width // img.width 
        else:
            size = img.size
        return cls(img.convert("RGB").resize(size))

    @classmethod
    def from_unicode(cls, mapping: Colors, grid: UnicodeGrid, *, ignore_unknown=False) -> Pixels:
        '''Constructs a `Pixels` instance from unicode pixel data.
        
        If `ignore_unknown` is `False` (the default), raises `ValueError` for unknown characters in the strings.
        Otherwise, unknown characters are interpreted as black pixels.
        '''
        width, height = grid.shape
        return cls(
            Image.fromarray(
                mapping.get_colors(grid, ignore_unknown=ignore_unknown).reshape((width, height, 3)),
                mode="RGB"
            )
        )

    def to_image(self) -> Image.Image:
        '''Returns the underlying image instance.'''
        return self.img
    
    def to_unicode(self, mapping: Colors) -> UnicodeGrid:
        '''Converts the underlying image into a list of unicode strings.'''
        width, height = self.img.size
        arr = np.asarray(self.img).reshape((width * height, 3))
        return mapping.get_chars(arr).reshape((height, width))

class Namespace(dict):
    '''Hack to hook into namespace access in the class body.
    
    This is the class through which unicode bitmaps are read into pixel data.
    '''
    parent: type[Meta]
    mapping: Colors
    @classmethod
    def with_parent(cls, parent: type[Meta]) -> Namespace:
        self = cls()
        self.parent = parent
        return self
    
    def __missing__(self, key: str):
        # this is "unconventional"
        if all(is_fullwidth_identifier(c) for c in key):
            self.parent.rows.append(list(key))
            return None
        raise KeyError(key)

class Meta(type):
    '''Metaclass that injects the custom `Namespace` namespace into class creation,
    replacing the newly created class with its parsed image data.

    For more information, see the readme.
    '''
    rows = []
    @classmethod
    def __prepare__(cls, name: str, bases: tuple[type, ...], *, font: str | None = None, **kwargs: Any) -> Mapping[str, Any]:
        cls.rows.clear()
        return Namespace.with_parent(cls)

    def __new__(cls, name: str, bases: tuple[type, ...], ns: dict[str, Any], *, font: str | None = None, **kwargs: Any) -> Meta | Image.Image:
        if len(bases) == 0:
            return super().__new__(cls, name, bases, ns)
        else:
            # this is "unconventional"
            return Pixels.from_unicode(get_mapping(font), np.array(cls.rows)).to_image()
class Bitmap(metaclass=Meta):
    '''The "base class" from which bitmap images are generated.
    Its subclasses are parsed as unicode art bitmap images.
    
    Subclassing `Bitmap` will yield instances of `PIL.Image.Image` rather than new classes. 
    This is due to the unconventional (see: "hacky") use of metaclasses, and is thus likely unsupported
    by most static type-checkers, expecting class creation to return a `type` instance.
    If you care about this, be prepared to add some `# type: ignore` lines.
    For more information, see the readme.

    A possible bitmap may look like:
    ```py
    class MyImage(Bitmap, font=...):
        攟丶丶丶攟丶丶攟攟攟丶丶攟
        攟丶丶丶攟丶丶丶攟丶丶丶攟
        攟攟攟攟攟丶丶丶攟丶丶丶攟
        攟丶丶丶攟丶丶丶攟丶丶丶丶
        攟丶丶丶攟丶丶攟攟攟丶丶攟

    # MyImage is now a `PIL.Image.Image` object
    MyImage.show()
    ```
    '''
