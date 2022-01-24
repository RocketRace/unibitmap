# Unibitmap

*A Python module & command-line interface for converting between RGB images and executable unicode art*

![Demo image](./demo.jpg?raw=true)

## Usage

Install the module using `python3 -m pip install git+https://github.com/RocketRace/unibitmap`. The interface is then accessible through `python -m unibitmap [options]`.

This help output is available through `python3 -m unibitmap --help`.

```
usage: python3 -m unibitmap [--help] [--height HEIGHT] [--width WIDTH] [--font FONT] [--name NAME] [--tabs] [--generate-mapping]
                 file [out]

Encodes images into executable character art which can be evaluated into the original image

positional arguments:
  file                  File to read image data from
  out                   Path of the resulting Python file

optional arguments:
  --help, -H            Show this help message and exit
  --height HEIGHT, -h HEIGHT
                        Rescaled height of the generated bitmap
  --width WIDTH, -w WIDTH
                        Rescaled width of generated bitmap
  --font FONT, -f FONT  Specify the font used for bitmap generation
  --name NAME, -n NAME  Specify the class name in resulting file
  --tabs, -t            Use tabs instead of 4 spaces in resulting file
  --generate-mapping    Generate or regenerate a color mapping for the current font, if one is missing
```

## API

The module exposes the `Bitmap` class intended for general use.
The `Pixels` class is also available, for more programmatic access to Unibitmap features.

* `Bitmap`: The "base class" from which bitmap images are generated. Its subclasses are parsed as executable character art. This class should only ever be subclassed, never accessed directly.

## Rationale

Unibitmap's primary goal is to create a textual encoding for image data that is both valid Python as well as visually similar to the image it encodes (akin to ASCII art). The project initially sparked by an interest in using Python's metaclass system and name lookup semantics in order to store arbitrary data into identifiers. An example is shown below:

```py
class MyImage(Bitmap):
    攟丶丶丶攟丶丶攟攟攟丶丶攟
    攟丶丶丶攟丶丶丶攟丶丶丶攟
    攟攟攟攟攟丶丶丶攟丶丶丶攟
    攟丶丶丶攟丶丶丶攟丶丶丶丶
    攟丶丶丶攟丶丶攟攟攟丶丶攟

# MyImage is now a `PIL.Image.Image` object
MyImage.show()
```

Unibitmap defines an encoding and decoding scheme based on characters in a given font. Encoding converts an image into executable unicode art, and relies on a `color -> character` mapping, where the characters for missing colors are approximated using a weighted nearest-neighbor search. Decoding art into an image relies on the reverse `character -> color` mapping, ignoring or raising an error on unknown characters. The encoding-decoding process is visualized below:

```
@========@     (color -> character)    @============@      (character -> color)     @========@
[        ]        +-----------+        [ Executable ]        +-------------+        [        ]
[ Source ] -------| Unibitmap |------> [ character  ] -------| Python      |------> [ Output ]
[ image  ] -------| CLI       |------> [ art        ] -------| interpreter |------> [ image  ]
[        ]        +-----------+        [            ]        +-------------+        [        ]
@========@      (Lossy conversion)     @============@      (Lossless conversion)    @========@
                        ^                                            ^
                        |        +-------------------------+         |
                        +--------| Font-generated mappings |---------+
                                 +-------------------------+
```

Both encoding and decoding mappings depend on a font file, used to determine the lightness of characters. If multiple characters share the same lightness, they are each given unique colors with that given lightness. Consequently, a font with more supported characters provides a greater color range for the resulting mappings. Unibitmap ships with a default mapping generated from the Noto Sans font, supporting a vast quantity of CJK characters.

Unibitmap executable character art is perfectly valid Python, requiring only the `unibitmap` module to be imported. Furthermore, as each bitmap is bound to a class scope, this can be used to include inline RGB images within Python scripts. (I pray that nobody does this in a serious project.) 

