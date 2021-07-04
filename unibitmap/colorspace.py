'''
Colorspace utilities to efficiently add color data to mappings
'''
from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Any, Generator, Iterable, Literal, TypeVar

import numpy as np
from scipy import optimize, spatial

np.seterr(divide='ignore')

__all__ = ()

Axis = Literal['r', 'g', 'b']
T = TypeVar('T')
U = TypeVar('U')

if TYPE_CHECKING:
    Float = np.dtype[np.float64]
    Int = np.dtype[np.int32]
    Uint = np.dtype[np.uint8]
    Shape2 = tuple[Literal[2]]
    Shape3 = tuple[Literal[3]]
    Shape23 = tuple[Literal[2], Literal[3]]
    Shape32 = tuple[Literal[3], Literal[2]]
    ShapeN2 = tuple[Any, Literal[2]]
    ShapeN22 = tuple[Any, Literal[2], Literal[2]]
    Edges = np.ndarray[ShapeN22, Float]
    Matrix = np.ndarray[Shape23, Float]
    InverseMatrix = np.ndarray[Shape32, Float]
    Int3Vec = np.ndarray[Shape3, Uint]
    Int2Vec = np.ndarray[Shape2, Int]
    Float3Vec = np.ndarray[Shape3, Float]
    Float2Vec = np.ndarray[Shape2, Float]
    Int2Vecs = np.ndarray[ShapeN2, Int]

RGB_TO_YCBCR_TRANSFORM = np.array([
    [0.299,           0.587,          0.114],
    [-0.299 / 1.772, -0.587 / 1.772,  0.5],
    [0.5,            -0.587 / 1.402, -0.114 / 1.402],
])

LUMA_TRANSFORM: Float3Vec = RGB_TO_YCBCR_TRANSFORM[0]

CHROMINANCE_TRANSFORM: Matrix = RGB_TO_YCBCR_TRANSFORM[1:]

INVERSE_CHROMINANCE_TRANSFORM: InverseMatrix = np.array([
    [0,              1.402],
    [-0.114 * 1.772, -0.299 * 1.402 / 0.587],
    [1.772,          0],
])

class Edge:
    '''One edge of the RGB color space cube.'''
    axis: Axis
    def __init__(self, axis: Axis, constants: tuple[int, int]) -> None:
        '''One edge of the RGB color space cube.

        `axis` represents the axis with variable coordinates.
        `constants` represents the axes with constant coordinates, 
        ordered from R to G to B.'''
        self.axis = axis
        self.constants = constants

    @classmethod
    def rgb(cls) -> Generator[Edge, None, None]:
        '''The 12 lines that each edge of the RGB cube lies on'''
        for point in itertools.product((0, 255), repeat=2):
            for axis in 'rgb':
                yield cls(axis, point) # type: ignore

class Plane:
    '''A plane in 3D space, representing YCbCr color space with constant Y'''
    def __init__(self, transform: Float3Vec, luma: int) -> None:
        self.transform = transform
        self.luma = luma

    def intersection(self, edge: Edge) -> Int3Vec | None:
        '''Returns the intersection point between this plane and the given line segment.
        
        If one does not exist, returns `None`.
        '''
        k_r, k_g, k_b = self.transform
        if edge.axis == 'r':
            g, b = edge.constants
            r = (self.luma - k_g * g - k_b * b) / k_r
            if 0 <= r <= 255:
                return np.array([round(r), g, b])
        elif edge.axis == 'g':
            r, b = edge.constants
            g = (self.luma - k_r * r - k_b * b) / k_g
            if 0 <= g <= 255:
                return np.array([r, round(g), b])
        elif edge.axis == 'b':
            r, g = edge.constants
            b = (self.luma - k_r * r - k_g * g) / k_b
            if 0 <= b <= 255:
                return np.array([r, g, round(b)])

    def rgb_intersections(self, edges: Iterable[Edge]) -> Iterable[Int3Vec]:
        '''Returns the RGB gamut from the intersection of a plane and the egdes of a cube'''
        return np.unique(list(filter(lambda x: x is not None, map(self.intersection, edges))), axis=0)

def ycbcr(point: tuple[int, int, int]) -> tuple[int, int, int]:
    '''Plain ycbcr transformation'''
    arr = np.array(point)
    y = LUMA_TRANSFORM.dot(arr)
    cb, cr = CHROMINANCE_TRANSFORM.dot(arr)
    return (round(y), round(cb), round(cr))

def as_cbcr(point: Int3Vec) -> Int2Vec:
    '''Converts a point from RGB space to CbCr space, discarding luma information.'''
    # Note: i32 used, to avoid overflow later
    return CHROMINANCE_TRANSFORM.dot(point).round().clip(-128, 127).astype('int32')

def as_rgb(point: Int2Vec, luma: int) -> Int3Vec:
    '''Converts a point from CbCr space to RGB space using a given luma value'''
    # Unlike above, u8 is used - this is because conversion to RGB is final
    return (INVERSE_CHROMINANCE_TRANSFORM.dot(point) + luma).round().clip(0, 255).astype('uint8')

def points_as_cbcr(rgb_points: Iterable[Int3Vec]) -> Int2Vecs:
    '''Converts an RGB color gamut into CbCr space, sorted counterclockwise around the origin'''
    return np.array(sorted((as_cbcr(x) for x in rgb_points), key=lambda point: math.atan2(point[1], point[0])))

def color_delta(a: Int2Vec, b: Int2Vec) -> float:
    '''The approximate difference between the two points.
    
    Less accurate but bunches more efficient than CIE DE transformations.
    '''
    return math.dist(a, b)

def ray_intersection(point: Int2Vec, a: Int2Vec, b: Int2Vec) -> bool | None:
    '''Does a horizontal ray from the point intersect the edge defined by two endpoints?
    
    Returns a boolean if the ray intersects with the edge. Returns None if the point is on the edge.
    '''
    x = point[0]
    y = point[1]
    a_x = a[0]
    a_y = a[1]
    b_x = b[0]
    b_y = b[1]
    
    # Degenerate case
    # This only occurs when the entire gamut is a single point
    # => said point is only inside the gamut
    if a_x == b_x and a_y == b_y:
        if a_x == x and a_y == y:
            return None
        else:
            return False

    # On the endpoints => inside the polygon
    if (a_x == x and a_y == y) or (b_x == x and b_y == y):
        return None

    # Bounding box
    if a_y > b_y:
        top = a_y
        bottom = b_y
    else:
        top = b_y
        bottom = a_y
    if a_x > b_x:
        right = a_x
        left = b_x
    else:
        right = b_x
        left = a_x

    # Between the y values
    # This inequality has to be inclusive on one end (doesn't matter which)
    # to avoid inaccuracies in reporting
    if bottom <= y < top:
        intersection = left + (right - left) / (top - bottom) * (top - y)
        # Overlap
        if x == intersection:
            return None
        if x < intersection:
            return True

    # On the same horizontal / vertical:
    # In the edge => In the polygon
    # Else, does not intersect
    if y == top == bottom:
        if left < x < right:
            return None
    
    if x == left == right:
        if bottom < y < top:
            return None

    return False

class Gamut:
    '''Represents a color gamut in CbCr space. Its edges represent a convex polygon.'''
    def __init__(self, edges: Edges, points: Int2Vecs) -> None:
        self.edges = edges
        self.points = points
        # Gamut bounding box
        axes = self.points.T
        self.min_x = axes[0].min()
        self.min_y = axes[1].min()
        max_x = axes[0].max()
        max_y = axes[1].max()
        self.max_x = 127 if max_x == 127 else max_x + 1
        self.max_y = 127 if max_y == 127 else max_y + 1

    @classmethod
    def from_rgb_points(cls, outer_points: Iterable[Int3Vec]) -> Gamut:
        '''Initialize the gamut from a set of points in RGB space.
        
        This converts the points into CbCr space, then into the edges of the polygon.
        '''
        points = points_as_cbcr(outer_points)
        edges = np.array([
            # pairs of consecutive points
            [p0, p1] for p0, p1 in zip(points, np.roll(points, -1, axis=0))
        ])
        return cls(edges, points)

    @property
    def area(self) -> int:
        '''How many pixels are encompassed by the gamut'''
        return (self.max_x - self.min_y) * (self.max_y - self.min_y)

    def __contains__(self, point: Int2Vec) -> bool:
        # Using the raycasting method to solve the point-in-polygon problem
        count = 0
        for edge in self.edges:
            result = ray_intersection(point, edge[0], edge[1])
            if result is None:
                return True
            else:
                count += result
        return count % 2 == 1

class Optimizer:
    '''Optimization!
    
    This finds the configuration of two variables (angle, distance) for 
    a hexagonal packing of points inside the gamut that minimizes the 
    `mean_distance` metric. These two variables define a linear transformation
    on a simple hexagonal tiling with 1 unit between adjacent points.
    '''
    def __init__(self, luma: int, n: int) -> None:
        '''`luma` in [0, 255), `n` positive'''
        self.luma = luma
        self.n = n
        self.gamut = Gamut.from_rgb_points(Plane(LUMA_TRANSFORM, luma).rgb_intersections(Edge.rgb()))
        # from 0 to pi/3
        self.angle = 0.
        # approximate initial distance (works best for close-to-square gamuts)
        self.distance = self.approximate_distance()
        self.points = np.array([])
        self.tree = None

    def approximate_distance(self):
        '''A decent guess for the optimal distance'''
        return max(0.1, math.sqrt(self.gamut.area / self.n))

    def get_packing_points(self) -> Int2Vecs:
        '''Standard hexagonal packing (or a triangular grid)
        that fits inside the gamut
        '''
        # compute an upper bound for the width & height of the grid
        size_hint = min(128, max(math.dist([0, 0], point) for point in self.gamut.points))
        unit_width = self.distance
        unit_height = self.distance * np.sqrt(3)/2
        
        # degenerate case
        if unit_width == unit_height == 0:
            return np.array([[0., 0.]])
        
        x_scale = size_hint - size_hint % unit_width + unit_width
        y_scale = size_hint - size_hint % unit_height + unit_height

        # The `+ unit_xyz` are used to make the range ends inclusive at `xyz_scale`.
        # The `* 1.5` are necessary due to floating point errors.
        # An extra `+0.5 unit_xyz` on top of the previous `+1 unit_xyz`
        # ensures that small errors don't lead to an off-by-one error
        # where an element can't "fit" into the grid, while still making sure
        # there isn't a superfluous element added. (This fix can't just
        # be a flat offset since unit_xyz can be arbitrarily small and
        # computing their absolute minimum is unnecessary & not future-proof)
        grid = np.mgrid[
            -x_scale : x_scale + unit_width * 1.5 : unit_width,
            -y_scale : y_scale + unit_height * 1.5 : unit_height
        ].reshape(2, -1).T

        # shift every other row right to make it a triangular grid
        # make sure to keep the (0, 0) row untouched
        epsilon_x = unit_width / 2
        epsilon_y = unit_height / 2
        origin_indices = [
            i for i, (x, y) in enumerate(grid)
            if abs(x) < epsilon_x and abs(y) < epsilon_y
        ]
        shift_offset = origin_indices[0] + 1

        grid[shift_offset % 2::2, 0] += unit_width / 2

        rot = np.array([
            [math.cos(self.angle), -math.sin(self.angle)],
            [math.sin(self.angle), math.cos(self.angle)]
        ])

        grid = np.array([rot.dot(point) for point in grid])

        # points outside the color space
        grid = grid[
              (grid[:, 0] >= -128) 
            & (grid[:, 1] >= -128)
            & (grid[:, 0] <= 127) 
            & (grid[:, 1] <= 127)
        ]

        # deduplication is necessary due to rounding
        points = self.points = np.unique(
            np.array([
                point for point in grid.round().astype('int8') if point in self.gamut
            ]),
            axis=0
        )
        self.tree = spatial.cKDTree(points) # type: ignore
        return points

    def optimize(self) -> tuple[float, float]:
        '''Find the range of distances & angles that ensures `self.count <= self.n`
        and roughly minimizes `self.mean_distance()`.
        '''
        if self.gamut.area == 0:
            return (0, 1)
        dist_min = 0
        dist_max = 128
        angle_min = 0
        angle_max = math.pi/3
        dist_precision = dist_max / 128
        angle_precision = angle_max / 16

        samples = {}
        # the behavior as angle changes is fairly unpredictable => plain iteration
        for angle in np.arange(angle_min, angle_max, angle_precision):
            self.angle = angle
            self.distance = self.approximate_distance()
            self.get_packing_points()
            lower_bound = dist_min
            upper_bound = dist_max
            # find the boundary distance after which self.count <= self.n
            # the count is nonincreasing as distance increases (discretely)
            while (upper_bound - lower_bound) > dist_precision:
                self.distance = (lower_bound + upper_bound) / 2
                self.get_packing_points()
                if self.count <= self.n:
                    # the minimal mean distance tends to be close to the boundary
                    # by reusing our calculations, we can get a close enough approximation
                    # without having to recompute the packing points
                    samples[angle, self.distance] = self.mean_distance()
                    upper_bound = self.distance
                else:
                    lower_bound = self.distance
        
        # a little unpack assignment abuse (as a treat)
        self.angle, self.distance = minimum = min(samples.items(), key=lambda a: a[1])[0]
        return minimum

    def get_arrangement_points(self) -> Generator[tuple[tuple[int, int, int], tuple[int, int, int]], None, None]:
        '''Get the packing points in (RGB, YCbCr) format (this is typically final)'''
        for point in self.get_packing_points():
            ycbcr = (self.luma, point[0], point[1])
            rgb = tuple(as_rgb(point, self.luma))
            yield (rgb, ycbcr) # type: ignore

    @property
    def count(self):
        '''Number of arranged points currently within the gamut'''
        return len(self.points)

    def sample_points(self, gap: int) -> Generator[Int2Vec, None, None]:
        '''Grid of points within the gamut with `gap` spaces between
        each other in the Cb and Cr axes. `gap` should be positive.
        '''
        yield from self.gamut.points
        yield from np.mgrid[
            self.gamut.min_x : self.gamut.max_x : gap,
            self.gamut.min_y : self.gamut.max_y : gap
        ].reshape(2,-1).T
    
    def debug(self):
        '''Render the current simulation state'''
        from PIL import Image
        img = Image.new('RGB', (256, 256))
        for point in self.gamut.points:
            img.putpixel(point + 128, (0, 0, 255))
        for point in self.get_packing_points():
            img.putpixel(point + 128, (255, 0, 0))

        img.putpixel((128, 128), (0, 255, 0))
        return img

    def mean_distance(self) -> float:
        '''Sample points in a grid, returning the maximal distance between
        some point and its closest neighbor in `self.points`.
        '''
        gap = min(15, max(1, self.gamut.area))
        dd, _ = self.tree.query( # type: ignore
            np.array([point for point in self.sample_points(gap) if point in self.gamut])
        )
        return np.mean(dd)

def optimize_points(luma: int, n: int) -> Generator[tuple[tuple[int, int, int], tuple[int, int, int]], None, None]:
    '''Generate colors required by the optimization.
    Yields (RGB, YCbCr) pairs.
    '''
    opt = Optimizer(luma, n)
    opt.optimize()
    print(luma, n)
    return opt.get_arrangement_points()
