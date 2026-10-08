import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
from shapely import Polygon, get_coordinates, make_valid

if TYPE_CHECKING:
    # import for type checking only to avoid circular runtime import
    from matplotlib.figure import Figure  # pragma: no cover

    from ..core.buildings_downloader import Building  # pragma: no cover


def make_surface_coords_valid(coordinates: np.ndarray) -> np.ndarray:
    """Return valid, single-polygon coordinates for a surface ring.

    LoD-2 source data occasionally contains self-intersecting surface polygons, which make shapely
    operations such as ``union_all`` raise a ``TopologyException``. This builds a polygon from the
    ``(N, 2)`` or ``(N, 3)`` coordinates and, if it is invalid, repairs it with
    ``make_valid(method="structure")`` (which preserves Z). If the repair splits the ring into
    several polygons, the largest by area is kept so the result stays a single ring, matching the
    single-polygon storage and database contract. The output keeps the same dimensionality (2D or
    3D) as the input.

    Args:
        coordinates: Array of shape ``(N, 2)`` or ``(N, 3)`` with 2D or 3D coordinates.

    Returns:
        The original coordinates when already valid or when no polygonal repair is possible
        otherwise the repaired single-ring ``(M, 2)`` or ``(M, 3)`` coordinates.
    """
    polygon = Polygon(coordinates)
    if polygon.is_valid:
        return coordinates
    repaired = make_valid(polygon, method="structure")
    if isinstance(repaired, Polygon):
        best = repaired
    else:
        polygons = [g for g in getattr(repaired, "geoms", []) if isinstance(g, Polygon)]
        if not polygons:
            return coordinates
        best = max(polygons, key=lambda g: g.area)
    if best.is_empty:
        return coordinates
    include_z = np.asarray(coordinates).shape[-1] == 3
    return get_coordinates(best.exterior, include_z=include_z)


def get_orientation_and_tilt(coordinates: List[List[float]]) -> Tuple[float, float]:
    """
    Calculate the azimuth angle (orientation) and the surface slope (tilt) of a 3D polygon.

    Args:
        coordinates: List of [x, y, z] coordinates representing the 3D polygon

    Returns:
        Tuple[float, float]: A tuple containing:
            - azimuth_degrees: The azimuth angle in degrees (0° = North, 90° = East, 180° =
                South, 270° = West)
            - slope_degrees: The surface tilt angle in degrees
    """

    # Convert to numpy array for easier manipulation
    polygon = np.array(coordinates)

    # Find the four most distant points from each other
    if len(polygon) > 4:
        # Calculate pairwise distances between all points
        n_points = len(polygon)
        distances = np.zeros((n_points, n_points))

        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = np.linalg.norm(polygon[i] - polygon[j])
                distances[i, j] = distances[j, i] = dist

        # Find the most distant points using a greedy approach
        selected_indices = []

        # First point: one end of the longest distance
        i, j = np.unravel_index(np.argmax(distances), distances.shape)
        selected_indices.append(i)

        # Second point: the other end of the longest distance
        selected_indices.append(j)

        # Third point: most distant from the first two
        remaining = list(set(range(n_points)) - set(selected_indices))
        max_dist = -1
        third_point = -1

        for idx in remaining:
            dist = distances[idx, selected_indices[0]] + distances[idx, selected_indices[1]]
            if dist > max_dist:
                max_dist = dist
                third_point = idx

        selected_indices.append(third_point)

        # Fourth point: most distant from the first three
        remaining = list(set(range(n_points)) - set(selected_indices))
        max_dist = -1
        fourth_point = -1

        for idx in remaining:
            dist = (
                distances[idx, selected_indices[0]]
                + distances[idx, selected_indices[1]]
                + distances[idx, selected_indices[2]]
            )
            if dist > max_dist:
                max_dist = dist
                fourth_point = idx

        selected_indices.append(fourth_point)

        # Use the selected points for calculation
        polygon = polygon[selected_indices]

    # Calculate the normal vector of the polygon
    # Using the first three points to define a plane
    v1 = polygon[1] - polygon[0]
    v2 = polygon[2] - polygon[0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)  # Normalize to unit vector

    # Ensure the normal vector points upwards (positive z-direction)
    if normal[2] < 0:
        normal = -normal  # Flip the normal if it points downward

    # Project the normal vector to the XY plane (2D)
    normal_2d = np.array([normal[0], normal[1], 0])
    if np.linalg.norm(normal_2d) > 0:
        normal_2d = normal_2d / np.linalg.norm(normal_2d)  # Normalize the 2D projection

    # Get the 2D normal vector components
    normal_x = normal_2d[0]
    normal_y = normal_2d[1]

    # Calculate azimuth (0° = North, 90° = East, 180° = South, 270° = West)
    # Using arctan2(x, y) because in our coordinate system:
    # North is +Y, East is +X, South is -Y, West is -X
    azimuth_degrees = (np.arctan2(normal_x, normal_y) * 180 / np.pi + 360) % 360

    # Calculate the surface tilt angle (slope)
    # This is the angle between the normal vector and the Z-axis
    slope_degrees = np.arccos(abs(normal[2])) * 180 / np.pi

    return azimuth_degrees, slope_degrees


def get_orientation(polygon: Polygon) -> float:
    """
    Return the azimuth of a polygon based on the minimum rotated rectangle. The orientation of the
    edge that faces the southernmost direction is used for calculating the orientation. If multiple
    edges are equally distant from south (e.g., one at 120° and another at 240°), the longer edge
    is chosen.

    Only tested with rectangular polygons so far, but should work for any polygon since the minimum
    rotated rectangle is taken for the calculations.

    Args:
        polygon (Polygon): The input polygon as a shapely Polygon.

    Returns:
        float: The azimuth angle in degrees (0° = North, 90° = East, 180° = South, 270° = West)
    """

    def _azimuth(point1, point2):
        """Azimuth between 2 points (interval 0 - 180)"""
        angle = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
        return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180

    def _get_edge_length(p1, p2):
        """Get the length of an edge"""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def _get_distance_from_south(edge):
        """Calculate how far an edge's orientation is from south (180°)"""
        az = _azimuth(edge[0], edge[1])
        base_az = (180 - az) % 360
        # Adjust to prefer southern direction range (90-270°)
        orientation = base_az + (180 if base_az < 90 else 0)
        # Calculate distance from south (180°)
        return min(
            abs(orientation - 180), abs(orientation - 180 + 360), abs(orientation - 180 - 360)
        )

    mrr = polygon.minimum_rotated_rectangle
    bbox = list(mrr.exterior.coords)

    # Get the four edges of the rectangle (excluding the duplicate last point)
    edges = [(bbox[i], bbox[(i + 1) % 4]) for i in range(4)]

    # Find edges with minimum distance from south
    min_distance_from_south = min(_get_distance_from_south(edge) for edge in edges)
    candidate_edges = [
        edge for edge in edges if _get_distance_from_south(edge) == min_distance_from_south
    ]

    # If multiple edges are equally distant from south, choose the longer one
    if len(candidate_edges) > 1:
        southernmost_edge = max(
            candidate_edges, key=lambda edge: _get_edge_length(edge[0], edge[1])
        )
    else:
        southernmost_edge = candidate_edges[0]

    # Calculate azimuth for the selected edge
    az = _azimuth(southernmost_edge[0], southernmost_edge[1])

    # now make sure that the azimuth is in the range 90 to 270°
    base_az = (180 - az) % 360
    return base_az + (180 if base_az < 90 else 0)


def sanitize_float(value: Optional[float]) -> Optional[float]:
    """
    Ensure the float value is valid and properly formatted.

    Args:
        value (float): The float value to sanitize.
    Returns:
        Optional[float]: The sanitized float value rounded to 2 decimal places, or None if the value
            is not finite.
    """
    if value is None or not np.isfinite(value):
        return None
    return round(value, 2)


def plot_building_wireframe(
    buildings: Union["Building", Iterable["Building"]],
    plot_grounds: bool = True,
    plot_roofs: bool = True,
    plot_walls: bool = True,
    marker: str = "o",
) -> "Figure":
    """Plot a Building's roof, wall and ground surfaces as a 3D wireframe.

    Args:
        buildings: One or more Building objects containing roofs, walls and grounds.
        plot_grounds: Whether to plot ground surfaces.
        plot_roofs: Whether to plot roof surfaces.
        plot_walls: Whether to plot wall surfaces.
        marker: Marker used for vertices.

    Returns:
        The created figure.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("'matplotlib' is required to use 'plot_building_wireframe()'") from e

    # normalize to iterable of buildings
    if not isinstance(buildings, Iterable):
        buildings = [buildings]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    all_points = []

    for b in buildings:
        surfaces = [
            (b.roofs, "tab:orange") if plot_roofs else None,
            (b.grounds, "tab:red") if plot_grounds else None,
            (b.walls, "tab:blue") if plot_walls else None,
        ]

        for item in surfaces:
            if item is None:
                continue

            surface_list, color = item

            for surface in surface_list:
                coords = surface.gml_surface_3darray
                all_points.append(coords)

                closed = np.vstack([coords, coords[0]])

                ax.plot(
                    closed[:, 0],
                    closed[:, 1],
                    closed[:, 2],
                    color=color,
                    marker=marker,
                )

    # equal metric scaling (1 m in X = 1 m in Y = 1 m in Z)
    if all_points:
        all_points = np.vstack(all_points)
        ax.set_box_aspect(np.ptp(all_points, axis=0))

    return fig
