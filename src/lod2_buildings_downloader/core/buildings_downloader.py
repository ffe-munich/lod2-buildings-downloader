import logging
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Union
from uuid import uuid4

try:
    import citydpc.core.object.surfacegml as citydpc_surfacegml
    from citydpc import Dataset, config
    from citydpc.core.input.citygmlInput import load_buildings_from_xml_file
    from citydpc.core.object import SurfaceConfig
    from citydpc.core.object.address import CoreAddress
    from citydpc.core.object.building import Building as CityDPCBuilding
    from citydpc.core.object.building import BuildingPart
    from citydpc.core.object.exceptions import SurfaceSplitDueToMultipleSurfaceMembers
    from citydpc.logger import logger as citydpc_logger
except ModuleNotFoundError as exc:
    if exc.name != "citydpc":
        raise
    citydpc_install_url = "git+https://github.com/ffe-munich/CityDPC.git@main"
    raise ImportError(
        "CityDPC is required to use lod2-buildings-downloader. Install it with "
        f'`pip install "{citydpc_install_url}"` or `uv add "{citydpc_install_url}"`. '
        "See the README for details."
    ) from exc

import numpy as np
from pyproj import Transformer
from requests import get
from requests.exceptions import RequestException
from shapely import (
    MultiPolygon,
    Polygon,
    box,
    buffer,
    envelope,
    intersects,
    union_all,
    within,
)

if TYPE_CHECKING:
    from orthophotos_downloader.data_scraping.image_download import (
        Image,
        ImageDownloader,
    )  # noqa: F401

from lod2_buildings_downloader.utils.helpers import (
    get_orientation,
    get_orientation_and_tilt,
    make_surface_coords_valid,
    sanitize_float,
)

log = logging.getLogger(__name__)


class _ErrorOnlyFilter(logging.Filter):
    """Passes only ERROR and CRITICAL records. Used to silence noisy third-party loggers."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.ERROR


_citydpc_configured = False


def _configure_citydpc() -> None:
    """Apply the global citydpc settings this library relies on.

    Kept out of import scope so that importing the package has no side effects (no third-party
    global mutation, no log output). Invoked lazily only from the public entry points (i.e. the
    building-generation functions).
    """
    global _citydpc_configured
    if _citydpc_configured:
        return

    # set this threshold value to a lower value than the default to avoid removal of meaningful
    # points; see: https://github.com/ffe-munich/CityDPC
    log.info(
        "Setting distance between line and point to 0.001 to avoid removal of meaningful points."
    )
    SurfaceConfig.set_distance_between_line_and_point(0.001)
    log.info("Suppressing 'SurfaceSplitDueToMultipleSurfaceMembers' warnings from citydpc package.")
    config.SUPPRESSED_WARNING_CATEGORIES.add(SurfaceSplitDueToMultipleSurfaceMembers)
    _citydpc_configured = True


VALID_ALLOWED_SURFACE_TYPES = {"RoofSurface", "GroundSurface", "WallSurface"}


@dataclass(frozen=True)
class _SurfaceBase:
    """Shared base for ground, roof and wall surfaces.

    The 2D coordinates are derived from the 3D coordinates in the ``gml_surface_3darray`` property,
    which is a view of ``gml_surface_3darray`` with the Z coordinate dropped.
    """

    gml_id: str
    gml_surface_3darray: np.ndarray  # holds 3D coordinates of the surface
    surface_area: float

    def __post_init__(self):
        """Ensures that the gml_surface_3darray is immutable after initialization."""
        self.gml_surface_3darray.setflags(write=False)

    @property
    def gml_surface_2darray(self) -> np.ndarray:
        """Derived from gml_surface_3darray by dropping the Z coordinate."""
        return self.gml_surface_3darray[:, :2]

    @property
    def gml_surface_2array(self) -> np.ndarray:
        """Proxy for gml_surface_2darray to maintain downward compatibility."""
        return self.gml_surface_2darray


@dataclass(frozen=True)
class GroundSurface(_SurfaceBase):
    """Represents a ground surface of a building.

    Attributes:
        gml_id: The GML ID of the surface.
        gml_surface_3darray: Array of shape (N, 3) containing the 3D coordinates of the surface.
            Made immutable after initialization.
        surface_area: The area of the surface in square meters.
        gml_surface_2darray: Array of shape (N, 2) containing the 2D coordinates of the surface,
            derived from ``gml_surface_3darray`` by dropping the Z coordinate.
        gml_surface_2array: Proxy for ``gml_surface_2darray`` to maintain downward compatibility.
    """

    pass  # nothing extra


@dataclass(frozen=True)
class WallSurface(_SurfaceBase):
    """Represents a wall surface of a building.

    Attributes:
        gml_id: The GML ID of the surface.
        gml_surface_3darray: Array of shape (N, 3) containing the 3D coordinates of the surface.
            Made immutable after initialization.
        surface_area: The area of the surface in square meters.
        gml_surface_2darray: Array of shape (N, 2) containing the 2D coordinates of the surface,
            derived from ``gml_surface_3darray`` by dropping the Z coordinate.
        gml_surface_2array: Proxy for ``gml_surface_2darray`` to maintain downward compatibility.
    """

    pass  # nothing extra


@dataclass(frozen=True)
class RoofSurface(_SurfaceBase):
    """Represents a roof surface of a building.

    Attributes:
        gml_id: The GML ID of the surface.
        gml_surface_3darray: Array of shape (N, 3) containing the 3D coordinates of the surface.
            Made immutable after initialization.
        surface_area: The area of the surface in square meters.
        gml_surface_2darray: Array of shape (N, 2) containing the 2D coordinates of the surface,
            derived from ``gml_surface_3darray`` by dropping the Z coordinate.
        gml_surface_2array: Proxy for ``gml_surface_2darray`` to maintain downward compatibility.
        surface_tilt: Cleaned tilt angle of the roof surface in degrees.
        surface_orientation: Cleaned orientation angle of the roof surface in degrees.
        surface_tilt_original: Original tilt angle from the source data, before any normalization.
            May be None if unavailable.
        surface_orientation_original: Original orientation angle from the source data, before any
            normalization. May be None if unavailable.
    """

    surface_tilt: float
    surface_orientation: float
    surface_tilt_original: Optional[float]
    surface_orientation_original: Optional[float]


@dataclass(frozen=True)
class Address:
    """A postal address of a building, extracted from a citydpc ``CoreAddress``.

    All fields are optional because CityGML address data is frequently incomplete.

    When instantiating an object via :meth:`from_citydpc_address`, the street and house-number might
    be merged in the streed field depending on the original CityGML data (e.g. ``"Bavariaring 29"``
    in the street field with an empty house-number field). In these cases, the method tries to split
    the house-number information off and store it in the house_number field, while the street field
    is cleaned up to only contain the street name. See :data:`_TRAILING_HOUSE_NUMBER_RE`.

    Attributes:
        street: Street/thoroughfare name (citydpc ``thoroughfareName``).
        house_number: House/building number (citydpc ``thoroughfareNumber``).
        postal_code: Postal/zip code (citydpc ``postalCodeNumber``).
        locality: City/municipality name (citydpc ``localityName``).
        country: Country name (citydpc ``countryName``).
    """

    # Matches a trailing house number so it can be split off a street that merged both, e.g.
    # "Musterstraße 29" -> ("Musterstraße", "29"), "Hauptstraße 31A" -> ("Hauptstraße", "31A"),
    # "Am Markt 12-14" -> ("Am Markt", "12-14"). A number-less street (e.g. "Im Körbchen") is left
    # untouched because the pattern requires a leading digit in the trailing token.
    _TRAILING_HOUSE_NUMBER_RE = re.compile(
        r"^(?P<street>.+?)\s+(?P<house_number>\d+(?:\s?[-/]\s?\d+)?\s?[a-zA-Z]?)$"
    )

    street: Optional[str] = None
    house_number: Optional[str] = None
    postal_code: Optional[str] = None
    locality: Optional[str] = None
    country: Optional[str] = None

    @classmethod
    def from_citydpc_address(cls, address: CoreAddress) -> "Address":
        """Creates an Address from a citydpc ``CoreAddress`` object.

        Splits a trailing house number off the street when the house number field is empty but the
        street carries both (e.g. ``"Musterstraße 29"``). See :data:`_TRAILING_HOUSE_NUMBER_RE`.
        """
        street = address.thoroughfareName
        house_number = address.thoroughfareNumber
        if street and not house_number:
            match = cls._TRAILING_HOUSE_NUMBER_RE.match(street.strip())
            if match:
                street = match.group("street").strip()
                house_number = match.group("house_number").strip()
        return cls(
            street=street,
            house_number=house_number,
            postal_code=address.postalCodeNumber,
            locality=address.localityName,
            country=address.countryName,
        )

    @property
    def is_empty(self) -> bool:
        """True if the address carries no information at all."""
        return not any(
            (self.street, self.house_number, self.postal_code, self.locality, self.country)
        )

    @property
    def formatted(self) -> str:
        """A single-line, human-readable address with missing parts omitted."""
        street_line = " ".join(part for part in (self.street, self.house_number) if part)
        locality_line = " ".join(part for part in (self.postal_code, self.locality) if part)
        return ", ".join(part for part in (street_line, locality_line, self.country) if part)

    def __str__(self) -> str:
        return self.formatted


class Building:
    """Represents a single LoD-2 building (or building part) with its surfaces and attributes.

    Buildings may form a 1:n parent <-> children respectively building <-> building-part
    relationship: a building with building parts acts as the group ``parent`` and holds its parts in
    ``children``, while each part references the parent via the ``parent`` attribute. A building
    without a parent and without children is just a standalone building.

    Semantic attributes are stored in backing fields (``_creation_date``, ``_function``, ``_usage``,
    ``_year_of_construction``, ``_roof_type``, ``_measured_height``, ``_storeys_above_ground``) and
    exposed through same-named properties. When a building part's own backing value is ``None``, the
    property falls back to the value stored on its parent building. A parent building only ever
    reports its own value and never pulls from its parts, since parts may legitimately disagree.
    Only backing fields are inspected during resolution, so there is no recursion between a parent
    and its parts. See :meth:`_resolve_grouped_attribute`.

    Addresses and additional attributes follow the same part -> parent idea via the
    :attr:`addresses` property (with :attr:`address` as a convenience for the first one) and the
    :attr:`additional_attributes` property, except that an empty collection -- an empty list or dict
    rather than ``None`` -- marks "unset": a building part with none of its own inherits its
    parent's.

    So if you need to read the original CityDPC data, access the backing fields directly (e.g.
    ``building._function``). If you want to read the group-resolved value, use the properties (e.g.
    ``building.function``).

    Attributes:
        gml_id: The GML ID of the building.
        epsg: The EPSG code of the CRS used by the building's surface coordinates.
        grounds: Ground surfaces of the building.
        roofs: Roof surfaces of the building.
        walls: Wall surfaces of the building.
        parent: The parent building if this building is a part, otherwise ``None``.
        children: The building parts if this building has any, otherwise ``None``.
        creation_date: Resolved creation date (see class description).
        function: Resolved building function.
        usage: Resolved building usage.
        year_of_construction: Resolved year of construction.
        roof_type: Resolved roof type.
        measured_height: Resolved measured height.
        storeys_above_ground: Resolved number of storeys above ground.
        additional_attributes: Resolved dict of additional attributes; a part inherits its parent's
            when it has none of its own (see class description).
        addresses: Resolved list of Address objects; a part inherits its parent's addresses when it
            has none of its own (see class description).
        address: The first resolved Address, or ``None`` if the building has no address.
    """

    FLAT_ROOF_TILT_THRESHOLD = 15.0
    MINIMUM_TILT = 0
    MAXIMUM_TILT = 70

    def __init__(
        self,
        gml_id: str,
        epsg: int,
        grounds: List[GroundSurface] = [],
        roofs: List[RoofSurface] = [],
        walls: List[WallSurface] = [],
        creation_date: Optional[str] = None,
        function: Optional[str] = None,
        usage: Optional[str] = None,
        year_of_construction: Optional[int] = None,
        roof_type: Optional[str] = None,
        measured_height: Optional[float] = None,
        storeys_above_ground: Optional[int] = None,
        additional_attributes: Optional[Dict[str, Union[str, float]]] = None,
        addresses: Optional[List[Address]] = None,
        parent: Optional["Building"] = None,
        children: Optional[List["Building"]] = None,
    ):
        """Initializes a Building object.

        This class is intended to be initialized via the :meth:`from_citydpc_building` class method,
        which converts a CityDPC Building or BuildingPart object to a Building object. The
        constructor is provided for direct initialization of a Building object, but it is
        recommended to use the class method for initialization.

        Args:
            gml_id: The GML ID of the building.
            epsg: The EPSG code of the CRS used in the coordinates in the grounds and roofs.
            grounds: List of GroundSurface objects representing the ground surfaces of the building.
            roofs: List of RoofSurface objects representing the roof surfaces of the building.
            walls: List of WallSurface objects representing the wall surfaces of the building.
            creation_date: Creation date according to the CityDPC object.
            function: Function of the building according to the CityDPC object.
            usage: Usage of the building according to the CityDPC object.
            year_of_construction: Year of construction according to the CityDPC object.
            roof_type: Roof type according to the CityDPC object.
            measured_height: Measured height according to the CityDPC object.
            storeys_above_ground: Number of storeys above ground according to the CityDPC object.
            additional_attributes: Optional dictionary of additional attributes for the building.
                This dict is built from the CityDPC object's genericDoubles and genericStrings
                attributes.
            addresses: Optional list of Address objects for the building, built from the CityDPC
                object's address collection.
            parent: Optional parent building if this is a building part.
            children: Optional list of child buildings if this building has building parts.
        """
        # attributes intended for external access
        self.gml_id = gml_id
        self.grounds = grounds
        self.roofs = roofs
        self.walls = walls
        self.epsg = epsg
        self.parent = parent
        self.children = children

        # internal attributes that serve as backing fields for the group-resolved properties (see
        # class docstring and `_resolve_grouped_attribute`); accessed via public properties
        self._creation_date = creation_date
        self._function = function
        self._usage = usage
        self._year_of_construction = year_of_construction
        self._roof_type = roof_type
        self._measured_height = measured_height
        self._storeys_above_ground = storeys_above_ground
        self._additional_attributes = (
            additional_attributes if additional_attributes is not None else {}
        )
        self._addresses = addresses if addresses is not None else []

    def _resolve_grouped_attribute(self, private_attr: str):
        """Resolves an attribute value, letting a building part inherit from its parent.

        Returns the building's own stored value if it is set. Otherwise, if this building is a part,
        it falls back to the parent building's stored value. A parent building never pulls values
        from its parts (that would be ambiguous when parts disagree), so it just returns its own
        value or ``None``.

        Only stored (underscore-prefixed) values are inspected, never the resolving properties, so
        there is no risk of infinite recursion between a parent and its parts.

        Args:
            private_attr: Name of the backing attribute to resolve, e.g. ``"_creation_date"``.

        Returns:
            The building's own value if set, otherwise the parent's value for a building part, or
            ``None`` if neither is available.
        """
        own = getattr(self, private_attr)
        if own is not None:
            return own
        # a building part inherits the value from its parent building
        if self.parent is not None:
            return getattr(self.parent, private_attr)
        return None

    @property
    def creation_date(self) -> Optional[str]:
        """Returns the creation date, falling back to the parent if possible."""
        return self._resolve_grouped_attribute("_creation_date")

    @property
    def function(self) -> Optional[str]:
        """Returns the function, falling back to the parent if possible."""
        return self._resolve_grouped_attribute("_function")

    @property
    def usage(self) -> Optional[str]:
        """Returns the usage, falling back to the parent if possible."""
        return self._resolve_grouped_attribute("_usage")

    @property
    def year_of_construction(self) -> Optional[int]:
        """Returns the year of construction, falling back to the parent if possible."""
        return self._resolve_grouped_attribute("_year_of_construction")

    @property
    def roof_type(self) -> Optional[str]:
        """Returns the roof type, falling back to the parent if possible."""
        return self._resolve_grouped_attribute("_roof_type")

    @property
    def measured_height(self) -> Optional[float]:
        """Returns the measured height, falling back to the parent if possible."""
        return self._resolve_grouped_attribute("_measured_height")

    @property
    def storeys_above_ground(self) -> Optional[int]:
        """Returns the number of storeys above ground, falling back to the parent if possible."""
        return self._resolve_grouped_attribute("_storeys_above_ground")

    @property
    def additional_attributes(self) -> dict:
        """Returns additional attributes, falling back to the parent's if this part has none.

        Like :attr:`addresses`, an empty dict -- rather than ``None`` -- marks "unset": a building
        part with no attributes of its own inherits its parent building's. A parent building only
        ever reports its own attributes. Returns an empty dict if none exist.

        When created via :meth:`from_citydpc_building`, the additional attributes are the union of
        CityDPC's ``genericDoubles`` and ``genericStrings`` key-value mappings stored as one dict.
        """
        if self._additional_attributes:
            return self._additional_attributes
        if self.parent is not None:
            return self.parent._additional_attributes
        return {}

    @property
    def addresses(self) -> List[Address]:
        """Returns the building's addresses, falling back to the parent's if this part has none.

        A building part with no address of its own inherits its parent building's addresses. A
        parent building only ever reports its own addresses. Returns an empty list if none exist.
        """
        if self._addresses:
            return self._addresses
        if self.parent is not None:
            return self.parent._addresses
        return []

    @property
    def address(self) -> Optional[Address]:
        """Returns the primary (first) resolved address, or None if the building has none."""
        resolved = self.addresses
        return resolved[0] if resolved else None

    @property
    def is_building_part(self) -> bool:
        """Checks if the building is part of another building (i.e. has parent).

        Returns:
            True if the building is a building part (i.e., has a parent), False otherwise.
        """
        return self.parent is not None

    @property
    def has_building_parts(self) -> bool:
        """Checks if the building has building parts (i.e. children).

        Returns:
            True if the building has building parts, False otherwise.
        """
        return self.children is not None and len(self.children) > 0

    @property
    def grounds_polygon(self) -> Optional[Polygon]:
        """Get the 2D ground surfaces (z-coordinate removed) merged as a single polygon."""
        if len(self.grounds) == 0:
            return None
        return union_all([Polygon(g.gml_surface_2darray) for g in self.grounds])

    @property
    def roofs_polygon(self) -> Optional[Polygon]:
        """Get the 2D roof surfaces (z-coordinate removed) merged as a single polygon."""
        if len(self.roofs) == 0:
            return None
        return union_all([Polygon(r.gml_surface_2darray) for r in self.roofs])

    @property
    def walls_polygon(self) -> Optional[Polygon]:
        """Get the 2D wall surfaces (z-coordinate removed) merged as a single polygon."""
        if len(self.walls) == 0:
            return None
        return union_all([Polygon(w.gml_surface_2darray) for w in self.walls])

    @property
    def all_roofs_have_valid_tilt_and_orientation(self) -> bool:
        """
        Checks if all roof surfaces have valid tilt and orientation values.

        Returns:
            True if all roof surfaces have valid tilt and orientation, False otherwise.
        """
        return all(Building.validate_roof_surface(roof) for roof in self.roofs)

    @staticmethod
    def clean_tilt(tilt: Optional[float]) -> float:
        """
        Cleans and normalizes the tilt value for a roof surface. Only checks if the tilt is a float
        within the valid range. Does not consider orientation, i.e. does not handle flat roofs.

        Args:
            tilt: The original tilt angle of the roof surface in degrees.

        Returns:
            The cleaned tilt value. Returns NaN if the tilt is not a valid float or not in a valid
            range, otherwise returns the tilt in the range [Building.MINIMUM_TILT,
            Building.MAXIMUM_TILT), i.e. angles > 90° are converted to this range.
        """
        if sanitize_float(tilt) is None or tilt is None:  # if tilt is not a valid float, return NaN
            return np.nan
        # for angles > 90° convert to the valid range for tilt angles
        elif tilt >= 90 and Building.MINIMUM_TILT <= 180 - tilt <= Building.MAXIMUM_TILT:
            return 180 - tilt
        # the tilt is already in the correct range
        elif Building.MINIMUM_TILT <= tilt < Building.MAXIMUM_TILT:
            return tilt
        # if none of the above conditions are met, the tilt is not in the valid range
        return np.nan

    @staticmethod
    def clean_orientation(orientation: Optional[float]) -> float:
        """
        Cleans and normalizes the orientation value for a roof surface. Only checks if the
        orientation is a float within the valid range.
        Does not consider tilt, i.e. does not handle flat roofs.

        Args:
            orientation: The original orientation of the roof surface in degrees.

        Returns:
            The cleaned orientation value. Returns 0 if orientation is 360, otherwise returns the
            orientation if in [0, 360]. Returns NaN if not in a valid range or not a valid float.
        """
        if sanitize_float(orientation) is None or orientation is None:
            return np.nan  # if orientation is not a valid float, return NaN
        elif 0 <= orientation <= 360:
            return orientation if orientation != 360 else 0  # set 360 to 0 degrees
        return np.nan  # if orientation is not in the valid range, return NaN

    @staticmethod
    def validate_tilt(tilt: Optional[float]) -> bool:
        """
        Validates the tilt for a roof surface by checking if the cleaned value is not NaN.

        Args:
            tilt: The tilt angle to validate.

        Returns:
            True if the tilt is valid, False otherwise.
        """
        return not np.isnan(Building.clean_tilt(tilt))

    @staticmethod
    def validate_orientation(orientation: Optional[float]) -> bool:
        """
        Validates the orientation for a roof surface by checking if the cleaned value is not NaN.

        Args:
            orientation: The orientation angle to validate.

        Returns:
            True if the orientation is valid, False otherwise.
        """
        return not np.isnan(Building.clean_orientation(orientation))

    @classmethod
    def validate_roof_surface(cls, roof: RoofSurface) -> bool:
        """
        Validates the roof surface by calling validate_tilt and validate_orientation methods.

        Args:
            roof: The roof surface to validate.

        Returns:
            True if both tilt and orientation are valid, False otherwise.
        """
        return cls.validate_tilt(roof.surface_tilt) and cls.validate_orientation(
            roof.surface_orientation
        )

    @classmethod
    def from_citydpc_building(
        cls,
        building: Union[CityDPCBuilding, BuildingPart],
        epsg: int,
        tilt_and_orientation_method: Optional[Callable] = None,
    ) -> "Building":
        """Converts a CityDPC Building object to an object of this Building class.

        Ground, wall and roof surfaces are extracted from the CityDPC Building object. For ground
        and wall surfaces, only the gml_id, geometry and area are extracted. For roof surfaces, the
        gml_id, geometry, area, surface_tilt and surface_orientation are extracted.

        Tilt and orientation values are validated using the validate_tilt and validate_orientation
        methods. If not valid, they are set to NaN.

        Args:
            building: The CityDPC Building or BuildingPart object to convert.
            epsg: The EPSG code for the CRS.
            tilt_and_orientation_method: A method to calculate the tilt and orientation from the 3D
                polygon coordinates. If not passed, the values from the CityDPC Building object are
                used.

        Returns:
            Building: The converted Building object.
        """
        grounds = [
            GroundSurface(k, make_surface_coords_valid(v.gml_surface_2array), float(v.surface_area))
            for k, v in building.grounds.items()
        ]

        walls = [
            WallSurface(k, make_surface_coords_valid(v.gml_surface_2array), float(v.surface_area))
            for k, v in building.walls.items()
        ]

        roofs = []
        for k, v in building.roofs.items():
            # values from citydpc
            surface_area = float(v.surface_area) if v.surface_area is not None else np.nan

            # use the provided tilt_and_orientation_method to calculate tilt and orientation or
            # read them from the CityDPC Building object if not provided
            if tilt_and_orientation_method is not None:
                calculated_values = tilt_and_orientation_method(v.gml_surface_2array)
                tilt = calculated_values[1]
                orientation = calculated_values[0]
            else:
                tilt = v.surface_tilt
                orientation = v.surface_orientation

            # clean values
            cleaned_tilt = float(cls.clean_tilt(tilt))
            cleaned_orientation = float(cls.clean_orientation(orientation))

            # check for flat roofs
            if cleaned_tilt < Building.FLAT_ROOF_TILT_THRESHOLD:
                # set flat roof tilt to 0 but instead of applying a default orientation of 180 like
                # we used to do, use the actual orientation of the longer axis of the polygon
                cleaned_tilt = 0
                cleaned_orientation = get_orientation(Polygon(v.gml_surface_2array[:, :2]))

            # create the RoofSurface object
            roof = RoofSurface(
                gml_id=k,
                gml_surface_3darray=make_surface_coords_valid(v.gml_surface_2array),
                surface_area=surface_area,
                surface_tilt=cleaned_tilt,
                surface_orientation=cleaned_orientation,
                surface_tilt_original=tilt,
                surface_orientation_original=orientation,
            )
            roofs.append(roof)

        addresses = [
            Address.from_citydpc_address(a) for a in building.addressCollection.get_adresses()
        ]

        return cls(
            gml_id=building.gml_id,
            grounds=grounds,
            roofs=roofs,
            walls=walls,
            epsg=epsg,
            creation_date=building.creationDate,
            function=building.function,
            usage=building.usage,
            year_of_construction=building.yearOfConstruction,
            roof_type=building.roofType,
            measured_height=building.measuredHeight,
            storeys_above_ground=building.storeysAboveGround,
            additional_attributes=building.genericDoubles | building.genericStrings,
            addresses=addresses,
        )

    def get_bbox(self, buffer_size: int) -> Polygon | MultiPolygon | None:
        """
        Returns the bounding box of the building with a given offset.

        Args:
            buffer_size: The size of the buffer to be added to the bounding box in meters.

        Returns:
            The bounding box as a Polygon, or None if no grounds or roofs exist.
        """
        if len(self.grounds) and len(self.roofs) == 0:
            return None
        return envelope(buffer(self.grounds_polygon or self.roofs_polygon, buffer_size))

    def download_orthophoto(
        self,
        image_downloader: "ImageDownloader",
        target_directory: Path | str,
        buffer_size: int = 3,
    ) -> "Image":
        """
        Downloads an orthophoto for the building using the provided ImageDownloader from the
        orthophotos-downloader package. Uses the ground surface(s) of the building and the provided
        buffer size to determine the bounding box for the orthophoto. If no ground surface is
        available, the roof surface(s) are used.

        Args:
            image_downloader: An instance of ImageDownloader instantiated with the desired
                WMS-Service used to download the orthophoto.
            target_directory: The directory where the downloaded orthophoto will be saved. Can be a
                Path or str.
            buffer_size: The buffer size in meters to extend the ground surface of the building.
                Default is 3.

        Returns:
            An Image object representing the downloaded orthophoto.

        Raises:
            Exception: If the CRS of the building does not match the CRS of the WMS.
            Exception: If the building is not within the bounds of the WMS.
            Exception: If no ground or roof surface is available for the building.
        """
        try:
            import orthophotos_downloader  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The 'orthophotos-downloader' package is required to use 'download_orthophoto()'. "
                "Install it with: pip install lod2-buildings-downloader[orthophotos]"
            ) from e

        # check if target_directory is a Path or str and convert to Path if necessary
        if isinstance(target_directory, str):
            target_directory = Path(target_directory)
        # check if target_directory exists and create it if not
        if not target_directory.exists():
            target_directory.mkdir(parents=True, exist_ok=True)

        # check if the epsg of the building matches the epsg of the wms
        if f"EPSG:{self.epsg}" != image_downloader.wms.crs:
            msg = (
                f"CRS of the building (EPSG:{self.epsg}) does not match the CRS of the WMS"
                f" ({image_downloader.wms.crs})."
            )
            log.error(msg)
            raise Exception(msg)

        # check if the building has a ground or roof surface
        if not self.grounds_polygon and not self.roofs_polygon:
            log.error(f"No ground or roof surface available for building {self.gml_id}.")
            raise Exception(f"No ground or roof surface available for building {self.gml_id}.")

        # buffer the grounds of roof surface by the buffer size and create a bounding box around it
        bbox = self.get_bbox(buffer_size=buffer_size)

        # check if the building is within the bounds of the WMS
        _xmin, _ymin, _xmax, _ymax = image_downloader.wms.wms.contents[
            image_downloader.wms.layer_name
        ].boundingBoxWGS84

        # if we use boundingBox instead of boundingBoxWGS84 we cannot be sure which CRS the WMS uses
        # so we use boundingBoxWGS84 which is always in WGS84 coordinates and transform it to the
        # EPSG of the building
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg}", always_xy=True)
        _xmin_t, _ymin_t = transformer.transform(_xmin, _ymin)
        _xmax_t, _ymax_t = transformer.transform(_xmax, _ymax)
        wms_bbox = Polygon(
            [(_xmin_t, _ymin_t), (_xmax_t, _ymin_t), (_xmax_t, _ymax_t), (_xmin_t, _ymax_t)]
        )

        if not within(bbox, wms_bbox):
            msg = f"Building {self.gml_id} is not within the bounds of the WMS."
            log.error(msg)
            raise Exception(msg)

        # using the resolution of the wms and the size of the bounding box in meters we calculate
        # the size of the image in pixels
        bbox_width, bbox_height = bbox.bounds[2] - bbox.bounds[0], bbox.bounds[3] - bbox.bounds[1]
        img_width = int(bbox_width / image_downloader.wms.resolution)
        img_height = int(bbox_height / image_downloader.wms.resolution)

        target_image_path = target_directory / f"{self.gml_id}.tif"
        return image_downloader.download_single_image(
            target_image_path, bbox, image_downloader.wms, img_width, img_height, driver="GTiff"
        )


class BuildingsDownloaderBase(ABC):
    """Abstract base class for buildings downloaders.

    Provides common functionality for downloading and processing GML files containing building data.
    """

    # CRS for the building data - must be defined by subclasses as a non-None int
    EPSG: int

    # Bounding box of the available data - must be defined by subclasses as (minx, miny, maxx, maxy)
    bounds: tuple

    # define the default allowed surface types; only these will be extracted from GML
    ALLOWED_SURFACE_TYPES = list(VALID_ALLOWED_SURFACE_TYPES)

    # default timeout in seconds for HTTP requests; individual get()/post() calls may override it
    DOWNLOAD_TIMEOUT = 60

    # enforce that subclasses define EPSG and bounds class attributes
    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "EPSG") or not isinstance(cls.EPSG, int):
            raise TypeError(f"Class '{cls.__name__}' must define an int class attribute 'EPSG'.")
        if not hasattr(cls, "bounds") or not isinstance(cls.bounds, tuple) or len(cls.bounds) != 4:
            raise TypeError(
                f"Class '{cls.__name__}' must define a tuple class attribute 'bounds' "
                "with four elements (minx, miny, maxx, maxy)."
            )

    def __init__(
        self,
        area_of_interest: Polygon | MultiPolygon,
        filter_by_aoi: bool = True,
        tilt_and_orientation_method: Callable = get_orientation_and_tilt,
        suppress_planarity_warnings: bool = True,
    ):
        """Initializes a buildings downloader class.

        Args:
            area_of_interest: The area of interest for downloading building data, represented as a
                shapely Polygon or MultiPolygon. Must be in the correct CRS for the specific
                downloader.
            filter_by_aoi: If True (default), filters the downloaded buildings to only include those
                that intersect with the area of interest. If False, all buildings in the downloaded
                GML files are returned, regardless of their location.
            tilt_and_orientation_method: A method to calculate the tilt and orientation from the
                LoD-2 data. If not passed, the values from the CityDPC Building object are used.
            suppress_planarity_warnings: If True (default), suppresses SurfacePlanarityWarning from
                citydpc. These warnings indicate that a roof surface is not perfectly planar, which
                may affect area and orientation accuracy.
        """

        # check if the area of interest is a valid shapely Polygon or MultiPolygon
        if not isinstance(area_of_interest, (Polygon, MultiPolygon)):
            raise TypeError(
                "area_of_interest must be a instance of shapely.Polygon or shapely.MultiPolygon "
                f"with EPSG:{self.EPSG} coordinates."
            )

        # check if the area of interest intersects with the bounds defined for the downloader
        if not intersects(area_of_interest, box(*self.bounds)):
            raise ValueError(
                f"area_of_interest does not intersect the available data bounds {self.bounds} "
                f"(EPSG:{self.EPSG})."
            )

        self.area_of_interest = area_of_interest
        self.tilt_and_orientation_method = tilt_and_orientation_method
        self.filter_by_aoi = filter_by_aoi
        self.suppress_planarity_warnings = suppress_planarity_warnings

        if suppress_planarity_warnings:
            log.info(
                "SurfacePlanarityWarning is suppressed. Non-planar roof surfaces will not be "
                "reported. To enable these warnings, set suppress_planarity_warnings=False."
            )

    @classmethod
    @abstractmethod
    def _get_gml_file_urls_for_single_polygon(cls, polygon: Polygon) -> List[str]:
        """Retrieves URLs for GML files containing building data for a single polygon.

        This method must be implemented by subclasses as it differs between services.

        Args:
            polygon: The area of interest represented as a shapely Polygon in the correct CRS.

        Returns:
            A list of URLs pointing to GML files that overlap with the polygon.
        """
        pass

    def _get_gml_file_urls_for_area_of_interest(self) -> List[str]:
        """Retrieves unique URLs for GML files containing building data for the area of interest.

        Dispatches to _get_gml_file_urls_for_single_polygon for each polygon in the area of
        interest and deduplicates the results. Can be overridden by subclasses that require
        different deduplication logic (e.g. same file served from multiple download servers).

        Returns:
            A list of unique URLs pointing to GML files that overlap with the area of interest.
        """
        if isinstance(self.area_of_interest, Polygon):
            all_urls = self.__class__._get_gml_file_urls_for_single_polygon(self.area_of_interest)
        else:  # MultiPolygon (type is validated in __init__)
            all_urls = []
            for polygon in self.area_of_interest.geoms:
                all_urls += self.__class__._get_gml_file_urls_for_single_polygon(polygon)
        return list(set(all_urls))

    def _download_gml_file_from_url(self, url: str, target_directory: Path) -> Optional[Path]:
        """Downloads a GML file from the given URL and saves it to the target directory.

        Args:
            url: The URL of the GML file to download.
            target_directory: The directory where the downloaded GML file will be saved.

        Returns:
            The path to the downloaded GML file if successful, otherwise None.
        """
        log.debug(f"Downloading GML file from {url}...")
        gml_file_path = Path(target_directory) / url.split("/")[-1]
        try:
            file_response = get(url, timeout=self.DOWNLOAD_TIMEOUT)
        except RequestException as e:
            log.error(f"Failed to download file from {url}: {e}")
            return None
        if file_response.status_code == 200:
            with open(gml_file_path, "wb") as file:
                file.write(file_response.content)
            log.debug(f"Download from {url} successful!")
            return gml_file_path
        log.error(f"Failed to download file from {url} Status code: {file_response.status_code}")
        return None

    def _filter_buildings_by_area_of_interest(self, buildings: List[Building]) -> List[Building]:
        """Filters buildings based on their intersection with the area of interest.

        Args:
            buildings: A list of Building objects to filter.

        Returns:
            A list of Building objects that intersect with the area of interest or belong to a
            parent-child group where at least one part intersects with the area of interest.
        """
        log.debug(f"Filtering {len(buildings)} buildings by area of interest...")

        filtered_buildings = []  # list that will hold the final result

        # first, find the gml_ids of all buildings which have a geometry intersecting with the AOI
        intersecting_gml_ids = {
            b.gml_id
            for b in buildings
            if (b.grounds_polygon and b.grounds_polygon.intersects(self.area_of_interest))
            or (b.roofs_polygon and b.roofs_polygon.intersects(self.area_of_interest))
            or (b.walls_polygon and b.walls_polygon.intersects(self.area_of_interest))
        }

        # a parent-child group is kept as a whole if the parent itself or any of its parts
        # intersects the AOI; collect the gml_ids of all parents whose group should be kept. This
        # ensures that sibling parts are not dropped and that the retained parent never references
        # parts that were removed from the result.
        kept_parent_gml_ids = {
            b.gml_id
            for b in buildings
            if b.has_building_parts
            and (
                b.gml_id in intersecting_gml_ids
                or any(child.gml_id in intersecting_gml_ids for child in b.children)
            )
        }

        # then, filter buildings that either directly intersect with the AOI or belong to a kept
        # parent-child group (i.e. building parts and their parent)
        for b in buildings:
            # direct intersection with AOI
            if b.gml_id in intersecting_gml_ids:
                filtered_buildings.append(b)

            # parent whose group is kept (any part intersects, even if the parent has no geometry)
            elif b.has_building_parts and b.gml_id in kept_parent_gml_ids:
                filtered_buildings.append(b)

            # building part belonging to a kept group (keeps siblings of an intersecting part too)
            elif b.is_building_part and b.parent.gml_id in kept_parent_gml_ids:
                filtered_buildings.append(b)

        log.debug(
            f"{len(filtered_buildings)} buildings remain after filtering by area of interest."
        )
        return filtered_buildings

    def _extract_buildings_from_gml_file(self, gml_file: Path, **kwargs) -> List[Building]:
        """Extracts building data from a GML file.

        Args:
            gml_file: The path to the GML file from which building data will be extracted.
            **kwargs: Additional keyword arguments passed to
                ``citydpc.core.input.citygmlInput.load_buildings_from_xml_file``.


        Returns:
            A list of Building objects extracted from the GML file.
        """
        # remove any parameters from kwargs that we set ourselves directly; otherwise they would be
        # passed twice to load_buildings_from_xml_file and raise a TypeError
        for param in ("dataset", "filepath", "allowed_surface_types"):
            if param in kwargs:
                log.debug(f"Ignoring '{param}' passed in kwargs; it is set by this method.")
                kwargs.pop(param)

        # verify that the ALLOWED_SURFACE_TYPES class attribute contains at least one allowed value
        if not set(self.__class__.ALLOWED_SURFACE_TYPES) & VALID_ALLOWED_SURFACE_TYPES:
            raise ValueError(
                f"ALLOWED_SURFACE_TYPES must contain one of {VALID_ALLOWED_SURFACE_TYPES}. "
                "Otherwise spatial intersection with the given area of interest won't work."
            )

        log.debug(f"Extracting buildings from GML file {gml_file.name}.")

        # sanitize the GML file if needed)
        _gml_file = self._sanitize_gml_file_if_needed(gml_file)
        ds = Dataset()

        # temporarily suppress citydpc INFO output (e.g. to keep progress logs clean) by
        # attaching a per-call filter rather than mutating the logger's level.
        _silence = _ErrorOnlyFilter()
        citydpc_logger.addFilter(_silence)
        try:
            load_buildings_from_xml_file(
                dataset=ds,
                filepath=_gml_file.as_posix(),
                allowed_surface_types=self.__class__.ALLOWED_SURFACE_TYPES,
                **kwargs,
            )
        finally:
            citydpc_logger.removeFilter(_silence)

        # iterate over all citydpb Building objects and extract relevant data to create our custom
        # Building objects
        buildings = []
        for b in ds.get_building_list():
            # handling of buildings which have building parts (i.e. sub-buildings)
            if b.has_building_parts():
                sub_buildings = b.get_building_parts()
                sub_buildings_to_append = []
                for sub_b in sub_buildings:
                    sub_buildings_to_append.append(
                        Building.from_citydpc_building(
                            sub_b,
                            epsg=self.EPSG,
                            tilt_and_orientation_method=self.tilt_and_orientation_method,
                        )
                    )

                # create a Building object for the parent building
                parent_building = Building.from_citydpc_building(
                    b,
                    epsg=self.EPSG,
                    tilt_and_orientation_method=self.tilt_and_orientation_method,
                )
                parent_building.children = []

                # set parent for building parts and vice versa
                for sub_b in sub_buildings_to_append:
                    sub_b.parent = parent_building
                    parent_building.children.append(sub_b)

                # add the parent building and its building parts to the list of buildings
                buildings.append(parent_building)
                buildings.extend(sub_buildings_to_append)

            # handling of buildings which do not have building parts
            else:
                buildings.append(
                    Building.from_citydpc_building(
                        b,
                        epsg=self.EPSG,
                        tilt_and_orientation_method=self.tilt_and_orientation_method,
                    )
                )

        log.debug(f"Extracted {len(buildings)} buildings from GML file {gml_file.name}.")

        # cleanup the sanitized file if it was created
        if _gml_file != gml_file:
            try:
                _gml_file.unlink()
                log.debug(f"Deleted temporary sanitized GML file {_gml_file}.")
            except Exception as e:
                log.warning(f"Failed to delete temporary sanitized GML file {_gml_file}: {e}")

        if self.filter_by_aoi:
            buildings = self._filter_buildings_by_area_of_interest(buildings)

        return buildings

    def _process_gml_file(self, url: str, tmp_dir: Path) -> Optional[List[Building]]:
        """
        Helper method to download a GML file and extract buildings from it.

        Args:
            url: The URL of the GML file to download.
            tmp_dir: The temporary directory where the GML file will be saved.

        Returns:
            A list of Building objects if successful, otherwise None.
        """
        gml_file = self._download_gml_file_from_url(url, tmp_dir)
        if gml_file:
            return self._extract_buildings_from_gml_file(gml_file)
        return None

    @staticmethod
    def _drop_duplicate_buildings(buildings: List[Building]) -> List[Building]:
        """Drop duplicate buildings by ``gml_id``, keeping the first occurrence of each.

        Overlapping tiles (or overlapping internal tiles aggregated into a single archive) can yield
        the same building more than once. After deduplication, parent/child references among the
        kept buildings are re-linked by ``gml_id`` so that a kept part always points at the kept
        parent object and a kept parent's ``children`` only holds kept parts. This is independent of
        the order the buildings appear in, so it does not rely on a parent being listed before its
        parts.
        """
        kept: Dict[str, Building] = {}
        for b in buildings:
            kept.setdefault(b.gml_id, b)

        # re-link the surviving group so no reference points at a dropped duplicate
        for b in kept.values():
            if b.parent is not None:
                b.parent = kept.get(b.parent.gml_id, b.parent)
            if b.children:
                b.children = [kept.get(child.gml_id, child) for child in b.children]

        dropped = len(buildings) - len(kept)
        if dropped:
            log.info(f"Removed {dropped} duplicate buildings (same gml_id).")
        return list(kept.values())

    @staticmethod
    def _sanitize_citygml(content: str) -> str:
        """Ensures that the content is a valid CityGML document by removing any extraneous content
        before the XML declaration and after the closing CityModel tag.

        The content is modified only if a complete CityGML document can be identified with high
        confidence. Otherwise the original content is returned unchanged.
        """

        start_xml = content.find("<?xml")
        citymodel_start = re.search(r"<(?:\w+:)?CityModel\b", content)
        citymodel_end = re.search(r"</(?:\w+:)?CityModel\s*>", content)

        # don't change anything unless all required markers exist
        if start_xml == -1 or citymodel_start is None or citymodel_end is None:
            return content

        # sanity check ordering
        if not (start_xml < citymodel_start.start() < citymodel_end.start()):
            return content

        sanitized = content[start_xml : citymodel_end.end()]

        # only return modified content if we actually removed something
        return sanitized

    def _sanitize_gml_file_if_needed(self, gml_file: Path) -> Path:
        """Returns a sanitized copy of the file if sanitization is required.

        The original file is never modified.
        """

        # "utf-8-sig" transparently strips a leading UTF-8 BOM if present and behaves like plain
        # "utf-8" otherwise, so a BOM alone never triggers sanitization.
        try:
            content = gml_file.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            log.warning(f"Skipping sanitization of {gml_file.name}: file is not valid UTF-8.")
            return gml_file

        stripped = content.strip()
        sanitized = self._sanitize_citygml(stripped)

        # only sanitize when genuine extraneous wrapper content was removed; a whitespace-only
        # difference means the file already parses fine as-is
        if sanitized == stripped:
            return gml_file

        sanitized_file = gml_file.with_name(
            f"{gml_file.stem}_sanitized_{uuid4().hex[:8]}{gml_file.suffix}"
        )

        log.debug(
            f"Detected malformed CityGML wrapper content in {gml_file.name}. "
            f"Using sanitized copy {sanitized_file.name}."
        )

        sanitized_file.write_text(sanitized, encoding="utf-8")

        return sanitized_file

    def generate_buildings_by_tile(self, keep_gml_files: bool = False) -> Iterator[List[Building]]:
        """Yields the buildings of the area of interest one download tile at a time.

        Uses the same download and extraction path as :meth:`generate_buildings`, but instead of
        accumulating every tile's buildings into one list it yields each tile's buildings as soon
        as that tile has been downloaded and parsed. This lets callers persist buildings
        incrementally, so a failure late in a large area does not discard everything downloaded so
        far.

        Only removes duplicates per tile. Does not remove potential duplicate buildings across
        tiles, since this requires all tiles to be processed first, breaking the whole point of this
        method. Use :meth:`generate_buildings_parallel` or :meth:`generate_buildings` if you want to
        remove duplicates and return a single list of buildings.

        Args:
            keep_gml_files: If True, the downloaded GML files are kept in the current directory. If
                False (default), they are stored in a temporary directory and deleted after
                processing.

        Yields:
            A list of Building objects for each tile that produced at least one building.
        """
        _configure_citydpc()

        gml_urls = self._get_gml_file_urls_for_area_of_interest()
        log.info(f"Found {len(gml_urls)} download URLs (tiles/archives) for the area of interest.")

        if len(gml_urls) == 0:
            log.warning(
                "No GML files found for the area of interest. Make sure that the area_of_interest "
                f"is in EPSG:{self.EPSG} coordinates and within the bounds of the service."
            )
            return

        previous_planarity_check = citydpc_surfacegml.CHECK_IF_SURFACES_ARE_PLANAR
        if self.suppress_planarity_warnings:
            citydpc_surfacegml.CHECK_IF_SURFACES_ARE_PLANAR = False
        try:
            if not keep_gml_files:
                with TemporaryDirectory() as tmp_dir:
                    for url in gml_urls:
                        result = self._process_gml_file(url, Path(tmp_dir))
                        if result:
                            yield self._drop_duplicate_buildings(result)
            else:
                for url in gml_urls:
                    result = self._process_gml_file(url, Path("."))
                    if result:
                        yield self._drop_duplicate_buildings(result)
        finally:
            citydpc_surfacegml.CHECK_IF_SURFACES_ARE_PLANAR = previous_planarity_check

    def generate_buildings_by_tile_parallel(
        self, num_workers: int = 4, keep_gml_files: bool = False
    ) -> Iterator[List[Building]]:
        """Yields buildings one tile at a time while downloading and parsing tiles in parallel.

        Combines the streaming shape of :meth:`generate_buildings_by_tile` with the parallelism of
        :meth:`generate_buildings_parallel`: up to ``num_workers`` tiles are downloaded and parsed
        concurrently, and each tile's buildings are yielded as soon as that tile finishes. Results
        are yielded in completion order. This lets a caller process tiles as soon as they are
        finished instead of waiting for all tiles to finish, which is useful when the area of
        interest is large.

        Only removes duplicates per tile. Does not remove potential duplicate buildings across
        tiles, since this requires all tiles to be processed first, breaking the whole point of this
        method. Use :meth:`generate_buildings_parallel` or :meth:`generate_buildings` if you want to
        remove duplicates and return a single list of buildings.

        Args:
            num_workers: The number of tiles to download and parse concurrently. Default is 4.
            keep_gml_files: If True, the downloaded GML files are kept in the current directory. If
                False (default), they are stored in a temporary directory and deleted when iteration
                completes.

        Yields:
            A list of Building objects for each tile that produced at least one building.
        """
        _configure_citydpc()

        gml_urls = self._get_gml_file_urls_for_area_of_interest()
        log.info(f"Found {len(gml_urls)} download URLs (tiles/archives) for the area of interest.")

        if len(gml_urls) == 0:
            log.warning(
                "No GML files found for the area of interest. Make sure that the area_of_interest "
                f"is in EPSG:{self.EPSG} coordinates and within the bounds of the service."
            )
            return

        previous_planarity_check = citydpc_surfacegml.CHECK_IF_SURFACES_ARE_PLANAR
        if self.suppress_planarity_warnings:
            citydpc_surfacegml.CHECK_IF_SURFACES_ARE_PLANAR = False

        # keep_gml_files reuses the cwd; otherwise a TemporaryDirectory is torn down once the
        # generator is exhausted or closed (the ExecutorContext shuts down first, joining workers).
        tmp_ctx = nullcontext(".") if keep_gml_files else TemporaryDirectory()
        try:
            with tmp_ctx as tmp_dir, ThreadPoolExecutor(max_workers=num_workers) as executor:
                tmp_path = Path(tmp_dir)
                futures = {
                    executor.submit(self._process_gml_file, url, tmp_path): url for url in gml_urls
                }
                total = len(futures)
                # emit at most ~50 INFO progress messages; log every completion on DEBUG
                step = max(1, total // 50)
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    result = future.result()
                    pct = 100 * completed // total
                    log.debug(f"{completed} out of {total} tiles finished ({pct}%).")
                    if completed % step == 0 or completed == total:
                        log.info(f"{completed} out of {total} tiles finished ({pct}%).")
                    if result:
                        yield self._drop_duplicate_buildings(result)
        finally:
            citydpc_surfacegml.CHECK_IF_SURFACES_ARE_PLANAR = previous_planarity_check

    def generate_buildings(self, keep_gml_files: bool = False) -> List[Building]:
        """Downloads GML files for the area of interest, extracts building data, and returns a list
        of Building objects.

        Eager counterpart of :meth:`generate_buildings_by_tile`: it drains that per-tile stream and
        removes duplicates, returning the full result in one list.

        Args:
            keep_gml_files: If True, the downloaded GML files will be kept in the current directory.
                If False, they will be stored in a temporary directory and deleted after processing.
                Default is False.

        Returns:
            A list of Building objects representing the buildings in the area of interest.
        """
        buildings = [b for tile in self.generate_buildings_by_tile(keep_gml_files) for b in tile]
        buildings = self._drop_duplicate_buildings(buildings)
        log.info(f"Finished generation of {len(buildings)} buildings for the area of interest.")
        return buildings

    def generate_buildings_parallel(self, num_workers: int = 4) -> List[Building]:
        """Downloads GML files for the area of interest, extracts building data in parallel, and
        returns a list of Building objects.

        Eager counterpart of :meth:`generate_buildings_by_tile_parallel`: it drains that per-tile
        stream (tiles downloaded and parsed concurrently by up to ``num_workers`` threads) and
        removes duplicates, returning the full result in one list.

        Args:
            num_workers: The number of parallel workers to use for processing. Default is 4.

        Returns:
            A list of Building objects representing the buildings in the area of interest.
        """
        buildings = [
            b for tile in self.generate_buildings_by_tile_parallel(num_workers) for b in tile
        ]
        buildings = self._drop_duplicate_buildings(buildings)
        log.info(f"Finished generation of {len(buildings)} buildings for the area of interest.")
        return buildings
