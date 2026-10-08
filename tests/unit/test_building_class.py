"""Unit tests for the Building class, the _SurfaceBase class, and its Address helper.

Focuses on the core mechanics of the Building class:

* ``Address`` emptiness and one-line formatting.
* Parent/part relationship helpers (``is_building_part`` / ``has_building_parts``).
* Group-attribute resolution: a building part inherits missing values from its parent, while a
    parent never pulls values from its parts.
* Address resolution, which follows the same part -> parent rule but treats an empty list (not
    ``None``) as "unset".
* Roof tilt/orientation normalisation and validation.
* Surface geometry: ``_SurfaceBase`` coordinate handling and the merged 2D polygon properties.
"""

import math

import numpy as np
import pytest
from citydpc.core.object.address import CoreAddress
from shapely import MultiPolygon, Polygon

from lod2_buildings_downloader.core.buildings_downloader import (
    Address,
    Building,
    GroundSurface,
    RoofSurface,
    WallSurface,
)

EPSG = 25832


def _core_address(thoroughfare_name=None, thoroughfare_number=None):
    """Builds a citydpc ``CoreAddress`` carrying only the thoroughfare fields under test."""
    address = CoreAddress()
    address.thoroughfareName = thoroughfare_name
    address.thoroughfareNumber = thoroughfare_number
    return address


# A representative attribute per Python type; all group-resolved scalar attributes share
# the same resolution logic, so testing one string/int/float value is sufficient.
GROUP_ATTRS = [
    ("function", "1000", "2000"),
    ("year_of_construction", 1990, 2000),
    ("measured_height", 10.0, 12.5),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _building(gml_id: str = "b", **kwargs) -> Building:
    """Create a minimal Building; surface geometry is irrelevant for these tests."""
    return Building(gml_id=gml_id, epsg=EPSG, **kwargs)


def _square_3d(x0: float = 0.0, y0: float = 0.0, size: float = 1.0, z: float = 10.0) -> np.ndarray:
    """Return the 3D corner coordinates of an axis-aligned square at constant height ``z``."""
    return np.array(
        [
            [x0, y0, z],
            [x0 + size, y0, z],
            [x0 + size, y0 + size, z],
            [x0, y0 + size, z],
        ],
        dtype=float,
    )


def _ground(**kwargs) -> GroundSurface:
    return GroundSurface("g", _square_3d(**kwargs), 1.0)


def _wall(**kwargs) -> WallSurface:
    return WallSurface("w", _square_3d(**kwargs), 1.0)


def _roof(**kwargs) -> RoofSurface:
    return RoofSurface("r", _square_3d(**kwargs), 1.0, 30.0, 180.0, 30.0, 180.0)


# ---------------------------------------------------------------------------
# Address
# ---------------------------------------------------------------------------


class TestAddress:
    def test_is_empty_true_when_all_fields_none(self):
        assert Address().is_empty

    def test_is_empty_false_when_any_field_set(self):
        assert not Address(locality="Aachen").is_empty

    def test_formatted_full_address(self):
        address = Address(
            street="Hauptstraße",
            house_number="42",
            postal_code="52062",
            locality="Aachen",
            country="Germany",
        )
        assert address.formatted == "Hauptstraße 42, 52062 Aachen, Germany"
        assert str(address) == address.formatted

    def test_formatted_omits_missing_parts(self):
        assert Address(street="Hauptstraße", locality="Aachen").formatted == "Hauptstraße, Aachen"

    def test_merged_street_and_house_number_is_split(self):
        address = Address.from_citydpc_address(_core_address("Bavariaring 29"))
        assert address.street == "Bavariaring"
        assert address.house_number == "29"

    def test_merged_street_with_letter_suffix_is_split(self):
        address = Address.from_citydpc_address(_core_address("Hauptstraße 31A"))
        assert address.street == "Hauptstraße"
        assert address.house_number == "31A"

    def test_merged_street_with_number_range_is_split(self):
        address = Address.from_citydpc_address(_core_address("Am Markt 12-14"))
        assert address.street == "Am Markt"
        assert address.house_number == "12-14"

    def test_street_without_number_is_left_untouched(self):
        address = Address.from_citydpc_address(_core_address("Im Körbchen"))
        assert address.street == "Im Körbchen"
        assert address.house_number is None

    def test_existing_house_number_is_not_overwritten(self):
        address = Address.from_citydpc_address(_core_address("Bavariaring 29", "7"))
        assert address.street == "Bavariaring 29"
        assert address.house_number == "7"

    def test_constructor_does_not_split(self):
        address = Address(street="Bavariaring 29")
        assert address.street == "Bavariaring 29"
        assert address.house_number is None


# ---------------------------------------------------------------------------
# Parent / part relationships
# ---------------------------------------------------------------------------


class TestRelationships:
    def test_standalone_building(self):
        building = _building()
        assert not building.is_building_part
        assert not building.has_building_parts

    def test_part_and_parent_flags(self):
        parent = _building("parent")
        part = _building("part", parent=parent)
        parent.children = [part]
        assert part.is_building_part
        assert not part.has_building_parts
        assert parent.has_building_parts
        assert not parent.is_building_part

    def test_has_building_parts_false_for_empty_children(self):
        assert not _building(children=[]).has_building_parts


# ---------------------------------------------------------------------------
# Group-attribute resolution (part -> parent fallback)
# ---------------------------------------------------------------------------


class TestGroupAttributeResolution:
    @pytest.mark.parametrize("attr, own, parent_value", GROUP_ATTRS)
    def test_own_value_takes_precedence(self, attr, own, parent_value):
        parent = _building("parent", **{attr: parent_value})
        part = _building("part", parent=parent, **{attr: own})
        assert getattr(part, attr) == own

    @pytest.mark.parametrize("attr, own, parent_value", GROUP_ATTRS)
    def test_part_inherits_from_parent_when_unset(self, attr, own, parent_value):
        parent = _building("parent", **{attr: parent_value})
        part = _building("part", parent=parent)
        assert getattr(part, attr) == parent_value

    @pytest.mark.parametrize("attr, own, parent_value", GROUP_ATTRS)
    def test_parent_never_pulls_from_parts(self, attr, own, parent_value):
        parent = _building("parent")
        part = _building("part", parent=parent, **{attr: own})
        parent.children = [part]
        assert getattr(parent, attr) is None

    def test_standalone_returns_none_when_unset(self):
        assert _building().function is None

    def test_zero_value_is_not_treated_as_unset(self):
        """A legitimate falsy value (0.0) must be returned, not overridden by fallback."""
        parent = _building("parent", measured_height=99.0)
        part = _building("part", parent=parent, measured_height=0.0)
        assert part.measured_height == 0.0


# ---------------------------------------------------------------------------
# Address resolution (empty list marks "unset")
# ---------------------------------------------------------------------------


class TestAddressResolution:
    def test_standalone_without_addresses(self):
        building = _building()
        assert building.addresses == []
        assert building.address is None

    def test_own_addresses_take_precedence(self):
        own = Address(locality="Aachen")
        parent = _building("parent", addresses=[Address(locality="Berlin")])
        part = _building("part", parent=parent, addresses=[own])
        assert part.addresses == [own]
        assert part.address is own

    def test_part_inherits_parent_addresses_when_empty(self):
        parent_address = Address(locality="Berlin")
        parent = _building("parent", addresses=[parent_address])
        part = _building("part", parent=parent)
        assert part.addresses == [parent_address]
        assert part.address is parent_address

    def test_parent_never_pulls_addresses_from_parts(self):
        parent = _building("parent")
        part = _building("part", parent=parent, addresses=[Address(locality="Aachen")])
        parent.children = [part]
        assert parent.addresses == []
        assert parent.address is None


# ---------------------------------------------------------------------------
# Additional-attributes resolution (empty dict marks "unset")
# ---------------------------------------------------------------------------


class TestAdditionalAttributesResolution:
    def test_standalone_without_attributes(self):
        assert _building().additional_attributes == {}

    def test_own_attributes_take_precedence(self):
        parent = _building("parent", additional_attributes={"b": 2.0})
        part = _building("part", parent=parent, additional_attributes={"a": 1.0})
        assert part.additional_attributes == {"a": 1.0}

    def test_part_inherits_parent_attributes_when_empty(self):
        parent = _building("parent", additional_attributes={"b": 2.0})
        part = _building("part", parent=parent)
        assert part.additional_attributes == {"b": 2.0}

    def test_parent_never_pulls_attributes_from_parts(self):
        parent = _building("parent")
        part = _building("part", parent=parent, additional_attributes={"a": 1.0})
        parent.children = [part]
        assert parent.additional_attributes == {}


# ---------------------------------------------------------------------------
# Roof tilt / orientation normalisation
# ---------------------------------------------------------------------------


class TestRoofNormalization:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            (30.0, 30.0),  # already within [MINIMUM_TILT, MAXIMUM_TILT)
            (170.0, 10.0),  # angle > 90 folded into the valid range
            (75.0, math.nan),  # above MAXIMUM_TILT -> invalid
            (None, math.nan),  # not a valid float -> invalid
        ],
    )
    def test_clean_tilt(self, raw, expected):
        result = Building.clean_tilt(raw)
        assert math.isnan(result) if math.isnan(expected) else result == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (180.0, 180.0),
            (360.0, 0.0),  # 360 normalised to 0
            (400.0, math.nan),  # out of range -> invalid
            (None, math.nan),  # not a valid float -> invalid
        ],
    )
    def test_clean_orientation(self, raw, expected):
        result = Building.clean_orientation(raw)
        assert math.isnan(result) if math.isnan(expected) else result == expected

    def test_validate_reflects_cleaned_values(self):
        assert Building.validate_tilt(30.0)
        assert not Building.validate_tilt(75.0)
        assert Building.validate_orientation(180.0)
        assert not Building.validate_orientation(400.0)


# ---------------------------------------------------------------------------
# _SurfaceBase coordinate handling
# ---------------------------------------------------------------------------


class TestSurfaceBase:
    def test_2darray_drops_z_coordinate(self):
        surface = _ground()
        expected = _square_3d()[:, :2]
        assert np.array_equal(surface.gml_surface_2darray, expected)
        assert surface.gml_surface_2darray.shape == (4, 2)

    def test_2array_is_proxy_for_2darray(self):
        surface = _ground()
        assert np.array_equal(surface.gml_surface_2array, surface.gml_surface_2darray)

    def test_3darray_is_immutable_after_init(self):
        surface = _ground()
        with pytest.raises(ValueError):
            surface.gml_surface_3darray[0, 0] = 99.0


# ---------------------------------------------------------------------------
# Merged 2D polygon properties
# ---------------------------------------------------------------------------


class TestSurfacePolygons:
    @pytest.mark.parametrize("prop", ["grounds_polygon", "roofs_polygon", "walls_polygon"])
    def test_polygon_is_none_without_surfaces(self, prop):
        assert getattr(_building(), prop) is None

    def test_grounds_polygon_matches_surface(self):
        polygon = _building(grounds=[_ground()]).grounds_polygon
        assert polygon.equals(Polygon(_square_3d()[:, :2]))

    def test_roofs_polygon_matches_surface(self):
        polygon = _building(roofs=[_roof()]).roofs_polygon
        assert polygon.equals(Polygon(_square_3d()[:, :2]))

    def test_walls_polygon_matches_surface(self):
        polygon = _building(walls=[_wall()]).walls_polygon
        assert polygon.equals(Polygon(_square_3d()[:, :2]))

    def test_polygon_merges_multiple_surfaces(self):
        """Two disjoint unit squares merge into a MultiPolygon with the combined area."""
        building = _building(grounds=[_ground(), _ground(x0=5.0)])
        assert isinstance(building.grounds_polygon, MultiPolygon)
        assert building.grounds_polygon.area == pytest.approx(2.0)
