"""Unit tests for state-specific BuildingsDownloaderBase subclasses.

Two kinds of tests live here:

1. ``TestBuildingsDownloaderBaseEnforcement``
   Tests that the ABC (``BuildingsDownloaderBase``) rejects non-compliant subclasses
   at *class-definition* time.  These run once, independently of any concrete
   state downloader.

2. ``StateDownloaderContractTestBase``
   A reusable base class that verifies the contract every concrete downloader
   must fulfil.  Adding tests for a new state only requires:

   a. Create a concrete subclass of ``StateDownloaderContractTestBase``.
   b. Set the ``downloader_class`` class attribute.
   c. Optionally add state-specific test methods.

   The base class name intentionally does *not* start with "Test" so pytest
   skips it during collection and only runs tests in the concrete subclasses.
"""

import logging

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Polygon

from lod2_buildings_downloader.core.buildings_downloader import Building, BuildingsDownloaderBase
from lod2_buildings_downloader.core.state_downloaders import (
    BuildingsDownloaderBB,
    BuildingsDownloaderBE,
    BuildingsDownloaderBW,
    BuildingsDownloaderBY,
    BuildingsDownloaderHB,
    BuildingsDownloaderHE,
    BuildingsDownloaderHH,
    BuildingsDownloaderMV,
    BuildingsDownloaderNI,
    BuildingsDownloaderNW,
    BuildingsDownloaderRP,
    BuildingsDownloaderSH,
    BuildingsDownloaderSL,
    BuildingsDownloaderSN,
    BuildingsDownloaderST,
    BuildingsDownloaderTH,
    get_downloaders_for_aoi,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _polygon_inside(bounds: tuple) -> Polygon:
    """Return a small Polygon centred inside *bounds*."""
    minx, miny, maxx, maxy = bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    delta = min(maxx - minx, maxy - miny) * 0.01
    return Polygon(
        [
            (cx - delta, cy - delta),
            (cx + delta, cy - delta),
            (cx + delta, cy + delta),
            (cx - delta, cy + delta),
        ]
    )


def _polygon_outside(bounds: tuple) -> Polygon:
    """Return a small Polygon clearly outside *bounds* (shifted east)."""
    minx, miny, maxx, maxy = bounds
    offset = (maxx - minx) * 2
    side = (maxy - miny) * 0.01
    return Polygon(
        [
            (maxx + offset, miny),
            (maxx + offset + side, miny),
            (maxx + offset + side, miny + side),
            (maxx + offset, miny + side),
        ]
    )


def _make_downloader_class(**attrs):
    """Dynamically create a minimal BuildingsDownloaderBase subclass with *attrs*.

    Used to test enforcement inside ``__init_subclass__`` without polluting the
    module namespace with broken class definitions.
    """

    def _get_gml_file_urls_for_area_of_interest(self):  # pragma: no cover
        return []

    return type(
        "DynamicDownloader",
        (BuildingsDownloaderBase,),
        {
            "_get_gml_file_urls_for_area_of_interest": _get_gml_file_urls_for_area_of_interest,
            **attrs,
        },
    )


EPSG = 25832


def _standalone(gml_id: str) -> Building:
    """A minimal standalone building (geometry is irrelevant for dedup)."""
    return Building(gml_id=gml_id, epsg=EPSG)


def _group(parent_id: str, *part_ids: str) -> list:
    """Return ``[parent, part, ...]`` wired as one parent/child group, in tile order."""
    parent = Building(gml_id=parent_id, epsg=EPSG, children=[])
    parts = [Building(gml_id=pid, epsg=EPSG, parent=parent) for pid in part_ids]
    parent.children = parts
    return [parent, *parts]


# ---------------------------------------------------------------------------
# 0. Duplicate-building removal
# ---------------------------------------------------------------------------


class TestDropDuplicateBuildings:
    """Verify ``BuildingsDownloaderBase._drop_duplicate_buildings`` deduplicates by ``gml_id``."""

    def test_empty_input_returns_empty_list(self):
        assert BuildingsDownloaderBase._drop_duplicate_buildings([]) == []

    def test_no_duplicates_are_preserved_in_order(self):
        a, b, c = _standalone("A"), _standalone("B"), _standalone("C")
        result = BuildingsDownloaderBase._drop_duplicate_buildings([a, b, c])
        assert result == [a, b, c]

    def test_duplicate_gml_ids_are_removed(self):
        result = BuildingsDownloaderBase._drop_duplicate_buildings(
            [_standalone("A"), _standalone("B"), _standalone("A")]
        )
        assert [b.gml_id for b in result] == ["A", "B"]

    def test_first_occurrence_object_is_kept(self):
        first, second = _standalone("A"), _standalone("A")
        result = BuildingsDownloaderBase._drop_duplicate_buildings([first, second])
        assert result == [first]
        assert result[0] is first

    def test_order_is_preserved_after_removing_duplicates(self):
        a1, b, c, a2 = _standalone("A"), _standalone("B"), _standalone("C"), _standalone("A")
        result = BuildingsDownloaderBase._drop_duplicate_buildings([a1, b, c, a2])
        assert result == [a1, b, c]

    def test_duplicate_group_from_overlapping_tiles_is_dropped_once(self):
        # same parent/child group extracted from two overlapping tiles
        combined = _group("P", "A", "B") + _group("P", "A", "B")
        result = BuildingsDownloaderBase._drop_duplicate_buildings(combined)
        assert [b.gml_id for b in result] == ["P", "A", "B"]

    def test_retained_part_references_a_retained_parent(self):
        combined = _group("P", "A", "B") + _group("P", "A", "B")
        result = BuildingsDownloaderBase._drop_duplicate_buildings(combined)
        parent = next(b for b in result if b.gml_id == "P")
        # each kept part must point at the kept parent object, not a dropped duplicate
        for part in (b for b in result if b.gml_id in {"A", "B"}):
            assert part.parent is parent


# ---------------------------------------------------------------------------
# 1. ABC enforcement tests
# ---------------------------------------------------------------------------


class TestBuildingsDownloaderBaseEnforcement:
    """Verify that ``BuildingsDownloaderBase.__init_subclass__`` rejects
    non-compliant subclass definitions."""

    _valid_attrs = {"EPSG": 25832, "bounds": (0, 0, 1000, 1000)}

    def test_missing_epsg_raises_type_error(self):
        with pytest.raises(TypeError, match="EPSG"):
            _make_downloader_class(bounds=(0, 0, 1000, 1000))

    def test_non_int_epsg_raises_type_error(self):
        with pytest.raises(TypeError, match="EPSG"):
            _make_downloader_class(EPSG="25832", bounds=(0, 0, 1000, 1000))

    def test_missing_bounds_raises_type_error(self):
        with pytest.raises(TypeError, match="bounds"):
            _make_downloader_class(EPSG=25832)

    def test_non_tuple_bounds_raises_type_error(self):
        with pytest.raises(TypeError, match="bounds"):
            _make_downloader_class(EPSG=25832, bounds=[0, 0, 1000, 1000])

    def test_bounds_wrong_length_raises_type_error(self):
        with pytest.raises(TypeError, match="bounds"):
            _make_downloader_class(EPSG=25832, bounds=(0, 0, 1000))

    def test_compliant_subclass_is_accepted(self):
        """A subclass with correct attributes must be defined without error."""
        cls = _make_downloader_class(**self._valid_attrs)
        assert issubclass(cls, BuildingsDownloaderBase)


# ---------------------------------------------------------------------------
# 2. Reusable per-state contract tests
# ---------------------------------------------------------------------------


class StateDownloaderContractTestBase:
    """Contract tests that every concrete ``BuildingsDownloaderBase`` subclass must pass.

    Subclasses **must** define:

    Attributes:
        downloader_class: The concrete ``BuildingsDownloaderBase`` subclass under test.
    """

    downloader_class: type

    # ------------------------------------------------------------------
    # Class-attribute contract
    # ------------------------------------------------------------------

    def test_epsg_attribute_is_int(self):
        assert isinstance(self.downloader_class.EPSG, int)

    def test_bounds_attribute_is_tuple(self):
        assert isinstance(self.downloader_class.bounds, tuple)

    def test_bounds_has_four_elements(self):
        assert len(self.downloader_class.bounds) == 4

    def test_bounds_values_are_ordered(self):
        minx, miny, maxx, maxy = self.downloader_class.bounds
        assert minx < maxx, "bounds minx must be less than maxx"
        assert miny < maxy, "bounds miny must be less than maxy"

    # ------------------------------------------------------------------
    # Successful instantiation
    # ------------------------------------------------------------------

    def test_instantiation_with_polygon_inside_bounds(self):
        polygon = _polygon_inside(self.downloader_class.bounds)
        downloader = self.downloader_class(area_of_interest=polygon)
        assert downloader.area_of_interest == polygon

    def test_instantiation_with_multipolygon_inside_bounds(self):
        poly = _polygon_inside(self.downloader_class.bounds)
        multi = MultiPolygon([poly])
        downloader = self.downloader_class(area_of_interest=multi)
        assert downloader.area_of_interest == multi

    # ------------------------------------------------------------------
    # Rejected instantiation
    # ------------------------------------------------------------------

    def test_raises_value_error_for_polygon_outside_bounds(self):
        polygon = _polygon_outside(self.downloader_class.bounds)
        with pytest.raises(ValueError, match="bounds"):
            self.downloader_class(area_of_interest=polygon)

    def test_raises_type_error_for_non_polygon_area_of_interest(self):
        with pytest.raises(TypeError):
            self.downloader_class(area_of_interest="not a polygon")

    def test_raises_type_error_for_none_area_of_interest(self):
        with pytest.raises(TypeError):
            self.downloader_class(area_of_interest=None)


# ---------------------------------------------------------------------------
# 3. Concrete subclass test suites
# ---------------------------------------------------------------------------


class TestBuildingsDownloaderBB(StateDownloaderContractTestBase):
    downloader_class = BuildingsDownloaderBB


class TestBuildingsDownloaderBE(StateDownloaderContractTestBase):
    downloader_class = BuildingsDownloaderBE


class TestBuildingsDownloaderBW(StateDownloaderContractTestBase):
    downloader_class = BuildingsDownloaderBW


class TestBuildingsDownloaderBY(StateDownloaderContractTestBase):
    downloader_class = BuildingsDownloaderBY

    # 20 km x 20 km square AOI well inside the Bavarian bounds
    _AOI = Polygon([(500000, 5300000), (520000, 5300000), (520000, 5320000), (500000, 5320000)])
    _TOO_LARGE = "Geometrie zu groß oder außerhalb der Grenzen."

    class _FakeResponse:
        def __init__(self, text: str):
            self.text = text

    @classmethod
    def _metalink_xml(cls, *urls: str) -> str:
        files = "".join(f"<file><url>{u}</url></file>" for u in urls)
        return f'<metalink xmlns="urn:ietf:params:xml:ns:metalink">{files}</metalink>'

    def _patch_post(self, monkeypatch, handler):
        monkeypatch.setattr("lod2_buildings_downloader.core.state_downloaders.post", handler)

    def test_single_polygon_accepted_returns_urls_without_splitting(self, monkeypatch):
        calls = []

        def fake_post(url, data, timeout):
            calls.append(data)
            return self._FakeResponse(self._metalink_xml("http://x/a.gml"))

        self._patch_post(monkeypatch, fake_post)
        urls = BuildingsDownloaderBY._get_gml_file_urls_for_single_polygon(self._AOI)
        assert urls == ["http://x/a.gml"]
        assert len(calls) == 1, "expected no splitting when the whole area is accepted"

    def test_area_too_large_splits_and_collects_tile_urls(self, monkeypatch):
        state = {"whole_rejected": False}

        def fake_post(url, data, timeout):
            if not state["whole_rejected"]:
                state["whole_rejected"] = True
                return self._FakeResponse(self._TOO_LARGE)
            return self._FakeResponse(self._metalink_xml("http://x/tile.gml"))

        self._patch_post(monkeypatch, fake_post)
        urls = BuildingsDownloaderBY._get_gml_file_urls_for_single_polygon(self._AOI)
        assert urls == ["http://x/tile.gml", "http://x/tile.gml"]

    def test_area_too_large_advances_to_finer_split_level(self, monkeypatch):
        from shapely import wkt

        # reject any query whose geometry is larger than a single 2x2 (level-4) tile,
        # forcing the whole polygon and the two level-2 tiles to be rejected.
        threshold = self._AOI.area / 4 * 1.5

        def fake_post(url, data, timeout):
            geom = wkt.loads(data.split(";", 1)[1])
            if geom.area > threshold:
                return self._FakeResponse(self._TOO_LARGE)
            return self._FakeResponse(self._metalink_xml(f"http://x/{geom.area:.0f}.gml"))

        self._patch_post(monkeypatch, fake_post)
        urls = BuildingsDownloaderBY._get_gml_file_urls_for_single_polygon(self._AOI)
        assert len(urls) == 4, "expected the 4-tile split level to succeed"

    def test_area_too_large_at_all_levels_returns_empty(self, monkeypatch):
        self._patch_post(
            monkeypatch, lambda url, data, timeout: self._FakeResponse(self._TOO_LARGE)
        )
        urls = BuildingsDownloaderBY._get_gml_file_urls_for_single_polygon(self._AOI)
        assert urls == []

    def test_split_polygon_into_chunks_returns_requested_count(self):
        assert len(BuildingsDownloaderBY._split_polygon_into_chunks(self._AOI, 2)) == 2
        assert len(BuildingsDownloaderBY._split_polygon_into_chunks(self._AOI, 4)) == 4
        assert len(BuildingsDownloaderBY._split_polygon_into_chunks(self._AOI, 8)) == 8

    def test_split_polygon_into_chunks_clips_to_aoi(self):
        # an L-shaped AOI: the envelope corner cell that lies entirely outside the AOI must not
        # produce a chunk, and the chunks must not extend beyond the AOI.
        l_shape = Polygon(
            [
                (500000, 5300000),
                (520000, 5300000),
                (520000, 5310000),
                (510000, 5310000),
                (510000, 5320000),
                (500000, 5320000),
            ]
        )
        chunks = BuildingsDownloaderBY._split_polygon_into_chunks(l_shape, 4)
        assert len(chunks) == 3, "the corner cell outside the L-shape must be dropped"
        for chunk in chunks:
            assert l_shape.buffer(0).contains(chunk.buffer(-1e-6))


class TestBuildingsDownloaderHB(StateDownloaderContractTestBase):
    downloader_class = BuildingsDownloaderHB

    # a Polygon inside HB bounds
    poly_hb = Polygon([(471742, 5897055), (474102, 5897055), (474102, 5898735), (471742, 5898735)])
    # a Polygon inside BHV bounds
    poly_bhv = Polygon([(471742, 5927219), (474102, 5927219), (474102, 5927319), (471742, 5927319)])
    # a Polygon overlapping both HB and BHV bounds
    poly_all = Polygon([(471742, 5897055), (474102, 5897055), (474102, 5927319), (471742, 5927319)])
    # a Polygon outside both HB and BHV bounds
    poly_out = Polygon([(471742, 5911121), (474102, 5911121), (474102, 5911321), (471742, 5911321)])

    def test_get_gml_file_urls_for_area_of_interest_returns_hb_url_for_polygon_inside_hb(self):
        downloader = self.downloader_class(area_of_interest=self.poly_hb)
        urls = downloader._get_gml_file_urls_for_area_of_interest()
        assert any("HB" in url for url in urls), "Expected at least one HB URL"
        assert self.downloader_class._url_hb in urls, (
            f"Expected HB URL {self.downloader_class._url_hb} not found in {urls}"
        )
        assert self.downloader_class._url_bhv not in urls, (
            f"Did not expect BHV URL {self.downloader_class._url_bhv} in {urls}"
        )

    def test_get_gml_file_urls_for_area_of_interest_returns_bhv_url_for_polygon_inside_bhv(self):
        downloader = self.downloader_class(area_of_interest=self.poly_bhv)
        urls = downloader._get_gml_file_urls_for_area_of_interest()
        assert any("BHV" in url for url in urls), "Expected at least one BHV URL"
        assert self.downloader_class._url_bhv in urls, (
            f"Expected BHV URL {self.downloader_class._url_bhv} not found in {urls}"
        )
        assert self.downloader_class._url_hb not in urls, (
            f"Did not expect HB URL {self.downloader_class._url_hb} in {urls}"
        )

    def test_get_gml_file_urls_for_area_of_interest_returns_no_urls_for_polygon_outside_both(self):
        downloader = self.downloader_class(area_of_interest=self.poly_out)
        urls = downloader._get_gml_file_urls_for_area_of_interest()
        assert len(urls) == 0, f"Expected no URLs, but got: {urls}"

    def test_get_gml_file_urls_for_area_of_interest_returns_both_urls_for_polygon_overlapping_both(
        self,
    ):
        downloader = self.downloader_class(area_of_interest=self.poly_all)
        urls = downloader._get_gml_file_urls_for_area_of_interest()
        assert any("HB" in url for url in urls), "Expected at least one HB URL"
        assert any("BHV" in url for url in urls), "Expected at least one BHV URL"
        assert self.downloader_class._url_hb in urls, (
            f"Expected HB URL {self.downloader_class._url_hb} not found in {urls}"
        )
        assert self.downloader_class._url_bhv in urls, (
            f"Expected BHV URL {self.downloader_class._url_bhv} not found in {urls}"
        )


class TestBuildingsDownloaderHE(StateDownloaderContractTestBase):
    downloader_class = BuildingsDownloaderHE

    def test_normalize_name_strips_spaces_and_punctuation(self):
        assert BuildingsDownloaderHE._normalize_name("Bad Homburg") == "badhomburg"

    def test_normalize_name_expands_ae_umlaut(self):
        assert BuildingsDownloaderHE._normalize_name("Bäder") == "baeder"

    def test_normalize_name_expands_oe_umlaut(self):
        assert BuildingsDownloaderHE._normalize_name("Höhe") == "hoehe"

    def test_normalize_name_expands_ue_umlaut(self):
        assert BuildingsDownloaderHE._normalize_name("Grünberg") == "gruenberg"

    def test_normalize_name_expands_sharp_s(self):
        assert BuildingsDownloaderHE._normalize_name("Straße") == "strasse"

    def test_normalize_name_expands_uppercase_umlauts(self):
        assert BuildingsDownloaderHE._normalize_name("Über") == "ueber"

    def test_normalize_name_is_lowercase(self):
        result = BuildingsDownloaderHE._normalize_name("Frankfurt")
        assert result == result.lower()

    def test_normalize_name_wfs_and_rest_api_names_match(self):
        """The critical invariant: WFS name and REST API name must map to the same key.

        WFS returns e.g. "Bad Homburg v. d. Höhe" (spaces, dots, umlaut).
        REST API filename is e.g. "Bad Homburg v.d. Hoehe" (no spaces around dots, ASCII oe).
        Both must normalize to the same string.
        """
        wfs_name = "Bad Homburg v. d. Höhe"
        rest_name = "Bad Homburg v.d. Hoehe"
        assert BuildingsDownloaderHE._normalize_name(
            wfs_name
        ) == BuildingsDownloaderHE._normalize_name(rest_name)

    def test_normalize_name_returns_only_alphanumeric(self):
        result = BuildingsDownloaderHE._normalize_name("Groß-Gerau / (Test)")
        assert result.isalnum(), f"Expected only alphanumeric chars, got: '{result}'"


class TestBuildingsDownloaderHH(StateDownloaderContractTestBase):
    downloader_class = BuildingsDownloaderHH


class TestBuildingsDownloaderMV(StateDownloaderContractTestBase):
    downloader_class = BuildingsDownloaderMV


class TestBuildingsDownloaderNI(StateDownloaderContractTestBase):
    downloader_class = BuildingsDownloaderNI


class TestBuildingsDownloaderNW(StateDownloaderContractTestBase):
    downloader_class = BuildingsDownloaderNW


class TestBuildingsDownloaderRP(StateDownloaderContractTestBase):
    downloader_class = BuildingsDownloaderRP


class TestBuildingsDownloaderSH(StateDownloaderContractTestBase):
    downloader_class = BuildingsDownloaderSH

    def test_bounds_x_coordinates_are_consistent_with_epsg25832(self):
        """SH tile bounds must use EPSG:25832 coordinates (x > 100_000).

        The SH GeoJSON tile index uses EPSG:25832 natively. This test guards
        against a future CRS drift where the index might accidentally switch to
        geographic coordinates (WGS84, x < 100).
        """
        minx, _miny, maxx, _maxy = BuildingsDownloaderSH.bounds
        assert minx > 100_000, (
            f"SH minx={minx} looks like a geographic coordinate — expected EPSG:25832 (x > 100000)"
        )
        assert maxx > 100_000, (
            f"SH maxx={maxx} looks like a geographic coordinate — expected EPSG:25832 (x > 100000)"
        )


class TestBuildingsDownloaderSL(StateDownloaderContractTestBase):
    downloader_class = BuildingsDownloaderSL

    # One small representative triangle per county, in EPSG:25832.
    # Coordinates are rounded from the saarkreise_testpolys.geojson fixture.
    # Maps county name → expected zip filename prefix → triangle vertices
    _COUNTY_CASES = {
        "Regionalverband Saarbrücken": (
            "SB_",
            [(340622, 5449567), (341391, 5450404), (342261, 5449220)],
        ),
        "Merzig-Wadern": (
            "MZG_",
            [(313371, 5487371), (314515, 5488000), (314446, 5486622)],
        ),
        "Neunkirchen": (
            "NK_",
            [(357951, 5472498), (355768, 5472920), (356630, 5473347)],
        ),
        "Saarlouis": (
            "SLS_",
            [(327311, 5467542), (329550, 5467900), (329161, 5466794)],
        ),
        "Saarpfalz-Kreis": (
            "SPK_",
            [(377257, 5446039), (378075, 5447000), (378744, 5446098)],
        ),
        "St. Wendel": (
            "WND_",
            [(364803, 5489822), (366618, 5490266), (366346, 5488453)],
        ),
    }

    def test_county_url_selection(self):
        """Each county test polygon returns exactly the URL for that county.

        For each entry in _COUNTY_CASES (one small triangle per county, EPSG:25832)
        verifies that _get_gml_file_urls_for_single_polygon returns exactly one URL
        whose filename starts with the expected county prefix.
        """
        for county_name, (expected_prefix, coords) in self._COUNTY_CASES.items():
            polygon = Polygon(coords)
            urls = BuildingsDownloaderSL._get_gml_file_urls_for_single_polygon(polygon)

            assert len(urls) == 1, (
                f"Expected exactly 1 URL for '{county_name}', got {len(urls)}: {urls}"
            )
            filename = urls[0].split("/")[-1]
            assert filename.startswith(expected_prefix), (
                f"Expected URL for '{county_name}' to start with '{expected_prefix}', "
                f"got '{filename}'"
            )


class TestBuildingsDownloaderSN(StateDownloaderContractTestBase):
    downloader_class = BuildingsDownloaderSN


class TestBuildingsDownloaderST(StateDownloaderContractTestBase):
    downloader_class = BuildingsDownloaderST

    def test_coords_to_label_formats_correctly(self):
        """Test that _coords_to_label produces the expected format."""
        label = BuildingsDownloaderST._coords_to_label(606000, 5760000)
        assert label == "326065760"

    def test_coords_to_label_truncates_digits(self):
        """Test that _coords_to_label truncates to 3 and 4 digits."""
        label = BuildingsDownloaderST._coords_to_label(1234567, 9876543)
        assert label == "321239876"

    def test_label_to_coords_parses_correctly(self):
        """Test that _label_to_coords extracts coordinates."""
        easting_km, northing_km = BuildingsDownloaderST._label_to_coords("327165652")
        assert easting_km == 716
        assert northing_km == 5652

    def test_label_to_coords_inverse_of_coords_to_label(self):
        """Test that _label_to_coords is the inverse of _coords_to_label."""
        original_easting_km, original_northing_km = 716, 5652
        easting_m, northing_m = original_easting_km * 1000, original_northing_km * 1000
        label = BuildingsDownloaderST._coords_to_label(easting_m, northing_m)
        parsed_easting_km, parsed_northing_km = BuildingsDownloaderST._label_to_coords(label)
        assert parsed_easting_km == original_easting_km
        assert parsed_northing_km == original_northing_km

    def test_load_label_to_id_mapping_returns_dataframe(self):
        """Test that _load_label_to_id_mapping returns a DataFrame."""
        df = BuildingsDownloaderST._load_label_to_id_mapping()
        assert isinstance(df, pd.DataFrame)
        assert "id" in df.columns
        assert df.index.name == "label"
        assert len(df) == 4487

    def test_load_label_to_id_mapping_is_cached(self):
        """Test that _load_label_to_id_mapping caches results."""
        df1 = BuildingsDownloaderST._load_label_to_id_mapping()
        df2 = BuildingsDownloaderST._load_label_to_id_mapping()
        # should return the exact same object (cached)
        assert df1 is df2

    def test_load_cell_geometries_is_cached(self):
        """Test that _load_cell_geometries caches results."""
        gdf1 = BuildingsDownloaderST._load_cell_geometries()
        gdf2 = BuildingsDownloaderST._load_cell_geometries()
        # should return the exact same object (cached)
        assert gdf1 is gdf2

    def test_load_cell_geometries_returns_geodataframe(self):
        """Test that _load_cell_geometries returns a GeoDataFrame."""
        gdf = BuildingsDownloaderST._load_cell_geometries()

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert "id" in gdf.columns
        assert "geometry" in gdf.columns
        assert gdf.index.name == "label"
        assert len(gdf) == 4487

    def test_load_cell_geometries_has_valid_geometries(self):
        """Test that _load_cell_geometries creates valid Polygon geometries."""
        gdf = BuildingsDownloaderST._load_cell_geometries()

        for geom in gdf.geometry.head(10):
            assert isinstance(geom, Polygon)
            assert geom.is_valid
            assert abs(geom.area - 4_000_000) < 1  # 2000 * 2000

    def test_load_cell_geometries_has_spatial_index(self):
        """Test that _load_cell_geometries builds a spatial index."""
        gdf = BuildingsDownloaderST._load_cell_geometries()
        assert gdf.sindex is not None

    def test_get_intersecting_cell_ids_returns_list_of_strings(self):
        """Test that _get_intersecting_cell_ids returns a list of string IDs."""
        # Create a small polygon in the middle of ST bounds
        # ST bounds: (607190, 5647911, 789276, 5880165)
        test_polygon = Polygon(
            [
                (700000, 5750000),
                (702000, 5750000),
                (702000, 5752000),
                (700000, 5752000),
            ]
        )
        cell_ids = BuildingsDownloaderST._get_intersecting_cell_ids(test_polygon)
        assert isinstance(cell_ids, list)
        assert len(cell_ids) > 0
        for cell_id in cell_ids:
            assert isinstance(cell_id, str)

    def test_get_intersecting_cell_ids_finds_expected_cells(self):
        """Test that _get_intersecting_cell_ids finds cells for a known region."""
        test_polygon = Polygon(
            [
                (666136, 5774232),
                (667870, 5774146),
                (667927, 5773703),
                (666130, 5773732),
            ]
        )
        cell_ids = BuildingsDownloaderST._get_intersecting_cell_ids(test_polygon)
        assert len(cell_ids) == 2
        assert "692444" in cell_ids, f"Expected cell ID '692444' not found in {cell_ids}"
        assert "692482" in cell_ids, f"Expected cell ID '692482' not found in {cell_ids}"

    def test_parse_tile_coords_valid_filename(self):
        """Test that _parse_tile_coords correctly parses valid filenames."""
        coords = BuildingsDownloaderST._parse_tile_coords("LoD2_326065760.gml")
        assert coords == (606, 5760)

    def test_parse_tile_coords_handles_path(self):
        """Test that _parse_tile_coords works with filenames containing paths."""
        coords = BuildingsDownloaderST._parse_tile_coords("LoD21/LoD2_327165652.gml")
        assert coords == (716, 5652)

    def test_parse_tile_coords_invalid_format(self):
        """Test that _parse_tile_coords returns None for invalid filenames."""
        assert BuildingsDownloaderST._parse_tile_coords("invalid.gml") is None
        assert BuildingsDownloaderST._parse_tile_coords("LoD2_123.gml") is None
        assert BuildingsDownloaderST._parse_tile_coords("LoD2_notanumber.gml") is None

    def test_parse_bulk_tile_coords_valid_filename(self):
        """Bulk zip members use the '32_{easting}_{northing}_{raster}_ST' scheme (not the API's
        packed 'LoD2_...' scheme); confirm the easting/northing km are extracted from it."""
        assert BuildingsDownloaderST._parse_bulk_tile_coords("32_658_5704_2_ST.gml") == (658, 5704)

    def test_parse_bulk_tile_coords_invalid_format(self):
        """Names that don't match the bulk scheme must return None so a service naming change is
        caught (logged) rather than silently mis-parsed. The API scheme must not match here."""
        assert BuildingsDownloaderST._parse_bulk_tile_coords("LoD2_326065760.gml") is None
        assert BuildingsDownloaderST._parse_bulk_tile_coords("invalid.gml") is None

    def test_download_via_api_batches_cell_ids(self, monkeypatch, tmp_path):
        """The prepare API is slow/URL-limited per request, so the cell IDs must be split into
        API_BATCH_SIZE-sized batches with one prepared zip per batch. Here 2*size+20 IDs must
        produce three batches sized [size, size, 20]."""
        size = BuildingsDownloaderST.API_BATCH_SIZE
        ids = [str(i) for i in range(2 * size + 20)]
        seen_sizes = []

        # stand in for the real per-batch download; just record how many IDs each batch received
        def fake_batch(cell_ids, tmp_dir, index):
            seen_sizes.append(len(cell_ids))
            return tmp_path / f"batch_{index}.zip"

        monkeypatch.setattr(BuildingsDownloaderST, "_download_api_batch", staticmethod(fake_batch))
        paths = BuildingsDownloaderST._download_via_api(ids, tmp_path)

        assert seen_sizes == [size, size, 20]
        assert len(paths) == 3

    def test_download_via_api_retries_failed_batch_once(self, monkeypatch, tmp_path):
        """A transient batch failure should not lose data: the batch is retried once, and the
        successful retry's zip is kept (so a single batch yields one path after two attempts)."""
        ids = [str(i) for i in range(10)]  # single batch (< API_BATCH_SIZE)
        attempts = []

        # fail the very first attempt for a batch, succeed on the retry
        def fake_batch(cell_ids, tmp_dir, index):
            attempts.append(index)
            if attempts.count(index) == 1:
                return None
            return tmp_path / f"batch_{index}.zip"

        monkeypatch.setattr(BuildingsDownloaderST, "_download_api_batch", staticmethod(fake_batch))
        paths = BuildingsDownloaderST._download_via_api(ids, tmp_path)

        assert attempts == [1, 1]  # exactly one failure followed by one successful retry
        assert len(paths) == 1

    def test_download_via_api_raises_when_batch_fails_after_retry(self, monkeypatch, tmp_path):
        """All-or-nothing: a batch that keeps failing (even after its retry) aborts the whole API
        download with a RuntimeError rather than returning a partial/incomplete result."""
        ids = [str(i) for i in range(BuildingsDownloaderST.API_BATCH_SIZE + 5)]  # 2 batches
        attempts = []

        # batch 1 always fails (both attempts); batch 2 would succeed but must never be reached
        def fake_batch(cell_ids, tmp_dir, index):
            attempts.append(index)
            return None if index == 1 else tmp_path / f"batch_{index}.zip"

        monkeypatch.setattr(BuildingsDownloaderST, "_download_api_batch", staticmethod(fake_batch))
        with pytest.raises(RuntimeError):
            BuildingsDownloaderST._download_via_api(ids, tmp_path)

        assert attempts == [1, 1]  # failed batch retried once, then aborted before batch 2


class TestBuildingsDownloaderTH(StateDownloaderContractTestBase):
    downloader_class = BuildingsDownloaderTH


# ---------------------------------------------------------------------------
# 4. Building-generation orchestration (serial / parallel / streaming)
# ---------------------------------------------------------------------------


class _OrchestrationDownloader(BuildingsDownloaderBase):
    """Minimal concrete downloader used to test the ``generate_buildings*`` orchestration.

    The two network seams (``_get_gml_file_urls_for_area_of_interest`` and ``_process_gml_file``)
    are replaced per test, so nothing is downloaded: only the serial/parallel/streaming iteration,
    flattening, deduplication and planarity-flag handling are exercised. Because these are the exact
    seams every state subclass overrides, passing here means the orchestration keeps honoring those
    overrides.
    """

    EPSG = 25832
    bounds = (0, 0, 1_000_000, 1_000_000)

    @classmethod
    def _get_gml_file_urls_for_single_polygon(cls, polygon):  # pragma: no cover - stubbed per test
        return []


def _make_orchestration_downloader(
    tile_map, url_order=None, *, suppress_planarity=True, on_process=None
):
    """Return an ``_OrchestrationDownloader`` whose seams serve ``tile_map`` (url -> buildings).

    ``url_order`` fixes the URL sequence (defaults to ``tile_map`` insertion order). ``on_process``,
    if given, is called as ``on_process(url)`` inside the stubbed ``_process_gml_file`` to observe
    state (e.g. the planarity flag) at processing time.
    """
    poly = _polygon_inside(_OrchestrationDownloader.bounds)
    downloader = _OrchestrationDownloader(
        area_of_interest=poly, suppress_planarity_warnings=suppress_planarity
    )
    order = list(url_order) if url_order is not None else list(tile_map)
    downloader._get_gml_file_urls_for_area_of_interest = lambda: list(order)

    def _process(url, tmp_dir):
        if on_process is not None:
            on_process(url)
        return tile_map.get(url)

    downloader._process_gml_file = _process
    return downloader


def _ids(buildings):
    return [b.gml_id for b in buildings]


class TestGenerateBuildingsByTile:
    """Streaming, serial: ``generate_buildings_by_tile``."""

    def test_yields_one_list_per_nonempty_tile_in_url_order(self):
        a, b, c = _standalone("A"), _standalone("B"), _standalone("C")
        # u2 -> None and u3 -> [] must both be skipped
        tile_map = {"u1": [a, b], "u2": None, "u3": [], "u4": [c]}
        downloader = _make_orchestration_downloader(tile_map, url_order=["u1", "u2", "u3", "u4"])
        tiles = list(downloader.generate_buildings_by_tile())
        assert [_ids(t) for t in tiles] == [["A", "B"], ["C"]]

    def test_empty_url_list_yields_nothing_and_warns(self, caplog):
        downloader = _make_orchestration_downloader({}, url_order=[])
        with caplog.at_level(logging.WARNING):
            assert list(downloader.generate_buildings_by_tile()) == []
        assert any("No GML files found" in r.message for r in caplog.records)

    def test_keep_gml_files_true_still_yields(self):
        downloader = _make_orchestration_downloader({"u1": [_standalone("A")]})
        tiles = list(downloader.generate_buildings_by_tile(keep_gml_files=True))
        assert [_ids(t) for t in tiles] == [["A"]]

    def test_duplicates_within_a_single_tile_are_removed(self):
        # a single _process_gml_file result can aggregate overlapping internal tiles
        tile_map = {"u1": [_standalone("A"), _standalone("B"), _standalone("A")]}
        downloader = _make_orchestration_downloader(tile_map)
        tiles = list(downloader.generate_buildings_by_tile())
        assert [_ids(t) for t in tiles] == [["A", "B"]]

    def test_duplicate_group_within_a_tile_keeps_parent_child_integrity(self):
        tile_map = {"u1": _group("P", "A", "B") + _group("P", "A", "B")}
        downloader = _make_orchestration_downloader(tile_map)
        [tile] = list(downloader.generate_buildings_by_tile())
        assert _ids(tile) == ["P", "A", "B"]
        parent = next(b for b in tile if b.gml_id == "P")
        for part in (b for b in tile if b.gml_id in {"A", "B"}):
            assert part.parent is parent


class TestGenerateBuildingsByTileParallel:
    """Streaming, parallel: ``generate_buildings_by_tile_parallel``."""

    def test_yields_every_nonempty_tile_exactly_once(self):
        a, b, c = _standalone("A"), _standalone("B"), _standalone("C")
        tile_map = {"u1": [a, b], "u2": None, "u3": [], "u4": [c]}
        downloader = _make_orchestration_downloader(tile_map)
        tiles = list(downloader.generate_buildings_by_tile_parallel(num_workers=4))
        # completion order is nondeterministic; compare as a multiset of tile contents
        got = sorted(tuple(_ids(t)) for t in tiles)
        assert got == sorted([("A", "B"), ("C",)])

    def test_empty_url_list_yields_nothing(self):
        downloader = _make_orchestration_downloader({}, url_order=[])
        assert list(downloader.generate_buildings_by_tile_parallel()) == []

    def test_keep_gml_files_true_still_yields(self):
        downloader = _make_orchestration_downloader({"u1": [_standalone("A")]})
        tiles = list(downloader.generate_buildings_by_tile_parallel(keep_gml_files=True))
        assert [_ids(t) for t in tiles] == [["A"]]

    def test_duplicates_within_a_single_tile_are_removed(self):
        tile_map = {"u1": [_standalone("A"), _standalone("B"), _standalone("A")]}
        downloader = _make_orchestration_downloader(tile_map)
        tiles = list(downloader.generate_buildings_by_tile_parallel())
        assert [_ids(t) for t in tiles] == [["A", "B"]]


class TestGenerateBuildings:
    """Eager, serial: ``generate_buildings`` drains the tile stream and deduplicates."""

    def test_flattens_all_tiles_in_url_order(self):
        tile_map = {"u1": [_standalone("A"), _standalone("B")], "u2": [_standalone("C")]}
        downloader = _make_orchestration_downloader(tile_map, url_order=["u1", "u2"])
        assert _ids(downloader.generate_buildings()) == ["A", "B", "C"]

    def test_deduplicates_across_tiles_keeping_first(self):
        first, dup = _standalone("A"), _standalone("A")
        tile_map = {"u1": [first, _standalone("B")], "u2": [dup, _standalone("C")]}
        downloader = _make_orchestration_downloader(tile_map, url_order=["u1", "u2"])
        result = downloader.generate_buildings()
        assert _ids(result) == ["A", "B", "C"]
        assert result[0] is first  # the first occurrence is the object kept

    def test_empty_url_list_returns_empty(self):
        downloader = _make_orchestration_downloader({}, url_order=[])
        assert downloader.generate_buildings() == []


class TestGenerateBuildingsParallel:
    """Eager, parallel: ``generate_buildings_parallel`` drains the parallel stream and deduplicates."""

    def test_flattens_and_deduplicates(self):
        tile_map = {
            "u1": [_standalone("A"), _standalone("B")],
            "u2": [_standalone("A"), _standalone("C")],
        }
        downloader = _make_orchestration_downloader(tile_map)
        result = downloader.generate_buildings_parallel(num_workers=4)
        # order is nondeterministic under threads; assert dedup happened and membership is correct
        assert sorted(_ids(result)) == ["A", "B", "C"]

    def test_empty_url_list_returns_empty(self):
        downloader = _make_orchestration_downloader({}, url_order=[])
        assert downloader.generate_buildings_parallel() == []


# ---------------------------------------------------------------------------
# get_downloaders_for_aoi
# ---------------------------------------------------------------------------


def _square(cx: float, cy: float, half: float = 0.5) -> Polygon:
    """Return an axis-aligned square Polygon centred at (cx, cy)."""
    return Polygon(
        [
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ]
    )


class TestGetDownloadersForAoi:
    """Unit tests for the AOI -> downloader resolution helper.

    These exercise the real bundled state-boundary GeoJSON. AOIs are built from interior points of
    the respective states (in EPSG:4326). Berlin (state_id 11) is fully enclosed by Brandenburg
    (state_id 12), so a small box over Berlin deterministically hits exactly those two states.
    """

    def test_returns_single_intersecting_downloader(self):
        aoi = _square(12.0461, 48.9186, half=0.05)  # interior of Bavaria
        assert get_downloaders_for_aoi(aoi) == [BuildingsDownloaderBY]

    def test_returns_multiple_ordered_by_id_region(self):
        # a box covering Berlin and its Brandenburg surroundings hits state_id 11 then 12
        aoi = Polygon([(13.0, 52.3), (13.8, 52.3), (13.8, 52.7), (13.0, 52.7)])
        assert get_downloaders_for_aoi(aoi) == [BuildingsDownloaderBE, BuildingsDownloaderBB]

    def test_returns_classes_not_instances(self):
        result = get_downloaders_for_aoi(_square(14.1601, 52.4554, half=0.05))  # Brandenburg
        assert result == [BuildingsDownloaderBB]
        assert all(isinstance(cls, type) for cls in result)

    def test_no_intersection_returns_empty_and_warns(self, caplog):
        aoi = _square(0.0, 0.0, half=0.1)  # valid lon/lat but no German state (Gulf of Guinea)
        with caplog.at_level(logging.WARNING):
            assert get_downloaders_for_aoi(aoi) == []
        assert any("No federal state intersects" in r.message for r in caplog.records)

    def test_multipolygon_is_accepted(self):
        aoi = MultiPolygon(
            [_square(12.0461, 48.9186, half=0.05), _square(14.1601, 52.4554, half=0.05)]
        )
        assert get_downloaders_for_aoi(aoi) == [BuildingsDownloaderBY, BuildingsDownloaderBB]

    @pytest.mark.parametrize("bad_aoi", ["not a polygon", 42, None, (1, 2, 3)])
    def test_non_polygon_raises_type_error(self, bad_aoi):
        with pytest.raises(TypeError):
            get_downloaders_for_aoi(bad_aoi)

    def test_out_of_range_coords_raise_value_error(self):
        # UTM-like coordinates fall well outside valid lon/lat ranges
        utm_aoi = Polygon(
            [(250148, 5690476), (250648, 5690476), (250648, 5690976), (250148, 5690976)]
        )
        with pytest.raises(ValueError):
            get_downloaders_for_aoi(utm_aoi)
