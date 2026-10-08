"""Integration tests for building extraction.

Two modes are supported:

Local mode (default)
    The ``buildings`` fixture loads a pre-downloaded GML file from disk via
    ``_extract_buildings_from_gml_file()``.  No network access is required.
    Run with the normal ``pytest`` invocation.

Download mode
    Tests marked ``@pytest.mark.download`` exercise the full
    ``generate_buildings()`` pipeline including network access.  They are
    **skipped by default** and must be opted into explicitly::

        pytest --download

Structure
---------
``BuildingExtractionTestBase`` is a reusable base class.  Adding tests for a new state only
requires:

1. Create a concrete subclass of ``BuildingExtractionTestBase``.
2. Set the required class attributes (see the base-class docstring).
3. Optionally add state-specific tests as extra methods.

The base class name intentionally does not start with "Test" so pytest skips
it during collection and only runs tests in the concrete subclasses.
"""

from pathlib import Path
from typing import override

import numpy as np
import pytest
from shapely import Polygon

from lod2_buildings_downloader.core.buildings_downloader import Building, GroundSurface, RoofSurface
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
)

# ---------------------------------------------------------------------------
# Base test class
# ---------------------------------------------------------------------------


class BuildingExtractionTestBase:
    """Reusable test suite for building extraction.

    By default all tests load buildings from a pre-downloaded local GML file
    (no network required).  When ``--download`` is passed to pytest the full
    suite is run a *second* time against data freshly produced by
    ``generate_buildings()``, so every assertion is exercised against both
    data sources.

    The GML fixture file for each subclass is resolved automatically from
    ``downloader_class.__name__``::

        tests/data/<DownloaderClassName>.gml

    Place a pre-downloaded GML file there and the offline fixtures work without
    any additional configuration.

    Subclasses **must** define the following class attributes:

    Attributes:
        downloader_class: The concrete ``BuildingsDownloaderBase`` subclass.
        sample_area_of_interest: A ``Polygon`` in the correct CRS for this state, used
            to initialise the downloader and filter downloaded buildings.
        expected_building_id: GML ID of the specific building to assert on.
        expected_n_grounds: Expected number of ground surfaces on that building.
        expected_n_roofs: Expected number of roof surfaces on that building.
        expected_ground_area: Expected area (m²) of the first ground surface.
        expected_bbox_bounds: Expected ``get_bbox(0).bounds`` as a 4-tuple.
    """

    downloader_class: type
    sample_area_of_interest: Polygon
    expected_building_id: str
    expected_n_grounds: int
    expected_n_roofs: int
    expected_ground_area: float
    expected_bbox_bounds: tuple

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture(scope="class")
    @classmethod
    def downloader(cls):
        return cls.downloader_class(cls.sample_area_of_interest)

    @classmethod
    def _local_gml_file(cls) -> Path:
        """Resolves the GML fixture path from the downloader class name.

        Convention: ``tests/data/<DownloaderClassName>.gml``
        """
        return Path(__file__).parents[1] / "data" / f"{cls.downloader_class.__name__}.gml"

    @pytest.fixture(scope="class", params=["local", "download"])
    @classmethod
    def buildings(cls, request, downloader):
        """Parametrized fixture that provides buildings from two sources.

        * ``local``    — parsed from a pre-downloaded GML file; always runs.
        * ``download`` — produced by ``generate_buildings()`` (network); only
          runs when ``--download`` is passed to pytest.
        """
        if request.param == "download":
            if not request.config.getoption("--download"):
                pytest.skip("Requires network access; run with --download to enable.")
            return downloader.generate_buildings()
        return downloader._extract_buildings_from_gml_file(
            cls._local_gml_file(), use_multiprocessing=False
        )

    @pytest.fixture(scope="class")
    @classmethod
    def target_building(cls, buildings):
        matches = [b for b in buildings if b.gml_id == cls.expected_building_id]
        assert matches, f"Building {cls.expected_building_id!r} not found in extracted buildings"
        return matches[0]

    # ------------------------------------------------------------------
    # General extraction tests
    # ------------------------------------------------------------------

    def test_any_buildings_exist(self, buildings):
        assert len(buildings) > 0, "No buildings were extracted from the GML file"
        assert all(isinstance(b, Building) for b in buildings)

    def test_target_building_exists(self, buildings):
        ids = [b.gml_id for b in buildings]
        assert self.expected_building_id in ids

    # ------------------------------------------------------------------
    # Target-building surface-type tests
    # ------------------------------------------------------------------

    def test_target_building_grounds_are_ground_surfaces(self, target_building):
        assert all(isinstance(g, GroundSurface) for g in target_building.grounds)

    def test_target_building_roofs_are_roof_surfaces(self, target_building):
        assert all(isinstance(r, RoofSurface) for r in target_building.roofs)

    # ------------------------------------------------------------------
    # Target-building count / geometry tests
    # ------------------------------------------------------------------

    def test_target_building_ground_count(self, target_building):
        assert len(target_building.grounds) == self.expected_n_grounds

    def test_target_building_roof_count(self, target_building):
        assert len(target_building.roofs) == self.expected_n_roofs

    def test_target_building_ground_area(self, target_building):
        assert np.isclose(
            target_building.grounds[0].surface_area, self.expected_ground_area, rtol=0.03
        )

    def test_target_building_all_roofs_have_valid_tilt_and_orientation(self, target_building):
        assert target_building.all_roofs_have_valid_tilt_and_orientation

    def test_target_building_epsg_matches_downloader(self, target_building):
        assert target_building.epsg == self.downloader_class.EPSG

    def test_target_building_bbox_is_not_none(self, target_building):
        assert target_building.get_bbox(0) is not None

    def test_target_building_bbox_bounds(self, target_building):
        bounds = target_building.get_bbox(0).bounds
        assert all(
            np.isclose(actual, expected, atol=0.001)
            for actual, expected in zip(bounds, self.expected_bbox_bounds)
        )


class TestBuildingExtractionBB(BuildingExtractionTestBase):
    downloader_class = BuildingsDownloaderBB
    sample_area_of_interest = Polygon(
        [
            (388393, 5747290),
            (388431, 5746614),
            (388916, 5746560),
            (388881, 5747348),
            (388393, 5747290),
        ]
    )
    expected_building_id = "GUID_6752028119458991664_3"
    expected_n_grounds = 1
    expected_n_roofs = 2
    expected_ground_area = 155.03
    expected_bbox_bounds = (388656.743, 5746992.408, 388671.702, 5747012.347)


class TestBuildingExtractionBE(BuildingExtractionTestBase):
    downloader_class = BuildingsDownloaderBE
    sample_area_of_interest = Polygon(
        [
            (409923, 5807678),
            (409707, 5807437),
            (409831, 5807322),
            (410059, 5807557),
            (409923, 5807678),
        ]
    )
    expected_building_id = "DEBE09YYP0003Zr4"
    expected_n_grounds = 1
    expected_n_roofs = 2
    expected_ground_area = 50.01
    expected_bbox_bounds = (409861.919, 5807588.941, 409870.007, 5807600.594)


class TestBuildingExtractionBW(BuildingExtractionTestBase):
    downloader_class = BuildingsDownloaderBW
    sample_area_of_interest = Polygon([(415600, 5271777), (416162.8, 5271841), (415856.7, 5271508)])
    expected_building_id = "DEBW_001000au6A1"
    expected_n_grounds = 1
    expected_n_roofs = 2
    expected_ground_area = 118.11
    expected_bbox_bounds = (416004.98, 5271685.81, 416018.45, 5271697.38)


class TestBuildingExtractionBY(BuildingExtractionTestBase):
    downloader_class = BuildingsDownloaderBY
    sample_area_of_interest = Polygon(
        [(690000, 5334000), (690100, 5334000), (690100, 5334100), (690000, 5334100)]
    )
    expected_building_id = "DEBY_LOD2_60537"
    expected_n_grounds = 1
    expected_n_roofs = 2
    expected_ground_area = 173.99
    expected_bbox_bounds = (690038.812, 5334071.851, 690063.749, 5334091.679)


class TestBuildingExtractionHB(BuildingExtractionTestBase):
    downloader_class = BuildingsDownloaderHB
    sample_area_of_interest = Polygon(
        [(471742, 5927219), (474102, 5927219), (474102, 5927319), (471742, 5927319)]
    )
    expected_building_id = "DEHB01AL40e0001V"
    expected_n_roofs = 3
    expected_n_grounds = 1
    expected_ground_area = 11.84
    expected_bbox_bounds = (474007.776, 5927300.199, 474012.35, 5927304.912)


class TestBuildingExtractionHE(BuildingExtractionTestBase):
    downloader_class = BuildingsDownloaderHE
    sample_area_of_interest = Polygon([(546511, 5685003), (547101, 5684985), (546763, 5684594)])
    expected_building_id = "DEHE06180000Hmbv"
    expected_n_grounds = 1
    expected_n_roofs = 2
    expected_ground_area = 83.31
    expected_bbox_bounds = (546556.599, 5684960.497, 546569.124, 5684972.942)


class TestBuildingExtractionHH(BuildingExtractionTestBase):
    downloader_class = BuildingsDownloaderHH
    sample_area_of_interest = Polygon([(575731, 5927635), (576579, 5927416), (576124, 5926811)])
    expected_building_id = "DEHHALKA7jr0002B"
    expected_n_grounds = 1
    expected_n_roofs = 2
    expected_ground_area = 20.53
    expected_bbox_bounds = (575909.179, 5927507.337, 575915.679, 5927513.767)

    @pytest.fixture(scope="class", params=["local"])
    @classmethod
    def buildings(cls, request, downloader):
        """Override to skip download mode for HH (too slow due to large state-wide zip)."""
        return downloader._extract_buildings_from_gml_file(
            cls._local_gml_file(), use_multiprocessing=False
        )


class TestBuildingExtractionMV(BuildingExtractionTestBase):
    downloader_class = BuildingsDownloaderMV
    sample_area_of_interest = Polygon(
        [
            (283624, 5933550),
            (283620, 5934069),
            (284143, 5934073),
            (284128, 5933539),
            (283624, 5933550),
        ]
    )
    expected_building_id = "DEMVAL7600edCoBc"
    expected_n_grounds = 1
    expected_n_roofs = 4
    expected_ground_area = 17.70
    expected_bbox_bounds = (283862.111, 5933970.795, 283868.106, 5933975.926)


class TestBuildingExtractionNI(BuildingExtractionTestBase):
    downloader_class = BuildingsDownloaderNI
    sample_area_of_interest = Polygon(
        [(561755, 5713085), (561755, 5713150), (561821, 5713150), (561821, 5713085)]
    )
    expected_building_id = "UUID_1d1f3e57-17aa-4669-8934-d77fc0dc8b77"
    expected_n_grounds = 1
    expected_n_roofs = 2
    expected_ground_area = 57.86
    expected_bbox_bounds = (561793.266, 5713123.357, 561800.236, 5713132.668)


class TestBuildingExtractionNW(BuildingExtractionTestBase):
    downloader_class = BuildingsDownloaderNW
    sample_area_of_interest = Polygon(
        [(429561, 5677176), (430338, 5677137), (430309, 5676595), (429444, 5676617)]
    )
    expected_building_id = "DENW07AL0000qRSm"
    expected_n_grounds = 1
    expected_n_roofs = 1
    expected_ground_area = 107.28
    expected_bbox_bounds = (429932.665, 5676882.328, 429943.265, 5676893.627)


class TestBuildingExtractionRP(BuildingExtractionTestBase):
    downloader_class = BuildingsDownloaderRP
    sample_area_of_interest = Polygon([(391281, 5522942), (391742, 5522773), (391472, 5522636)])
    expected_building_id = "DERPLP020000rJbv"
    expected_n_grounds = 1
    expected_n_roofs = 2
    expected_ground_area = 53.66
    expected_bbox_bounds = (391654.396, 5522763.535, 391664.764, 5522773.866)


class TestBuildingExtractionSH(BuildingExtractionTestBase):
    downloader_class = BuildingsDownloaderSH
    sample_area_of_interest = Polygon([(541118, 6002684), (541817, 6002005), (541000, 6001905)])
    expected_building_id = "DESHPDHK0001rSVD"
    expected_n_grounds = 1
    expected_n_roofs = 3
    expected_ground_area = 141.06
    expected_bbox_bounds = (541333.707, 6002313.84, 541351.06, 6002327.465)


class TestBuildingExtractionSL(BuildingExtractionTestBase):
    downloader_class = BuildingsDownloaderSL
    sample_area_of_interest = Polygon(
        [(311731, 5486163), (311638.5, 5486910), (312552.3, 5486832), (312615.6, 5486166)]
    )
    expected_building_id = "DESLT2884A8438CB"
    expected_n_grounds = 2
    expected_n_roofs = 2
    expected_ground_area = 13.46
    expected_bbox_bounds = (312151.946, 5486856.859, 312157.798, 5486861.865)

    @override
    @pytest.fixture(scope="class")
    @classmethod
    def downloader(cls):
        """Problems with DNS resolution on test runners might cause the SL Nextcloud share token
        retrieval to fail.

        In that case, we hardcode the last known working token to allow the tests to proceed.
        """
        dl = cls.downloader_class(cls.sample_area_of_interest)
        if not dl.__class__.SHARE_TOKEN:
            dl.__class__.SHARE_TOKEN = "NK8ndP55qAqGEZD"
        return dl


class TestBuildingExtractionSN(BuildingExtractionTestBase):
    downloader_class = BuildingsDownloaderSN
    sample_area_of_interest = Polygon([(378951, 5673866), (379444, 5673784), (379177, 5673436)])
    expected_building_id = "DESNATPU1000GzUa"
    expected_n_grounds = 1
    expected_n_roofs = 1
    expected_ground_area = 37.43
    expected_bbox_bounds = (379140.25, 5673762.88, 379149.12, 5673768.77)


class TestBuildingExtractionST(BuildingExtractionTestBase):
    downloader_class = BuildingsDownloaderST
    sample_area_of_interest = Polygon([(665849, 5774635), (667576, 5774300), (666033, 5774030)])
    expected_building_id = "DEST_DESTLIKA0003rfjY"
    expected_n_grounds = 1
    expected_n_roofs = 4
    expected_ground_area = 26.18
    expected_bbox_bounds = (666691.096, 5774163.288, 666696.462, 5774171.111)


class TestBuildingExtractionTH(BuildingExtractionTestBase):
    downloader_class = BuildingsDownloaderTH
    sample_area_of_interest = Polygon([(613076, 5587925), (613275, 5587932), (613109, 5587659)])
    expected_building_id = "DETHL57P0000PTeI"
    expected_n_grounds = 1
    expected_n_roofs = 2
    expected_ground_area = 80.38
    expected_bbox_bounds = (613227.797, 5587899.739, 613240.461, 5587912.083)
