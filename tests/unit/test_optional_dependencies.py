"""Tests for optional-dependency import guards.

Each optional extra (matplotlib, orthophotos-downloader, …) is imported lazily inside the function
that needs it. These tests verify that a clear ``ImportError`` with an actionable message is raised
when the package is absent.
"""

import builtins
import subprocess
import sys
import textwrap
from re import escape
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lod2_buildings_downloader.core.buildings_downloader import Building, RoofSurface
from lod2_buildings_downloader.utils.helpers import plot_building_wireframe


def _make_building_with_roof(epsg: int = 4326) -> Building:
    """Return a minimal Building with one roof surface."""
    coords_3d = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    roof = RoofSurface(
        gml_id="r1",
        gml_surface_3darray=coords_3d,
        surface_area=1.0,
        surface_tilt=30.0,
        surface_orientation=180.0,
        surface_tilt_original=30.0,
        surface_orientation_original=180.0,
    )
    return Building(gml_id="test_building", epsg=epsg, roofs=[roof])


@pytest.mark.parametrize(
    ("blocked_module", "error_match", "call"),
    [
        pytest.param(
            "matplotlib.pyplot",
            escape("'matplotlib' is required to use 'plot_building_wireframe()'"),
            lambda: plot_building_wireframe(MagicMock()),
            id="matplotlib",
        ),
        pytest.param(
            "orthophotos_downloader",
            escape("'orthophotos-downloader' package is required to use 'download_orthophoto()'"),
            lambda: _make_building_with_roof().download_orthophoto(MagicMock(), "/tmp/test"),
            id="orthophotos_downloader",
        ),
    ],
)
def test_raises_importerror_when_optional_dependency_missing(blocked_module, error_match, call):
    """ImportError with a helpful message is raised when an optional package is not installed."""
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"No module named '{blocked_module}'")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=_blocked_import):
        with pytest.raises(ImportError, match=error_match):
            call()


def test_package_import_requires_citydpc_with_install_instructions():
    """Importing the package without CityDPC gives users an actionable error."""
    script = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "citydpc" or name.startswith("citydpc."):
                raise ModuleNotFoundError("No module named 'citydpc'", name="citydpc")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocked_import

        try:
            import lod2_buildings_downloader
        except ImportError as error:
            assert "CityDPC is required" in str(error)
            assert "pip install" in str(error)
        else:
            raise AssertionError("Expected a helpful CityDPC ImportError on package import")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
