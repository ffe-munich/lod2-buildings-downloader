"""Unit tests for helper functions in ``lod2_buildings_downloader.utils.helpers``."""

import numpy as np
import pytest
from shapely import Polygon

from lod2_buildings_downloader.utils.helpers import make_surface_coords_valid


class TestMakeSurfaceCoordsValid:
    """Tests for ``make_surface_coords_valid`` with 2D and 3D coordinates."""

    def test_valid_3d_returned_unchanged(self):
        coords = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [1.0, 1.0, 5.0], [0.0, 1.0, 5.0]])
        result = make_surface_coords_valid(coords)
        assert result is coords
        np.testing.assert_array_equal(result, coords)

    def test_valid_2d_returned_unchanged(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        result = make_surface_coords_valid(coords)
        assert result is coords
        np.testing.assert_array_equal(result, coords)

    def test_invalid_3d_is_repaired_to_single_valid_ring(self):
        # Self-intersecting "bow-tie" ring with a constant Z of 1.0.
        coords = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        result = make_surface_coords_valid(coords)

        assert result.shape[1] == 3
        assert Polygon(result).is_valid
        # Z is preserved and finite (no spurious NaN column).
        assert np.all(np.isfinite(result))
        np.testing.assert_array_equal(result[:, 2], 1.0)

    def test_invalid_2d_is_repaired_and_stays_2d(self):
        # Same bow-tie in 2D: result must stay 2D and not gain a NaN Z column.
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        result = make_surface_coords_valid(coords)

        assert result.shape[1] == 2
        assert Polygon(result).is_valid
        assert np.all(np.isfinite(result))

    def test_too_few_coordinates_raises(self):
        coords = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.raises(ValueError):
            make_surface_coords_valid(coords)

    def test_single_coordinate_raises(self):
        coords = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError):
            make_surface_coords_valid(coords)

    def test_one_dimensional_input_raises(self):
        coords = np.array([0.0, 1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            make_surface_coords_valid(coords)
