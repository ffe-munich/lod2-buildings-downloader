"""Short demo that can be used to generate reference data for tests.

The local tests use a small sample of a GML file per downloader to test the building extraction.
This script can be used to generate those GML files and the expected values for the tests.

It downloads buildings for a small area of interest and prints out the expected values for one of
the buildings.

Use as follows:
1. Set the desired downloader class in the import section below
2. Get the test area of interest (i.e. Polygon) from the test file
    ``tests/integration/test_building_extraction.py`` for that downloader class
3. Run STEP 1 of this script to download the buildings (GML files are kept on disk).
4. Run STEP 2 of this script to print out the expected values for the chosen building.
5. Choose a building ID from the printed list and set it in the variable ``gml_id`` in STEP 3.
6. Run STEP 3 of this script to print out the expected values for that building.
7. Since this is only a test fixture, we do not want large files, so copy only the part of the GML
    file for that building from the download folder to ``tests/data/BuildingsDownloaderXY.gml``.
    Make sure to keep the XML header and surrounding tags.
8. Set the expected values in
   ``tests/integration/test_building_extraction.py::TestBuildingExtractionXY``.
9. Run the affected tests to verify.
"""

# %% STEP 1
from shapely.geometry import Polygon

from lod2_buildings_downloader.state_downloaders import BuildingsDownloaderHE

# retrieve polygon from the test file tests/integration/test_building_extraction.py
poly = Polygon([(546511, 5685003), (547101, 5684985), (546763, 5684594)])
d = BuildingsDownloaderHE(area_of_interest=poly)
buildings = d.generate_buildings(keep_gml_files=True)

# %% STEP 2
sorted(
    [(b.gml_id, len(b.roofs), len(b.grounds)) for b in buildings], key=lambda r: r[1], reverse=True
)

# %% Print out expected values for a specific building
gml_id = "<insert-gml-id-of-building-to-extract-here>"
building = [b for b in buildings if b.gml_id == gml_id][0]

print(f'expected_building_id = "{building.gml_id}"')
print(f"expected_n_roofs = {len(building.roofs)}")
print(f"expected_n_grounds = {len(building.grounds)}")
print(f"expected_ground_area = {round(building.grounds[0].surface_area, 2)}")
print(f"expected_bbox_bounds = {building.grounds_polygon.bounds}")
