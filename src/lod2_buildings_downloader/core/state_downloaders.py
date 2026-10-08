import hashlib
import io
import json
import logging
import math
import re
import tempfile
import warnings
import xml.etree.ElementTree as ET
from importlib import resources
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, override
from zipfile import BadZipFile, ZipFile

import geopandas as gpd
import pandas as pd
from bs4 import BeautifulSoup
from pyproj import Transformer
from requests import Session, get, post
from requests.exceptions import ConnectionError, RequestException
from shapely import MultiPolygon, Point, Polygon, box, intersects
from shapely.geometry import shape

from lod2_buildings_downloader.core.buildings_downloader import (
    Building,
    BuildingsDownloaderBase,
)
from lod2_buildings_downloader.utils.helpers import get_orientation_and_tilt

log = logging.getLogger(__name__)


class BuildingsDownloaderBB(BuildingsDownloaderBase):
    """Buildings downloader for Brandenburg, Germany.

    Tiles are listed in an HTML directory index page. Each tile is a zip archive
    containing a single CityGML .gml file.

    Tile index:
        https://data.geobasis-bb.de/geobasis/daten/3d_gebaeude/lod2_gml/

    Tile download URL pattern:
        https://data.geobasis-bb.de/geobasis/daten/3d_gebaeude/lod2_gml/lod2_33{x}-{y}.zip

    EPSG: 25833 (ETRS89 / UTM Zone 33N)
    License: © GeoBasis-DE/BB — dl-de/by-2-0
    """

    INDEX_URL: str = "https://data.geobasis-bb.de/geobasis/daten/3d_gebaeude/lod2_gml/"
    CITYGML_SERVER: str = "https://data.geobasis-bb.de/geobasis/daten/3d_gebaeude/lod2_gml"

    EPSG: int = 25833
    bounds: Tuple[int, int, int, int] = (250148, 5690476, 483776, 5935094)

    _available_tiles: Optional[set] = None

    @classmethod
    def _fetch_available_tiles(cls) -> set:
        """Scrape the HTML tile index and return the set of available zip filenames."""
        if cls._available_tiles is not None:
            return cls._available_tiles
        log.debug("Fetching Brandenburg tile index...")
        response = get(cls.INDEX_URL, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        cls._available_tiles = {
            a["href"] for a in soup.find_all("a", href=True) if a["href"].endswith(".zip")
        }
        log.debug(f"Brandenburg tile index loaded: {len(cls._available_tiles)} tiles available.")
        return cls._available_tiles

    @staticmethod
    def _coords_to_tile(x_utm: float, y_utm: float) -> Tuple[int, int]:
        """Convert UTM coordinates (EPSG:25833) to 1x1 km tile indices."""
        return (math.floor(x_utm / 1000), math.floor(y_utm / 1000))

    @classmethod
    @override
    def _get_gml_file_urls_for_single_polygon(cls, polygon: Polygon) -> List[str]:
        """
        Return download URLs for Brandenburg tiles overlapping the given polygon.

        Args:
            polygon: The area of interest in EPSG:25833 coordinates.

        Returns:
            A list of zip file URLs for tiles that overlap the polygon.
        """
        available = cls._fetch_available_tiles()
        minx, miny, maxx, maxy = polygon.bounds
        x1, y1 = cls._coords_to_tile(minx, miny)
        x2, y2 = cls._coords_to_tile(maxx, maxy)

        urls = []
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                if not box(x * 1000, y * 1000, (x + 1) * 1000, (y + 1) * 1000).intersects(polygon):
                    continue
                filename = f"lod2_33{x}-{y}.zip"
                if filename in available:
                    urls.append(f"{cls.CITYGML_SERVER}/{filename}")
                else:
                    log.debug(f"Tile {filename} not in Brandenburg tile index, skipping.")
        log.debug(f"Found {len(urls)} Brandenburg tile URLs for polygon.")
        return urls

    @override
    def _process_gml_file(self, url: str, tmp_dir: Path) -> Optional[List[Building]]:
        """
        Download a Brandenburg tile zip and extract the CityGML file.

        Args:
            url: URL of the tile zip archive.
            tmp_dir: Temporary directory for extracted files.

        Returns:
            A list of Building objects, or None on download/parse failure.
        """
        log.debug(f"Downloading Brandenburg zip from {url}...")
        response = get(url, timeout=self.DOWNLOAD_TIMEOUT)
        if response.status_code != 200:
            log.error(f"Failed to download {url}. Status: {response.status_code}")
            return None

        try:
            z = ZipFile(io.BytesIO(response.content))
        except BadZipFile as e:
            log.error(f"Failed to open zip from {url}: {e}")
            return None

        gml_names = [f for f in z.namelist() if f.endswith(".gml")]
        if not gml_names:
            log.warning(f"No GML file found in {url.split('/')[-1]}.")
            return None

        gml_path = Path(tmp_dir) / Path(gml_names[0]).name
        gml_path.write_bytes(z.read(gml_names[0]))
        return self._extract_buildings_from_gml_file(gml_path)


class BuildingsDownloaderBE(BuildingsDownloaderBase):
    """Buildings downloader for Berlin, Germany.

    Tiles are indexed via an INSPIRE ATOM feed. Each tile is a zip archive containing a single
    CityGML file with a .xml extension (renamed to .gml before parsing).

    ATOM feed:
        https://gdi.berlin.de/data/a_lod2/atom/0.atom

    Tile download URL pattern:
        https://gdi.berlin.de/data/a_lod2/atom/LoD2_{x}_{y}.zip

    EPSG: 25833 (ETRS89 / UTM Zone 33N)
    License: © Geoportal Berlin / SenStadtWohn — dl-de/by-2-0
    """

    ATOM_FEED_URL: str = "https://gdi.berlin.de/data/a_lod2/atom/0.atom"
    CITYGML_SERVER: str = "https://gdi.berlin.de/data/a_lod2/atom"

    EPSG: int = 25833
    bounds: Tuple[int, int, int, int] = (369999, 5799519, 415741, 5837245)

    _available_tiles: Optional[set] = None

    @classmethod
    def _fetch_available_tiles(cls) -> set:
        """Fetch the ATOM feed and return the set of available tile filenames."""
        if cls._available_tiles is not None:
            return cls._available_tiles
        log.debug("Fetching Berlin tile index from ATOM feed...")
        response = get(cls.ATOM_FEED_URL, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        cls._available_tiles = {
            link.attrib["title"]
            for link in root.findall(".//atom:link[@rel='section']", ns)
            if "title" in link.attrib
        }
        log.debug(f"Berlin tile index loaded: {len(cls._available_tiles)} tiles available.")
        return cls._available_tiles

    @staticmethod
    def _coords_to_tile(x_utm: float, y_utm: float) -> Tuple[int, int]:
        """Convert UTM coordinates (EPSG:25833) to 1x1 km tile indices."""
        return (math.floor(x_utm / 1000), math.floor(y_utm / 1000))

    @classmethod
    @override
    def _get_gml_file_urls_for_single_polygon(cls, polygon: Polygon) -> List[str]:
        """
        Return download URLs for Berlin tiles overlapping the given polygon.

        Checks all 1x1 km tile boxes covering the polygon bounding box against the
        ATOM tile index and returns URLs for matching known tiles.

        Args:
            polygon: The area of interest in EPSG:25833 coordinates.

        Returns:
            A list of zip file URLs for tiles that overlap the polygon.
        """
        available = cls._fetch_available_tiles()
        minx, miny, maxx, maxy = polygon.bounds
        x1, y1 = cls._coords_to_tile(minx, miny)
        x2, y2 = cls._coords_to_tile(maxx, maxy)

        urls = []
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                if not box(x * 1000, y * 1000, (x + 1) * 1000, (y + 1) * 1000).intersects(polygon):
                    continue
                filename = f"LoD2_{x}_{y}.zip"
                if filename in available:
                    urls.append(f"{cls.CITYGML_SERVER}/{filename}")
                else:
                    log.debug(f"Tile {filename} not in Berlin tile index, skipping.")
        log.debug(f"Found {len(urls)} Berlin tile URLs for polygon.")
        return urls

    @override
    def _process_gml_file(self, url: str, tmp_dir: Path) -> Optional[List[Building]]:
        """
        Download a Berlin tile zip and extract the CityGML file.

        Berlin tiles contain a single .xml file in CityGML format. The file is
        renamed to .gml so citydpc can recognise it.

        Args:
            url: URL of the tile zip archive.
            tmp_dir: Temporary directory for extracted files.

        Returns:
            A list of Building objects, or None on download/parse failure.
        """
        log.debug(f"Downloading Berlin zip from {url}...")
        response = get(url, timeout=self.DOWNLOAD_TIMEOUT)
        if response.status_code != 200:
            log.error(f"Failed to download {url}. Status: {response.status_code}")
            return None

        try:
            z = ZipFile(io.BytesIO(response.content))
        except BadZipFile as e:
            log.error(f"Failed to open zip from {url}: {e}")
            return None

        xml_names = [f for f in z.namelist() if f.endswith(".xml")]
        if not xml_names:
            log.warning(f"No XML file found in {url.split('/')[-1]}.")
            return None

        gml_path = Path(tmp_dir) / (Path(xml_names[0]).stem + ".gml")
        gml_path.write_bytes(z.read(xml_names[0]))
        return self._extract_buildings_from_gml_file(gml_path)


class BuildingsDownloaderBW(BuildingsDownloaderBase):
    """Buildings downloader for Baden-Württemberg, Germany.

    Downloads and processes LoD-2 building data provided as zipped GML files on a public server.

    The service does not provide an API to query the files based on a polygon, so we need to
    calculate the URLs for the files based on the bounding box of the area of interest and the
    tiling scheme of the data. The files are named like this: "LoD2_32_400_500_2_bw.zip" where the
    numbers indicate the tile coordinates and the tiling scheme is based on a grid with a cell size
    of 1000m x 1000m.
    """

    # URL of the BW LoD2 service providing zipped GML files with building data
    service_url: str = "https://opengeodata.lgl-bw.de/data/lod2"

    # the CRS in which the LoD2 data for this state is provided
    EPSG: int = 25832

    # bounding box of this state in this state's CRS (minx, miny, maxx, maxy)
    bounds: Tuple[int, int, int, int] = (388573, 5265194, 610061, 5515628)

    # the service provides tiles in a raster of 2, so we need to round down the tile coordinates to
    # the nearest multiple of 2 (see _coords_to_tile method)
    raster_size: int = 2

    @classmethod
    def _coords_to_tile(cls, lon: float, lat: float) -> tuple[int, int]:
        """
        Convert WGS84 coordinates to tile indices for the BW LoD2 tiling scheme.

        The tiling scheme is based on a grid with a cell size of 1000m x 1000m. Tile coordinates
        are calculated by dividing the coordinates by 1000 and rounding down to the nearest multiple
        of the raster size (2). The service provides tiles in a raster of 2, so we need to round
        down the tile coordinates to the nearest multiple of 2.
        """
        to_utm = Transformer.from_crs(4326, cls.EPSG, always_xy=True)
        x, y = to_utm.transform(lon, lat)
        x = math.floor(x / 1000) - 1
        y = math.floor(y / 1000)
        x -= x % cls.raster_size
        y -= y % cls.raster_size
        return (x + 1, y)

    @classmethod
    def _get_bounding_tiles(cls, polygon: Polygon) -> tuple[int, int, int, int]:
        """
        Calculate the bounding tile indices for a polygon.

        The area of interest must be in EPSG:25832. The bounding box is transformed to WGS84 to
        determine the tile indices.
        """
        to_wgs84 = Transformer.from_crs(cls.EPSG, 4326, always_xy=True)
        minx, miny, maxx, maxy = polygon.bounds
        min_lon, min_lat = to_wgs84.transform(minx, miny)
        max_lon, max_lat = to_wgs84.transform(maxx, maxy)
        x1, y1 = cls._coords_to_tile(min_lon, min_lat)
        x2, y2 = cls._coords_to_tile(max_lon, max_lat)
        return (x1, x2, y1, y2)

    @classmethod
    @override
    def _get_gml_file_urls_for_single_polygon(cls, polygon: Polygon) -> List[str]:
        """
        Calculate URLs for zipped GML files covering the given polygon.

        The files are named like "LoD2_32_<x>_<y>_2_bw.zip" where x and y are tile coordinates.
        Only tiles whose center is within a certain distance of the polygon are included.
        """
        x1, x2, y1, y2 = cls._get_bounding_tiles(polygon)
        urls = []
        for x in range(x1, x2 + 1, cls.raster_size):
            for y in range(y1, y2 + 1, cls.raster_size):
                tile_center = Point((x + 1) * 1000, (y + 1) * 1000)
                if polygon.distance(tile_center) > cls.raster_size * 1000:
                    continue
                zip_name = f"LoD2_32_{x}_{y}_{cls.raster_size}_bw.zip"
                urls.append(f"{cls.service_url}/{zip_name}")
        log.debug(f"Found {len(urls)} BW tile URLs for polygon.")
        return urls

    @override
    def _process_gml_file(self, url: str, tmp_dir: Path) -> Optional[List[Building]]:
        """
        Download, unzip, and process a zipped GML file from the BW LoD2 server.

        The zip file is expected to contain one or more GML files. Each GML file is extracted and
        processed to extract building data.
        """
        log.debug(f"Downloading BW zip from {url}...")
        response = get(url, timeout=self.DOWNLOAD_TIMEOUT)
        if response.status_code != 200:
            log.error(
                f"Failed to download {url}. Status: {response.status_code}. If other requests "
                f"work, this tile is probably outside the service bounds!"
            )
            return None

        try:
            z = ZipFile(io.BytesIO(response.content))
        except BadZipFile as e:
            log.error(f"Failed to open zip from {url}: {e}")
            return None

        gml_names = [f for f in z.namelist() if f.endswith(".gml")]
        log.debug(f"Extracted {len(gml_names)} GML files from {url.split('/')[-1]}.")

        buildings = []
        for gml_name in gml_names:
            gml_path = Path(tmp_dir) / Path(gml_name).name
            gml_path.write_bytes(z.read(gml_name))
            buildings += self._extract_buildings_from_gml_file(gml_path)
        return buildings


class BuildingsDownloaderBY(BuildingsDownloaderBase):
    """Buildings downloader for Bayern (Bavaria), Germany.

    Uses the Bavarian LoD2 service to retrieve building data.
    """

    # derived from the download functionality on this page:
    # https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=lod2
    service_url: str = "https://geoservices.bayern.de/services/poly2metalink/metalink/lod2"

    # the CRS in which the LoD2 data for this state is provided
    EPSG: int = 25832

    # bounding box of this state in this state's CRS (minx, miny, maxx, maxy)
    bounds: Tuple[int, int, int, int] = (498126, 5235939, 855902, 5601943)

    # exact response body the metalink service returns when the posted geometry exceeds the
    # maximum area it will process (or lies outside its bounds)
    _AREA_TOO_LARGE_RESPONSE: str = "Geometrie zu groß oder außerhalb der Grenzen."

    # numbers of chunks the polygon is split into, tried in order, when the service rejects the
    # area as too large
    _SPLIT_LEVELS: Tuple[int, ...] = (2, 4, 8)

    @classmethod
    def _split_polygon_into_chunks(cls, polygon: Polygon, num_chunks: int) -> List[Polygon]:
        """Split the polygon into ``num_chunks`` chunks clipped to an envelope grid.

        The bounding box is divided into a ``cols`` x ``rows`` grid (with ``cols * rows ==
        num_chunks``) chosen to keep the grid cells as square as possible. Each cell is intersected
        with the polygon and only the non-empty, positive-area intersections are returned, so no
        chunk covers area outside the polygon (degenerate boundary-only touches are dropped).

        Args:
            polygon: The area of interest in EPSG:25832 coordinates.
            num_chunks: The number of grid cells to split the envelope into.

        Returns:
            A list of polygon chunks (the polygon clipped to each grid cell).
        """
        minx, miny, maxx, maxy = polygon.bounds
        width, height = maxx - minx, maxy - miny

        # pick the factorisation cols * rows == num_chunks that yields the most square cells
        best_cols, best_rows, best_aspect = 1, num_chunks, float("inf")
        for cols in range(1, num_chunks + 1):
            if num_chunks % cols != 0:
                continue
            rows = num_chunks // cols
            tile_w, tile_h = width / cols, height / rows
            aspect = max(tile_w, tile_h) / min(tile_w, tile_h)
            if aspect < best_aspect:
                best_cols, best_rows, best_aspect = cols, rows, aspect

        tile_w, tile_h = width / best_cols, height / best_rows
        chunks = []
        for i in range(best_cols):
            for j in range(best_rows):
                cell = box(
                    minx + i * tile_w,
                    miny + j * tile_h,
                    minx + (i + 1) * tile_w,
                    miny + (j + 1) * tile_h,
                )
                chunk = cell.intersection(polygon)
                if not chunk.is_empty and chunk.area > 0:
                    chunks.append(chunk)
        return chunks

    @classmethod
    def _post_metalink(cls, polygon: Polygon) -> Optional[List[str]]:
        """Post a single polygon to the metalink service and return the GML URLs it responds with.

        Args:
            polygon: The geometry to query, in EPSG:25832 coordinates.

        Returns:
            A list of GML URLs, or ``None`` if the service rejected the geometry as too large
            (response body equal to ``_AREA_TOO_LARGE_RESPONSE``).
        """
        post_data = f"SRID={cls.EPSG};{str(polygon)}"
        metalink_response = post(cls.service_url, data=post_data, timeout=cls.DOWNLOAD_TIMEOUT)

        if metalink_response.text.strip() == cls._AREA_TOO_LARGE_RESPONSE:
            return None

        # extract the URLs from the XML response
        namespace = {"ml": "urn:ietf:params:xml:ns:metalink"}
        try:
            root = ET.fromstring(metalink_response.text)
        except ET.ParseError as e:
            log.error(f"Failed to parse XML response from {cls.service_url}: {e}")
            log.error("Response content: %s", metalink_response.text)
            return []
        return [url.text for url in root.findall(".//ml:url", namespace) if url.text is not None]

    @classmethod
    @override
    def _get_gml_file_urls_for_single_polygon(cls, polygon: Polygon) -> List[str]:
        """
        Retrieves URLs for GML files containing building data for a single polygon by posting the
        polygon to the service URL which responds with a list of URLs.

        The service rejects geometries that are too large with the response body
        ``_AREA_TOO_LARGE_RESPONSE``. When that happens, the polygon is split into progressively
        more chunks (2, then 4, then 8) clipped to an envelope grid and each chunk is queried
        individually. The URLs collected at a given split level are only returned if *no* chunk at
        that level was rejected as too large; otherwise the next, finer split level is attempted.

        Args:
            polygon: The area of interest represented as a shapely Polygon in EPSG:25832
                coordinates.

        Returns:
            A list of URLs pointing to GML files that overlap with the polygon. The list might
            contain URLs that point to the same file but from different download servers.
        """
        # first try the whole polygon; if the service accepts it we are done
        urls = cls._post_metalink(polygon)
        if urls is not None:
            return urls

        # the service probably rejected the area as too large: split the polygon into progressively
        # more chunks and query each one
        for num_chunks in cls._SPLIT_LEVELS:
            log.debug(
                f"Service rejected area as too large; retrying with {num_chunks} polygon chunks."
            )
            chunks = cls._split_polygon_into_chunks(polygon, num_chunks)
            collected: List[str] = []
            too_large = False
            for chunk in chunks:
                chunk_urls = cls._post_metalink(chunk)
                if chunk_urls is None:
                    too_large = True
                    break
                collected += chunk_urls
            if not too_large:
                log.debug(
                    f"Collected {len(collected)} URLs after splitting into {num_chunks} chunks."
                )
                return collected

        log.error(
            f"Service still rejects the area as too large after splitting into "
            f"{cls._SPLIT_LEVELS[-1]} chunks. Returning no URLs for this polygon. Try a smaller "
            "area of interest."
        )
        return []

    @override
    def _get_gml_file_urls_for_area_of_interest(self) -> List[str]:
        """
        Retrieves URLs for GML files containing building data for the area of interest by posting
        the area of interest to the service URL which responds with a list of URLs.

        The Bavarian service may return multiple URLs pointing to the same file from different
        download servers (e.g. download1.bayernwolke.de and download2.bayernwolke.de). The base
        class dedup via set() would keep both, so we deduplicate by filename instead.

        Returns:
            A list of unique URLs pointing to GML files that overlap with the area of interest.
        """
        all_urls = super()._get_gml_file_urls_for_area_of_interest()
        unique_urls = {}
        for url in all_urls:
            file_name = url.split("/")[-1]
            if file_name not in unique_urls:
                unique_urls[file_name] = url
        return list(unique_urls.values())


class BuildingsDownloaderHB(BuildingsDownloaderBase):
    """
    Downloads LoD2 building data for the state of Bremen (Bremen + Bremerhaven).

    Both cities are provided as a single zip-within-a-zip download containing 2x2 km CityGML tiles.
    Both zips are downloaded and all tiles overlapping the area of interest are processed.
    """

    # URLs of the two zip files containing the Bremen LoD2 data
    _url_hb: str = "https://gdi2.geo.bremen.de/inspire/download/LoD/data/LOD2_CITYGML_HB.zip"
    _url_bhv: str = "https://gdi2.geo.bremen.de/inspire/download/LoD/data/LOD2_CITYGML_BHV.zip"

    # the CRS in which the LoD2 data for this state is provided
    EPSG: int = 25832

    # bounding box of this state in this state's CRS (minx, miny, maxx, maxy)
    bounds: Tuple[int, int, int, int] = (465587, 5873496, 499380, 5940262)

    # bounding box of Bremen (HB) and Bremerhaven (BHV)
    # these are used to determine which zip files to download based on the area of interest in order
    # to avoid unnecessary downloads
    _bounds_hb: Tuple[int, int, int, int] = (465587, 5873496, 499393, 5897861)
    _bounds_bhv: Tuple[int, int, int, int] = (465587, 5924806, 477490, 5940262)

    @classmethod
    @override
    def _get_gml_file_urls_for_single_polygon(cls, polygon: Polygon) -> List[str]:
        if intersects(polygon, box(*cls._bounds_hb)) and not intersects(
            polygon, box(*cls._bounds_bhv)
        ):
            return [cls._url_hb]
        elif intersects(polygon, box(*cls._bounds_bhv)) and not intersects(
            polygon, box(*cls._bounds_hb)
        ):
            return [cls._url_bhv]
        elif intersects(polygon, box(*cls._bounds_hb)) and intersects(
            polygon, box(*cls._bounds_bhv)
        ):
            return [cls._url_hb, cls._url_bhv]
        else:
            log.warning(
                "Area of interest does not intersect with either Bremen or Bremerhaven bounds."
            )
            return []

    @staticmethod
    def _tile_coords_from_filename(filename: str) -> Optional[Tuple[int, int, int]]:
        """Parse tile coordinates from a GML filename.

        Expects filenames like "LoD2_32_<x>_<y>_<raster>.gml" and returns the tile
        coordinates as a tuple (x, y, raster). Returns None if parsing fails.
        """
        parts = Path(filename).stem.split("_")
        try:
            x = int(parts[2])
            y = int(parts[3])
            raster = int(parts[4])
            return x, y, raster
        except (IndexError, ValueError):
            return None

    @override
    def _process_gml_file(self, url: str, tmp_dir: Path) -> Optional[List[Building]]:
        """Download and process a zip-within-a-zip archive from the Bremen LoD2 server.

        The outer zip contains an inner zip, which in turn contains 2x2 km CityGML tiles. Only tiles
        that intersect the area of interest are processed.
        """
        log.debug(f"Downloading zip file from {url}...")
        response = get(url, timeout=300)
        if response.status_code != 200:
            log.error(f"Failed to download {url}. Status: {response.status_code}")
            return None

        try:
            outer_zip = ZipFile(io.BytesIO(response.content))
        except BadZipFile as e:
            log.error(f"Failed to open outer zip from {url}: {e}")
            return None

        inner_zip_names = [f for f in outer_zip.namelist() if f.endswith(".zip")]
        if not inner_zip_names:
            log.error(f"No inner zip found in {url}.")
            return None

        try:
            inner_zip = ZipFile(io.BytesIO(outer_zip.read(inner_zip_names[0])))
        except BadZipFile as e:
            log.error(f"Failed to open inner zip from {url}: {e}")
            return None

        gml_names = [f for f in inner_zip.namelist() if f.endswith(".gml")]
        log.debug(f"Found {len(gml_names)} GML tiles in {url.split('/')[-1]}.")

        buildings = []
        for gml_name in gml_names:
            coords = self._tile_coords_from_filename(gml_name)
            if coords is not None:
                x, y, raster = coords
                tile_polygon = Polygon(
                    [
                        (x * 1000, y * 1000),
                        ((x + raster) * 1000, y * 1000),
                        ((x + raster) * 1000, (y + raster) * 1000),
                        (x * 1000, (y + raster) * 1000),
                    ]
                )
                if not tile_polygon.intersects(self.area_of_interest):
                    continue

            gml_path = Path(tmp_dir) / Path(gml_name).name
            gml_path.write_bytes(inner_zip.read(gml_name))
            buildings += self._extract_buildings_from_gml_file(gml_path)

        return buildings


class BuildingsDownloaderHE(BuildingsDownloaderBase):
    """Buildings downloader for Hessen (Hesse), Germany.

    Operates on county level. Download URLs for all counties are fetched fresh from the
    gds.hessen.de REST API, which embeds a daily-rotating date token in each URI automatically.
    County boundaries are fetched from a WFS endpoint to spatially restrict downloads to only those
    counties that overlap the AOI.

    Each county zip contains multiple municipality zips which in turn contain the CityGML .gml
    file(s).

    REST API:
        https://gds.hessen.de/INTERSHOP/rest/WFS/HLBG-Geodaten-Site/-/downloadcenter

    WFS endpoint:
        https://basisdienste.geoportal.hessen.de/ogc/borders
        Layer: borders:landkreiseHE_wfs

    EPSG: 25832 (ETRS89 / UTM Zone 32N)
    License: © Hessische Verwaltung für Bodenmanagement und Geoinformation — dl-de/by-2-0
    """

    REST_API_URL: str = (
        "https://gds.hessen.de/INTERSHOP/rest/WFS/HLBG-Geodaten-Site/-/downloadcenter"
    )
    LOD2_REST_PATH: str = "3D-Daten/3D-Gebäudemodelle/3D-Gebäudemodelle LoD2"
    BASE_DOWNLOAD_URL: str = "https://gds.hessen.de"

    WFS_URL: str = "https://basisdienste.geoportal.hessen.de/ogc/borders"
    WFS_COUNTY_LAYER: str = "borders:landkreiseHE_wfs"

    # this dict maps normalized names used in the WFS county geometries to the normalized names used
    # in the REST API download URLs.
    # usually this would be a 1:1 mappping, but due to a single edge case we need a 1:n map here
    # edge case: "Kreisfreie Stadt Hanau" is NOT included in the county geometries, but
    # is covered by the geometry of the "Main-Kinzig-Kreis" county. The REST API on the
    # other hand does provides a distinct download URL for it, so we need to add it
    # manually when the AOI intersects "Main-Kinzig-Kreis".
    _GEOM_TO_URL_INDEX_NAME_MAP: Dict[str, List[str]] = {
        "hochtaunus": ["hochtaunus"],
        "kreisfreiestadtdarmstadt": ["kreisfreiestadtdarmstadt"],
        "kreisfreiestadtfrankfurtammain": ["kreisfreiestadtfrankfurt"],
        "kreisfreiestadtkassel": ["kreisfreiestadtkassel"],
        "kreisfreiestadtoffenbachammain": ["kreisfreiestadtoffenbachammain"],
        "landeshauptstadtwiesbaden": ["kreisfreiestadtwiesbaden"],
        "lahndill": ["lahndillkreis"],
        "bergstrasse": ["landkreisbergstrasse"],
        "darmstadtdieburg": ["landkreisdarmstadtdieburg"],
        "fulda": ["landkreisfulda"],
        "giessen": ["landkreisgiessen"],
        "grossgerau": ["landkreisgrossgerau"],
        "hersfeldrotenburg": ["landkreishersfeldrotenburg"],
        "kassel": ["landkreiskassel"],
        "limburgweilburg": ["landkreislimburgweilburg"],
        "marburgbiedenkopf": ["landkreismarburgbiedenkopf"],
        "offenbach": ["landkreisoffenbach"],
        "waldeckfrankenberg": ["landkreiswaldeckfrankenberg"],
        "mainkinzig": ["mainkinzigkreis", "kreisfreiestadthanau"],
        "maintaunus": ["maintaunuskreis"],
        "odenwaldkreis": ["odenwaldkreis"],
        "rheingautaunus": ["rheingautaunuskreis"],
        "schwalmeder": ["schwalmederkreis"],
        "vogelsberg": ["vogelsbergkreis"],
        "werrameissner": ["werrameissnerkreis"],
        "wetterau": ["wetteraukreis"],
    }

    EPSG: int = 25832
    bounds: Tuple[int, int, int, int] = (412054, 5471323, 586276, 5722999)

    # class-level cache for county geometries: normalized name -> geometry.
    # county boundaries are stable, so caching at class level is safe.
    _county_geometries: Optional[Dict[str, Polygon | MultiPolygon]] = None

    # instance-level cache: URLs contain a daily-rotating date token, so each instance
    # (typically short-lived) fetches its own fresh index. The class attribute is only the
    # unset sentinel — _get_county_url_index() assigns the real value via self.
    _county_url_index: Optional[Dict[str, str]] = None

    # lazily-created per-instance requests Session (see the _session property)
    _http_session: Optional[Session] = None

    @property
    def _session(self) -> Session:
        """Lazily create and cache a per-instance requests Session with a browser User-Agent."""
        if self._http_session is None:
            self._http_session = Session()
            self._http_session.headers.update({"User-Agent": "Mozilla/5.0"})
        return self._http_session

    # ------------------------------------------------------------------
    # Name normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a county name to a canonical lowercase ASCII key for matching.

        Expands German umlauts and ß, converts to lowercase, then removes all characters that are
        not ASCII letters or digits. This allows matching WFS names like "Bad Homburg v. d. Höhe"
        against download filenames like "Bad Homburg v.d. Hoehe" — both normalize to
        "badhomburgvdhoehe".

        Args:
            name: The raw county name string.

        Returns:
            A lowercase alphanumeric-only string.
        """
        replacements = [
            ("ä", "ae"),
            ("ö", "oe"),
            ("ü", "ue"),
            ("ß", "ss"),
            ("Ä", "Ae"),
            ("Ö", "Oe"),
            ("Ü", "Ue"),
        ]
        for src, dst in replacements:
            name = name.replace(src, dst)
        name = name.lower()
        return "".join(ch for ch in name if ch.isascii() and ch.isalnum())

    # ------------------------------------------------------------------
    # REST API — county URL index
    # ------------------------------------------------------------------

    def _build_county_url_index(self) -> Dict[str, str]:
        """Fetch all county download URLs from the gds.hessen.de REST API.

        Queries the navigation tree to discover all 27 county sub-pages (level-4 navigation
        items), then fetches each county page to collect its individual county download URIs.
        The REST API embeds the current date token in each URI, so no separate date probing is
        required.

        Returns:
            A dict mapping normalized county names to their full download URLs.
        """
        log.debug("Fetching Hessen county URL index from REST API...")
        resp = self._session.get(
            self.REST_API_URL,
            params={"path": self.LOD2_REST_PATH, "navigation": "all"},
            timeout=30,
        )
        resp.raise_for_status()

        district_items = {
            self._normalize_name(item["name"]): item["uri"]
            for item in resp.json().get("navigation", [])
            if item.get("level") == 4
        }
        log.debug(f"Found {len(district_items)} county entries in REST API navigation.")

        index: Dict[str, str] = {}
        for norm_name, uri in district_items.items():
            lk_resp = self._session.get(self.BASE_DOWNLOAD_URL + uri, timeout=30)
            lk_resp.raise_for_status()
            downloads = lk_resp.json().get("searchresult", {}).get("packages", [])
            for dl in downloads:
                if not re.match(r"Alle Dateien zu .* herunterladen", dl.get("name", "")):
                    continue
                download_link = dl.get("downloadLink") or {}
                if not download_link.get("uri"):
                    continue
                index[norm_name] = self.BASE_DOWNLOAD_URL + download_link["uri"]

        log.debug(f"County URL index built: {len(index)} entries.")
        return index

    def _get_county_url_index(self) -> Dict[str, str]:
        """Return the cached county URL index, building it if necessary."""
        if self._county_url_index is None:
            self._county_url_index = self._build_county_url_index()
        return self._county_url_index

    # ------------------------------------------------------------------
    # WFS — county geometry fetch
    # ------------------------------------------------------------------

    @classmethod
    def _fetch_county_geometries(cls) -> Dict[str, Polygon | MultiPolygon]:
        """Fetch Hessen county boundaries from the WFS endpoint.

        Returns a dict mapping normalized county names to their geometries in EPSG:25832, or an
        empty dict on failure (caller handles fallback).

        Returns:
            A dict mapping normalized county names to their Shapely geometries.
        """
        log.debug("Fetching Hessen county boundaries from WFS...")
        params = {
            "SERVICE": "WFS",
            "VERSION": "2.0.0",
            "REQUEST": "GetFeature",
            "TYPENAMES": cls.WFS_COUNTY_LAYER,
            "SRSNAME": "EPSG:25832",
            "outputFormat": "application/json; subtype=geojson",
        }
        try:
            resp = get(cls.WFS_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning(f"WFS county fetch failed: {e}. Will fall back to downloading all.")
            return {}

        if "features" not in data:
            log.warning("WFS response missing 'features'. Will fall back to downloading all.")
            return {}

        result: Dict[str, Polygon | MultiPolygon] = {}
        for feat in data["features"]:
            gmde_name = feat["properties"].get("KREIS_BZ")
            if not gmde_name:
                continue
            result[cls._normalize_name(gmde_name)] = shape(feat["geometry"])

        log.debug(f"Loaded {len(result)} county geometries from WFS.")
        return result

    @classmethod
    def _get_county_geometries(cls) -> Dict[str, Polygon | MultiPolygon]:
        """Return the cached county geometries, fetching them if necessary."""
        if cls._county_geometries is None:
            cls._county_geometries = cls._fetch_county_geometries()
        return cls._county_geometries

    # ------------------------------------------------------------------
    # URL and download logic
    # ------------------------------------------------------------------

    @classmethod
    @override
    def _get_gml_file_urls_for_single_polygon(cls, polygon: Polygon) -> List[str]:
        """Not used — BuildingsDownloaderHE overrides _get_gml_file_urls_for_area_of_interest
        directly because URL construction requires instance state (session, URL index).

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "BuildingsDownloaderHE does not support per-polygon URL lookup. Use "
            "generate_buildings(), generate_buildings_by_tile(), generate_buildings_parallel() or "
            "generate_buildings_by_tile_parallel() instead."
        )

    @override
    def _get_gml_file_urls_for_area_of_interest(self) -> List[str]:
        """Return download URLs for all counties whose geometry intersects the AOI.

        If the WFS is unavailable, falls back to returning all county URLs (safe over-download).

        Returns:
            A list of zip file URLs, one per matching county.
        """
        url_index = self._get_county_url_index()
        geometries = self._get_county_geometries()

        if not geometries:
            log.warning(
                "County geometries unavailable — downloading all Hessen counties as fallback."
            )
            return list(url_index.values())

        matching_geom_names = [
            geo_name
            for geo_name, geom in geometries.items()
            if geom.intersects(self.area_of_interest)
        ]
        log.debug(f"AOI intersects {len(matching_geom_names)} counties.")

        urls = []
        for geom_name in matching_geom_names:
            url_names = self._GEOM_TO_URL_INDEX_NAME_MAP.get(geom_name)
            if not url_names:
                log.warning(
                    f"No download URL found for county with normalized name '{geom_name}'. WFS "
                    "name may not match any REST API entry."
                )
                continue
            urls.extend(url for url_name in url_names if (url := url_index.get(url_name)))
        return urls

    @override
    def _process_gml_file(self, url: str, tmp_dir: Path) -> Optional[List[Building]]:
        """Download one Hessen county zip and extract buildings from it.

        Each county zip contains multiple municipality zips which in turn contain the actual .gml
        CityGML file(s).

        Args:
            url: URL of the county zip archive.
            tmp_dir: Temporary directory for the extracted GML file.

        Returns:
            A list of Building objects, or None on download/parse failure.
        """
        log.debug(f"Downloading {url.split('/')[-1]}...")
        r = self._session.get(url, timeout=120)
        if r.status_code != 200:
            log.error(f"Failed to download {url}: HTTP {r.status_code}")
            return None

        try:
            county_zip = ZipFile(io.BytesIO(r.content))
        except BadZipFile:
            log.error(f"Failed to open zip from {url}.")
            return None

        municipality_zip_names = [n for n in county_zip.namelist() if n.endswith(".zip")]
        if not municipality_zip_names:
            log.warning(f"No municipality zip found in {url.split('/')[-1]}.")
            return None

        buildings = []
        for municipality_zip_name in municipality_zip_names:
            try:
                municipality_zip = ZipFile(io.BytesIO(county_zip.read(municipality_zip_name)))
            except BadZipFile:
                log.error(f"Failed to open municipality zip {municipality_zip_name} from {url}.")
                continue

            gml_names = [n for n in municipality_zip.namelist() if n.endswith(".gml")]
            if not gml_names:
                log.warning(f"No GML file found in {municipality_zip_name}.")
                continue

            for gml_name in gml_names:
                gml_path = Path(tmp_dir) / Path(gml_name).name
                gml_path.write_bytes(municipality_zip.read(gml_name))
                buildings += self._extract_buildings_from_gml_file(gml_path)

        return buildings


class BuildingsDownloaderHH(BuildingsDownloaderBase):
    """Buildings downloader for Hamburg, Germany.

    The entire city is distributed as a single zip archive containing tile-based xml files in
    CityGML format.

    Tile filename pattern inside zip:
        LoD2_32_{x}_{y}_1_HH.xml  (CityGML format, .xml extension)

    Download URL:
        https://daten-hamburg.de/opendata/3d_stadtmodell_lod2/LoD2-DE_HH_2025-03-14.zip

    EPSG: 25832 (ETRS89 / UTM Zone 32N)
    License: © Freie und Hansestadt Hamburg, LGV — dl-de/by-2-0
    """

    DOWNLOAD_URL: str = (
        "https://daten-hamburg.de/opendata/3d_stadtmodell_lod2/LoD2-DE_HH_2025-03-14.zip"
    )

    EPSG: int = 25832
    bounds: Tuple[int, int, int, int] = (462228, 5917065, 587991, 5979771)

    @staticmethod
    def _tile_coords_from_filename(filename: str) -> Optional[Tuple[int, int]]:
        """
        Parse (x_km, y_km) tile coordinates from a Hamburg tile filename.

        Expects filenames like "LoD2_32_{x}_{y}_1_HH.xml".

        Args:
            filename: The tile filename (basename only).

        Returns:
            A tuple (x_km, y_km) or None if parsing fails.
        """
        parts = Path(filename).stem.split("_")
        try:
            return int(parts[2]), int(parts[3])
        except (IndexError, ValueError):
            return None

    def _get_relevant_tiles(self, z: ZipFile) -> List[str]:
        """
        Return names of tiles whose 1x1 km bbox intersects the AOI.

        Args:
            z: An open ZipFile of the Hamburg state archive.

        Returns:
            A list of zip-internal filenames for tiles that overlap the AOI.
        """
        relevant = []
        for name in z.namelist():
            if not (name.endswith(".xml") or name.endswith(".gml")):
                continue
            coords = self._tile_coords_from_filename(name)
            if coords is None:
                continue
            x, y = coords
            tile_polygon = Polygon(
                [
                    (x * 1000, y * 1000),
                    ((x + 1) * 1000, y * 1000),
                    ((x + 1) * 1000, (y + 1) * 1000),
                    (x * 1000, (y + 1) * 1000),
                ]
            )
            if tile_polygon.intersects(self.area_of_interest):
                relevant.append(name)
        log.debug(f"Found {len(relevant)} relevant tiles out of {len(z.namelist())} total.")
        return relevant

    @classmethod
    @override
    def _get_gml_file_urls_for_single_polygon(cls, polygon: Polygon) -> List[str]:
        """
        Return the single Hamburg state-wide zip URL.

        AOI filtering is performed in _process_gml_file after the zip is opened.

        Args:
            polygon: Unused — the single state-wide zip URL is always returned.

        Returns:
            A list containing only DOWNLOAD_URL.
        """
        return [cls.DOWNLOAD_URL]

    @override
    def _process_gml_file(self, url: str, tmp_dir: Path) -> Optional[List[Building]]:
        """
        Download the Hamburg zip file to disk and extract all AOI-matching tiles.

        Args:
            url: URL of the Hamburg state-wide zip archive.
            tmp_dir: Temporary directory for the downloaded zip and extracted files.

        Returns:
            A list of Building objects from all matching tiles, or None on failure.
        """
        zip_path = Path(tmp_dir) / "hamburg_lod2.zip"

        log.debug("Downloading Hamburg zip...")
        r = get(url, stream=True, timeout=600)
        if r.status_code != 200:
            log.error(f"Failed to download {url}. Status: {r.status_code}")
            return None
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        log.debug(f"Downloaded {zip_path.stat().st_size / 1e6:.1f} MB.")

        z = ZipFile(zip_path)
        relevant = self._get_relevant_tiles(z)
        if not relevant:
            log.warning("No tiles overlap the area of interest.")
            return []

        buildings = []
        for name in relevant:
            gml_path = Path(tmp_dir) / Path(name).with_suffix(".gml").name
            gml_path.write_bytes(z.read(name))
            buildings += self._extract_buildings_from_gml_file(gml_path)

        return buildings


class BuildingsDownloaderMV(BuildingsDownloaderBase):
    """Buildings downloader for Mecklenburg-Vorpommern, Germany.

    Tiles are indexed via an INSPIRE ATOM feed with WGS84 bounding boxes used for AOI
    filtering. Each tile is a zip archive containing a single CityGML file.

    ATOM service feed:
        https://www.geodaten-mv.de/dienste/gebaeude_atom
    Dataset feed:
        https://www.geodaten-mv.de/dienste/gebaeude_atom?type=dataset&id=<DATASET_ID>

    EPSG: 25833 (ETRS89 / UTM Zone 33N)
    License: © GeoBasis-DE/M-V — dl-de/by-2-0
    """

    ATOM_SERVICE_URL: str = "https://www.geodaten-mv.de/dienste/gebaeude_atom"
    DATASET_ID: str = "8397b554-5cb9-4274-8be8-c20490d9a6e8"

    EPSG: int = 25833
    bounds: Tuple[int, int, int, int] = (206892, 5890618, 460866, 6060876)

    _tile_index: Optional[List[Tuple[str, Polygon]]] = None

    @classmethod
    def _fetch_tile_index(cls) -> List[Tuple[str, Polygon]]:
        """Fetch the dataset ATOM feed and parse all tile URLs and WGS84 bounding boxes."""
        log.debug("Fetching Mecklenburg-Vorpommern tile index from ATOM feed...")
        url = f"{cls.ATOM_SERVICE_URL}?type=dataset&id={cls.DATASET_ID}"
        resp = get(url, timeout=cls.DOWNLOAD_TIMEOUT)
        resp.raise_for_status()

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.content)
        tiles = []

        for entry in root.findall("atom:entry", ns):
            for link in entry.findall("atom:link[@rel='section']", ns):
                href = link.get("href")
                bbox_str = link.get("bbox")  # "minLat minLon maxLat maxLon"
                if not href or not bbox_str:
                    continue
                try:
                    min_lat, min_lon, max_lat, max_lon = map(float, bbox_str.split())
                    tiles.append((href, box(min_lon, min_lat, max_lon, max_lat)))
                except (ValueError, TypeError) as e:
                    log.warning(f"Could not parse bbox '{bbox_str}': {e}")

        log.debug(f"Mecklenburg-Vorpommern tile index loaded: {len(tiles)} tiles available.")
        return tiles

    @classmethod
    def _get_tile_index(cls) -> List[Tuple[str, Polygon]]:
        """Return the cached tile index, fetching it if necessary."""
        if cls._tile_index is None:
            cls._tile_index = cls._fetch_tile_index()
        return cls._tile_index

    @classmethod
    @override
    def _get_gml_file_urls_for_single_polygon(cls, polygon: Polygon) -> List[str]:
        """
        Return download URLs for Mecklenburg-Vorpommern tiles overlapping the given polygon.

        The polygon (in EPSG:25833) is transformed to WGS84 before comparing against
        the WGS84 bounding boxes in the tile index.

        Args:
            polygon: The area of interest in EPSG:25833 coordinates.

        Returns:
            A list of tile zip download URLs that overlap the polygon.
        """
        transformer = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)
        coords_wgs84 = [transformer.transform(x, y) for x, y in polygon.exterior.coords]
        poly_wgs84 = Polygon(coords_wgs84)
        tiles = cls._get_tile_index()
        matching = [url for url, tile_box in tiles if tile_box.intersects(poly_wgs84)]
        log.debug(
            f"Found {len(matching)} of {len(tiles)} Mecklenburg-Vorpommern tiles for polygon."
        )
        return matching

    @override
    def _process_gml_file(self, url: str, tmp_dir: Path) -> Optional[List[Building]]:
        """
        Download a Mecklenburg-Vorpommern tile zip and extract the CityGML file.

        Args:
            url: URL of the tile zip archive.
            tmp_dir: Temporary directory for extracted files.

        Returns:
            A list of Building objects, or None on download/parse failure.
        """
        r = get(url, timeout=120)
        if r.status_code != 200:
            log.error(f"Failed to download {url}. Status: {r.status_code}")
            return None

        try:
            z = ZipFile(io.BytesIO(r.content))
        except BadZipFile:
            log.error(f"Failed to open zip from {url}.")
            return None

        gml_names = [n for n in z.namelist() if n.endswith(".gml") or n.endswith(".xml")]
        if not gml_names:
            log.warning(f"No GML or XML file found in zip from {url}.")
            return None

        gml_path = Path(tmp_dir) / Path(gml_names[0]).with_suffix(".gml").name
        gml_path.write_bytes(z.read(gml_names[0]))
        return self._extract_buildings_from_gml_file(gml_path)


class BuildingsDownloaderNI(BuildingsDownloaderBase):
    """Buildings downloader for Niedersachsen (Lower Saxony), Germany.

    Uses the ArcGIS REST service to retrieve building data from the LGLN (Landesamt für
    Geoinformation und Landesvermessung Niedersachsen).
    """

    # ArcGIS REST service URL for Niedersachsen LoD2 data
    service_url: str = (
        "https://services-eu1.arcgis.com/4v3xxN52w88W065F/arcgis/rest/services/"
        "lgln_opengeodata_lod2/FeatureServer/0/query"
    )

    # the CRS in which the LoD2 data for this state is provided
    EPSG: int = 25832

    # bounding box of this state in this state's CRS (minx, miny, maxx, maxy)
    bounds: Tuple[int, int, int, int] = (342764, 5683146, 674154, 5971879)

    @classmethod
    @override
    def _get_gml_file_urls_for_single_polygon(cls, polygon: Polygon) -> List[str]:
        """
        Retrieves URLs for GML files containing building data for a single polygon by querying the
        ArcGIS REST service with the polygon geometry.

        The query is sent as a POST request with the parameters in the request body, so the
        (potentially large) geometry does not run into the URL-length limits that a GET query
        string would hit.

        Args:
            polygon: The area of interest represented as a shapely Polygon in EPSG:25832
                coordinates.

        Returns:
            A list of URLs pointing to GML files that overlap with the polygon.
        """
        geometry = json.dumps(
            {"rings": [list(polygon.exterior.coords)], "spatialReference": {"wkid": cls.EPSG}}
        )
        params = {
            "f": "json",
            "geometry": geometry,
            "geometryType": "esriGeometryPolygon",
            "spatialRel": "esriSpatialRelIntersects",
            "inSR": str(cls.EPSG),
            "returnGeometry": "false",
            "outSR": str(cls.EPSG),
            "outFields": "xml",
            "where": "1=1",
        }

        try:
            response = post(cls.service_url, data=params, timeout=cls.DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            log.error(f"Failed to query ArcGIS service at {cls.service_url}: {e}")
            return []

        if "features" not in data:
            log.warning(f"No features found in ArcGIS response. Response: {data}")
            return []

        urls = [
            f["attributes"]["xml"]
            for f in data["features"]
            if "attributes" in f and "xml" in f["attributes"]
        ]
        log.debug(f"Found {len(urls)} GML file URLs for polygon")
        return urls


class BuildingsDownloaderNW(BuildingsDownloaderBase):
    """Buildings downloader for Nordrhein-Westfalen (North Rhine-Westphalia), Germany.

    Tiles are listed in an XML directory index. Each tile is a direct CityGML .gml file
    (no zip) which is downloaded and parsed directly by the base class.

    Tile index:
        https://www.opengeodata.nrw.de/produkte/geobasis/3dg/lod2_gml/lod2_gml/

    Tile download URL pattern:
        https://www.opengeodata.nrw.de/.../LoD2_32_{x}_{y}_1_NW.gml

    EPSG: 25832 (ETRS89 / UTM Zone 32N)
    License: © Geobasis NRW — dl-de/zero-2-0
    """

    INDEX_URL: str = "https://www.opengeodata.nrw.de/produkte/geobasis/3dg/lod2_gml/lod2_gml/"
    CITYGML_SERVER: str = "https://www.opengeodata.nrw.de/produkte/geobasis/3dg/lod2_gml/lod2_gml"

    EPSG: int = 25832
    bounds: Tuple[int, int, int, int] = (280400, 5577534, 531762, 5820385)

    _available_tiles: Optional[set] = None

    @classmethod
    def _fetch_available_tiles(cls) -> set:
        """Fetch the XML tile index and return the set of available GML filenames."""
        if cls._available_tiles is not None:
            return cls._available_tiles
        log.debug("Fetching NRW tile index...")
        response = get(cls.INDEX_URL, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        cls._available_tiles = {f.attrib["name"] for f in root.findall(".//file")}
        log.debug(f"NRW tile index loaded: {len(cls._available_tiles)} tiles available.")
        return cls._available_tiles

    @staticmethod
    def _coords_to_tile(x_utm: float, y_utm: float) -> Tuple[int, int]:
        """Convert UTM coordinates (EPSG:25832) to 1x1 km tile indices."""
        return (math.floor(x_utm / 1000), math.floor(y_utm / 1000))

    @classmethod
    @override
    def _get_gml_file_urls_for_single_polygon(cls, polygon: Polygon) -> List[str]:
        """
        Return download URLs for NRW tiles overlapping the given polygon.

        Args:
            polygon: The area of interest in EPSG:25832 coordinates.

        Returns:
            A list of direct GML file URLs for tiles that overlap the polygon.
        """
        available = cls._fetch_available_tiles()
        minx, miny, maxx, maxy = polygon.bounds
        x1, y1 = cls._coords_to_tile(minx, miny)
        x2, y2 = cls._coords_to_tile(maxx, maxy)

        urls = []
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                if not box(x * 1000, y * 1000, (x + 1) * 1000, (y + 1) * 1000).intersects(polygon):
                    continue
                filename = f"LoD2_32_{x}_{y}_1_NW.gml"
                if filename in available:
                    urls.append(f"{cls.CITYGML_SERVER}/{filename}")
                else:
                    log.debug(f"Tile {filename} not in NRW tile index, skipping.")
        log.debug(f"Found {len(urls)} NRW tile URLs for polygon.")
        return urls


class BuildingsDownloaderRP(BuildingsDownloaderBase):
    """Buildings downloader for Rheinland-Pfalz (Rhineland-Palatinate), Germany.

    Tiles are indexed via an INSPIRE ATOM feed with WGS84 bounding boxes. Each tile
    is a direct CityGML .gml file (no zip) served by geobasis-rlp.de.

    Dataset feed:
        https://www.geoportal.rlp.de/mapbender/php/mod_inspireDownloadFeed.php
        ?id=0b28684d-b2ce-4b0b-b080-928025588c61&type=DATASET&generateFrom=remotelist

    Tile download URL pattern:
        https://geobasis-rlp.de/data/geb3dlo/current/gml/LoD2_32_{x}_{y}_2_RP.gml

    EPSG: 25832 (ETRS89 / UTM Zone 32N)
    License: dl-de/by-2-0 — © GeoBasis-DE / LVermGeoRP
    """

    DATASET_FEED_URL: str = (
        "https://www.geoportal.rlp.de/mapbender/php/mod_inspireDownloadFeed.php"
        "?id=0b28684d-b2ce-4b0b-b080-928025588c61&type=DATASET&generateFrom=remotelist"
    )

    EPSG: int = 25832
    bounds: Tuple[int, int, int, int] = (293349, 5424032, 466037, 5644110)

    _tile_index: Optional[List[Tuple[str, Polygon]]] = None

    @classmethod
    def _fetch_tile_index(cls) -> List[Tuple[str, Polygon]]:
        """Fetch the dataset ATOM feed and parse all tile GML URLs and WGS84 bounding boxes."""
        log.debug("Fetching Rheinland-Pfalz tile index from ATOM feed...")
        resp = get(cls.DATASET_FEED_URL, timeout=cls.DOWNLOAD_TIMEOUT)
        resp.raise_for_status()

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.content)
        tiles = []

        for entry in root.findall("atom:entry", ns):
            for link in entry.findall("atom:link[@rel='section']", ns):
                href = link.get("href")
                bbox_str = link.get("bbox")  # "minLat, minLon, maxLat, maxLon"
                if not href or not bbox_str:
                    continue
                try:
                    parts = [float(v.strip()) for v in bbox_str.split(",")]
                    min_lat, min_lon, max_lat, max_lon = parts
                    tiles.append((href, box(min_lon, min_lat, max_lon, max_lat)))
                except (ValueError, TypeError) as e:
                    log.warning(f"Could not parse bbox '{bbox_str}': {e}")

        log.debug(f"Rheinland-Pfalz tile index loaded: {len(tiles)} tiles available.")
        return tiles

    @classmethod
    def _get_tile_index(cls) -> List[Tuple[str, Polygon]]:
        """Return the cached tile index, fetching it if necessary."""
        if cls._tile_index is None:
            cls._tile_index = cls._fetch_tile_index()
        return cls._tile_index

    @classmethod
    @override
    def _get_gml_file_urls_for_single_polygon(cls, polygon: Polygon) -> List[str]:
        """
        Return download URLs for Rheinland-Pfalz tiles overlapping the given polygon.

        The polygon (in EPSG:25832) is transformed to WGS84 before comparing against
        the WGS84 bounding boxes in the tile index.

        Args:
            polygon: The area of interest in EPSG:25832 coordinates.

        Returns:
            A list of direct GML file URLs for tiles that overlap the polygon.
        """
        transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
        coords_wgs84 = [transformer.transform(x, y) for x, y in polygon.exterior.coords]
        poly_wgs84 = Polygon(coords_wgs84)
        tiles = cls._get_tile_index()
        matching = [url for url, tile_box in tiles if tile_box.intersects(poly_wgs84)]
        log.debug(f"Found {len(matching)} of {len(tiles)} Rheinland-Pfalz tiles for polygon.")
        return matching


class BuildingsDownloaderSH(BuildingsDownloaderBase):
    """Buildings downloader for Schleswig-Holstein, Germany.

    Tiles are indexed via a GeoJSON file. The tile polygons are in EPSG:25832
    (same CRS as the AOI), so no coordinate transformation is required for filtering.
    Each tile is a CityGML file served as raw XML over HTTP.

    GeoJSON tile index:
        https://geodaten.schleswig-holstein.de/gaialight-sh/_apps/dladownload/
        single.php?file=LOD2_SH_Massendownload.geojson&id=4

    EPSG: 25832 (ETRS89 / UTM Zone 32N)
    License: © GeoBasis-DE/SH — dl-de/by-2-0
    """

    GEOJSON_INDEX_URL: str = (
        "https://geodaten.schleswig-holstein.de/gaialight-sh/_apps/dladownload/"
        "single.php?file=LOD2_SH_Massendownload.geojson&id=4"
    )

    EPSG: int = 25832
    bounds: Tuple[int, int, int, int] = (426522, 5913656, 650183, 6101270)

    _tile_index: Optional[List[Tuple[str, Polygon]]] = None

    @classmethod
    def _fetch_tile_index(cls) -> List[Tuple[str, Polygon]]:
        """Fetch the GeoJSON tile index and return a list of (download_url, tile_polygon) pairs.

        The tile polygons in the GeoJSON index are in EPSG:25832. If this changes in
        future data releases, the AOI comparison in _get_gml_file_urls_for_single_polygon
        would silently return wrong results — the unit test guards against this.
        """
        log.debug("Fetching Schleswig-Holstein tile index from GeoJSON...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resp = get(cls.GEOJSON_INDEX_URL, verify=False, timeout=cls.DOWNLOAD_TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        tiles = []
        for feat in data.get("features", []):
            url = feat.get("properties", {}).get("data_link")
            geom = feat.get("geometry")
            if url and geom:
                tiles.append((url, shape(geom)))

        log.debug(f"Schleswig-Holstein tile index loaded: {len(tiles)} tiles available.")
        return tiles

    @classmethod
    def _get_tile_index(cls) -> List[Tuple[str, Polygon]]:
        """Return the cached tile index, fetching it if necessary."""
        if cls._tile_index is None:
            cls._tile_index = cls._fetch_tile_index()
        return cls._tile_index

    @classmethod
    @override
    def _get_gml_file_urls_for_single_polygon(cls, polygon: Polygon) -> List[str]:
        """
        Return download URLs for Schleswig-Holstein tiles overlapping the given polygon.

        The tile polygons from the GeoJSON index are in EPSG:25832 (same CRS as the AOI),
        so no coordinate transformation is applied.

        Args:
            polygon: The area of interest in EPSG:25832 coordinates.

        Returns:
            A list of tile download URLs that overlap the polygon.
        """
        tiles = cls._get_tile_index()
        matching = [url for url, tile_poly in tiles if tile_poly.intersects(polygon)]
        log.debug(f"Found {len(matching)} of {len(tiles)} Schleswig-Holstein tiles for polygon.")
        return matching

    @override
    def _process_gml_file(self, url: str, tmp_dir: Path) -> Optional[List[Building]]:
        """
        Download a Schleswig-Holstein CityGML file and save it with a .gml extension.

        The service returns raw CityGML XML over HTTP (no zip). SSL verification is
        disabled because the server uses a self-signed certificate.

        Args:
            url: URL of the CityGML tile.
            tmp_dir: Temporary directory for downloaded files.

        Returns:
            A list of Building objects, or None on download/parse failure.
        """
        filename = url.split("file=")[-1].split("&")[0] if "file=" in url else url.split("/")[-1]
        gml_path = Path(tmp_dir) / Path(filename).with_suffix(".gml").name

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = get(url, verify=False, timeout=120)

        if r.status_code != 200:
            log.error(f"Failed to download {url}. Status: {r.status_code}")
            return None

        gml_path.write_bytes(r.content)
        return self._extract_buildings_from_gml_file(gml_path)


class BuildingsDownloaderSL(BuildingsDownloaderBase):
    """Buildings downloader for Saarland, Germany.

    Data is split across 6 zip files by Landkreis (county), hosted on a Saarland Nextcloud share.
    Each zip contains 1x1 km CityGML tiles in EPSG:25832.

    Only Landkreise whose hardcoded bounding box (derived from official county boundaries) overlaps
    the AOI are downloaded, avoiding unnecessary transfers.

    Tile filename pattern:
        LoD2_32_{x}_{y}_1_SL.gml

    Nextcloud share (contains a rotating sharetoken which needs to be update periodically):
        https://www.shop.lvgl.saarland.de/cloud/index.php/s/<share-token>?dir=/OD_Geb%C3%A4udemodelle_LoD2_gml_LK

    EPSG: 25832 (ETRS89 / UTM Zone 32N)
    License: © LVGL Saarland — dl-de/by-2-0
    """

    SHARE_TOKEN: Optional[str] = None
    SHARED_GEODATA_URL = "https://www.shop.lvgl.saarland.de/cloud/freiegeobasisdaten"
    DOWNLOAD_BASE_URL: str = "https://www.shop.lvgl.saarland.de/cloud/public.php/dav/files"
    DOWNLOAD_DIRECTORY: str = "OD_Geb%C3%A4udemodelle_LoD2_gml_LK"

    # Mapping of zip filename → county bounding box (minx, miny, maxx, maxy) in EPSG:25832.
    # Bounds are derived from the official Saarland regional boundaries dataset.
    LANDKREIS_ZIPS: Dict[str, Tuple[int, int, int, int]] = {
        # Saarbrücken
        "SB_LOD2BWK_gml.zip": (333231, 5441716, 364342, 5471474),
        # Merzig-Wadern
        "MZG_LOD2BWK_gml.zip": (308754, 5471381, 352692, 5497787),
        # Neunkirchen
        "NK_LOD2BWK_gml.zip": (348909, 5460983, 376144, 5478060),
        # Saarlouis
        "SLS_LOD2BWK_gml.zip": (321360, 5453943, 354231, 5485330),
        # Saarpfalz-Kreis
        "SPK_LOD2BWK_gml.zip": (359022, 5441633, 384063, 5474241),
        # St. Wendel
        "WND_LOD2BWK_gml.zip": (347615, 5474756, 377684, 5500412),
    }
    EPSG: int = 25832
    bounds: Tuple[int, int, int, int] = (308721, 5441570, 384032, 5500420)

    def __init__(
        self,
        area_of_interest: Polygon | MultiPolygon,
        filter_by_aoi: bool = True,
        tilt_and_orientation_method: Callable = get_orientation_and_tilt,
        suppress_planarity_warnings: bool = True,
    ):
        """Initializes a BuildingsDownloaderSL instance.

        Tries to retrieve the current share token for the Saarland Nextcloud share. If the token
        cannot be retrieved, an error is logged.

        Args:
            area_of_interest: The area of interest for downloading
                building data, represented as a shapely Polygon or MultiPolygon. Must be in the
                correct CRS for the specific downloader.
            filter_by_aoi: If True (default), filters the downloaded buildings to only include those
                that intersect with the area of interest. If False, all buildings in the downloaded
                GML files are returned, regardless of their location.
            tilt_and_orientation_method: A method to calculate the tilt and orientation from the
            LoD-2 data. If not passed, the values from the CityDPC Building object are used.
            suppress_planarity_warnings: If True (default), suppresses SurfacePlanarityWarning from
                citydpc. These warnings indicate that a roof surface is not perfectly planar, which
                may affect area and orientation accuracy.
        """
        super().__init__(
            area_of_interest,
            filter_by_aoi=filter_by_aoi,
            tilt_and_orientation_method=tilt_and_orientation_method,
            suppress_planarity_warnings=suppress_planarity_warnings,
        )

        # retrieve the current share token and store it in the class variable
        try:
            self.__class__.SHARE_TOKEN = self._get_share_token()
        except ConnectionError as e:
            log.error(f"Failed to retrieve SL Nextcloud share token: {e}")
        finally:
            if not self.__class__.SHARE_TOKEN:
                log.error("Failed to retrieve SL Nextcloud share token. Downloader may not work!")

    @classmethod
    def _get_share_token(cls) -> Optional[str]:
        """Get the current share token for the Saarland Nextcloud share.

        Fetches the shared geodata page which redirects to the actual share URL. Extracts the share
        token from the URLs in the redirect history.

        Returns:
            The current share token as a string.
        """
        response = get(cls.SHARED_GEODATA_URL, timeout=cls.DOWNLOAD_TIMEOUT)
        for redirect in response.history:
            if "Location" in redirect.headers:
                # the URL containing the share token looks like this:
                # http://www.shop.lvgl.saarland.de/cloud/index.php/s/<share-token>
                match = re.search(r"/s/([^/?]+)", redirect.headers["Location"])
                if match:
                    return match.group(1)
        return None

    @staticmethod
    def _parse_tile_coords(filename: str) -> Optional[Tuple[int, int]]:
        """Parse (x_km, y_km) tile coordinates from a Saarland tile filename.

        Expects filenames like "LoD2_32_{x}_{y}_1_SL.gml".

        Args:
            filename: The tile filename (basename only).

        Returns:
            A tuple (x_km, y_km) or None if parsing fails.
        """
        parts = Path(filename).stem.split("_")
        try:
            return int(parts[2]), int(parts[3])
        except (IndexError, ValueError):
            return None

    def _get_matching_tiles(self, z: ZipFile) -> List[str]:
        """Return zip-internal paths of GML tiles whose 1x1 km bbox intersects the AOI.

        Args:
            z: An open ZipFile of one of the Saarland Landkreis archives.

        Returns:
            A list of zip-internal paths for tiles that overlap the AOI.
        """
        matching = []
        for name in z.namelist():
            if not name.endswith(".gml"):
                continue
            coords = self._parse_tile_coords(Path(name).name)
            if coords is None:
                continue
            x, y = coords
            tile = box(x * 1000, y * 1000, (x + 1) * 1000, (y + 1) * 1000)
            if tile.intersects(self.area_of_interest):
                matching.append(name)
        return matching

    @classmethod
    @override
    def _get_gml_file_urls_for_single_polygon(cls, polygon: Polygon) -> List[str]:
        """Return download URLs for Saarland Landkreis zip archives that overlap the polygon.

        Each Landkreis has a hardcoded bounding box (derived from official county boundaries) that
        is checked against the AOI to skip unnecessary downloads.

        Args:
            polygon: The area of interest in EPSG:25832 coordinates.

        Returns:
            A list of WebDAV URLs for the Landkreis archives that overlap the polygon.
        """
        matching = [
            f"{cls.DOWNLOAD_BASE_URL}/{cls.SHARE_TOKEN}/{cls.DOWNLOAD_DIRECTORY}/{name}"
            for name, bounds in cls.LANDKREIS_ZIPS.items()
            if box(*bounds).intersects(polygon)
        ]
        log.debug(f"AOI overlaps {len(matching)} of {len(cls.LANDKREIS_ZIPS)} Saarland Landkreise.")
        return matching

    @override
    def _process_gml_file(self, url: str, tmp_dir: Path) -> Optional[List[Building]]:
        """
        Download one Saarland Landkreis zip and extract all AOI-matching tiles.

        Args:
            url: URL of one of the 6 Landkreis zip archives.
            tmp_dir: Temporary directory for extracted tile files.

        Returns:
            A list of Building objects from all matching tiles, or None on failure.
        """
        log.debug(f"Downloading {url.split('/')[-1]} ...")
        r = get(url, timeout=300)
        if r.status_code != 200:
            log.error(f"Failed to download {url}. Status: {r.status_code}")
            return None
        log.debug(f"Downloaded {len(r.content) / 1e6:.1f} MB.")

        try:
            z = ZipFile(io.BytesIO(r.content))
        except BadZipFile as e:
            log.error(f"Failed to open zip from {url}: {e}")
            return None

        matching = self._get_matching_tiles(z)
        log.debug(f"{url.split('/')[-1]}: {len(matching)} tile(s) overlap AOI.")
        if not matching:
            return None

        buildings = []
        for tile_path in matching:
            gml_path = Path(tmp_dir) / Path(tile_path).name
            gml_path.write_bytes(z.read(tile_path))
            result = self._extract_buildings_from_gml_file(gml_path)
            buildings.extend(result)
            log.debug(f"  {Path(tile_path).name}: {len(result)} buildings")

        return buildings


class BuildingsDownloaderSN(BuildingsDownloaderBase):
    """Buildings downloader for Sachsen (Free State of Saxony), Germany.

    Tile download URLs are retrieved from an ArcGIS MapServer that indexes all 4,938
    LoD2 tiles. The MapServer is queried with the AOI bounding box (paginated) so only
    overlapping tiles are fetched. Each tile is a zip archive containing a single .gml
    CityGML file.

    Sachsen GML files lack a root <gml:boundedBy> envelope. The envelope is injected
    by _process_gml_file before parsing. Empty tiles are identified by a self-closing
    <core:CityModel .../> tag and skipped.

    MapServer:
        https://geodienste.sachsen.de/ags-relay/ArcGISServer/guest/arcgis/rest/
        services/geosn/rest_geosn_downloadlinks/MapServer/3

    EPSG: 25833 (ETRS89 / UTM Zone 33N)
    License: © GeoSN — dl-de/by-2-0
    """

    MAPSERVER_URL: str = (
        "https://geodienste.sachsen.de/ags-relay/ArcGISServer/guest/arcgis/rest/"
        "services/geosn/rest_geosn_downloadlinks/MapServer/3"
    )
    MAPSERVER_SRID: int = 25833

    GEOCLOUD_TOKEN: str = "AyJqXpJAZJXomCb"
    GEOCLOUD_BASE: str = "https://geocloud.landesvermessung.sachsen.de/public.php/dav/files"

    EPSG: int = 25833
    bounds: Tuple[int, int, int, int] = (278328, 5561109, 502917, 5728202)

    @classmethod
    @override
    def _get_gml_file_urls_for_single_polygon(cls, polygon: Polygon) -> List[str]:
        """
        Query the ArcGIS MapServer with the polygon bounding box and return tile URLs.

        Uses pagination (1000 records per page) to handle large result sets.

        Args:
            polygon: The area of interest in EPSG:25833 coordinates.

        Returns:
            A list of geocloud download URLs for tiles overlapping the polygon.
        """
        minx, miny, maxx, maxy = polygon.bounds
        geometry = f"{minx},{miny},{maxx},{maxy}"

        urls = []
        offset = 0
        page_size = 1000

        while True:
            params = {
                "where": "1=1",
                "geometry": geometry,
                "geometryType": "esriGeometryEnvelope",
                "inSR": cls.MAPSERVER_SRID,
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "Kachel,Download_CityGML",
                "resultOffset": offset,
                "resultRecordCount": page_size,
                "f": "json",
            }
            resp = get(f"{cls.MAPSERVER_URL}/query", params=params, timeout=cls.DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            features = data.get("features", [])
            for feat in features:
                kachel = feat.get("attributes", {}).get("Kachel")
                if kachel:
                    easting = kachel[:3]
                    northing = kachel[3:]
                    filename = f"lod2_33{easting}_{northing}_2_sn_citygml.zip"
                    urls.append(f"{cls.GEOCLOUD_BASE}/{cls.GEOCLOUD_TOKEN}/{filename}")

            log.debug(f"MapServer page offset={offset}: got {len(features)} features.")
            if len(features) < page_size:
                break
            offset += page_size

        log.debug(f"Found {len(urls)} Sachsen tile URLs for polygon.")
        return urls

    @override
    def _extract_buildings_from_gml_file(self, gml_file: Path, **kwargs) -> List[Building]:
        """Overrides the base class method by passing ignoreRefSystem=True to
        ``citydpc.core.input.citygmlInput.load_buildings_from_xml_file``.

        Sachsen GML files lack a root <gml:boundedBy> envelope that citydpc normally requires to
        determine the reference system. So, this override passes ignoreRefSystem=True to
        ``citydpc.core.input.citygmlInput.load_buildings_from_xml_file`` so that parsing succeeds
        even after envelope injection may have been skipped due to an error.

        Args:
            gml_file: Path to the CityGML file.
            **kwargs: Additional keyword arguments passed to
                ``citydpc.core.input.citygmlInput.load_buildings_from_xml_file``.

        Returns:
            A list of Building objects extracted from the GML file.
        """
        if "ignoreRefSystem" in kwargs:
            log.warning("ignoreRefSystem argument is ignored for Sachsen GML files.")
            kwargs.pop("ignoreRefSystem")
        return super()._extract_buildings_from_gml_file(gml_file, ignoreRefSystem=True, **kwargs)

    @override
    def _process_gml_file(self, url: str, tmp_dir: Path) -> Optional[List[Building]]:
        """
        Download a Sachsen tile zip and inject a missing CRS envelope before parsing.

        Empty tiles (self-closing <core:CityModel .../>) are detected and skipped.
        For non-empty tiles, a <gml:boundedBy> envelope derived from the filename
        is injected after the opening <core:CityModel> tag if no envelope is present.

        Args:
            url: URL of the tile zip archive.
            tmp_dir: Temporary directory for extracted and modified GML files.

        Returns:
            A list of Building objects, or None on download/parse failure.
        """
        r = get(url, timeout=120)
        if r.status_code != 200:
            log.error(f"Failed to download {url}. Status: {r.status_code}")
            return None

        try:
            z = ZipFile(io.BytesIO(r.content))
        except BadZipFile as e:
            log.error(f"Failed to open zip from {url}: {e}")
            return None

        gml_names = [n for n in z.namelist() if n.endswith(".gml")]
        if not gml_names:
            log.warning(f"No GML file found in zip from {url}.")
            return None

        gml_bytes = z.read(gml_names[0])
        gml_str = gml_bytes.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")

        # Locate the opening <core:CityModel> tag
        m_start = re.search(r"<core:CityModel\b", gml_str)
        if not m_start:
            log.warning(f"No CityModel tag found in {gml_names[0]}, skipping.")
            return []

        # Find the closing > of the opening tag, respecting quoted attribute values
        tag_end = None
        in_attr = False
        for i, ch in enumerate(gml_str[m_start.start() :], m_start.start()):
            if ch == '"':
                in_attr = not in_attr
            elif ch == ">" and not in_attr:
                tag_end = i
                break

        if tag_end is None:
            log.warning(f"Could not find end of CityModel tag in {gml_names[0]}, skipping.")
            return []

        # A self-closing tag means an empty tile — nothing to parse
        if gml_str[tag_end - 1] == "/":
            log.debug(f"{gml_names[0]}: empty tile (self-closing), skipping.")
            return []

        # Inject missing <gml:boundedBy> envelope derived from the tile filename
        if "gml:Envelope" not in gml_str[: tag_end + 200]:
            try:
                stem = Path(gml_names[0]).stem
                parts = stem.split("_")  # e.g. ['lod2', '33306', '5678', '2', 'sn']
                easting_km = int(parts[1][2:])  # strip '33' zone prefix → e.g. 306
                northing_km = int(parts[2])
                minx_e, miny_e = easting_km * 1000, northing_km * 1000
                maxx_e, maxy_e = minx_e + 2000, miny_e + 2000
                envelope = (
                    f"\n  <gml:boundedBy>"
                    f'\n    <gml:Envelope srsName="urn:ogc:def:crs,crs:EPSG:6.12:25833,'
                    f'crs:EPSG:6.12:7837" srsDimension="3">'
                    f'\n      <gml:lowerCorner srsDimension="3">'
                    f"{minx_e} {miny_e} 0.0</gml:lowerCorner>"
                    f'\n      <gml:upperCorner srsDimension="3">'
                    f"{maxx_e} {maxy_e} 500.0</gml:upperCorner>"
                    f"\n    </gml:Envelope>"
                    f"\n  </gml:boundedBy>"
                )
                gml_str = gml_str[: tag_end + 1] + envelope + gml_str[tag_end + 1 :]
                gml_bytes = gml_str.encode("utf-8")
            except Exception as e:
                log.error(f"Envelope injection failed for {gml_names[0]}: {type(e).__name__}: {e}")

        gml_path = Path(tmp_dir) / Path(gml_names[0]).name
        gml_path.write_bytes(gml_bytes)
        return self._extract_buildings_from_gml_file(gml_path)


class BuildingsDownloaderST(BuildingsDownloaderBase):
    """Buildings downloader for Sachsen-Anhalt (Saxony-Anhalt), Germany.

    Two download paths are used depending on the size of the area of interest (AOI).

    API path (small AOIs, <= ``cell_threshold`` cells):
        The 2x2 km grid cells intersecting the AOI are looked up in the bundled tile grid
        (``cell_label_to_id_ST.csv``) and requested by ID from the prepare API. Cell IDs are
        requested in batches of ``API_BATCH_SIZE`` (one prepared zip per batch). The API path is
        all-or-nothing: a batch that fails is retried once, and if it still fails the whole API
        download is aborted with an error rather than returning an incomplete result.
        Every ``.gml`` member of a batch zip is a requested tile, so all members are processed. API
        zip members are named ``LoD2_{label}.gml`` with the packed label
        ``32{easting_km:03d}{northing_km:04d}`` (e.g. ``LoD2_326065760.gml`` → easting 606 km,
        northing 5760 km). Parsed by ``_parse_tile_coords`` / ``_label_to_coords`` via
        ``_API_TILE_RE``.

    Non-API / bulk path (large AOIs, > ``cell_threshold`` cells):
        Four state-wide bulk zip archives are downloaded and only the tiles overlapping the AOI
        are extracted. Bulk zip members are named ``32_{easting_km}_{northing_km}_{raster}_ST.gml``
        (e.g. ``32_658_5704_2_ST.gml`` → easting 658 km, northing 5704 km, 2 km raster). Parsed by
        ``_parse_bulk_tile_coords`` via ``_BULK_TILE_RE``.

    If a downloaded ``.gml`` member does not match the expected naming scheme, a warning is logged
    (the source service may have changed its naming) and only that member is skipped — mismatches
    are never silently ignored.

    Bulk zip download URLs:
        https://www.geodatenportal.sachsen-anhalt.de/gfds_webshare/download/
        LVermGeo/Geodatenportal/Online-Bereitstellung-LVermGeo/3D/LoD2-{1..4}.zip

    API endpoint:
        https://www.lvermgeo.sachsen-anhalt.de/de/mod/4,1965,501/ajax/1/prepare/

    EPSG: 25832 (ETRS89 / UTM Zone 32N)
    License: © LVermGeo ST — dl-de/by-2-0
    """

    ZIP_URLS: List[str] = [
        "https://www.geodatenportal.sachsen-anhalt.de/gfds_webshare/download/"
        f"LVermGeo/Geodatenportal/Online-Bereitstellung-LVermGeo/3D/LoD2-{i}.zip"
        for i in range(1, 5)
    ]

    API_URL: str = "https://www.lvermgeo.sachsen-anhalt.de/de/mod/4,1965,501/ajax/1/prepare/"

    # csv file name for tile grid mapping (bundled in package data)
    CSV_FILENAME: str = "cell_label_to_id_ST.csv"

    # default threshold for choosing API vs bulk download (number of cells)
    DEFAULT_CELL_THRESHOLD: int = 400

    # cells requested per API "prepare" call. The endpoint costs ~0.16 s/cell server-side (measured)
    # and rejects long item lists via a URL-length limit, so requests are batched to stay well under
    # DOWNLOAD_TIMEOUT (60 s) and the URL-length ceiling.
    API_BATCH_SIZE: int = 100

    EPSG: int = 25832
    bounds: Tuple[int, int, int, int] = (607190, 5647911, 789276, 5880165)

    # class-level cache for label-to-ID mapping (label → API ID)
    _label_to_id_cache: Optional[pd.DataFrame] = None

    # class-level cache for grid cell geometries with spatial index
    _cell_geometries_cache: Optional[gpd.GeoDataFrame] = None

    # Directory for caching bulk zip files on disk; defaults to a subdirectory of the system
    # temp dir when None. Override at class level before instantiation to use a custom path.
    BULK_CACHE_DIR: Optional[Path] = None

    # in-process cache: URL -> path of the cached bulk zip file
    _bulk_zip_cache: Dict[str, Path] = {}

    # API/CSV grid-label scheme: "32" + 3-digit easting_km + 4-digit northing_km (e.g.
    # "326065760"); API zip members prefix it with "LoD2_" (e.g. "LoD2_326065760.gml"). Groups
    # capture easting_km and northing_km. Matched against a bare label or a filename stem.
    _API_TILE_RE: re.Pattern = re.compile(r"^(?:LoD2_)?32(\d{3})(\d{4})$", re.IGNORECASE)

    # Bulk zip member scheme: "32_{easting_km}_{northing_km}_{raster}_ST" (e.g. "32_658_5704_2_ST").
    # Groups capture easting_km and northing_km. Matched against a filename stem.
    _BULK_TILE_RE: re.Pattern = re.compile(r"^(?:LoD2_)?32_(\d+)_(\d+)_\d+_ST$", re.IGNORECASE)

    def __init__(
        self,
        area_of_interest: Polygon | MultiPolygon,
        filter_by_aoi: bool = True,
        tilt_and_orientation_method: Callable = get_orientation_and_tilt,
        cell_threshold: Optional[int] = None,
        suppress_planarity_warnings: bool = True,
    ):
        """
        Initializes the BuildingsDownloaderST class.

        Args:
            area_of_interest: The area of interest for downloading building data, represented as a
                shapely Polygon or MultiPolygon. Must be in EPSG:25832 coordinates.
            filter_by_aoi: If True (default), filters the downloaded buildings to only include those
                that intersect with the area of interest. If False, all buildings in the downloaded
                GML files are returned, regardless of their location.
            tilt_and_orientation_method: A method to calculate the tilt and orientation from the
                LoD-2 data. If not passed, the values from the CityDPC Building object are used.
            cell_threshold: Maximum number of cells for using API approach. If the number of
                intersecting cells exceeds this value, bulk download is used. Defaults to
                ``DEFAULT_CELL_THRESHOLD``.
            suppress_planarity_warnings: If True (default), suppresses SurfacePlanarityWarning
                from citydpc. Non-planar roof surfaces will not be reported.
        """
        super().__init__(
            area_of_interest=area_of_interest,
            filter_by_aoi=filter_by_aoi,
            tilt_and_orientation_method=tilt_and_orientation_method,
            suppress_planarity_warnings=suppress_planarity_warnings,
        )
        self.cell_threshold = (
            cell_threshold if cell_threshold is not None else self.DEFAULT_CELL_THRESHOLD
        )

    @classmethod
    def _load_label_to_id_mapping(cls) -> pd.DataFrame:
        """
        Load and cache the CSV mapping cell labels to API IDs.

        Returns:
            DataFrame with 'id' and 'label' columns indexed by label.
        """
        if cls._label_to_id_cache is None:
            data_files = resources.files("lod2_buildings_downloader") / "data"
            csv_file = data_files / cls.CSV_FILENAME

            # Read CSV using traversable path
            with csv_file.open("r", encoding="utf-8") as f:
                cls._label_to_id_cache = pd.read_csv(f, index_col="label", dtype={"id": str})

            log.debug(
                f"Loaded label-to-ID mapping with {len(cls._label_to_id_cache)} cells "
                f"from bundled {cls.CSV_FILENAME}"
            )
        return cls._label_to_id_cache

    @classmethod
    def _load_cell_geometries(cls) -> gpd.GeoDataFrame:
        """
        Load and cache grid cell geometries as a GeoDataFrame with spatial index.

        Returns:
            GeoDataFrame with 'label', 'id', and 'geometry' columns, indexed by label.
        """
        if cls._cell_geometries_cache is None:
            label_to_id = cls._load_label_to_id_mapping()
            data = []

            for label_int in label_to_id.index:
                label_str = str(label_int)
                easting_km, northing_km = cls._label_to_coords(label_str)

                # convert to meters and create 2x2 km cell polygon
                x, y = easting_km * 1000, northing_km * 1000
                geometry = box(x, y, x + 2000, y + 2000)

                data.append(
                    {
                        "label": label_str,
                        "id": str(label_to_id.loc[label_int, "id"]),
                        "geometry": geometry,
                    }
                )

            # create GeoDataFrame with spatial index
            cls._cell_geometries_cache = gpd.GeoDataFrame(
                data, crs=f"EPSG:{cls.EPSG}", geometry="geometry"
            ).set_index("label")
            cls._cell_geometries_cache.sindex

            log.debug(
                f"Created GeoDataFrame with {len(cls._cell_geometries_cache)} grid cells "
                f"and spatial index"
            )

        return cls._cell_geometries_cache

    @staticmethod
    def _coords_to_label(easting: int, northing: int) -> str:
        """
        Convert grid cell coordinates to label string. The label format is
        ``32{easting_km:03d}{northing_km:04d}`` where easting_km and northing_km are the easting and
        northing in kilometers (truncated to 3 and 4 digits respectively) of the bottom left corner
        of the cell.

        Args:
            easting: X coordinate (left edge of 2x2 km cell).
            northing: Y coordinate (bottom edge of 2x2 km cell).

        Returns:
            Label string like "326065760" for cell at (606000, 5760000).
        """
        return f"32{str(easting)[:3]}{str(northing)[:4]}"

    @classmethod
    def _label_to_coords(cls, label: str) -> Tuple[int, int]:
        """Parse easting and northing in kilometers from an API/CSV grid label (``_API_TILE_RE``).

        The label format is ``32{easting_km:03d}{northing_km:04d}`` (an optional ``LoD2_`` prefix
        from API zip member names is tolerated). This operates on trusted grid data, so a label
        that does not match the expected pattern raises rather than returning a bogus coordinate —
        a mismatch signals that the bundled grid or the service label format has changed.

        Args:
            label: Label string like "327165652".

        Returns:
            Tuple of (easting_km, northing_km), e.g. (716, 5652) for label "327165652".

        Raises:
            ValueError: If ``label`` does not match ``_API_TILE_RE``.
        """
        match = cls._API_TILE_RE.match(label)
        if match is None:
            raise ValueError(
                f"Label {label!r} does not match the expected Sachsen-Anhalt grid-label pattern "
                f"{cls._API_TILE_RE.pattern!r}."
            )
        return int(match.group(1)), int(match.group(2))

    @classmethod
    def _get_bulk_zip(cls, url: str) -> Optional[ZipFile]:
        """Return an open ZipFile for the given bulk archive URL, downloading if not cached.

        Uses a two-layer cache: an in-process class-level dict (survives across instances in the
        same process) backed by a disk cache in ``BULK_CACHE_DIR`` (survives across runs). Each
        of the four fixed bulk URLs maps to a stable filename derived from a URL hash.

        Args:
            url: URL of one of the four Sachsen-Anhalt state-wide bulk zip archives.

        Returns:
            An open ZipFile, or None on download or parse failure.
        """
        # in-process cache hit
        cached_path = cls._bulk_zip_cache.get(url)
        if cached_path is not None and cached_path.exists():
            log.debug(f"Bulk zip cache hit (in-process): {cached_path.name}")
            return ZipFile(cached_path)

        # disk cache hit
        cache_dir = cls.BULK_CACHE_DIR or Path(tempfile.gettempdir()) / "lod2_buildings_downloader"
        cache_dir.mkdir(parents=True, exist_ok=True)
        url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
        cache_path = cache_dir / f"st_bulk_{url_hash}.zip"

        if cache_path.exists():
            log.debug(f"Bulk zip cache hit (disk): {cache_path.name}")
            try:
                z = ZipFile(cache_path)
                cls._bulk_zip_cache[url] = cache_path
                return z
            except BadZipFile:
                log.warning(f"Cached bulk zip {cache_path.name} is corrupt; re-downloading.")
                cache_path.unlink(missing_ok=True)

        # download and populate both cache layers
        log.debug(f"Downloading {url.split('/')[-1]} ...")
        r = get(url, timeout=600)
        if r.status_code != 200:
            log.error(f"Failed to download {url}. Status: {r.status_code}")
            return None
        log.debug(f"Downloaded {len(r.content) / 1e6:.1f} MB.")

        cache_path.write_bytes(r.content)
        try:
            z = ZipFile(cache_path)
        except BadZipFile as e:
            log.error(f"Failed to open zip from {url}: {e}")
            cache_path.unlink(missing_ok=True)
            return None

        cls._bulk_zip_cache[url] = cache_path
        log.debug(f"Cached bulk zip to {cache_path}")
        return z

    @classmethod
    def _parse_bulk_tile_coords(cls, filename: str) -> Optional[Tuple[int, int]]:
        """Parse easting and northing in km from a bulk zip tile filename (``_BULK_TILE_RE``).

        Expects filenames like "32_658_5704_2_ST.gml", where 658 is the easting and 5704 the
        northing in kilometres (EPSG:25832). The bulk archives use this naming scheme instead of
        the packed "LoD2_326065760.gml" scheme handled by _parse_tile_coords.

        Args:
            filename: The tile filename (may include a path; only the basename is used).

        Returns:
            A tuple (easting_km, northing_km), or None if the name does not match the expected
            pattern (a warning is logged in that case, as the service may have changed its naming).
        """
        match = cls._BULK_TILE_RE.match(Path(filename).stem)
        if match is None:
            log.warning(
                f"Bulk tile filename {filename!r} does not match the expected Sachsen-Anhalt "
                f"pattern {cls._BULK_TILE_RE.pattern!r}; the download service may have changed its "
                f"naming scheme. Skipping this tile."
            )
            return None
        return int(match.group(1)), int(match.group(2))

    @classmethod
    def _get_intersecting_cell_ids(cls, polygon: Polygon) -> List[str]:
        """
        Get IDs of 2x2 km grid cells that intersect the polygon using spatial indexing.

        Args:
            polygon: Area of interest in EPSG:25832.

        Returns:
            List of cell IDs (as strings) that can be requested from the API.
        """
        cell_geometries = cls._load_cell_geometries()
        possible_matches_idx = cell_geometries.sindex.query(polygon, predicate="intersects")
        intersecting_cells = cell_geometries.iloc[possible_matches_idx]
        available_ids = intersecting_cells["id"].tolist()
        log.debug(f"Found {len(available_ids)} intersecting cells available in tile grid")
        return available_ids

    @classmethod
    def _download_via_api(cls, cell_ids: List[str], tmp_dir: Path) -> List[Path]:
        """Download the given cells via the prepare API, one zip per batch.

        The prepare endpoint costs ~0.16 s per cell server-side and rejects overly long item lists
        (URL-length limit), so cells are requested in batches of ``API_BATCH_SIZE``. The API path is
        all-or-nothing: a batch that fails is retried once, and if it still fails a ``RuntimeError``
        is raised so no incomplete result is returned.

        Args:
            cell_ids: List of cell IDs to request.
            tmp_dir: Directory to save the downloaded zip files.

        Returns:
            One zip path per batch (only returned when every batch succeeded).

        Raises:
            RuntimeError: If any batch could not be downloaded, even after one retry.
        """
        batches = [
            cell_ids[i : i + cls.API_BATCH_SIZE]
            for i in range(0, len(cell_ids), cls.API_BATCH_SIZE)
        ]
        log.debug(f"Requesting {len(cell_ids)} cells via API in {len(batches)} batch(es).")

        zip_paths: List[Path] = []
        for index, batch in enumerate(batches, 1):
            zip_path = cls._download_api_batch(batch, tmp_dir, index)
            if zip_path is None:
                log.warning(f"API batch {index}/{len(batches)} failed; retrying once...")
                zip_path = cls._download_api_batch(batch, tmp_dir, index)
            if zip_path is None:
                raise RuntimeError(
                    f"API batch {index}/{len(batches)} failed after retry. Aborting the ST "
                    f"API download (all-or-nothing); no partial result is returned."
                )
            zip_paths.append(zip_path)
        return zip_paths

    @classmethod
    def _download_api_batch(cls, cell_ids: List[str], tmp_dir: Path, index: int) -> Optional[Path]:
        """Prepare and download a single batch of cells via the API.

        Args:
            cell_ids: The cell IDs for this batch (at most ``API_BATCH_SIZE``).
            tmp_dir: Directory to save the downloaded zip file.
            index: Zero-based batch index, used to name the output file uniquely.

        Returns:
            Path to the downloaded zip file, or None on failure.
        """
        log.debug(f"Preparing API batch {index} with {len(cell_ids)} cell(s)...")

        # ask the service to prepare a zip; the response body is the download URL
        try:
            response = get(
                cls.API_URL,
                params={"items": ",".join(cell_ids), "format": "zip"},
                timeout=cls.DOWNLOAD_TIMEOUT,
            )
        except RequestException as e:
            log.error(f"API prepare request for batch {index} failed: {e}")
            return None
        if response.status_code != 200:
            log.error(f"API prepare request for batch {index} failed: HTTP {response.status_code}")
            return None

        download_url = response.text.strip()
        if not download_url.startswith("http"):
            log.error(
                f"API prepare request for batch {index} did not return a download URL "
                f"(got {download_url[:120]!r}); the service may have changed."
            )
            return None

        # download the prepared zip
        try:
            download_response = get(download_url, stream=True, timeout=600)
        except RequestException as e:
            log.error(f"Download of API batch {index} failed: {e}")
            return None
        if download_response.status_code != 200:
            log.error(f"Download of API batch {index} failed: HTTP {download_response.status_code}")
            return None

        zip_path = Path(tmp_dir) / f"st_api_download_{index}.zip"
        with open(zip_path, "wb") as f:
            for chunk in download_response.iter_content(chunk_size=8192):
                f.write(chunk)

        log.debug(f"Downloaded API batch {index}: {zip_path.stat().st_size / 1e6:.1f} MB")
        return zip_path

    @classmethod
    def _parse_tile_coords(cls, filename: str) -> Optional[Tuple[int, int]]:
        """
        Parse easting and northing in km from an API zip tile filename (``_API_TILE_RE``).

        Expects filenames like "LoD2_326065760.gml" where the digits encode a 3-digit easting and
        4-digit northing in kilometres (EPSG:25832).

        Args:
            filename: The tile filename (may include a path; only the basename is used).

        Returns:
            A tuple (easting_km, northing_km), or None if the name does not match the expected
            pattern (a warning is logged in that case, as the service may have changed its naming).
        """
        try:
            return cls._label_to_coords(Path(filename).stem)
        except ValueError:
            log.warning(
                f"API tile filename {filename!r} does not match the expected Sachsen-Anhalt "
                f"pattern {cls._API_TILE_RE.pattern!r}; the download service may have changed its "
                f"naming scheme. Skipping this tile."
            )
            return None

    def _get_matching_tiles(self, z: ZipFile) -> List[str]:
        """
        Return zip-internal paths of GML tiles whose 2x2 km bbox intersects the AOI.

        Args:
            z: An open ZipFile of one of the Sachsen-Anhalt bulk archives.

        Returns:
            A list of zip-internal paths for tiles that overlap the AOI.
        """
        matching = []
        for name in z.namelist():
            if not name.endswith(".gml"):
                continue
            coords = self._parse_bulk_tile_coords(Path(name).name)
            if coords is None:
                continue
            e, n = coords
            tile = box(e * 1000, n * 1000, (e + 2) * 1000, (n + 2) * 1000)
            if tile.intersects(self.area_of_interest):
                matching.append(name)
        return matching

    @classmethod
    @override
    def _get_gml_file_urls_for_single_polygon(cls, polygon: Polygon) -> List[str]:
        """
        Return URLs based on the size of the area. For small areas (few cells),
        returns a special marker for API-based download. For large areas, returns
        the bulk zip URLs.

        Args:
            polygon: The area of interest in EPSG:25832.

        Returns:
            Either a marker for API download or the list of bulk zip URLs.
        """
        # This method is called per polygon, but we need instance-level threshold.
        # Since this is a class method, we'll use the default threshold here.
        # The actual decision is made in _get_gml_file_urls_for_area_of_interest.
        return ["__API_DOWNLOAD__"]

    @override
    def _get_gml_file_urls_for_area_of_interest(self) -> List[str]:
        """
        Decide whether to use API or bulk download based on the number of intersecting cells.

        Returns:
            Either a marker for API download or the list of bulk zip URLs.
        """
        # Get all intersecting cells for the AOI
        if isinstance(self.area_of_interest, Polygon):
            cell_ids = self._get_intersecting_cell_ids(self.area_of_interest)
        else:  # MultiPolygon
            all_ids = []
            for polygon in self.area_of_interest.geoms:
                all_ids.extend(self._get_intersecting_cell_ids(polygon))
            cell_ids = list(set(all_ids))  # Deduplicate

        num_cells = len(cell_ids)
        log.debug(
            f"Area requires {num_cells} cells. Threshold: {self.cell_threshold}. "
            f"Using {'API' if num_cells <= self.cell_threshold else 'bulk'} download."
        )

        if num_cells <= self.cell_threshold:
            # Store cell IDs for later use in _process_gml_file
            self._api_cell_ids = cell_ids
            return ["__API_DOWNLOAD__"]
        else:
            # Use bulk download
            self._api_cell_ids = None
            return list(self.ZIP_URLS)

    @override
    def _process_gml_file(self, url: str, tmp_dir: Path) -> Optional[List[Building]]:
        """
        Download and process GML files. Handles both API and bulk download approaches.

        Args:
            url: Either "__API_DOWNLOAD__" marker or URL of a bulk zip archive.
            tmp_dir: Temporary directory for extracted tile files.

        Returns:
            A list of Building objects from all matching tiles, or None on failure.
        """
        if url == "__API_DOWNLOAD__":
            # Use API approach
            if not hasattr(self, "_api_cell_ids") or not self._api_cell_ids:
                log.error("API download requested but no cell IDs available")
                return None

            # all-or-nothing: _download_via_api raises if any batch could not be downloaded
            zip_paths = self._download_via_api(self._api_cell_ids, tmp_dir)

            buildings = []
            for zip_path in zip_paths:
                z = ZipFile(zip_path)
                # every member of a batch zip was requested specifically, so process them all
                gml_files = [name for name in z.namelist() if name.endswith(".gml")]
                log.debug(f"{zip_path.name} contains {len(gml_files)} GML file(s)")

                for tile_path in gml_files:
                    gml_path = Path(tmp_dir) / Path(tile_path).name
                    gml_path.write_bytes(z.read(tile_path))
                    result = self._extract_buildings_from_gml_file(gml_path)
                    buildings.extend(result)
                    log.debug(f"  {Path(tile_path).name}: {len(result)} buildings")

            return buildings

        else:
            # Use bulk download approach (with disk+in-process cache via _get_bulk_zip)
            z = self._get_bulk_zip(url)
            if z is None:
                return None

            matching = self._get_matching_tiles(z)
            log.debug(f"{url.split('/')[-1]}: {len(matching)} tile(s) overlap AOI.")
            if not matching:
                return None

            buildings = []
            for tile_path in matching:
                gml_path = Path(tmp_dir) / Path(tile_path).name
                gml_path.write_bytes(z.read(tile_path))
                result = self._extract_buildings_from_gml_file(gml_path)
                buildings.extend(result)
                log.debug(f"  {Path(tile_path).name}: {len(result)} buildings")

            return buildings


class BuildingsDownloaderTH(BuildingsDownloaderBase):
    """Buildings downloader for Thüringen (Thuringia), Germany.

    Tiles are indexed via an INSPIRE ATOM feed with WGS84 bounding boxes. The feed has
    two entries (LoD1 and LoD2); the LoD2 entry is selected by LOD2_ENTRY_INDEX. Each
    tile is a zip archive containing a single CityGML .gml file.

    ATOM feed:
        https://geoportal.geoportal-th.de/dienste/atom_th_gebaeude
        ?type=dataset&id=97d152b8-9e00-49f3-9ae4-8bbb30873562

    EPSG: 25832 (ETRS89 / UTM Zone 32N)
    License: © GDI-Th — dl-de/by-2-0
    """

    ATOM_FEED_URL: str = (
        "https://geoportal.geoportal-th.de/dienste/atom_th_gebaeude"
        "?type=dataset&id=97d152b8-9e00-49f3-9ae4-8bbb30873562"
    )
    LOD2_ENTRY_INDEX: int = 1  # Entry 0 = LoD1, Entry 1 = LoD2

    EPSG: int = 25832
    bounds: Tuple[int, int, int, int] = (562176, 5562839, 756700, 5723937)

    _tile_index: Optional[List[Tuple[str, Polygon]]] = None

    @classmethod
    def _fetch_tile_index(cls) -> List[Tuple[str, Polygon]]:
        """Fetch the ATOM feed and parse all LoD2 tile URLs and WGS84 bounding boxes."""
        log.debug("Fetching Thüringen tile index from ATOM feed...")
        resp = get(cls.ATOM_FEED_URL, timeout=cls.DOWNLOAD_TIMEOUT)
        resp.raise_for_status()

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.content)
        entries = root.findall("atom:entry", ns)

        if len(entries) <= cls.LOD2_ENTRY_INDEX:
            raise RuntimeError(
                f"Expected at least {cls.LOD2_ENTRY_INDEX + 1} entries in ATOM feed, "
                f"got {len(entries)}."
            )

        lod2_entry = entries[cls.LOD2_ENTRY_INDEX]
        title = lod2_entry.findtext("atom:title", namespaces=ns)
        log.debug(f"Using ATOM entry {cls.LOD2_ENTRY_INDEX}: {title!r}")

        tiles = []
        for link in lod2_entry.findall("atom:link[@rel='section']", ns):
            href = link.get("href")
            bbox_str = link.get("bbox")  # "minLat minLon maxLat maxLon"
            if not href or not bbox_str:
                continue
            try:
                min_lat, min_lon, max_lat, max_lon = map(float, bbox_str.split())
                tiles.append((href, box(min_lon, min_lat, max_lon, max_lat)))
            except (ValueError, TypeError) as e:
                log.warning(f"Could not parse bbox '{bbox_str}': {e}")

        log.debug(f"Thüringen tile index loaded: {len(tiles)} tiles available.")
        return tiles

    @classmethod
    def _get_tile_index(cls) -> List[Tuple[str, Polygon]]:
        """Return the cached tile index, fetching it if necessary."""
        if cls._tile_index is None:
            cls._tile_index = cls._fetch_tile_index()
        return cls._tile_index

    @classmethod
    @override
    def _get_gml_file_urls_for_single_polygon(cls, polygon: Polygon) -> List[str]:
        """
        Return download URLs for Thüringen tiles overlapping the given polygon.

        The polygon (in EPSG:25832) is transformed to WGS84 before comparing against
        the WGS84 bounding boxes in the tile index.

        Args:
            polygon: The area of interest in EPSG:25832 coordinates.

        Returns:
            A list of tile zip download URLs that overlap the polygon.
        """
        transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
        coords_wgs84 = [transformer.transform(x, y) for x, y in polygon.exterior.coords]
        poly_wgs84 = Polygon(coords_wgs84)
        tiles = cls._get_tile_index()
        matching = [url for url, tile_box in tiles if tile_box.intersects(poly_wgs84)]
        log.debug(f"Found {len(matching)} of {len(tiles)} Thüringen tiles for polygon.")
        return matching

    @override
    def _process_gml_file(self, url: str, tmp_dir: Path) -> Optional[List[Building]]:
        """
        Download a Thüringen tile zip and extract the CityGML file.

        Args:
            url: URL of the tile zip archive.
            tmp_dir: Temporary directory for extracted files.

        Returns:
            A list of Building objects, or None on download/parse failure.
        """
        r = get(url, timeout=120)
        if r.status_code != 200:
            log.error(f"Failed to download {url}. Status: {r.status_code}")
            return None

        try:
            z = ZipFile(io.BytesIO(r.content))
        except BadZipFile as e:
            log.error(f"Failed to open zip from {url}: {e}")
            return None

        gml_names = [n for n in z.namelist() if n.endswith(".gml")]
        if not gml_names:
            log.warning(f"No GML file found in zip from {url}.")
            return None

        gml_path = Path(tmp_dir) / Path(gml_names[0]).name
        gml_path.write_bytes(z.read(gml_names[0]))
        return self._extract_buildings_from_gml_file(gml_path)


# ---------------------------------------------------------------------------
# AOI -> downloader resolution
# ---------------------------------------------------------------------------

# bundled GeoJSON of German federal-state boundaries in EPSG:4326. Each feature must carry an
# integer "state_id" property (1-16) identifying the state; see _ID_REGION_TO_DOWNLOADER.
_STATE_BOUNDARIES_RESOURCE = "german_federal_states.geojson"

# maps the federal-state key (state_id, 1-16) to the downloader serving that state
_ID_REGION_TO_DOWNLOADER: Dict[int, Type[BuildingsDownloaderBase]] = {
    1: BuildingsDownloaderSH,
    2: BuildingsDownloaderHH,
    3: BuildingsDownloaderNI,
    4: BuildingsDownloaderHB,
    5: BuildingsDownloaderNW,
    6: BuildingsDownloaderHE,
    7: BuildingsDownloaderRP,
    8: BuildingsDownloaderBW,
    9: BuildingsDownloaderBY,
    10: BuildingsDownloaderSL,
    11: BuildingsDownloaderBE,
    12: BuildingsDownloaderBB,
    13: BuildingsDownloaderMV,
    14: BuildingsDownloaderSN,
    15: BuildingsDownloaderST,
    16: BuildingsDownloaderTH,
}

_state_boundaries: Optional[gpd.GeoDataFrame] = None


def _load_state_boundaries() -> gpd.GeoDataFrame:
    """Loads and caches the bundled federal-state boundary GeoDataFrame (EPSG:4326)."""
    global _state_boundaries
    if _state_boundaries is None:
        resource = resources.files("lod2_buildings_downloader.data").joinpath(
            _STATE_BOUNDARIES_RESOURCE
        )
        with resources.as_file(resource) as path:
            _state_boundaries = gpd.read_file(path)
    return _state_boundaries


def get_downloaders_for_aoi(
    area_of_interest: Polygon | MultiPolygon,
) -> List[Type[BuildingsDownloaderBase]]:
    """Returns the downloader classes whose federal state borders intersect the area of interest.

    The returned classes are not instantiated; it is up to the caller to construct each one with an
    AOI in that downloader's own CRS (see each class's ``EPSG``).

    Args:
        area_of_interest: The area of interest as a shapely Polygon or MultiPolygon. Must be in
            EPSG:4326 (WGS84).

    Returns:
        A list of BuildingsDownloaderBase subclasses (uninstantiated) whose state boundary
        intersects the AOI, ordered by federal-state key (state_id). Empty if no state is touched.

    Raises:
        TypeError: If area_of_interest is not a shapely Polygon or MultiPolygon.
        ValueError: If the AOI coordinates fall outside valid lon/lat ranges, hinting the AOI is not
            in EPSG:4326.
    """
    if not isinstance(area_of_interest, (Polygon, MultiPolygon)):
        raise TypeError(
            "area_of_interest must be a shapely.Polygon or shapely.MultiPolygon in EPSG:4326 "
            "coordinates."
        )

    # sanity check that coordinates look like lon/lat, i.e. the AOI is really in EPSG:4326
    minx, miny, maxx, maxy = area_of_interest.bounds
    if not (-180 <= minx <= maxx <= 180 and -90 <= miny <= maxy <= 90):
        raise ValueError(
            f"area_of_interest bounds {area_of_interest.bounds} are outside valid lon/lat ranges "
            "([-180, 180], [-90, 90]). The AOI must be in EPSG:4326 (WGS84) coordinates."
        )

    boundaries = _load_state_boundaries()
    intersecting = boundaries[boundaries.intersects(area_of_interest)]
    matched_ids = sorted(int(v) for v in intersecting["state_id"])

    downloaders = [
        _ID_REGION_TO_DOWNLOADER[i] for i in matched_ids if i in _ID_REGION_TO_DOWNLOADER
    ]

    if not downloaders:
        log.warning(
            "No federal state intersects the area of interest. Make sure the AOI is in EPSG:4326 "
            "coordinates and located within Germany."
        )
    return downloaders
