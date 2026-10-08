import logging

from .core.state_downloaders import (
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

# attach only a NullHandler so importing the package has no logging sideeffects. Applications
# configure output themselves (e.g. logging.basicConfig / dictConfig).
logging.getLogger(__name__).addHandler(logging.NullHandler())


__all__ = [
    "BuildingsDownloaderBB",
    "BuildingsDownloaderBE",
    "BuildingsDownloaderBW",
    "BuildingsDownloaderBY",
    "BuildingsDownloaderHB",
    "BuildingsDownloaderHE",
    "BuildingsDownloaderHH",
    "BuildingsDownloaderMV",
    "BuildingsDownloaderNI",
    "BuildingsDownloaderNW",
    "BuildingsDownloaderRP",
    "BuildingsDownloaderSH",
    "BuildingsDownloaderSL",
    "BuildingsDownloaderSN",
    "BuildingsDownloaderST",
    "BuildingsDownloaderTH",
    "get_downloaders_for_aoi",
]
