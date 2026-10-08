<a name="readme-top"></a>

<br />
<div align="center">

<h3 align="center">lod2-buildings-downloader</h3>

  <p align="center">
    Python library for downloading and parsing LoD2 building data from German open-geodata services. Given an area of interest, it connects to the relevant services, retrieves the necessary files and returns structured <code>Building</code> objects.
    <br />
    <br />
    &middot;
    <a href="https://github.com/ffe-munich/lod2-buildings-downloader/issues/new?labels=bug">Report Bug</a>
    &middot;
    <a href="https://github.com/ffe-munich/lod2-buildings-downloader/issues/new?labels=enhancement">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About the project</a></li>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#getting-started">Getting started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li>
      <a href="#for-developers">For developers</a>
      <ol>
        <li><a href="#local-development-setup">Local development setup</a></li>
        <li><a href="#tests">Tests</a></li>
        <li><a href="#code-style">Code style</a></li>
      </ol>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About the project

Germany's LoD2 building data provides 3D geometries for virtually every building in the country.
However, each federal state publishes its data through different services and formats,
making nationwide usage cumbersome.

This library offers a central, user-friendly solution that downloads LoD2 data for a given area of
interest, parses the building geometries, and returns structured `Building` objects
— ready to use for solar potential analysis, urban planning, or any other application that needs
roof surface data.

The parsing of CityGML files is heavily powered by the work of the developers and maintainers
from the [CityDPC](https://github.com/ffe-munich/CityDPC) project. So thanks for that!

This is a sibling project of [orthophotos-downloader](https://github.com/ffe-munich/orthophotos-downloader)
and integrates directly with it via the `Building.download_orthophoto()` method, which instantly
downloads an image of the `Building` - especially usefull for use cases involving computer vison or object detection.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Prerequisites

This package requires Python ≥ 3.12.

> [!IMPORTANT]
> CityDPC is required at runtime but is not available from PyPI. Install it before installing the
> downloader:
>
> ```sh
> pip install "git+https://github.com/ffe-munich/CityDPC.git@main"
> # or with uv
> uv add "git+https://github.com/ffe-munich/CityDPC.git@main"
> ```
>
> **Using uv?** Run the CityDPC `uv add` command first. It records CityDPC in your project's
> dependencies and lockfile, so `--no-sync` is not needed for a normal user install. That warning
> applies only to the separate developer setup below.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Getting started

Ensure to read the [Prerequisites](#prerequisites) section and install the downloader:

```sh
pip install lod2-buildings-downloader
# or with uv
uv add lod2-buildings-downloader
```

With orthophoto support (integrates with [orthophotos-downloader](https://github.com/ffe-munich/orthophotos-downloader)):

```sh
pip install "lod2-buildings-downloader[orthophotos]"
# or with uv
uv add "lod2-buildings-downloader[orthophotos]"
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Usage

```python
from shapely.geometry import Polygon
from lod2_buildings_downloader import BuildingsDownloaderBY

area = Polygon([(690000, 5334000), (690100, 5334000), (690100, 5334100), (690000, 5334100)])
downloader = BuildingsDownloaderBY(area)
buildings = downloader.generate_buildings()

for building in buildings:
    print(building.gml_id)
    print(f"  Grounds: {len(building.grounds)}, Roofs: {len(building.roofs)}")
    for roof in building.roofs:
        print(
            f"  Roof area: {roof.surface_area:.1f} m², tilt: {roof.surface_tilt}°, orientation: {roof.surface_orientation}°"
        )
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## For developers

### Local development setup

1. Clone the repo
   ```sh
   git clone https://github.com/ffe-munich/lod2-buildings-downloader.git
   ```
2. Install all dependencies including the dev dependencies
   ```sh
   # requires pip>=25.1 for --group support
   pip install -e . --group dev
   # or with uv
   uv sync --group dev
   ```
3. Optionally install the orthophotos extra along with the dev dependencies
  ```sh
  # requires pip>=25.1 for --group support
  pip install -e ".[orthophotos]" --group dev
  # or with uv
  uv sync --group dev --extra orthophotos
  ```
4. Install the CityDPC revision required by this project
  ```sh
  pip install "git+https://github.com/ffe-munich/CityDPC.git@main"
  # or with uv
  uv pip install "git+https://github.com/ffe-munich/CityDPC.git@main"
  ```

**Important for uv:** CityDPC is installed outside `uv.lock`. After installing it, environment-syncing
commands such as `uv sync` or `uv run` without `--no-sync` can remove it. Run subsequent project
commands with `uv run --no-sync ...`. Install optional extras before CityDPC.

### Tests

The test suite uses pre-downloaded GML fixture files stored in `tests/data/` so no network
access is required by default.

Because CityDPC is installed separately from the uv lockfile, use `uv run --no-sync` to keep uv
from removing it when running project commands.

Run tests with these commands.
```sh
# offline (default) — fast, no network required
pytest
# or with uv
uv run --no-sync pytest

# also run the full download pipeline against the live service
pytest --download
# or with uv
uv run --no-sync pytest --download
```

Alternatively, when using VSCode you can use the included [settings.json example](.vscode/settings.json.example) to configure the [Python Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) with the Test Explorer integration and run / debug tests from there.

When `--download` is passed, every integration test is run a second time where it actually connects to the real services and downloads _live_
data. So, in this case the full pipeline is covered by the tests. Unit tests are unaffected by this flag.

### Code style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Its configuration in `pyproject.toml` sets a 100-character line length and checks for style issues (`E`), likely mistakes such as unused names (`F`), warnings (`W`), and import ordering (`I`).

#### Linting

```sh
# check for linting issues
uv run --no-sync ruff check

# apply fixes if possible
uv run --no-sync ruff check --fix
```

#### Formatting

```sh
# check formatting
uv run --no-sync ruff format --check .

# apply formatting
uv run --no-sync ruff format .
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire,
and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull
request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## License

Distributed under the MIT License. See [`LICENSE.md`](./LICENSE) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
