def pytest_addoption(parser):
    parser.addoption(
        "--download",
        action="store_true",
        default=False,
        help="Run tests that require downloading data from the internet.",
    )
