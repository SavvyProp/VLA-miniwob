import pytest


def pytest_addoption(parser):
    parser.addoption("--run-browser", action="store_true", help="Run real Chrome integration tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-browser"):
        for item in items:
            if "browser" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Use --run-browser to launch Chrome"))
