# ============================================================================
# conftest.py - Pytest configuration and fixtures
# ============================================================================
import pytest
import logging

logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    """Add custom command-line options for pytest"""
    parser.addoption(
        "--verbose-tests",
        action="store_true",
        default=False,
        help="Enable verbose logging during tests"
    )


@pytest.fixture(scope="session", autouse=True)
def configure_logging(request):
    """Configure logging based on --verbose-tests flag"""
    verbose = request.config.getoption("--verbose-tests")
    
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S',
            force=True
        )
        logger.info("=" * 80)
        logger.info("VERBOSE TEST MODE ENABLED")
        logger.info("=" * 80)
    else:
        logging.basicConfig(level=logging.WARNING, force=True)
    
    return verbose


@pytest.fixture
def verbose_mode(request):
    """Fixture to check if verbose mode is enabled"""
    return request.config.getoption("--verbose-tests")
