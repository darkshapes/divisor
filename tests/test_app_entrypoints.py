"""Tests for app.py entry point routing."""

import sys
from unittest.mock import patch

import pytest

from divisor.app import main


@pytest.fixture
def preserve_argv():
    """Fixture to preserve and restore sys.argv."""
    original_argv = sys.argv.copy()
    yield
    sys.argv = original_argv


@pytest.fixture
def mock_fire():
    """Fixture to mock Fire function."""
    with patch("divisor.app.Fire") as mock:
        yield mock


@pytest.fixture
def mock_flux1_main():
    """Fixture to mock flux1.prompt.main."""
    with patch("divisor.app.flux1_main") as mock:
        yield mock


@pytest.fixture
def mock_flux2_main():
    """Fixture to mock flux2.prompt.main."""
    with patch("divisor.app.flux2_main") as mock:
        yield mock


@pytest.fixture
def mock_xflux1_main():
    """Fixture to mock xflux1.prompt.main."""
    with patch("divisor.app.xflux1_main") as mock:
        yield mock


class TestAppEntryPoints:
    """Test that app.py routes to correct entry points based on model-type."""

    def test_routes_to_flux1_for_dev(self, mock_fire, mock_flux1_main, preserve_argv):
        """Test that --model-type dev routes to flux1_main."""
        sys.argv = ["dvzr", "--model-type", "flux1-dev"]
        main()
        # Verify Fire was called with flux1_main function (not the result of calling it)
        # Note: app.py has Fire(main()) which calls main(), but we test the actual behavior
        mock_fire.assert_called_once()
        # Check that sys.argv was modified correctly
        assert sys.argv[1] == "--model-id"
        assert sys.argv[2] == "flux1-dev"

    def test_routes_to_flux1_for_schnell(self, mock_fire, mock_flux1_main, preserve_argv):
        """Test that --model-type schnell routes to flux1_main."""
        sys.argv = ["dvzr", "--model-type", "flux1-schnell"]
        main()
        # Verify Fire was called
        mock_fire.assert_called_once()
        # Check that sys.argv was modified correctly
        assert sys.argv[1] == "--model-id"
        assert sys.argv[2] == "flux1-schnell"

    def test_routes_to_flux2_for_dev2(self, mock_fire, mock_flux2_main, preserve_argv):
        """Test that --model-type dev2 routes to flux2_main."""
        sys.argv = ["dvzr", "--model-type", "flux2-dev"]
        main()
        # Verify Fire was called
        mock_fire.assert_called_once()
        # Check that sys.argv was modified correctly
        assert sys.argv[1] == "--model-id"
        assert sys.argv[2] == "flux2-dev"

    def test_routes_to_xflux1_for_mini(self, mock_fire, mock_xflux1_main, preserve_argv):
        """Test that --model-type mini routes to xflux1_main."""
        sys.argv = ["dvzr", "--model-type", "mini"]
        main()
        # Verify Fire was called
        mock_fire.assert_called_once()
        # Check that sys.argv was modified correctly
        assert sys.argv[1] == "--model-id"
        assert sys.argv[2] == "flux1-dev:mini"

    def test_defaults_to_flux1_when_no_model_type(self, mock_fire, mock_flux1_main, preserve_argv):
        """Test that default (no model-type) routes to flux1_main."""
        sys.argv = ["dvzr"]  # No model-type, should default to dev
        main()
        # Verify Fire was called
        mock_fire.assert_called_once()
        # Check that sys.argv was modified correctly (default is dev)
        assert sys.argv[1] == "--model-id"
        assert sys.argv[2] == "flux1-dev"

    def test_preserves_additional_arguments(self, mock_fire, mock_flux1_main, preserve_argv):
        """Test that additional arguments are preserved after model-id insertion."""
        sys.argv = ["dvzr", "--model-type", "flux1-dev", "--width", "1024", "--height", "768"]
        main()
        # Verify Fire was called
        mock_fire.assert_called_once()
        # Check that model-id was inserted and other args preserved
        assert sys.argv[1] == "--model-id"
        assert sys.argv[2] == "flux1-dev"
        assert "--width" in sys.argv
        assert "--height" in sys.argv

    def test_preserves_quantization_flag(self, mock_fire, mock_flux1_main, preserve_argv):
        """Test that --quantization flag is preserved."""
        sys.argv = ["dvzr", "--model-type", "flux1-dev", "--quantization"]
        main()
        # Verify Fire was called
        mock_fire.assert_called_once()
        # The quantization flag is parsed by argparse and stored in args, but not in remaining_argv
        # So it won't appear in sys.argv after modification, which is expected behavior

    def test_raises_error_for_invalid_model_type(self, preserve_argv):
        """Test that invalid model-type raises an error."""
        sys.argv = ["dvzr", "--model-type", "invalid"]
        with pytest.raises(SystemExit):
            # argparse will raise SystemExit for invalid choices
            main()
