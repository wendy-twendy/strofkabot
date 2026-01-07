"""
Tests for the main() function and setup_logging in llumi.py.
"""
import logging
from unittest.mock import patch

import pytest

from strofkabot.llumi import setup_logging


class TestSetupLoggingDetails:
    """Additional tests for setup_logging function."""

    def test_creates_stream_handler(self):
        """Test that setup_logging creates a StreamHandler."""
        # Clear any existing handlers
        logger = logging.getLogger('LlumiBot')
        logger.handlers.clear()

        result = setup_logging('INFO')

        # Check a handler was added
        assert len(result.handlers) >= 1
        # Find StreamHandler
        stream_handlers = [h for h in result.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) >= 1

    def test_sets_formatter(self):
        """Test that setup_logging sets a formatter with timestamp."""
        # Clear any existing handlers
        logger = logging.getLogger('LlumiBot')
        logger.handlers.clear()

        result = setup_logging('DEBUG')

        # Check formatter is set
        for handler in result.handlers:
            if isinstance(handler, logging.StreamHandler):
                assert handler.formatter is not None
                # Format string includes timestamp, name, level
                format_string = handler.formatter._fmt
                assert 'asctime' in format_string
                assert 'name' in format_string
                assert 'levelname' in format_string


class TestMainFunction:
    """Tests for the main() function."""

    @pytest.mark.asyncio
    async def test_missing_token_returns_early(self):
        """Test that main() exits when LLUMI_BOT_TOKEN is not set."""
        from strofkabot.llumi import main

        with patch.dict('os.environ', {}, clear=True):
            # Remove the token if it exists
            with patch('os.getenv', return_value=None):
                with patch('strofkabot.llumi.setup_logging') as mock_logging:
                    mock_logger = logging.getLogger('test')
                    mock_logging.return_value = mock_logger

                    # Mock argparse to avoid command line issues
                    with patch('argparse.ArgumentParser.parse_args') as mock_args:
                        mock_args.return_value.log_level = 'INFO'

                        # main() should return early without crashing
                        await main()

                        # Verify it logged the error (by checking setup_logging was called)
                        mock_logging.assert_called_once()

    @pytest.mark.asyncio
    async def test_parses_log_level_argument(self):
        """Test that main() parses the --log-level argument."""
        from strofkabot.llumi import main

        with patch('argparse.ArgumentParser.parse_args') as mock_args:
            mock_args.return_value.log_level = 'DEBUG'

            with patch('strofkabot.llumi.setup_logging') as mock_logging:
                mock_logger = logging.getLogger('test')
                mock_logging.return_value = mock_logger

                # Missing token so it returns early
                with patch('os.getenv', return_value=None):
                    await main()

                    # Verify setup_logging was called with the parsed level
                    mock_logging.assert_called_once_with('DEBUG')
