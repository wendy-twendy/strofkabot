"""Tests for the !ask command in AICog."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strofkabot.cogs.ai import AICog
from strofkabot.discord_db import Database
from strofkabot.openrouter_client import OpenRouterResponse


@pytest.fixture
def mock_logger() -> logging.Logger:
    """Create a mock logger for testing."""
    logger = logging.getLogger("test_ask_logger")
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def mock_db():
    """Create a mock database."""
    return AsyncMock(spec=Database)


@pytest.fixture
def mock_bot():
    """Create a mock bot."""
    return MagicMock()


@pytest.fixture
def mock_ctx():
    """Create a mock Discord context."""
    ctx = MagicMock()
    ctx.author = MagicMock()
    ctx.author.mention = "@TestUser"
    ctx.author.display_name = "TestUser"
    ctx.author.roles = []
    ctx.message = MagicMock()
    ctx.message.id = 12345
    ctx.guild = MagicMock()
    ctx.guild.name = "Test Guild"
    ctx.channel = MagicMock()
    ctx.channel.name = "test-channel"
    ctx.send = AsyncMock()

    # Mock typing() as async context manager
    typing_mock = MagicMock()
    typing_mock.__aenter__ = AsyncMock()
    typing_mock.__aexit__ = AsyncMock()
    ctx.typing = MagicMock(return_value=typing_mock)

    return ctx


@pytest.fixture
def ai_cog(mock_bot, mock_db, mock_logger):
    """Create an AICog instance for testing."""
    return AICog(mock_bot, mock_db, mock_logger)


async def call_ask(cog, ctx, question):
    """Call the ask_question command's underlying callback."""
    # Access the callback directly to bypass discord.py's command wrapper
    return await cog.ask_question.callback(cog, ctx, question=question)


class TestAskLockMechanism:
    """Tests for the !ask command lock mechanism."""

    @pytest.mark.asyncio
    async def test_lock_rejects_concurrent_request(self, ai_cog, mock_ctx):
        """Test that a second request is rejected while first is processing."""
        mock_response = OpenRouterResponse(
            text="Hello!",
            success=True,
            model_used="test-model",
        )

        async def slow_ask(*args, **kwargs):
            await asyncio.sleep(0.5)
            return mock_response

        mock_client = MagicMock()
        mock_client.ask_with_context = slow_ask
        mock_client.classify_query = AsyncMock(return_value=None)
        ai_cog._openrouter_client = mock_client

        with (
            patch(
                "strofkabot.cogs.ai.fetch_context_messages", new_callable=AsyncMock
            ) as mock_fetch,
            patch("strofkabot.cogs.ai.prepare_context", new_callable=AsyncMock) as mock_prepare,
            patch(
                "strofkabot.cogs.ai.extract_images_from_messages", new_callable=AsyncMock
            ) as mock_images,
            patch(
                "strofkabot.cogs.ai.extract_urls_from_messages", new_callable=AsyncMock
            ) as mock_urls,
        ):
            mock_fetch.return_value = []
            mock_prepare.return_value = []
            mock_images.return_value = []
            mock_urls.return_value = {}

            # Start first request
            first_ctx = mock_ctx
            first_ctx.send = AsyncMock()
            task1 = asyncio.create_task(call_ask(ai_cog, first_ctx, "What is 2+2?"))

            # Wait for first request to acquire lock
            await asyncio.sleep(0.1)

            # Second request with different context
            second_ctx = MagicMock()
            second_ctx.author = MagicMock()
            second_ctx.author.mention = "@SecondUser"
            second_ctx.send = AsyncMock()

            # Try second request while first is still processing
            await call_ask(ai_cog, second_ctx, "What is 3+3?")

            # Second request should get busy message
            second_ctx.send.assert_called_once()
            call_args = second_ctx.send.call_args[0][0]
            assert "jam duke shkruar o kar" in call_args
            assert "prit radhen" in call_args

            # Wait for first request to finish
            await task1

    @pytest.mark.asyncio
    async def test_busy_message_mentions_user(self, ai_cog, mock_ctx):
        """Test that the busy message mentions the requesting user."""

        async def slow_ask(*args, **kwargs):
            await asyncio.sleep(1)
            return OpenRouterResponse(text="Done", success=True, model_used="test")

        mock_client = MagicMock()
        mock_client.ask_with_context = slow_ask
        mock_client.classify_query = AsyncMock(return_value=None)
        ai_cog._openrouter_client = mock_client

        with (
            patch(
                "strofkabot.cogs.ai.fetch_context_messages", new_callable=AsyncMock
            ) as mock_fetch,
            patch("strofkabot.cogs.ai.prepare_context", new_callable=AsyncMock) as mock_prepare,
            patch(
                "strofkabot.cogs.ai.extract_images_from_messages", new_callable=AsyncMock
            ) as mock_images,
            patch(
                "strofkabot.cogs.ai.extract_urls_from_messages", new_callable=AsyncMock
            ) as mock_urls,
        ):
            mock_fetch.return_value = []
            mock_prepare.return_value = []
            mock_images.return_value = []
            mock_urls.return_value = {}

            # Start first request in background
            task1 = asyncio.create_task(call_ask(ai_cog, mock_ctx, "First"))
            await asyncio.sleep(0.1)

            # Second request should get busy message with user mention
            second_ctx = MagicMock()
            second_ctx.author = MagicMock()
            second_ctx.author.mention = "@BusyUser"
            second_ctx.send = AsyncMock()

            await call_ask(ai_cog, second_ctx, "Second")

            call_args = second_ctx.send.call_args[0][0]
            assert "@BusyUser" in call_args

            # Clean up
            task1.cancel()
            try:
                await task1
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_lock_released_after_request(self, ai_cog, mock_ctx):
        """Test that the lock is released after a request completes."""
        mock_response = OpenRouterResponse(
            text="Hello!",
            success=True,
            model_used="test-model",
        )

        mock_client = MagicMock()
        mock_client.ask_with_context = AsyncMock(return_value=mock_response)
        ai_cog._openrouter_client = mock_client

        with (
            patch(
                "strofkabot.cogs.ai.fetch_context_messages", new_callable=AsyncMock
            ) as mock_fetch,
            patch("strofkabot.cogs.ai.prepare_context", new_callable=AsyncMock) as mock_prepare,
            patch(
                "strofkabot.cogs.ai.extract_images_from_messages", new_callable=AsyncMock
            ) as mock_images,
            patch(
                "strofkabot.cogs.ai.extract_urls_from_messages", new_callable=AsyncMock
            ) as mock_urls,
        ):
            mock_fetch.return_value = []
            mock_prepare.return_value = []
            mock_images.return_value = []
            mock_urls.return_value = {}

            await call_ask(ai_cog, mock_ctx, "What is 2+2?")

            # Lock should be released
            assert not ai_cog._ask_lock.locked()

    @pytest.mark.asyncio
    async def test_lock_released_on_error(self, ai_cog, mock_ctx):
        """Test that the lock is released even when an API error occurs."""
        mock_response = OpenRouterResponse(
            text="",
            success=False,
            error_message="API Error",
        )

        mock_client = MagicMock()
        mock_client.ask_with_context = AsyncMock(return_value=mock_response)
        ai_cog._openrouter_client = mock_client

        with (
            patch(
                "strofkabot.cogs.ai.fetch_context_messages", new_callable=AsyncMock
            ) as mock_fetch,
            patch("strofkabot.cogs.ai.prepare_context", new_callable=AsyncMock) as mock_prepare,
            patch(
                "strofkabot.cogs.ai.extract_images_from_messages", new_callable=AsyncMock
            ) as mock_images,
            patch(
                "strofkabot.cogs.ai.extract_urls_from_messages", new_callable=AsyncMock
            ) as mock_urls,
        ):
            mock_fetch.return_value = []
            mock_prepare.return_value = []
            mock_images.return_value = []
            mock_urls.return_value = {}

            await call_ask(ai_cog, mock_ctx, "What is 2+2?")

            # Lock should still be released
            assert not ai_cog._ask_lock.locked()

    @pytest.mark.asyncio
    async def test_lock_released_on_exception(self, ai_cog, mock_ctx):
        """Test that the lock is released even when an exception is raised."""
        mock_client = MagicMock()
        mock_client.ask_with_context = AsyncMock(side_effect=Exception("Test error"))
        ai_cog._openrouter_client = mock_client

        with (
            patch(
                "strofkabot.cogs.ai.fetch_context_messages", new_callable=AsyncMock
            ) as mock_fetch,
            patch("strofkabot.cogs.ai.prepare_context", new_callable=AsyncMock) as mock_prepare,
            patch(
                "strofkabot.cogs.ai.extract_images_from_messages", new_callable=AsyncMock
            ) as mock_images,
            patch(
                "strofkabot.cogs.ai.extract_urls_from_messages", new_callable=AsyncMock
            ) as mock_urls,
        ):
            mock_fetch.return_value = []
            mock_prepare.return_value = []
            mock_images.return_value = []
            mock_urls.return_value = {}

            await call_ask(ai_cog, mock_ctx, "What is 2+2?")

            # Lock should still be released
            assert not ai_cog._ask_lock.locked()

    @pytest.mark.asyncio
    async def test_no_clients_returns_config_error(self, ai_cog, mock_ctx):
        """Test that missing clients returns config error."""
        ai_cog._openrouter_client = None
        ai_cog._gemini_client = None

        # Override the lazy initialization properties to return None
        with patch.object(type(ai_cog), "openrouter_client", property(lambda self: None)):
            with patch.object(type(ai_cog), "gemini_client", property(lambda self: None)):
                await call_ask(ai_cog, mock_ctx, "Test?")

                mock_ctx.send.assert_called_once()
                call_args = mock_ctx.send.call_args[0][0]
                assert "configured" in call_args.lower()

    @pytest.mark.asyncio
    async def test_empty_question_returns_error(self, ai_cog, mock_ctx):
        """Test that empty question returns error."""
        mock_client = MagicMock()
        ai_cog._openrouter_client = mock_client

        await call_ask(ai_cog, mock_ctx, "")

        mock_ctx.send.assert_called_once()
        # Lock should not be held for validation errors
        assert not ai_cog._ask_lock.locked()

    @pytest.mark.asyncio
    async def test_question_too_long_returns_error(self, ai_cog, mock_ctx):
        """Test that very long question returns error."""
        mock_client = MagicMock()
        ai_cog._openrouter_client = mock_client

        long_question = "x" * 25000
        await call_ask(ai_cog, mock_ctx, long_question)

        mock_ctx.send.assert_called_once()
        assert not ai_cog._ask_lock.locked()
