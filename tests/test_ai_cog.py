"""Tests for AICog class."""

import datetime
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strofkabot.cogs.ai import AICog


class TestAICogInit:
    """Tests for AICog initialization."""

    def test_init_stores_dependencies(self):
        """Test that all dependencies are stored correctly."""
        bot = MagicMock()
        db = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.cogs.ai.load_nicknames", return_value={}):
            cog = AICog(bot, db, logger)

        assert cog.bot is bot
        assert cog.db is db
        assert cog.logger is logger

    def test_init_sets_clients_to_none(self):
        """Test that AI clients are initially None."""
        bot = MagicMock()
        db = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.cogs.ai.load_nicknames", return_value={}):
            cog = AICog(bot, db, logger)

        assert cog._gemini_client is None
        assert cog._openrouter_client is None

    def test_init_loads_nicknames(self):
        """Test that nicknames are loaded."""
        bot = MagicMock()
        db = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.cogs.ai.load_nicknames", return_value={123: ["Nick"]}) as mock_load:
            cog = AICog(bot, db, logger)
            mock_load.assert_called_once()
            assert cog._nicknames == {123: ["Nick"]}


class TestOpenRouterClientProperty:
    """Tests for openrouter_client lazy initialization."""

    @pytest.fixture
    def cog(self):
        bot = MagicMock()
        db = MagicMock()
        logger = logging.getLogger("test")
        with patch("strofkabot.cogs.ai.load_nicknames", return_value={}):
            return AICog(bot, db, logger)

    def test_initializes_client_on_first_access(self, cog):
        """Test that client is initialized on first access."""
        with patch("strofkabot.cogs.ai.OpenRouterClient") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            result = cog.openrouter_client

            MockClient.assert_called_once()
            assert result is mock_client

    def test_returns_cached_client_on_subsequent_access(self, cog):
        """Test that cached client is returned on subsequent access."""
        with patch("strofkabot.cogs.ai.OpenRouterClient") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            # First access
            result1 = cog.openrouter_client
            # Second access
            result2 = cog.openrouter_client

            # Should only be called once
            MockClient.assert_called_once()
            assert result1 is result2

    def test_returns_none_on_value_error(self, cog):
        """Test that None is returned when client raises ValueError."""
        with patch("strofkabot.cogs.ai.OpenRouterClient") as MockClient:
            MockClient.side_effect = ValueError("API key not configured")

            result = cog.openrouter_client

            assert result is None


class TestGeminiClientProperty:
    """Tests for gemini_client lazy initialization."""

    @pytest.fixture
    def cog(self):
        bot = MagicMock()
        db = MagicMock()
        logger = logging.getLogger("test")
        with patch("strofkabot.cogs.ai.load_nicknames", return_value={}):
            return AICog(bot, db, logger)

    def test_initializes_client_on_first_access(self, cog):
        """Test that client is initialized on first access."""
        with patch("strofkabot.cogs.ai.GeminiClient") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            result = cog.gemini_client

            MockClient.assert_called_once()
            assert result is mock_client

    def test_returns_cached_client_on_subsequent_access(self, cog):
        """Test that cached client is returned on subsequent access."""
        with patch("strofkabot.cogs.ai.GeminiClient") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            result1 = cog.gemini_client
            result2 = cog.gemini_client

            MockClient.assert_called_once()
            assert result1 is result2

    def test_returns_none_on_value_error(self, cog):
        """Test that None is returned when client raises ValueError."""
        with patch("strofkabot.cogs.ai.GeminiClient") as MockClient:
            MockClient.side_effect = ValueError("API key not configured")

            result = cog.gemini_client

            assert result is None


class TestAskQuestion:
    """Tests for ask_question command."""

    @pytest.fixture
    def cog(self):
        bot = MagicMock()
        db = MagicMock()
        logger = logging.getLogger("test")
        with patch("strofkabot.cogs.ai.load_nicknames", return_value={}):
            cog = AICog(bot, db, logger)
            # Manually set clients to avoid lazy init issues
            cog._openrouter_client = None
            cog._gemini_client = None
            return cog

    @pytest.fixture
    def mock_ctx(self):
        ctx = MagicMock()
        ctx.author = MagicMock()
        ctx.author.id = 12345
        ctx.author.display_name = "TestUser"
        ctx.author.mention = "@TestUser"
        ctx.message = MagicMock()
        ctx.message.id = 100
        ctx.message.attachments = []
        ctx.message.guild = MagicMock()
        ctx.message.guild.name = "Test Guild"
        ctx.channel = MagicMock()
        ctx.channel.name = "test-channel"
        ctx.channel.send = AsyncMock()
        ctx.channel.typing = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(), __aexit__=AsyncMock())
        )
        ctx.message.channel = ctx.channel
        ctx.message.author = ctx.author
        ctx.guild = MagicMock()
        ctx.guild.name = "Test Guild"
        ctx.send = AsyncMock()
        ctx.typing = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(), __aexit__=AsyncMock())
        )
        return ctx

    async def test_rejects_when_lock_held(self, cog, mock_ctx):
        """Test that concurrent requests in same channel are rejected."""
        # Acquire the per-channel lock
        channel_lock = cog._get_channel_lock(mock_ctx.channel.id)
        await channel_lock.acquire()

        try:
            await cog.ask_question.callback(cog, mock_ctx, question="Test question")
            mock_ctx.send.assert_called_once()
            assert "prit radhen" in mock_ctx.send.call_args[0][0]
        finally:
            channel_lock.release()

    async def test_rejects_empty_question(self, cog, mock_ctx):
        """Test that empty question is rejected."""
        await cog.ask_question.callback(cog, mock_ctx, question="")

        mock_ctx.send.assert_called_once()
        assert "!ask" in mock_ctx.send.call_args[0][0]

    async def test_rejects_whitespace_only_question(self, cog, mock_ctx):
        """Test that whitespace-only question is rejected."""
        await cog.ask_question.callback(cog, mock_ctx, question="   ")

        mock_ctx.send.assert_called_once()
        assert "!ask" in mock_ctx.send.call_args[0][0]

    async def test_rejects_too_long_question(self, cog, mock_ctx):
        """Test that question over 20000 chars is rejected."""
        long_question = "A" * 20001

        await cog.ask_question.callback(cog, mock_ctx, question=long_question)

        mock_ctx.send.assert_called_once()
        assert "20,000" in mock_ctx.send.call_args[0][0]

    async def test_rejects_when_no_clients_available(self, cog, mock_ctx):
        """Test that request is rejected when no AI clients are available."""
        cog._openrouter_client = None
        cog._gemini_client = None

        # Mock the properties to return None
        with (
            patch.object(type(cog), "openrouter_client", property(lambda self: None)),
            patch.object(type(cog), "gemini_client", property(lambda self: None)),
        ):
            await cog.ask_question.callback(cog, mock_ctx, question="Test question")

        mock_ctx.send.assert_called_once()
        assert "configured" in mock_ctx.send.call_args[0][0].lower()

    async def test_successful_response_with_openrouter(self, cog, mock_ctx):
        """Test successful response using OpenRouter."""
        mock_openrouter = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.text = "This is the AI response"
        mock_response.model_used = "test-model"
        mock_response.search_used = False
        mock_response.thinking_used = False
        mock_openrouter.ask_with_context = AsyncMock(return_value=mock_response)
        mock_openrouter.classify_query = AsyncMock(return_value=None)

        with (
            patch.object(type(cog), "openrouter_client", property(lambda self: mock_openrouter)),
            patch.object(type(cog), "gemini_client", property(lambda self: None)),
            patch(
                "strofkabot.cogs.ai.fetch_context_messages", new_callable=AsyncMock, return_value=[]
            ),
            patch("strofkabot.cogs.ai.prepare_context", new_callable=AsyncMock, return_value=[]),
            patch(
                "strofkabot.cogs.ai.extract_images_from_messages",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "strofkabot.cogs.ai.extract_urls_from_messages",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            await cog.ask_question.callback(cog, mock_ctx, question="Test question")

        # Should send the response via channel (from _handle_ask)
        mock_ctx.channel.send.assert_called_with("This is the AI response")

    async def test_falls_back_to_gemini_on_openrouter_failure(self, cog, mock_ctx):
        """Test that Gemini is used as fallback when OpenRouter fails."""
        mock_openrouter = MagicMock()
        mock_openrouter_response = MagicMock()
        mock_openrouter_response.success = False
        mock_openrouter_response.error_message = "OpenRouter error"
        mock_openrouter.ask_with_context = AsyncMock(return_value=mock_openrouter_response)
        mock_openrouter.classify_query = AsyncMock(return_value=None)

        mock_gemini = MagicMock()
        mock_gemini_response = MagicMock()
        mock_gemini_response.success = True
        mock_gemini_response.text = "Gemini response"
        mock_gemini_response.model_used = "gemini-model"
        mock_gemini.ask_with_context = AsyncMock(return_value=mock_gemini_response)

        with (
            patch.object(type(cog), "openrouter_client", property(lambda self: mock_openrouter)),
            patch.object(type(cog), "gemini_client", property(lambda self: mock_gemini)),
            patch(
                "strofkabot.cogs.ai.fetch_context_messages", new_callable=AsyncMock, return_value=[]
            ),
            patch("strofkabot.cogs.ai.prepare_context", new_callable=AsyncMock, return_value=[]),
            patch(
                "strofkabot.cogs.ai.extract_images_from_messages",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "strofkabot.cogs.ai.extract_urls_from_messages",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            await cog.ask_question.callback(cog, mock_ctx, question="Test question")

        mock_ctx.channel.send.assert_called_with("Gemini response")

    async def test_handles_rate_limit_error(self, cog, mock_ctx):
        """Test that rate limit error is handled."""
        mock_openrouter = MagicMock()
        mock_response = MagicMock()
        mock_response.success = False
        mock_response.error_message = "rate limit exceeded"
        mock_openrouter.ask_with_context = AsyncMock(return_value=mock_response)
        mock_openrouter.classify_query = AsyncMock(return_value=None)

        # Gemini also fails with rate limit so we get the rate_limit response
        mock_gemini = MagicMock()
        mock_gemini_response = MagicMock()
        mock_gemini_response.success = False
        mock_gemini_response.error_message = "rate limit exceeded"
        mock_gemini.ask_with_context = AsyncMock(return_value=mock_gemini_response)

        with (
            patch.object(type(cog), "openrouter_client", property(lambda self: mock_openrouter)),
            patch.object(type(cog), "gemini_client", property(lambda self: mock_gemini)),
            patch(
                "strofkabot.cogs.ai.fetch_context_messages", new_callable=AsyncMock, return_value=[]
            ),
            patch("strofkabot.cogs.ai.prepare_context", new_callable=AsyncMock, return_value=[]),
            patch(
                "strofkabot.cogs.ai.extract_images_from_messages",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "strofkabot.cogs.ai.extract_urls_from_messages",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            await cog.ask_question.callback(cog, mock_ctx, question="Test question")

        assert "wait" in mock_ctx.channel.send.call_args[0][0].lower()

    async def test_handles_unexpected_exception(self, cog, mock_ctx):
        """Test that unexpected exceptions are handled."""
        mock_openrouter = MagicMock()
        mock_openrouter.classify_query = AsyncMock(side_effect=Exception("Unexpected error"))

        with (
            patch.object(type(cog), "openrouter_client", property(lambda self: mock_openrouter)),
            patch.object(type(cog), "gemini_client", property(lambda self: None)),
            patch(
                "strofkabot.cogs.ai.fetch_context_messages", new_callable=AsyncMock, return_value=[]
            ),
            patch("strofkabot.cogs.ai.prepare_context", new_callable=AsyncMock, return_value=[]),
            patch(
                "strofkabot.cogs.ai.extract_images_from_messages",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "strofkabot.cogs.ai.extract_urls_from_messages",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            await cog.ask_question.callback(cog, mock_ctx, question="Test question")

        # Should send an error response via channel
        mock_ctx.channel.send.assert_called()

    async def test_includes_images_from_recent_messages(self, cog, mock_ctx):
        """Test that images from recent messages are included."""
        mock_openrouter = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.text = "Response"
        mock_response.model_used = "test-model"
        mock_response.search_used = False
        mock_response.thinking_used = False
        mock_openrouter.ask_with_context = AsyncMock(return_value=mock_response)
        mock_openrouter.classify_query = AsyncMock(return_value=None)

        mock_images = [{"data": "base64", "mime_type": "image/jpeg"}]

        with (
            patch.object(type(cog), "openrouter_client", property(lambda self: mock_openrouter)),
            patch.object(type(cog), "gemini_client", property(lambda self: None)),
            patch(
                "strofkabot.cogs.ai.fetch_context_messages",
                new_callable=AsyncMock,
                return_value=[MagicMock(), MagicMock()],
            ),
            patch("strofkabot.cogs.ai.prepare_context", new_callable=AsyncMock, return_value=[]),
            patch(
                "strofkabot.cogs.ai.extract_images_from_messages",
                new_callable=AsyncMock,
                return_value=mock_images,
            ),
            patch(
                "strofkabot.cogs.ai.extract_urls_from_messages",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            await cog.ask_question.callback(cog, mock_ctx, question="Test question")

        # Check that images were passed to the client
        call_args = mock_openrouter.ask_with_context.call_args
        assert call_args[1]["images"] == mock_images
        # Also verify response was sent via channel
        mock_ctx.channel.send.assert_called_with("Response")


class TestOnMessageMention:
    """Tests for on_message bot mention handler."""

    @pytest.fixture
    def cog(self):
        bot = MagicMock()
        bot.user = MagicMock()
        bot.user.id = 99999
        db = MagicMock()
        logger = logging.getLogger("test")
        with patch("strofkabot.cogs.ai.load_nicknames", return_value={}):
            cog = AICog(bot, db, logger)
            cog._openrouter_client = None
            cog._gemini_client = None
            return cog

    @pytest.fixture
    def mock_message(self, cog):
        message = MagicMock()
        message.author = MagicMock()
        message.author.id = 12345
        message.author.display_name = "TestUser"
        message.author.bot = False
        message.author.mention = "@TestUser"
        message.id = 100
        message.attachments = []
        message.mentions = [cog.bot.user]
        message.content = f"<@{cog.bot.user.id}> What is the weather?"
        message.guild = MagicMock()
        message.guild.name = "Test Guild"
        message.channel = MagicMock()
        message.channel.name = "test-channel"
        message.channel.send = AsyncMock()
        message.channel.typing = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(), __aexit__=AsyncMock())
        )
        return message

    async def test_ignores_bot_messages(self, cog, mock_message):
        """Test that bot messages are ignored."""
        mock_message.author.bot = True

        await cog.on_message(mock_message)

        mock_message.channel.send.assert_not_called()

    async def test_ignores_messages_without_mention(self, cog, mock_message):
        """Test that messages without bot mention are ignored."""
        mock_message.mentions = []

        await cog.on_message(mock_message)

        mock_message.channel.send.assert_not_called()

    async def test_rejects_when_lock_held(self, cog, mock_message):
        """Test that concurrent mention requests in same channel are rejected."""
        channel_lock = cog._get_channel_lock(mock_message.channel.id)
        await channel_lock.acquire()

        try:
            await cog.on_message(mock_message)
            mock_message.channel.send.assert_called_once()
            assert "prit radhen" in mock_message.channel.send.call_args[0][0]
        finally:
            channel_lock.release()

    async def test_rejects_empty_question_after_mention(self, cog, mock_message):
        """Test that mention with no question text is rejected."""
        mock_message.content = f"<@{cog.bot.user.id}>"

        await cog.on_message(mock_message)

        mock_message.channel.send.assert_called_once()
        assert "!ask" in mock_message.channel.send.call_args[0][0]

    async def test_extracts_question_from_mention(self, cog, mock_message):
        """Test that question is correctly extracted from mention."""
        mock_openrouter = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.text = "The weather is sunny"
        mock_response.model_used = "test-model"
        mock_response.search_used = False
        mock_response.thinking_used = False
        mock_openrouter.ask_with_context = AsyncMock(return_value=mock_response)
        mock_openrouter.classify_query = AsyncMock(return_value=None)

        with (
            patch.object(type(cog), "openrouter_client", property(lambda self: mock_openrouter)),
            patch.object(type(cog), "gemini_client", property(lambda self: None)),
            patch(
                "strofkabot.cogs.ai.fetch_context_messages", new_callable=AsyncMock, return_value=[]
            ),
            patch("strofkabot.cogs.ai.prepare_context", new_callable=AsyncMock, return_value=[]),
            patch(
                "strofkabot.cogs.ai.extract_images_from_messages",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "strofkabot.cogs.ai.extract_urls_from_messages",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            await cog.on_message(mock_message)

        # Verify response was sent
        mock_message.channel.send.assert_called_with("The weather is sunny")
        # Verify correct question was extracted (without mention)
        call_args = mock_openrouter.ask_with_context.call_args
        assert call_args[1]["question"] == "What is the weather?"

    async def test_handles_nickname_mention_format(self, cog, mock_message):
        """Test that nickname mention format is also handled."""
        mock_message.content = f"<@!{cog.bot.user.id}> Hello bot"
        mock_openrouter = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.text = "Hello!"
        mock_response.model_used = "test-model"
        mock_response.search_used = False
        mock_response.thinking_used = False
        mock_openrouter.ask_with_context = AsyncMock(return_value=mock_response)
        mock_openrouter.classify_query = AsyncMock(return_value=None)

        with (
            patch.object(type(cog), "openrouter_client", property(lambda self: mock_openrouter)),
            patch.object(type(cog), "gemini_client", property(lambda self: None)),
            patch(
                "strofkabot.cogs.ai.fetch_context_messages", new_callable=AsyncMock, return_value=[]
            ),
            patch("strofkabot.cogs.ai.prepare_context", new_callable=AsyncMock, return_value=[]),
            patch(
                "strofkabot.cogs.ai.extract_images_from_messages",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "strofkabot.cogs.ai.extract_urls_from_messages",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            await cog.on_message(mock_message)

        # Verify correct question was extracted
        call_args = mock_openrouter.ask_with_context.call_args
        assert call_args[1]["question"] == "Hello bot"


class TestMakePrediction:
    """Tests for make_prediction command."""

    @pytest.fixture
    def cog(self):
        bot = MagicMock()
        db = MagicMock()
        db.add_prediction = AsyncMock(return_value=1)
        logger = logging.getLogger("test")
        with patch("strofkabot.cogs.ai.load_nicknames", return_value={}):
            return AICog(bot, db, logger)

    @pytest.fixture
    def mock_ctx(self):
        ctx = MagicMock()
        ctx.author = MagicMock()
        ctx.author.id = 12345
        ctx.author.display_name = "TestUser"
        ctx.channel = MagicMock()
        ctx.channel.id = 67890
        ctx.send = AsyncMock()
        return ctx

    async def test_shows_usage_for_empty_args(self, cog, mock_ctx):
        """Test that usage is shown for empty args."""
        await cog.make_prediction.callback(cog, mock_ctx, args="")

        mock_ctx.send.assert_called_once()
        assert "Usage" in mock_ctx.send.call_args[0][0]

    async def test_rejects_unparseable_date(self, cog, mock_ctx):
        """Test that unparseable date is rejected."""
        with patch("strofkabot.cogs.ai.parse_prediction_date", return_value=(None, "")):
            await cog.make_prediction.callback(cog, mock_ctx, args="gibberish some text")

        assert "couldn't understand" in mock_ctx.send.call_args[0][0].lower()

    async def test_rejects_empty_prediction_text(self, cog, mock_ctx):
        """Test that empty prediction text is rejected."""
        future_date = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=30)

        with patch("strofkabot.cogs.ai.parse_prediction_date", return_value=(future_date, "   ")):
            await cog.make_prediction.callback(cog, mock_ctx, args="tomorrow")

        assert "prediction text" in mock_ctx.send.call_args[0][0].lower()

    async def test_rejects_past_date(self, cog, mock_ctx):
        """Test that past date is rejected."""
        past_date = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=1)

        with patch(
            "strofkabot.cogs.ai.parse_prediction_date", return_value=(past_date, "Some prediction")
        ):
            await cog.make_prediction.callback(cog, mock_ctx, args="yesterday Some prediction")

        assert "past" in mock_ctx.send.call_args[0][0].lower()

    async def test_rejects_date_too_far_in_future(self, cog, mock_ctx):
        """Test that date more than 5 years in future is rejected."""
        far_future = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=365 * 6)

        with patch(
            "strofkabot.cogs.ai.parse_prediction_date", return_value=(far_future, "Some prediction")
        ):
            await cog.make_prediction.callback(cog, mock_ctx, args="01-01-2030 Some prediction")

        assert "5 years" in mock_ctx.send.call_args[0][0].lower()

    async def test_stores_valid_prediction(self, cog, mock_ctx):
        """Test that valid prediction is stored."""
        future_date = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=30)

        with patch(
            "strofkabot.cogs.ai.parse_prediction_date", return_value=(future_date, "My prediction")
        ):
            await cog.make_prediction.callback(cog, mock_ctx, args="tomorrow My prediction")

        cog.db.add_prediction.assert_called_once()
        call_args = cog.db.add_prediction.call_args[1]
        assert call_args["author_id"] == 12345
        assert call_args["prediction_text"] == "My prediction"

    async def test_sends_confirmation_embed(self, cog, mock_ctx):
        """Test that confirmation embed is sent."""
        future_date = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=30)

        with patch(
            "strofkabot.cogs.ai.parse_prediction_date", return_value=(future_date, "My prediction")
        ):
            await cog.make_prediction.callback(cog, mock_ctx, args="tomorrow My prediction")

        # Check that embed was sent
        call_args = mock_ctx.send.call_args
        assert "embed" in call_args[1]
        embed = call_args[1]["embed"]
        assert "Prediction Recorded" in embed.title

    async def test_handles_database_error(self, cog, mock_ctx):
        """Test that database errors are handled."""
        future_date = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=30)
        cog.db.add_prediction = AsyncMock(side_effect=Exception("DB error"))

        with patch(
            "strofkabot.cogs.ai.parse_prediction_date", return_value=(future_date, "My prediction")
        ):
            await cog.make_prediction.callback(cog, mock_ctx, args="tomorrow My prediction")

        assert "error" in mock_ctx.send.call_args[0][0].lower()
