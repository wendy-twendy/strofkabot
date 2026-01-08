"""Tests for the Gemini client module."""

import datetime
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestGeminiUsageTracker:
    """Tests for GeminiUsageTracker class."""

    @pytest.fixture
    def usage_file(self, tmp_path: Path) -> Path:
        """Provide a temporary usage file path."""
        return tmp_path / "gemini_usage.json"

    @pytest.fixture
    def tracker(self, usage_file: Path):
        """Create a GeminiUsageTracker with a temp file."""
        from strofkabot.gemini_client import GeminiUsageTracker

        return GeminiUsageTracker(usage_file=usage_file)

    def test_get_available_model_returns_first_model_when_empty(self, tracker):
        """Test that first model is returned when no usage recorded."""
        from strofkabot.config import GEMINI_MODELS

        model = tracker.get_available_model()
        assert model == GEMINI_MODELS[0]

    def test_increment_usage_creates_file(self, tracker, usage_file: Path):
        """Test that incrementing usage creates the usage file."""
        from strofkabot.config import GEMINI_MODELS

        tracker.increment_usage(GEMINI_MODELS[0])
        assert usage_file.exists()

        with open(usage_file, encoding="utf-8") as f:
            data = json.load(f)
        assert data["models"][GEMINI_MODELS[0]] == 1

    def test_increment_usage_accumulates(self, tracker):
        """Test that usage accumulates correctly."""
        from strofkabot.config import GEMINI_MODELS

        model = GEMINI_MODELS[0]
        tracker.increment_usage(model)
        tracker.increment_usage(model)
        tracker.increment_usage(model)

        remaining = tracker.get_remaining_requests()
        assert remaining[model] == 17  # 20 - 3

    def test_get_available_model_skips_exhausted(self, tracker, usage_file: Path):
        """Test that exhausted models are skipped."""
        from strofkabot.config import GEMINI_MODELS, GEMINI_RPD_LIMIT

        today = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
        data = {
            "date": today,
            "models": {GEMINI_MODELS[0]: GEMINI_RPD_LIMIT},
        }
        with open(usage_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        model = tracker.get_available_model()
        assert model == GEMINI_MODELS[1]

    def test_get_available_model_returns_none_when_all_exhausted(self, tracker, usage_file: Path):
        """Test that None is returned when all models are exhausted."""
        from strofkabot.config import GEMINI_MODELS, GEMINI_RPD_LIMIT

        today = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
        data = {
            "date": today,
            "models": {model: GEMINI_RPD_LIMIT for model in GEMINI_MODELS},
        }
        with open(usage_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        model = tracker.get_available_model()
        assert model is None

    def test_resets_on_new_day(self, tracker, usage_file: Path):
        """Test that usage resets on a new day."""
        from strofkabot.config import GEMINI_MODELS, GEMINI_RPD_LIMIT

        yesterday = "2020-01-01"
        data = {
            "date": yesterday,
            "models": {model: GEMINI_RPD_LIMIT for model in GEMINI_MODELS},
        }
        with open(usage_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        model = tracker.get_available_model()
        assert model == GEMINI_MODELS[0]

    def test_get_remaining_requests(self, tracker):
        """Test getting remaining requests per model."""
        from strofkabot.config import GEMINI_MODELS, GEMINI_RPD_LIMIT

        remaining = tracker.get_remaining_requests()

        for model in GEMINI_MODELS:
            assert remaining[model] == GEMINI_RPD_LIMIT

    def test_handles_corrupt_json(self, tracker, usage_file: Path):
        """Test that corrupt JSON is handled gracefully."""
        from strofkabot.config import GEMINI_MODELS

        with open(usage_file, "w", encoding="utf-8") as f:
            f.write("not valid json{{{")

        model = tracker.get_available_model()
        assert model == GEMINI_MODELS[0]


class TestGeminiClient:
    """Tests for GeminiClient class."""

    def test_init_without_api_key_raises(self):
        """Test that missing API key raises ValueError."""
        from strofkabot.gemini_client import GeminiClient

        with patch.dict("os.environ", {}, clear=True):
            with patch("os.getenv", return_value=None):
                with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                    GeminiClient()

    def test_init_with_api_key_succeeds(self):
        """Test successful initialization with API key."""
        from strofkabot.gemini_client import GeminiClient

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("google.genai.Client"):
                client = GeminiClient()
                assert client is not None

    @pytest.mark.asyncio
    async def test_ask_with_context_success(self, tmp_path: Path):
        """Test successful API call."""
        from strofkabot.gemini_client import GeminiClient, GeminiUsageTracker

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("google.genai.Client") as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                mock_response = Mock()
                mock_response.text = "Test response"
                mock_client.models.generate_content.return_value = mock_response

                client = GeminiClient()
                client._usage_tracker = GeminiUsageTracker(usage_file=tmp_path / "usage.json")
                response = await client.ask_with_context(
                    question="Test question",
                    system_prompt="Test prompt",
                    context_messages=[],
                )

                assert response.success is True
                assert response.text == "Test response"

    @pytest.mark.asyncio
    async def test_ask_with_context_api_error(self, tmp_path: Path):
        """Test API error handling."""
        from strofkabot.gemini_client import GeminiClient, GeminiUsageTracker

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("google.genai.Client") as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                mock_client.models.generate_content.side_effect = Exception("API Error")

                client = GeminiClient()
                client._usage_tracker = GeminiUsageTracker(usage_file=tmp_path / "usage.json")
                response = await client.ask_with_context(
                    question="Test",
                    system_prompt="Test",
                    context_messages=[],
                )

                assert response.success is False
                assert "API Error" in response.error_message

    @pytest.mark.asyncio
    async def test_ask_with_context_no_text_output(self, tmp_path: Path):
        """Test handling when API returns no text output."""
        from strofkabot.gemini_client import GeminiClient, GeminiUsageTracker

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("google.genai.Client") as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                mock_response = Mock()
                mock_response.text = None  # No text in response
                mock_client.models.generate_content.return_value = mock_response

                client = GeminiClient()
                client._usage_tracker = GeminiUsageTracker(usage_file=tmp_path / "usage.json")
                response = await client.ask_with_context(
                    question="Test",
                    system_prompt="Test",
                    context_messages=[],
                )

                assert response.success is False
                assert "No text response" in response.error_message

    @pytest.mark.asyncio
    async def test_ask_with_context_all_models_exhausted(self, tmp_path: Path):
        """Test when all models are exhausted for the day."""
        from strofkabot.config import GEMINI_MODELS, GEMINI_RPD_LIMIT
        from strofkabot.gemini_client import GeminiClient, GeminiUsageTracker

        usage_file = tmp_path / "usage.json"
        today = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
        data = {
            "date": today,
            "models": {model: GEMINI_RPD_LIMIT for model in GEMINI_MODELS},
        }
        with open(usage_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("google.genai.Client"):
                client = GeminiClient()
                # Replace the usage tracker with one pointing to our test file
                client._usage_tracker = GeminiUsageTracker(usage_file=usage_file)

                response = await client.ask_with_context(
                    question="Test",
                    system_prompt="Test",
                    context_messages=[],
                )

                assert response.success is False
                assert "Daily limit reached" in response.error_message


class TestContextFormatting:
    """Tests for context formatting."""

    @pytest.fixture
    def client(self, tmp_path: Path):
        """Create a GeminiClient for testing formatting methods."""
        from strofkabot.gemini_client import GeminiClient

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("google.genai.Client"):
                with patch(
                    "strofkabot.gemini_client.GEMINI_USAGE_FILE",
                    tmp_path / "usage.json",
                ):
                    return GeminiClient()

    def test_format_context_with_messages(self, client):
        """Test formatting of context messages."""
        context = [
            {"author": "User1", "content": "Hello", "timestamp": "10:30"},
            {"author": "User2", "content": "Hi there", "timestamp": "10:31"},
        ]

        formatted = client._format_context("System prompt", context)

        assert "System prompt" in formatted
        assert "User1" in formatted
        assert "Hello" in formatted
        assert "User2" in formatted
        assert "10:30" in formatted

    def test_format_context_with_reply(self, client):
        """Test formatting of context with reply info."""
        context = [
            {
                "author": "User1",
                "content": "Response",
                "timestamp": "10:30",
                "reply_to_author": "User2",
            },
        ]

        formatted = client._format_context("System", context)

        assert "replying to User2" in formatted

    def test_format_context_with_images(self, client):
        """Test formatting of context with image count."""
        context = [
            {
                "author": "User1",
                "content": "Check this",
                "timestamp": "10:30",
                "image_count": 2,
            },
        ]

        formatted = client._format_context("System", context)

        assert "[+2 image(s)]" in formatted

    def test_format_context_empty_messages(self, client):
        """Test formatting with no context messages."""
        formatted = client._format_context("System prompt only", [])

        assert "System prompt only" in formatted
        assert "Conversation History" not in formatted

    def test_build_contents_with_images(self, client):
        """Test building contents with images."""
        import base64

        # Use valid base64 data
        test_data = base64.b64encode(b"test image data").decode()
        images = [
            {"data": test_data, "mime_type": "image/jpeg"},
        ]

        with patch("strofkabot.gemini_client.types") as mock_types:
            mock_text_part = Mock()
            mock_image_part = Mock()
            mock_types.Part.from_text.return_value = mock_text_part
            mock_types.Part.from_bytes.return_value = mock_image_part

            contents = client._build_contents(
                question="What is this?",
                system_prompt="You are helpful",
                context_messages=[],
                images=images,
            )

            # Should have: system+context text, image, question text
            assert len(contents) == 3
            assert mock_types.Part.from_text.call_count == 2
            assert mock_types.Part.from_bytes.call_count == 1

    def test_build_contents_without_images(self, client):
        """Test building contents without images."""
        with patch("strofkabot.gemini_client.types") as mock_types:
            mock_text_part = Mock()
            mock_types.Part.from_text.return_value = mock_text_part

            contents = client._build_contents(
                question="Hello?",
                system_prompt="System",
                context_messages=[],
                images=None,
            )

            # Should have: system+context text, question text
            assert len(contents) == 2
            assert mock_types.Part.from_text.call_count == 2
