"""Tests for ImageProcessor class."""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from strofkabot.image_processor import (
    AUDIO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    ImageProcessor,
)


class TestFileTypeDetection:
    """Tests for file type detection methods."""

    @pytest.fixture
    def processor(self):
        return ImageProcessor()

    @pytest.mark.parametrize(
        "filename",
        ["video.mp4", "video.MP4", "file.webm", "movie.mov", "clip.avi", "video.mkv"],
    )
    def test_is_video_returns_true_for_video_files(self, processor, filename):
        """Test that is_video returns True for video files."""
        assert processor.is_video(filename) is True

    @pytest.mark.parametrize(
        "filename",
        ["image.jpg", "document.pdf", "audio.mp3", "file.txt"],
    )
    def test_is_video_returns_false_for_non_video_files(self, processor, filename):
        """Test that is_video returns False for non-video files."""
        assert processor.is_video(filename) is False

    @pytest.mark.parametrize(
        "filename",
        ["song.mp3", "audio.MP3", "music.wav", "sound.ogg", "track.flac", "audio.m4a"],
    )
    def test_is_audio_returns_true_for_audio_files(self, processor, filename):
        """Test that is_audio returns True for audio files."""
        assert processor.is_audio(filename) is True

    @pytest.mark.parametrize(
        "filename",
        ["image.jpg", "document.pdf", "video.mp4", "file.txt"],
    )
    def test_is_audio_returns_false_for_non_audio_files(self, processor, filename):
        """Test that is_audio returns False for non-audio files."""
        assert processor.is_audio(filename) is False

    @pytest.mark.parametrize(
        "filename",
        ["photo.jpg", "image.JPEG", "pic.png", "animation.gif", "photo.webp", "img.bmp"],
    )
    def test_is_image_returns_true_for_image_files(self, processor, filename):
        """Test that is_image returns True for image files."""
        assert processor.is_image(filename) is True

    @pytest.mark.parametrize(
        "filename",
        ["video.mp4", "document.pdf", "audio.mp3", "file.txt"],
    )
    def test_is_image_returns_false_for_non_image_files(self, processor, filename):
        """Test that is_image returns False for non-image files."""
        assert processor.is_image(filename) is False

    def test_video_extensions_constant(self):
        """Test that VIDEO_EXTENSIONS contains expected values."""
        assert ".mp4" in VIDEO_EXTENSIONS
        assert ".webm" in VIDEO_EXTENSIONS
        assert ".mov" in VIDEO_EXTENSIONS

    def test_audio_extensions_constant(self):
        """Test that AUDIO_EXTENSIONS contains expected values."""
        assert ".mp3" in AUDIO_EXTENSIONS
        assert ".wav" in AUDIO_EXTENSIONS
        assert ".ogg" in AUDIO_EXTENSIONS

    def test_image_extensions_constant(self):
        """Test that IMAGE_EXTENSIONS contains expected values."""
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".png" in IMAGE_EXTENSIONS
        assert ".gif" in IMAGE_EXTENSIONS


class TestProcessAttachment:
    """Tests for process_attachment method."""

    @pytest.fixture
    def processor(self):
        return ImageProcessor()

    async def test_skips_non_image_files(self, processor, tmp_path):
        """Test that non-image files are skipped."""
        result = await processor.process_attachment(
            url="http://example.com/video.mp4",
            attachment_id=123,
            filename="video.mp4",
            output_dir=tmp_path,
        )
        assert result == (None, None)

    async def test_returns_none_on_download_failure(self, processor, tmp_path):
        """Test that download failure returns None."""
        with patch.object(processor, "_download", new_callable=AsyncMock) as mock_download:
            mock_download.return_value = None

            result = await processor.process_attachment(
                url="http://example.com/image.jpg",
                attachment_id=123,
                filename="image.jpg",
                output_dir=tmp_path,
            )

            assert result == (None, None)

    async def test_saves_gif_without_compression(self, processor, tmp_path):
        """Test that GIF files are saved without compression."""
        gif_data = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"

        with patch.object(processor, "_download", new_callable=AsyncMock) as mock_download:
            mock_download.return_value = gif_data

            result_path, result_size = await processor.process_attachment(
                url="http://example.com/animation.gif",
                attachment_id=456,
                filename="animation.gif",
                output_dir=tmp_path,
            )

            assert result_path == tmp_path / "456.gif"
            assert result_size == len(gif_data)
            assert result_path.exists()

    async def test_compresses_jpeg_images(self, processor, tmp_path):
        """Test that JPEG images are compressed."""
        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()

        with patch.object(processor, "_download", new_callable=AsyncMock) as mock_download:
            mock_download.return_value = image_data

            result_path, result_size = await processor.process_attachment(
                url="http://example.com/photo.jpg",
                attachment_id=789,
                filename="photo.jpg",
                output_dir=tmp_path,
            )

            assert result_path == tmp_path / "789.jpg"
            assert result_size > 0
            assert result_path.exists()

    async def test_returns_none_on_compression_failure(self, processor, tmp_path):
        """Test that compression failure returns None."""
        with (
            patch.object(processor, "_download", new_callable=AsyncMock) as mock_download,
            patch.object(processor, "_compress_image") as mock_compress,
        ):
            mock_download.return_value = b"fake image data"
            mock_compress.return_value = None

            result = await processor.process_attachment(
                url="http://example.com/image.jpg",
                attachment_id=123,
                filename="image.jpg",
                output_dir=tmp_path,
            )

            assert result == (None, None)

    async def test_creates_output_directory(self, processor, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        nested_dir = tmp_path / "nested" / "output"
        gif_data = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"

        with patch.object(processor, "_download", new_callable=AsyncMock) as mock_download:
            mock_download.return_value = gif_data

            await processor.process_attachment(
                url="http://example.com/animation.gif",
                attachment_id=123,
                filename="animation.gif",
                output_dir=nested_dir,
            )

            assert nested_dir.exists()

    async def test_handles_exception_gracefully(self, processor, tmp_path):
        """Test that exceptions are handled and return None."""
        with patch.object(processor, "_download", new_callable=AsyncMock) as mock_download:
            mock_download.side_effect = Exception("Network error")

            result = await processor.process_attachment(
                url="http://example.com/image.jpg",
                attachment_id=123,
                filename="image.jpg",
                output_dir=tmp_path,
            )

            assert result == (None, None)


class TestDownload:
    """Tests for _download method."""

    @pytest.fixture
    def processor(self):
        return ImageProcessor()

    async def test_successful_download(self, processor):
        """Test successful file download."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"image data")

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with patch("strofkabot.image_processor.aiohttp.ClientSession", return_value=mock_session):
            result = await processor._download("http://example.com/image.jpg")
            assert result == b"image data"

    async def test_download_returns_none_on_non_200_status(self, processor):
        """Test that non-200 status returns None."""
        mock_response = AsyncMock()
        mock_response.status = 404

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with patch("strofkabot.image_processor.aiohttp.ClientSession", return_value=mock_session):
            result = await processor._download("http://example.com/notfound.jpg")
            assert result is None

    async def test_download_returns_none_on_exception(self, processor):
        """Test that exceptions return None."""
        with patch("strofkabot.image_processor.aiohttp.ClientSession") as mock_session_class:
            mock_session_class.side_effect = Exception("Connection error")

            result = await processor._download("http://example.com/image.jpg")
            assert result is None


class TestCompressImage:
    """Tests for _compress_image method."""

    @pytest.fixture
    def processor(self):
        return ImageProcessor()

    def test_compresses_rgb_image(self, processor):
        """Test compression of RGB image."""
        img = Image.new("RGB", (100, 100), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = buffer.getvalue()

        result = processor._compress_image(image_data)

        assert result is not None
        # Verify it's a valid JPEG
        result_img = Image.open(io.BytesIO(result))
        assert result_img.format == "JPEG"

    def test_converts_rgba_to_rgb(self, processor):
        """Test that RGBA images are converted to RGB."""
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = buffer.getvalue()

        result = processor._compress_image(image_data)

        assert result is not None
        result_img = Image.open(io.BytesIO(result))
        assert result_img.mode == "RGB"

    def test_converts_palette_to_rgb(self, processor):
        """Test that palette mode images are converted to RGB."""
        img = Image.new("P", (100, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = buffer.getvalue()

        result = processor._compress_image(image_data)

        assert result is not None
        result_img = Image.open(io.BytesIO(result))
        assert result_img.mode == "RGB"

    def test_returns_none_on_invalid_image(self, processor):
        """Test that invalid image data returns None."""
        result = processor._compress_image(b"not an image")
        assert result is None

    def test_returns_compressed_data_even_if_large(self, processor):
        """Test that compression returns data even if it can't meet size limit."""
        # Create a large noisy image that won't compress well
        img = Image.new("RGB", (1000, 1000))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = buffer.getvalue()

        result = processor._compress_image(image_data)

        # Should still return something even if over limit
        assert result is not None
