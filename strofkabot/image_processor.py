"""Image processing utilities for downloading and compressing Discord attachments."""

import io
import logging
from pathlib import Path

import aiohttp
from PIL import Image

from strofkabot.config import (
    IMAGE_MAX_SIZE_BYTES,
    IMAGE_QUALITY_MIN,
    IMAGE_QUALITY_START,
)

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".wmv", ".flv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".wma"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}


class ImageProcessor:
    """Handles image downloading, compression, and file type detection."""

    @staticmethod
    def is_video(filename: str) -> bool:
        """Check if file is a video based on extension."""
        return Path(filename).suffix.lower() in VIDEO_EXTENSIONS

    @staticmethod
    def is_audio(filename: str) -> bool:
        """Check if file is audio based on extension."""
        return Path(filename).suffix.lower() in AUDIO_EXTENSIONS

    @staticmethod
    def is_image(filename: str) -> bool:
        """Check if file is an image based on extension."""
        return Path(filename).suffix.lower() in IMAGE_EXTENSIONS

    async def process_attachment(
        self,
        url: str,
        attachment_id: int,
        filename: str,
        output_dir: Path,
    ) -> tuple[Path | None, int | None]:
        """
        Download and compress an image attachment.

        Args:
            url: URL to download the attachment from
            attachment_id: Discord attachment ID
            filename: Original filename
            output_dir: Directory to save the compressed image

        Returns:
            Tuple of (output_path, file_size) or (None, None) if failed
        """
        if not self.is_image(filename):
            logger.debug("Skipping non-image file: %s", filename)
            return None, None

        try:
            image_data = await self._download(url)
            if image_data is None:
                return None, None

            output_dir.mkdir(parents=True, exist_ok=True)

            if filename.lower().endswith(".gif"):
                output_path = output_dir / f"{attachment_id}.gif"
                output_path.write_bytes(image_data)
                file_size = len(image_data)
                logger.debug("Saved GIF without compression: %s (%d bytes)", output_path, file_size)
                return output_path, file_size

            compressed_data = self._compress_image(image_data)
            if compressed_data is None:
                return None, None

            output_path = output_dir / f"{attachment_id}.jpg"
            output_path.write_bytes(compressed_data)
            file_size = len(compressed_data)
            logger.debug("Saved compressed image: %s (%d bytes)", output_path, file_size)
            return output_path, file_size

        except Exception:
            logger.exception("Failed to process attachment %d", attachment_id)
            return None, None

    async def _download(self, url: str) -> bytes | None:
        """Download file from URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.warning("Failed to download %s: HTTP %d", url, response.status)
                        return None
                    return await response.read()
        except Exception:
            logger.exception("Failed to download %s", url)
            return None

    def _compress_image(self, image_data: bytes) -> bytes | None:
        """
        Compress image to JPEG, reducing quality until it fits within size limit.

        Returns compressed image data or None if compression failed.
        """
        try:
            image = Image.open(io.BytesIO(image_data))

            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")

            quality = IMAGE_QUALITY_START
            while quality >= IMAGE_QUALITY_MIN:
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=quality, optimize=True)
                compressed_size = buffer.tell()

                if compressed_size <= IMAGE_MAX_SIZE_BYTES:
                    logger.debug("Compressed to %d bytes at quality %d", compressed_size, quality)
                    return buffer.getvalue()

                quality -= 10

            logger.warning(
                "Could not compress image below %d bytes (got %d at quality %d)",
                IMAGE_MAX_SIZE_BYTES,
                compressed_size,
                IMAGE_QUALITY_MIN,
            )
            return buffer.getvalue()

        except Exception:
            logger.exception("Failed to compress image")
            return None
