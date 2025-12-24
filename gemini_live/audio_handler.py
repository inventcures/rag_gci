"""
Audio Handler for Gemini Live API

Handles conversion between various audio formats:
- WhatsApp audio (OGG, MP3) -> PCM for Gemini input
- PCM from Gemini -> MP3/OGG for playback

Gemini Live API Requirements:
- Input:  16-bit PCM, 16kHz, mono, little-endian
- Output: 16-bit PCM, 24kHz, mono, little-endian
"""

import io
import logging
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    AUDIO_CHANNELS,
    DEFAULT_CHUNK_SIZE,
)

logger = logging.getLogger(__name__)


class AudioHandlerError(Exception):
    """Exception raised for audio processing errors."""
    pass


class AudioHandler:
    """
    Audio format conversion for Gemini Live API.

    Uses FFmpeg for format conversion and numpy for resampling.

    Input Requirements (to Gemini):
        - Format: Raw PCM
        - Sample Rate: 16kHz
        - Bit Depth: 16-bit (signed)
        - Channels: Mono
        - Byte Order: Little-endian

    Output Format (from Gemini):
        - Format: Raw PCM
        - Sample Rate: 24kHz
        - Bit Depth: 16-bit (signed)
        - Channels: Mono
        - Byte Order: Little-endian
    """

    # Supported input formats for conversion
    SUPPORTED_INPUT_FORMATS = ["ogg", "mp3", "wav", "m4a", "webm", "flac"]

    # Supported output formats
    SUPPORTED_OUTPUT_FORMATS = ["mp3", "ogg", "wav"]

    def __init__(self):
        """Initialize AudioHandler and verify FFmpeg is available."""
        self.ffmpeg_path = self._find_ffmpeg()
        if not self.ffmpeg_path:
            logger.warning(
                "FFmpeg not found. Audio conversion will fail. "
                "Install FFmpeg: brew install ffmpeg (macOS) or "
                "apt-get install ffmpeg (Linux)"
            )

    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFmpeg executable path."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            logger.debug(f"Found FFmpeg at: {ffmpeg}")
            return ffmpeg

        # Try common paths
        common_paths = [
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/opt/homebrew/bin/ffmpeg",
        ]
        for path in common_paths:
            if Path(path).exists():
                logger.debug(f"Found FFmpeg at: {path}")
                return path

        return None

    def convert_to_pcm(
        self,
        audio_data: bytes,
        input_format: str = "ogg",
        sample_rate: int = INPUT_SAMPLE_RATE
    ) -> bytes:
        """
        Convert audio to PCM format for Gemini Live API input.

        Args:
            audio_data: Input audio bytes in source format
            input_format: Source format (ogg, mp3, wav, m4a, webm)
            sample_rate: Target sample rate (default: 16kHz)

        Returns:
            Raw PCM bytes (16-bit, mono, little-endian)

        Raises:
            AudioHandlerError: If conversion fails
        """
        if not self.ffmpeg_path:
            raise AudioHandlerError("FFmpeg not available for audio conversion")

        if input_format.lower() not in self.SUPPORTED_INPUT_FORMATS:
            raise AudioHandlerError(
                f"Unsupported input format: {input_format}. "
                f"Supported: {self.SUPPORTED_INPUT_FORMATS}"
            )

        try:
            # FFmpeg command for conversion to PCM
            cmd = [
                self.ffmpeg_path,
                "-hide_banner",
                "-loglevel", "error",
                "-f", input_format,        # Input format
                "-i", "pipe:0",            # Read from stdin
                "-f", "s16le",             # Output: signed 16-bit little-endian
                "-acodec", "pcm_s16le",    # PCM codec
                "-ar", str(sample_rate),   # Sample rate
                "-ac", str(AUDIO_CHANNELS), # Mono
                "pipe:1"                    # Write to stdout
            ]

            logger.debug(f"Converting {input_format} to PCM ({sample_rate}Hz)")

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            pcm_data, error = process.communicate(input=audio_data)

            if process.returncode != 0:
                error_msg = error.decode() if error else "Unknown error"
                raise AudioHandlerError(f"FFmpeg conversion failed: {error_msg}")

            logger.debug(
                f"Converted {len(audio_data)} bytes {input_format} -> "
                f"{len(pcm_data)} bytes PCM"
            )

            return pcm_data

        except subprocess.SubprocessError as e:
            raise AudioHandlerError(f"FFmpeg subprocess error: {e}")

    def convert_from_pcm(
        self,
        pcm_data: bytes,
        output_format: str = "mp3",
        sample_rate: int = OUTPUT_SAMPLE_RATE,
        bitrate: str = "128k"
    ) -> bytes:
        """
        Convert PCM audio from Gemini Live API to playable format.

        Args:
            pcm_data: Raw PCM bytes (16-bit, mono, little-endian)
            output_format: Target format (mp3, ogg, wav)
            sample_rate: Input sample rate (default: 24kHz for Gemini output)
            bitrate: Output bitrate for compressed formats

        Returns:
            Audio bytes in target format

        Raises:
            AudioHandlerError: If conversion fails
        """
        if not self.ffmpeg_path:
            raise AudioHandlerError("FFmpeg not available for audio conversion")

        if output_format.lower() not in self.SUPPORTED_OUTPUT_FORMATS:
            raise AudioHandlerError(
                f"Unsupported output format: {output_format}. "
                f"Supported: {self.SUPPORTED_OUTPUT_FORMATS}"
            )

        try:
            # FFmpeg command for conversion from PCM
            cmd = [
                self.ffmpeg_path,
                "-hide_banner",
                "-loglevel", "error",
                "-f", "s16le",             # Input: signed 16-bit little-endian
                "-ar", str(sample_rate),   # Input sample rate
                "-ac", str(AUDIO_CHANNELS), # Mono
                "-i", "pipe:0",            # Read from stdin
            ]

            # Add format-specific options
            if output_format == "mp3":
                cmd.extend(["-f", "mp3", "-b:a", bitrate])
            elif output_format == "ogg":
                cmd.extend(["-f", "ogg", "-c:a", "libvorbis", "-b:a", bitrate])
            elif output_format == "wav":
                cmd.extend(["-f", "wav"])

            cmd.append("pipe:1")  # Write to stdout

            logger.debug(f"Converting PCM ({sample_rate}Hz) to {output_format}")

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            output_data, error = process.communicate(input=pcm_data)

            if process.returncode != 0:
                error_msg = error.decode() if error else "Unknown error"
                raise AudioHandlerError(f"FFmpeg conversion failed: {error_msg}")

            logger.debug(
                f"Converted {len(pcm_data)} bytes PCM -> "
                f"{len(output_data)} bytes {output_format}"
            )

            return output_data

        except subprocess.SubprocessError as e:
            raise AudioHandlerError(f"FFmpeg subprocess error: {e}")

    def chunk_audio(
        self,
        audio_data: bytes,
        chunk_size: int = DEFAULT_CHUNK_SIZE
    ) -> List[bytes]:
        """
        Split audio data into chunks for streaming.

        Args:
            audio_data: Audio bytes to split
            chunk_size: Size of each chunk in bytes

        Returns:
            List of audio chunks
        """
        chunks = []
        for i in range(0, len(audio_data), chunk_size):
            chunks.append(audio_data[i:i + chunk_size])

        logger.debug(
            f"Split {len(audio_data)} bytes into {len(chunks)} chunks "
            f"(chunk_size={chunk_size})"
        )

        return chunks

    def merge_chunks(self, chunks: List[bytes]) -> bytes:
        """
        Merge audio chunks back into a single buffer.

        Args:
            chunks: List of audio chunks

        Returns:
            Merged audio bytes
        """
        return b"".join(chunks)

    def resample(
        self,
        audio_data: bytes,
        from_rate: int,
        to_rate: int
    ) -> bytes:
        """
        Resample audio data using numpy interpolation.

        Args:
            audio_data: PCM audio bytes (16-bit, mono)
            from_rate: Source sample rate
            to_rate: Target sample rate

        Returns:
            Resampled PCM bytes
        """
        if from_rate == to_rate:
            return audio_data

        # Convert bytes to numpy array
        samples = np.frombuffer(audio_data, dtype=np.int16)

        # Calculate new length
        new_length = int(len(samples) * to_rate / from_rate)

        # Resample using linear interpolation
        indices = np.linspace(0, len(samples) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(samples)), samples)

        logger.debug(
            f"Resampled {len(samples)} samples ({from_rate}Hz) -> "
            f"{len(resampled)} samples ({to_rate}Hz)"
        )

        return resampled.astype(np.int16).tobytes()

    def get_audio_duration(
        self,
        audio_data: bytes,
        sample_rate: int = INPUT_SAMPLE_RATE,
        sample_width: int = 2  # 16-bit = 2 bytes
    ) -> float:
        """
        Calculate duration of PCM audio in seconds.

        Args:
            audio_data: PCM audio bytes
            sample_rate: Sample rate in Hz
            sample_width: Bytes per sample (2 for 16-bit)

        Returns:
            Duration in seconds
        """
        num_samples = len(audio_data) // sample_width
        duration = num_samples / sample_rate
        return duration

    def detect_format(self, audio_data: bytes) -> Optional[str]:
        """
        Detect audio format from magic bytes.

        Args:
            audio_data: Audio bytes

        Returns:
            Detected format string or None
        """
        if len(audio_data) < 12:
            return None

        # Check magic bytes
        if audio_data[:4] == b"OggS":
            return "ogg"
        elif audio_data[:3] == b"ID3" or audio_data[:2] == b"\xff\xfb":
            return "mp3"
        elif audio_data[:4] == b"RIFF" and audio_data[8:12] == b"WAVE":
            return "wav"
        elif audio_data[:4] == b"fLaC":
            return "flac"
        elif audio_data[4:8] == b"ftyp":
            return "m4a"
        elif audio_data[:4] == b"\x1aE\xdf\xa3":
            return "webm"

        return None

    def validate_pcm(
        self,
        pcm_data: bytes,
        expected_rate: int = INPUT_SAMPLE_RATE
    ) -> Tuple[bool, str]:
        """
        Validate PCM audio data.

        Args:
            pcm_data: PCM bytes to validate
            expected_rate: Expected sample rate

        Returns:
            Tuple of (is_valid, message)
        """
        if not pcm_data:
            return False, "Empty audio data"

        if len(pcm_data) % 2 != 0:
            return False, "Invalid PCM: odd number of bytes (should be 16-bit)"

        # Check for reasonable duration (0.1s to 60s)
        duration = self.get_audio_duration(pcm_data, expected_rate)
        if duration < 0.1:
            return False, f"Audio too short: {duration:.2f}s"
        if duration > 60:
            return False, f"Audio too long: {duration:.2f}s (max 60s)"

        # Check for silence (all zeros)
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        if np.all(samples == 0):
            return False, "Audio is silent (all zeros)"

        # Check for clipping (too many max values)
        clip_threshold = 0.1  # 10% of samples at max value
        clipped = np.sum(np.abs(samples) >= 32767) / len(samples)
        if clipped > clip_threshold:
            return False, f"Audio is clipped: {clipped*100:.1f}% at max value"

        return True, f"Valid PCM audio: {duration:.2f}s, {len(samples)} samples"

    def normalize_audio(
        self,
        pcm_data: bytes,
        target_db: float = -3.0
    ) -> bytes:
        """
        Normalize audio volume to target dB level.

        Args:
            pcm_data: PCM audio bytes
            target_db: Target dB level (default: -3dB)

        Returns:
            Normalized PCM bytes
        """
        samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)

        # Calculate current RMS level
        rms = np.sqrt(np.mean(samples ** 2))
        if rms == 0:
            return pcm_data

        # Calculate target RMS
        target_rms = 32767 * (10 ** (target_db / 20))

        # Scale samples
        scale = target_rms / rms
        normalized = samples * scale

        # Clip to valid range
        normalized = np.clip(normalized, -32768, 32767)

        return normalized.astype(np.int16).tobytes()


# Convenience function for quick conversion
def convert_whatsapp_audio_to_pcm(audio_data: bytes) -> bytes:
    """
    Convert WhatsApp audio (typically OGG) to PCM for Gemini.

    Args:
        audio_data: Audio bytes from WhatsApp

    Returns:
        PCM bytes ready for Gemini Live API
    """
    handler = AudioHandler()

    # Detect format
    fmt = handler.detect_format(audio_data)
    if not fmt:
        fmt = "ogg"  # Default for WhatsApp

    return handler.convert_to_pcm(audio_data, input_format=fmt)


def convert_gemini_audio_to_mp3(pcm_data: bytes) -> bytes:
    """
    Convert Gemini PCM output to MP3 for WhatsApp.

    Args:
        pcm_data: PCM bytes from Gemini (24kHz)

    Returns:
        MP3 bytes for WhatsApp
    """
    handler = AudioHandler()
    return handler.convert_from_pcm(
        pcm_data,
        output_format="mp3",
        sample_rate=OUTPUT_SAMPLE_RATE
    )
