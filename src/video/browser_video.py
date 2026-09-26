"""Convert OpenCV MP4 output into a browser-compatible H.264 file."""

import os
from pathlib import Path
import subprocess
import tempfile

import imageio_ffmpeg


class BrowserVideoError(RuntimeError):
    """Raised when the H.264 conversion cannot produce a usable file."""


def make_browser_playable(
    source_path: str | Path,
    destination_path: str | Path,
) -> Path:
    """Transcode one local video to H.264/yuv420p and replace atomically."""
    source = Path(source_path).resolve()
    destination = Path(destination_path).resolve()
    if not source.is_file():
        raise BrowserVideoError(f"source video does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.stem}-",
            suffix=".mp4",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary_path),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=300,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if completed.returncode != 0 or temporary_path.stat().st_size == 0:
            message = completed.stderr.decode("utf-8", errors="replace").strip()
            raise BrowserVideoError(f"browser video conversion failed: {message}")
        os.replace(temporary_path, destination)
        return destination
    except (OSError, subprocess.SubprocessError) as error:
        raise BrowserVideoError("browser video conversion failed") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
