"""TTS engine — wraps Piper as a subprocess for local text-to-speech.

Piper reads text from stdin and writes WAV to stdout. We use
asyncio.create_subprocess_exec to keep the daemon non-blocking.

speak() calls synthesize() then plays via macOS `afplay`.
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path


async def _run_piper(model_path: Path, text: str) -> bytes:
    """Run Piper subprocess: text → WAV bytes."""
    # Find piper in the same venv as the current Python
    piper_bin = Path(sys.executable).parent / "piper"
    if not piper_bin.exists():
        raise FileNotFoundError(f"Piper not found at {piper_bin}")

    proc = await asyncio.create_subprocess_exec(
        str(piper_bin),
        "--model", str(model_path),
        "--output-raw",  # raw PCM; we'll skip this and use --output_file -
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=text.encode("utf-8"))
    if proc.returncode != 0:
        raise RuntimeError(f"Piper failed (rc={proc.returncode}): {stderr.decode()}")
    return stdout


class TTSEngine:
    """Text-to-speech engine using Piper."""

    def __init__(self, voice_model: Path) -> None:
        self.voice_model = voice_model

    async def synthesize(self, text: str) -> bytes:
        """Convert text to WAV audio bytes via Piper subprocess."""
        if not text.strip():
            return b""
        return await _run_piper(self.voice_model, text)

    async def speak(self, text: str) -> None:
        """Synthesize text and play through Mac speakers."""
        wav_data = await self.synthesize(text)
        if not wav_data:
            return
        # Write to temp file and play via afplay (macOS)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_data)
            tmp_path = f.name
        proc = await asyncio.create_subprocess_exec("afplay", tmp_path)
        await proc.wait()
        import os
        os.unlink(tmp_path)
