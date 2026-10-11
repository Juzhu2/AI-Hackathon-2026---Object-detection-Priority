"""Read inputtest.txt and save Kokoro speech to output/output.wav."""

import argparse
import math
import os
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = 24000


def generate_audio(input_path, output_path, voice="af_heart", speed=1.0):
    """Synthesize every text chunk and combine them into one mono WAV file."""
    text = Path(input_path).read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError("The input text file is empty. Add text before running the script.")
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("Speech speed must be a positive finite number.")
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("Input and output must be different files.")

    # Keep downloaded models out of Git and together with this component.
    os.environ.setdefault("HF_HOME", str(APP_DIR / "weights" / "huggingface"))
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cpu")
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+"):
        if audio is not None:
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
            if chunk.size:
                chunks.append(chunk)
    if not chunks:
        raise RuntimeError("Kokoro produced no audio. Check the input text and voice.")

    audio = np.concatenate(chunks)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return len(audio) / SAMPLE_RATE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=APP_DIR / "inputtest.txt")
    parser.add_argument("--output", type=Path, default=APP_DIR / "output" / "output.wav")
    parser.add_argument("--voice", default="af_heart", help="American English Kokoro voice")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    try:
        duration = generate_audio(args.input, args.output, args.voice, args.speed)
    except ImportError as error:
        print(f"Missing dependency: {error}. Install: python -m pip install -r TTS/requirements.txt", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Unable to generate speech: {error}", file=sys.stderr)
        return 1
    print(f"Saved audio: {args.output.resolve()} ({duration:.1f} seconds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
