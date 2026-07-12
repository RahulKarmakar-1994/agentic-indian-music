"""Render tokenized sargam to a simple WAV file.

This is intentionally dependency-free. The output is a plain synth tone for
listening and practice; product-quality timbre should later use a SoundFont or
dedicated sampler.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math
import struct
import wave

try:
    from .sargam_tokens import tokens_to_midi_notes
    from .simple_midi import DEFAULT_TICKS_PER_BEAT
except ImportError:
    from sargam_tokens import tokens_to_midi_notes
    from simple_midi import DEFAULT_TICKS_PER_BEAT


def midi_to_freq(note: int) -> float:
    return 440.0 * (2 ** ((note - 69) / 12))


def render_tokens_to_wav(tokens: list[str], output: str | Path, sa: str = "C4", sample_rate: int = 44_100) -> dict[str, float | int | str]:
    notes, tempo_us = tokens_to_midi_notes(tokens, sa=sa)
    seconds_per_tick = (tempo_us / 1_000_000) / DEFAULT_TICKS_PER_BEAT
    end_time = max((note.end * seconds_per_tick for note in notes), default=2.0) + 0.4
    samples = [0.0] * int(end_time * sample_rate)

    for note in notes:
        start = int(note.start * seconds_per_tick * sample_rate)
        end = int(note.end * seconds_per_tick * sample_rate)
        freq = midi_to_freq(note.pitch)
        amp = min(0.34, max(0.1, note.velocity / 127 * 0.3))
        length = max(1, end - start)
        attack = max(1, int(0.018 * sample_rate))
        release = max(1, int(0.075 * sample_rate))
        for index in range(length):
            sample_index = start + index
            if sample_index >= len(samples):
                break
            t = index / sample_rate
            env = 1.0
            if index < attack:
                env = index / attack
            elif index > length - release:
                env = max(0.0, (length - index) / release)
            value = math.sin(2 * math.pi * freq * t) * 0.72
            value += math.sin(2 * math.pi * freq * 2 * t) * 0.14
            value += math.sin(2 * math.pi * freq * 3 * t) * 0.04
            samples[sample_index] += amp * env * value

    peak = max((abs(sample) for sample in samples), default=1.0)
    scale = 0.85 / peak if peak > 0 else 1.0
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for sample in samples:
            clipped = max(-1.0, min(1.0, sample * scale))
            wav.writeframes(struct.pack("<h", int(clipped * 32767)))

    return {"path": str(output), "notes": len(notes), "seconds": round(end_time, 2), "sample_rate": sample_rate}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokens", help="Path to a tokenized sargam file")
    parser.add_argument("--output", default="generated/sargam.wav")
    parser.add_argument("--sa", default="C4")
    parser.add_argument("--sequence-index", type=int, default=0)
    args = parser.parse_args()

    sequences = [line.split() for line in Path(args.tokens).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not sequences:
        raise SystemExit(f"No token sequences found in {args.tokens}")
    if not 0 <= args.sequence_index < len(sequences):
        raise SystemExit(f"--sequence-index must be between 0 and {len(sequences) - 1}")

    result = render_tokens_to_wav(sequences[args.sequence_index], args.output, sa=args.sa)
    print(f"wrote WAV to {result['path']}")
    print(f"notes={result['notes']} seconds={result['seconds']}")


if __name__ == "__main__":
    main()
