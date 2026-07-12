"""Render tokenized sargam to a simple WAV file.

This is intentionally dependency-free. The output is a plain synth tone for
listening and practice; product-quality timbre should later use a SoundFont or
dedicated sampler. A light Sa/Pa drone is included by default to make practice
phrases feel less sterile.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math
import struct
import wave

try:
    from .sargam_tokens import parse_sa, tokens_to_midi_notes
    from .simple_midi import DEFAULT_TICKS_PER_BEAT
except ImportError:
    from sargam_tokens import parse_sa, tokens_to_midi_notes
    from simple_midi import DEFAULT_TICKS_PER_BEAT


def midi_to_freq(note: int) -> float:
    return 440.0 * (2 ** ((note - 69) / 12))


def melody_value(freq: float, t: float, vibrato_phase: float = 0.0) -> float:
    vibrato = 1.0 + 0.0028 * math.sin(2 * math.pi * 5.2 * t + vibrato_phase)
    f = freq * vibrato
    value = math.sin(2 * math.pi * f * t) * 0.62
    value += math.sin(2 * math.pi * f * 2 * t) * 0.18
    value += math.sin(2 * math.pi * f * 3 * t) * 0.07
    value += math.sin(2 * math.pi * f * 4 * t) * 0.025
    return value


def add_drone(samples: list[float], sa: str, sample_rate: int, amp: float = 0.11) -> None:
    sa_midi = parse_sa(sa)
    drone_notes = [sa_midi - 12, sa_midi - 5, sa_midi]
    drone_freqs = [midi_to_freq(note) for note in drone_notes]
    for sample_index in range(len(samples)):
        t = sample_index / sample_rate
        pulse = 0.72 + 0.28 * (0.5 + 0.5 * math.sin(2 * math.pi * 0.42 * t))
        shimmer = 0.5 + 0.5 * math.sin(2 * math.pi * 2.1 * t)
        value = 0.0
        for index, freq in enumerate(drone_freqs):
            phase = index * 0.73
            value += math.sin(2 * math.pi * freq * t + phase) * 0.58
            value += math.sin(2 * math.pi * freq * 2 * t + phase) * 0.13
            value += math.sin(2 * math.pi * freq * 3 * t + phase) * 0.04 * shimmer
        samples[sample_index] += amp * pulse * value / len(drone_freqs)


def render_tokens_to_wav(
    tokens: list[str],
    output: str | Path,
    sa: str = "C4",
    sample_rate: int = 44_100,
    include_drone: bool = True,
) -> dict[str, float | int | str | bool]:
    notes, tempo_us = tokens_to_midi_notes(tokens, sa=sa)
    seconds_per_tick = (tempo_us / 1_000_000) / DEFAULT_TICKS_PER_BEAT
    end_time = max((note.end * seconds_per_tick for note in notes), default=2.0) + 0.8
    samples = [0.0] * int(end_time * sample_rate)

    if include_drone:
        add_drone(samples, sa, sample_rate)

    for note in notes:
        start = int(note.start * seconds_per_tick * sample_rate)
        end = int(note.end * seconds_per_tick * sample_rate)
        freq = midi_to_freq(note.pitch)
        amp = min(0.3, max(0.08, note.velocity / 127 * 0.25))
        length = max(1, end - start)
        attack = max(1, int(0.035 * sample_rate))
        release = max(1, int(0.12 * sample_rate))
        vibrato_phase = (note.pitch % 12) * 0.31
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
            value = melody_value(freq, t, vibrato_phase)
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

    return {
        "path": str(output),
        "notes": len(notes),
        "seconds": round(end_time, 2),
        "sample_rate": sample_rate,
        "drone": include_drone,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokens", help="Path to a tokenized sargam file")
    parser.add_argument("--output", default="generated/sargam.wav")
    parser.add_argument("--sa", default="C4")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--no-drone", action="store_true", help="Disable the Sa/Pa drone layer")
    args = parser.parse_args()

    sequences = [line.split() for line in Path(args.tokens).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not sequences:
        raise SystemExit(f"No token sequences found in {args.tokens}")
    if not 0 <= args.sequence_index < len(sequences):
        raise SystemExit(f"--sequence-index must be between 0 and {len(sequences) - 1}")

    result = render_tokens_to_wav(sequences[args.sequence_index], args.output, sa=args.sa, include_drone=not args.no_drone)
    print(f"wrote WAV to {result['path']}")
    print(f"notes={result['notes']} seconds={result['seconds']} drone={result['drone']}")


if __name__ == "__main__":
    main()
