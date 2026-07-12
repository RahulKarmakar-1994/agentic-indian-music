"""Tokenize symbolic music for a small GPT-style music language model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math

try:
    from .simple_midi import DEFAULT_TICKS_PER_BEAT, MidiNote, read_midi, write_midi
except ImportError:  # Allows `python src/music_tokens.py ...`
    from simple_midi import DEFAULT_TICKS_PER_BEAT, MidiNote, read_midi, write_midi


STEPS_PER_BEAT = 4
STEPS_PER_BAR = 16
PITCH_MIN = 21
PITCH_MAX = 108
VELOCITY_BUCKETS = [32, 48, 64, 80, 96, 112]


@dataclass(frozen=True)
class TokenEvent:
    start_step: int
    duration_steps: int
    pitch: int
    velocity: int


def quantize_tick(tick: int, ticks_per_beat: int) -> int:
    ticks_per_step = ticks_per_beat / STEPS_PER_BEAT
    return int(round(tick / ticks_per_step))


def step_to_tick(step: int, ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT) -> int:
    return int(round(step * ticks_per_beat / STEPS_PER_BEAT))


def velocity_bucket(velocity: int) -> int:
    return min(VELOCITY_BUCKETS, key=lambda bucket: abs(bucket - velocity))


def midi_to_tokens(path: str | Path, max_bars: int | None = None) -> list[str]:
    song = read_midi(path)
    events: list[TokenEvent] = []
    max_step = 0
    for note in song.notes:
        if not PITCH_MIN <= note.pitch <= PITCH_MAX:
            continue
        start = quantize_tick(note.start, song.ticks_per_beat)
        end = max(start + 1, quantize_tick(note.end, song.ticks_per_beat))
        duration = min(32, max(1, end - start))
        if max_bars is not None and start >= max_bars * STEPS_PER_BAR:
            continue
        events.append(TokenEvent(start, duration, note.pitch, velocity_bucket(note.velocity)))
        max_step = max(max_step, start + duration)

    bars = max(1, math.ceil(max_step / STEPS_PER_BAR))
    if max_bars is not None:
        bars = min(bars, max_bars)

    tokens = ["BOS", f"TEMPO_{round(60_000_000 / song.tempo_us_per_beat)}", "TS_4_4"]
    events_by_bar: dict[int, list[TokenEvent]] = {}
    for event in events:
        events_by_bar.setdefault(event.start_step // STEPS_PER_BAR, []).append(event)

    for bar in range(bars):
        tokens.append("BAR")
        bar_events = sorted(events_by_bar.get(bar, []), key=lambda event: (event.start_step, event.pitch))
        for event in bar_events:
            position = event.start_step % STEPS_PER_BAR
            tokens.extend([
                f"POS_{position}",
                f"NOTE_{event.pitch}",
                f"DUR_{event.duration_steps}",
                f"VEL_{event.velocity}",
            ])

    tokens.append("EOS")
    return tokens


def tokens_to_events(tokens: list[str]) -> tuple[list[MidiNote], int]:
    notes: list[MidiNote] = []
    current_bar = -1
    current_pos = 0
    tempo_bpm = 120
    pitch: int | None = None
    duration = 1
    velocity = 80

    def flush() -> None:
        nonlocal pitch, duration, velocity
        if pitch is None or current_bar < 0:
            pitch = None
            duration = 1
            velocity = 80
            return
        start_step = current_bar * STEPS_PER_BAR + current_pos
        start = step_to_tick(start_step)
        end = step_to_tick(start_step + max(1, duration))
        notes.append(MidiNote(start=start, end=end, pitch=pitch, velocity=velocity, channel=0))
        pitch = None
        duration = 1
        velocity = 80

    for token in tokens:
        if token == "BAR":
            flush()
            current_bar += 1
        elif token.startswith("TEMPO_"):
            try:
                tempo_bpm = max(40, min(220, int(token.split("_", 1)[1])))
            except ValueError:
                tempo_bpm = 120
        elif token.startswith("POS_"):
            flush()
            current_pos = max(0, min(STEPS_PER_BAR - 1, int(token.split("_", 1)[1])))
        elif token.startswith("NOTE_"):
            pitch = max(PITCH_MIN, min(PITCH_MAX, int(token.split("_", 1)[1])))
        elif token.startswith("DUR_"):
            duration = max(1, min(32, int(token.split("_", 1)[1])))
        elif token.startswith("VEL_"):
            velocity = max(1, min(127, int(token.split("_", 1)[1])))
        elif token == "EOS":
            break
    flush()
    tempo_us_per_beat = int(60_000_000 / tempo_bpm)
    return notes, tempo_us_per_beat


def write_token_file(tokens: list[str], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(" ".join(tokens) + "\n", encoding="utf-8")


def read_token_sequences(path: str | Path) -> list[list[str]]:
    sequences: list[list[str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            sequences.append(line.split())
    return sequences


def build_vocab(sequences: list[list[str]]) -> dict[str, int]:
    vocab = sorted({token for sequence in sequences for token in sequence})
    return {token: index for index, token in enumerate(vocab)}


def save_vocab(vocab: dict[str, int], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(vocab, indent=2, sort_keys=True), encoding="utf-8")


def load_vocab(path: str | Path) -> dict[str, int]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def command_encode(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output = Path(args.output)
    max_bars = args.max_bars
    sequences = []
    for midi_path in sorted(input_dir.glob("**/*.mid")) + sorted(input_dir.glob("**/*.midi")):
        tokens = midi_to_tokens(midi_path, max_bars=max_bars)
        if len(tokens) > 4:
            sequences.append(tokens)
            print(f"encoded {midi_path} -> {len(tokens)} tokens")
    if not sequences:
        raise SystemExit(f"No MIDI files found in {input_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(" ".join(sequence) for sequence in sequences) + "\n", encoding="utf-8")
    save_vocab(build_vocab(sequences), args.vocab)
    print(f"wrote {len(sequences)} sequences to {output}")
    print(f"wrote vocabulary to {args.vocab}")


def command_decode(args: argparse.Namespace) -> None:
    sequences = [line.split() for line in Path(args.tokens).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not sequences:
        raise SystemExit(f"No token sequences found in {args.tokens}")
    if not 0 <= args.sequence_index < len(sequences):
        raise SystemExit(f"--sequence-index must be between 0 and {len(sequences) - 1}")
    tokens = sequences[args.sequence_index]
    notes, tempo = tokens_to_events(tokens)
    write_midi(args.output, notes, tempo_us_per_beat=tempo)
    print(f"wrote {len(notes)} notes to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    encode = subparsers.add_parser("encode-midi", help="Tokenize a folder of MIDI files")
    encode.add_argument("--input-dir", default="data/raw_midi")
    encode.add_argument("--output", default="data/tokenized/corpus.txt")
    encode.add_argument("--vocab", default="data/tokenized/vocab.json")
    encode.add_argument("--max-bars", type=int, default=16)
    encode.set_defaults(func=command_encode)

    decode = subparsers.add_parser("decode-tokens", help="Render one token sequence to MIDI")
    decode.add_argument("--tokens", required=True)
    decode.add_argument("--output", default="generated/from_tokens.mid")
    decode.add_argument("--sequence-index", type=int, default=0)
    decode.set_defaults(func=command_decode)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
