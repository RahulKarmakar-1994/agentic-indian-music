"""Tokenize, validate, and render Indian sargam notation.

The friendly source format is one phrase per line:

    RAGA=YAMAN TALA=TEENTAL TEMPO=84 :: S:1 R:1 G:1 M^:1 | P:1 D:1 N:1 S':2

Durations are in matras. Swaras are relative to Sa, so the same sequence can be
rendered in any tonic by passing --sa C4, --sa D4, and so on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import re

try:
    from .simple_midi import DEFAULT_TICKS_PER_BEAT, MidiNote, write_midi
    from .music_tokens import build_vocab
except ImportError:
    from simple_midi import DEFAULT_TICKS_PER_BEAT, MidiNote, write_midi
    from music_tokens import build_vocab


SWARA_INTERVALS = {
    "SA": 0,
    "RE_KOMAL": 1,
    "RE": 2,
    "GA_KOMAL": 3,
    "GA": 4,
    "MA": 5,
    "MA_TIVRA": 6,
    "PA": 7,
    "DHA_KOMAL": 8,
    "DHA": 9,
    "NI_KOMAL": 10,
    "NI": 11,
}

SWARA_ALIASES = {
    "S": "SA",
    "SA": "SA",
    "R": "RE",
    "RE": "RE",
    "R_SHUDDHA": "RE",
    "r": "RE_KOMAL",
    "RE_KOMAL": "RE_KOMAL",
    "G": "GA",
    "GA": "GA",
    "G_SHUDDHA": "GA",
    "g": "GA_KOMAL",
    "GA_KOMAL": "GA_KOMAL",
    "M": "MA",
    "MA": "MA",
    "M_SHUDDHA": "MA",
    "M^": "MA_TIVRA",
    "M#": "MA_TIVRA",
    "MA_T": "MA_TIVRA",
    "MA_TIVRA": "MA_TIVRA",
    "P": "PA",
    "PA": "PA",
    "D": "DHA",
    "DHA": "DHA",
    "D_SHUDDHA": "DHA",
    "d": "DHA_KOMAL",
    "DHA_KOMAL": "DHA_KOMAL",
    "N": "NI",
    "NI": "NI",
    "N_SHUDDHA": "NI",
    "n": "NI_KOMAL",
    "NI_KOMAL": "NI_KOMAL",
}

RAGA_DEFINITIONS = {
    "BILAWAL": {
        "allowed": {"SA", "RE", "GA", "MA", "PA", "DHA", "NI"},
        "vadi": "DHA",
        "samvadi": "GA",
        "hint": "All shuddha swaras; useful as a simple major-scale teaching base.",
    },
    "YAMAN": {
        "allowed": {"SA", "RE", "GA", "MA_TIVRA", "PA", "DHA", "NI"},
        "vadi": "GA",
        "samvadi": "NI",
        "hint": "Use tivra Ma; avoid shuddha Ma in basic Yaman exercises.",
    },
    "BHAIRAV": {
        "allowed": {"SA", "RE_KOMAL", "GA", "MA", "PA", "DHA_KOMAL", "NI"},
        "vadi": "DHA_KOMAL",
        "samvadi": "RE_KOMAL",
        "hint": "Komal Re and komal Dha are important identity notes.",
    },
    "KAFI": {
        "allowed": {"SA", "RE", "GA_KOMAL", "MA", "PA", "DHA", "NI_KOMAL"},
        "vadi": "PA",
        "samvadi": "SA",
        "hint": "Komal Ga and komal Ni create the Kafi color.",
    },
    "BHUPALI": {
        "allowed": {"SA", "RE", "GA", "PA", "DHA"},
        "vadi": "GA",
        "samvadi": "DHA",
        "hint": "Audav raga: omit Ma and Ni.",
    },
    "DURGA": {
        "allowed": {"SA", "RE", "MA", "PA", "DHA"},
        "vadi": "MA",
        "samvadi": "SA",
        "hint": "Audav raga: omit Ga and Ni.",
    },
    "KALYANI": {
        "allowed": {"SA", "RE", "GA", "MA_TIVRA", "PA", "DHA", "NI"},
        "vadi": "GA",
        "samvadi": "NI",
        "hint": "Carnatic prati madhyamam equivalent of the Yaman pitch set.",
    },
    "MAYAMALAVAGOWLA": {
        "allowed": {"SA", "RE_KOMAL", "GA", "MA", "PA", "DHA_KOMAL", "NI"},
        "vadi": "GA",
        "samvadi": "NI",
        "hint": "Common beginner Carnatic melakarta with R1, G3, D1, N3.",
    },
    "TODI": {
        "allowed": {"SA", "RE_KOMAL", "GA_KOMAL", "MA_TIVRA", "PA", "DHA_KOMAL", "NI"},
        "vadi": "DHA_KOMAL",
        "samvadi": "GA_KOMAL",
        "hint": "Use komal Re, komal Ga, tivra Ma, and komal Dha.",
    },
    "POORVI": {
        "allowed": {"SA", "RE_KOMAL", "GA", "MA_TIVRA", "PA", "DHA_KOMAL", "NI"},
        "vadi": "GA",
        "samvadi": "NI",
        "hint": "Use komal Re, tivra Ma, and komal Dha.",
    },
}

TALA_MATRAS = {
    "TEENTAL": 16,
    "TRITAAL": 16,
    "KEHERWA": 8,
    "KEHEWA": 8,
    "DADRA": 6,
    "RUPAK": 7,
    "ADI": 8,
    "EKTAL": 12,
    "EKTAAL": 12,
    "JHAPTAAL": 10,
    "JHAAPTAAL": 10,
    "TILWADA": 16,
    "TILWARA": 16,
    "JHOOMRA": 14,
    "CHOUTAAL": 12,
    "CHAU_TAAL": 12,
    "ADA_CHOUTAAL": 14,
    "DHAMAAR": 14,
    "DHAMAR": 14,
    "SULTAAL": 10,
    "BRAHMATAAL": 28,
}

NOTE_BASE = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}


@dataclass(frozen=True)
class SargamEvent:
    start_matra: int
    duration_matras: int
    swara: str
    octave: int = 0
    velocity: int = 84


def normalize_key(value: str) -> str:
    return value.strip().upper().replace("-", "_")


def parse_swara(value: str) -> tuple[str, int]:
    raw = value.strip()
    octave = raw.count("'") - raw.count(".")
    raw = raw.replace("'", "").replace(".", "")
    if raw in SWARA_ALIASES:
        return SWARA_ALIASES[raw], octave
    upper = normalize_key(raw)
    if upper in SWARA_ALIASES:
        return SWARA_ALIASES[upper], octave
    raise ValueError(f"Unknown swara: {value}")


def parse_note_atom(atom: str) -> tuple[str, int, int]:
    if ":" in atom:
        swara_part, duration_part = atom.split(":", 1)
        duration = int(duration_part)
    else:
        swara_part = atom
        duration = 1
    swara, octave = parse_swara(swara_part)
    return swara, octave, max(1, duration)


def parse_source_line(line: str) -> list[str]:
    line = line.strip()
    if not line or line.startswith("#"):
        return []
    if "::" not in line:
        raise ValueError("Sargam lines must contain metadata and notation separated by '::'")

    metadata_text, notation_text = line.split("::", 1)
    metadata: dict[str, str] = {}
    for item in metadata_text.split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        metadata[normalize_key(key)] = normalize_key(value)

    raga = metadata.get("RAGA", "BILAWAL")
    tala = metadata.get("TALA", "TEENTAL")
    tempo = int(metadata.get("TEMPO", "84"))
    laya = metadata.get("LAYA", "MADHYA")
    if raga not in RAGA_DEFINITIONS:
        raise ValueError(f"Unsupported raga {raga}; known: {', '.join(sorted(RAGA_DEFINITIONS))}")
    if tala not in TALA_MATRAS:
        raise ValueError(f"Unsupported tala {tala}; known: {', '.join(sorted(TALA_MATRAS))}")

    matras_per_cycle = TALA_MATRAS[tala]
    tokens = ["BOS", f"RAGA_{raga}", f"TALA_{tala}", f"LAYA_{laya}", f"TEMPO_{tempo}", "BAR"]
    matra = 1
    for atom in notation_text.replace("|", " | ").split():
        if atom == "|":
            tokens.append("VIBHAG")
            continue
        if matra > matras_per_cycle:
            tokens.append("BAR")
            matra = 1
        if atom.upper() in {"REST", "-"}:
            swara, octave, duration = "REST", 0, 1
        else:
            swara, octave, duration = parse_note_atom(atom)
        tokens.extend([f"MATRA_{matra}", f"SWARA_{swara}", f"OCT_{octave}", f"DUR_{duration}"])
        matra += duration
    tokens.append("EOS")
    return tokens


def read_sargam_source(path: str | Path) -> list[list[str]]:
    sequences = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        try:
            tokens = parse_source_line(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if tokens:
            sequences.append(tokens)
    return sequences


def sequence_metadata(tokens: list[str]) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {"raga": "UNKNOWN", "tala": "UNKNOWN", "tempo": 84}
    for token in tokens:
        if token.startswith("RAGA_"):
            metadata["raga"] = token.removeprefix("RAGA_")
        elif token.startswith("TALA_"):
            metadata["tala"] = token.removeprefix("TALA_")
        elif token.startswith("TEMPO_"):
            try:
                metadata["tempo"] = int(token.removeprefix("TEMPO_"))
            except ValueError:
                metadata["tempo"] = 84
    return metadata


def tokens_to_sargam_events(tokens: list[str]) -> tuple[list[SargamEvent], dict[str, str | int]]:
    events: list[SargamEvent] = []
    current_bar = -1
    matras_per_cycle = 16
    current_matra = 1
    swara: str | None = None
    octave = 0
    duration = 1
    metadata = sequence_metadata(tokens)
    tala = str(metadata.get("tala", "TEENTAL"))
    matras_per_cycle = TALA_MATRAS.get(tala, matras_per_cycle)

    def flush() -> None:
        nonlocal swara, octave, duration
        if swara is None or swara == "REST" or current_bar < 0:
            swara = None
            octave = 0
            duration = 1
            return
        start = current_bar * matras_per_cycle + max(0, current_matra - 1)
        events.append(SargamEvent(start_matra=start, duration_matras=max(1, duration), swara=swara, octave=octave))
        swara = None
        octave = 0
        duration = 1

    for token in tokens:
        if token == "BAR":
            flush()
            current_bar += 1
        elif token.startswith("MATRA_"):
            flush()
            try:
                current_matra = max(1, int(token.removeprefix("MATRA_")))
            except ValueError:
                current_matra = 1
        elif token.startswith("SWARA_"):
            swara = token.removeprefix("SWARA_")
        elif token.startswith("OCT_"):
            try:
                octave = int(token.removeprefix("OCT_"))
            except ValueError:
                octave = 0
        elif token.startswith("DUR_"):
            try:
                duration = max(1, int(token.removeprefix("DUR_")))
            except ValueError:
                duration = 1
        elif token == "EOS":
            break
    flush()
    return events, metadata


def parse_sa(value: str) -> int:
    match = re.fullmatch(r"([A-Ga-g])([#bB]?)(-?\d+)", value.strip())
    if not match:
        raise ValueError("Sa must look like C4, D#3, or Bb3")
    name = (match.group(1).upper() + match.group(2).upper()).replace("B", "b").upper()
    octave = int(match.group(3))
    if name not in NOTE_BASE:
        raise ValueError(f"Unknown tonic note: {value}")
    return 12 * (octave + 1) + NOTE_BASE[name]


def swara_to_midi(swara: str, octave: int, sa_midi: int) -> int:
    if swara not in SWARA_INTERVALS:
        raise ValueError(f"Cannot render unknown swara {swara}")
    return sa_midi + SWARA_INTERVALS[swara] + octave * 12


def tokens_to_midi_notes(tokens: list[str], sa: str = "C4") -> tuple[list[MidiNote], int]:
    events, metadata = tokens_to_sargam_events(tokens)
    tempo_bpm = int(metadata.get("tempo", 84))
    tempo_us = int(60_000_000 / max(30, min(240, tempo_bpm)))
    sa_midi = parse_sa(sa)
    notes = [
        MidiNote(
            start=event.start_matra * DEFAULT_TICKS_PER_BEAT,
            end=(event.start_matra + event.duration_matras) * DEFAULT_TICKS_PER_BEAT,
            pitch=swara_to_midi(event.swara, event.octave, sa_midi),
            velocity=event.velocity,
        )
        for event in events
    ]
    return notes, tempo_us


def write_token_sequences(sequences: list[list[str]], output: str | Path) -> None:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(" ".join(sequence) for sequence in sequences) + "\n", encoding="utf-8")


def command_encode_text(args: argparse.Namespace) -> None:
    sequences = read_sargam_source(args.input)
    if not sequences:
        raise SystemExit(f"No sargam sequences found in {args.input}")
    write_token_sequences(sequences, args.output)
    Path(args.vocab).parent.mkdir(parents=True, exist_ok=True)
    Path(args.vocab).write_text(json.dumps(build_vocab(sequences), indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(sequences)} tokenized sargam sequences to {args.output}")
    print(f"wrote vocabulary to {args.vocab}")


def command_decode_midi(args: argparse.Namespace) -> None:
    sequences = [line.split() for line in Path(args.tokens).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not sequences:
        raise SystemExit(f"No token sequences found in {args.tokens}")
    if not 0 <= args.sequence_index < len(sequences):
        raise SystemExit(f"--sequence-index must be between 0 and {len(sequences) - 1}")
    notes, tempo_us = tokens_to_midi_notes(sequences[args.sequence_index], sa=args.sa)
    write_midi(args.output, notes, tempo_us_per_beat=tempo_us)
    print(f"wrote {len(notes)} notes to {args.output} with Sa={args.sa}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    encode = subparsers.add_parser("encode-text", help="Tokenize a friendly sargam text corpus")
    encode.add_argument("--input", default="data/sargam/seed_sargam.txt")
    encode.add_argument("--output", default="data/tokenized/seed_sargam.txt")
    encode.add_argument("--vocab", default="data/tokenized/seed_sargam_vocab.json")
    encode.set_defaults(func=command_encode_text)

    decode = subparsers.add_parser("decode-midi", help="Render tokenized sargam to MIDI")
    decode.add_argument("--tokens", required=True)
    decode.add_argument("--output", default="generated/sargam.mid")
    decode.add_argument("--sa", default="C4")
    decode.add_argument("--sequence-index", type=int, default=0)
    decode.set_defaults(func=command_decode_midi)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
