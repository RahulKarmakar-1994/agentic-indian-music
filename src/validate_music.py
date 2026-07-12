"""Validation and scoring for generated music-token sequences."""

from __future__ import annotations

from pathlib import Path
import argparse

try:
    from .music_tokens import STEPS_PER_BAR, tokens_to_events
except ImportError:
    from music_tokens import STEPS_PER_BAR, tokens_to_events


def validate_tokens(tokens: list[str]) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    bar_count = tokens.count("BAR")
    notes, _ = tokens_to_events(tokens)

    if not tokens or tokens[0] != "BOS":
        warnings.append("sequence does not start with BOS")
    if "EOS" not in tokens:
        warnings.append("sequence does not contain EOS")
    if bar_count == 0:
        errors.append("sequence contains no BAR tokens")
    if not notes:
        errors.append("sequence contains no playable notes")

    positions = []
    for token in tokens:
        if token.startswith("POS_"):
            try:
                value = int(token.split("_", 1)[1])
            except ValueError:
                errors.append(f"bad position token {token}")
                continue
            positions.append(value)
            if not 0 <= value < STEPS_PER_BAR:
                errors.append(f"position out of range: {token}")

    pitch_values = [note.pitch for note in notes]
    unique_pitches = len(set(pitch_values))
    if unique_pitches <= 2 and len(notes) > 6:
        warnings.append("low pitch variety")

    large_leaps = 0
    for left, right in zip(pitch_values, pitch_values[1:]):
        if abs(right - left) > 12:
            large_leaps += 1
    if large_leaps > max(2, len(notes) // 3):
        warnings.append("many large leaps")

    density = len(notes) / max(1, bar_count)
    if density > 24:
        warnings.append("very dense texture for a melody-first model")

    score = 100 - len(errors) * 30 - len(warnings) * 8
    return {
        "valid": not errors,
        "score": max(0, score),
        "bars": bar_count,
        "notes": len(notes),
        "unique_pitches": unique_pitches,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokens", help="Path to a token text file")
    args = parser.parse_args()
    sequences = [line.split() for line in Path(args.tokens).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not sequences:
        raise SystemExit(f"No token sequences found in {args.tokens}")
    for index, tokens in enumerate(sequences, start=1):
        result = validate_tokens(tokens)
        print(f"sequence {index}: {result}")


if __name__ == "__main__":
    main()
