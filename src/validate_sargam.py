"""Raga-aware validation and teaching hints for tokenized sargam sequences."""

from __future__ import annotations

from pathlib import Path
import argparse

try:
    from .sargam_tokens import RAGA_DEFINITIONS, TALA_MATRAS, sequence_metadata, tokens_to_sargam_events
except ImportError:
    from sargam_tokens import RAGA_DEFINITIONS, TALA_MATRAS, sequence_metadata, tokens_to_sargam_events


def validate_sargam_tokens(tokens: list[str]) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    metadata = sequence_metadata(tokens)
    raga = str(metadata.get("raga", "UNKNOWN"))
    tala = str(metadata.get("tala", "UNKNOWN"))
    events, _ = tokens_to_sargam_events(tokens)
    swaras = [event.swara for event in events]

    if not tokens or tokens[0] != "BOS":
        warnings.append("sequence does not start with BOS")
    if "EOS" not in tokens:
        warnings.append("sequence does not contain EOS")
    if raga not in RAGA_DEFINITIONS:
        errors.append(f"unknown raga: {raga}")
    if tala not in TALA_MATRAS:
        errors.append(f"unknown tala: {tala}")
    if not events:
        errors.append("sequence contains no playable swaras")

    if raga in RAGA_DEFINITIONS:
        definition = RAGA_DEFINITIONS[raga]
        allowed = definition["allowed"]
        outside = sorted({swara for swara in swaras if swara not in allowed})
        if outside:
            errors.append(f"swaras outside {raga}: {', '.join(outside)}")

        used = set(swaras)
        vadi = str(definition["vadi"])
        samvadi = str(definition["samvadi"])
        if vadi not in used and len(events) >= 6:
            warnings.append(f"phrase does not touch vadi swara {vadi}")
        if samvadi not in used and len(events) >= 8:
            warnings.append(f"phrase does not touch samvadi swara {samvadi}")

    if tala in TALA_MATRAS:
        matras = TALA_MATRAS[tala]
        overflows = [event for event in events if event.start_matra % matras + event.duration_matras > matras + 1]
        if overflows:
            warnings.append("some notes cross the tala cycle boundary")

    octave_jumps = 0
    for left, right in zip(events, events[1:]):
        if abs(right.octave - left.octave) > 1:
            octave_jumps += 1
    if octave_jumps:
        warnings.append("contains sudden multi-octave movement")

    unique_swaras = len(set(swaras))
    if unique_swaras <= 2 and len(events) > 6:
        warnings.append("low swara variety")

    score = 100 - len(errors) * 30 - len(warnings) * 8
    return {
        "valid": not errors,
        "score": max(0, score),
        "raga": raga,
        "tala": tala,
        "notes": len(events),
        "unique_swaras": unique_swaras,
        "errors": errors,
        "warnings": warnings,
        "teaching_hint": RAGA_DEFINITIONS.get(raga, {}).get("hint", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokens", help="Path to a tokenized sargam text file")
    args = parser.parse_args()
    sequences = [line.split() for line in Path(args.tokens).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not sequences:
        raise SystemExit(f"No token sequences found in {args.tokens}")
    for index, tokens in enumerate(sequences, start=1):
        print(f"sequence {index}: {validate_sargam_tokens(tokens)}")


if __name__ == "__main__":
    main()
