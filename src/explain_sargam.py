"""Create simple teaching notes for a generated or hand-written sargam phrase."""

from __future__ import annotations

from pathlib import Path
import argparse

try:
    from .sargam_tokens import RAGA_DEFINITIONS, SWARA_INTERVALS, TALA_MATRAS, tokens_to_sargam_events
except ImportError:
    from sargam_tokens import RAGA_DEFINITIONS, SWARA_INTERVALS, TALA_MATRAS, tokens_to_sargam_events


SWARA_LABELS = {
    "SA": "Sa",
    "RE_KOMAL": "komal Re",
    "RE": "Re",
    "GA_KOMAL": "komal Ga",
    "GA": "Ga",
    "MA": "Ma",
    "MA_TIVRA": "tivra Ma",
    "PA": "Pa",
    "DHA_KOMAL": "komal Dha",
    "DHA": "Dha",
    "NI_KOMAL": "komal Ni",
    "NI": "Ni",
}


def event_label(swara: str, octave: int) -> str:
    suffix = "'" if octave > 0 else "." if octave < 0 else ""
    return f"{SWARA_LABELS.get(swara, swara)}{suffix}"


def explain_sargam_tokens(tokens: list[str]) -> str:
    events, metadata = tokens_to_sargam_events(tokens)
    raga = str(metadata.get("raga", "UNKNOWN"))
    tala = str(metadata.get("tala", "UNKNOWN"))
    tempo = int(metadata.get("tempo", 84))
    definition = RAGA_DEFINITIONS.get(raga, {})
    matras = TALA_MATRAS.get(tala, 0)
    swaras = [event.swara for event in events]
    unique = []
    for swara in swaras:
        if swara not in unique:
            unique.append(swara)

    phrase = " ".join(event_label(event.swara, event.octave) for event in events)
    cells = []
    for start in range(0, len(events), 4):
        cell = events[start : start + 4]
        if cell:
            cells.append(" ".join(event_label(event.swara, event.octave) for event in cell))

    lines = [
        f"Raga: {raga}",
        f"Tala: {tala}" + (f" ({matras} matras)" if matras else ""),
        f"Tempo: {tempo} BPM",
        f"Swaras used: {', '.join(SWARA_LABELS.get(swara, swara) for swara in unique)}",
    ]
    if definition:
        lines.append(f"Raga color: {definition['hint']}")
        lines.append(f"Vadi/Samvadi focus: {SWARA_LABELS.get(str(definition['vadi']), definition['vadi'])} / {SWARA_LABELS.get(str(definition['samvadi']), definition['samvadi'])}")
    lines.extend([
        "",
        "Phrase:",
        phrase,
        "",
        "Practice cells:",
    ])
    lines.extend(f"{index}. {cell}" for index, cell in enumerate(cells, start=1))
    lines.extend([
        "",
        "Practice method:",
        "1. Clap the matras first without singing.",
        "2. Sing only Sa and the vadi swara slowly.",
        "3. Add the full sargam cell by cell.",
        "4. Repeat the final cell three times and land clearly on Sa.",
    ])

    outside = [swara for swara in unique if swara not in SWARA_INTERVALS]
    if outside:
        lines.append(f"Check these unknown swaras before practice: {', '.join(outside)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokens", help="Path to a tokenized sargam file")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--out", help="Optional markdown/text output path")
    args = parser.parse_args()

    sequences = [line.split() for line in Path(args.tokens).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not sequences:
        raise SystemExit(f"No token sequences found in {args.tokens}")
    if not 0 <= args.sequence_index < len(sequences):
        raise SystemExit(f"--sequence-index must be between 0 and {len(sequences) - 1}")

    lesson = explain_sargam_tokens(sequences[args.sequence_index])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(lesson + "\n", encoding="utf-8")
        print(f"wrote lesson to {args.out}")
    else:
        print(lesson)


if __name__ == "__main__":
    main()
