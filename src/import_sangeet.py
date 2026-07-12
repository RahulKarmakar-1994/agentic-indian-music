"""Convert the SANGEET XML dataset into the repo's sargam training format.

The source dataset uses Bhatkhande-style symbolic XML. This importer keeps a
conservative first-pass representation: recognizable swara symbols are extracted
from each composition and emitted as equal-matra sargam events. Ornament marks,
HTML-style rendering tags, and fine engraving details are ignored.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re
import xml.etree.ElementTree as ET


RAGA_SYMBOL_MAP = {
    "BHAIRAV": {
        "s": "S",
        "R": "r",
        "g": "G",
        "m": "M",
        "p": "P",
        "D": "d",
        "n": "N",
    },
    "TODI": {
        "s": "S",
        "R": "r",
        "G": "g",
        "M": "M^",
        "p": "P",
        "D": "d",
        "n": "N",
    },
    "POORVI": {
        "s": "S",
        "R": "r",
        "g": "G",
        "M": "M^",
        "p": "P",
        "D": "d",
        "n": "N",
    },
}

TALA_ALIASES = {
    "TRITAAL": "TRITAAL",
    "TEENTAAL": "TEENTAL",
    "TEEN TAAL": "TEENTAL",
    "EKTAAL": "EKTAAL",
    "EK TAAL": "EKTAAL",
    "JHAAPTAAL": "JHAAPTAAL",
    "JHAPTAAL": "JHAPTAAL",
    "JHAPTAL": "JHAPTAAL",
    "DADRA": "DADRA",
    "RUPAK": "RUPAK",
    "ROOPAK": "RUPAK",
    "KEHEWA": "KEHEWA",
    "KEHERWA": "KEHERWA",
    "TILWADA": "TILWADA",
    "TILWARA": "TILWARA",
    "JHOOMRA": "JHOOMRA",
    "CHOUTAAL": "CHOUTAAL",
    "CHAU TAAL": "CHAU_TAAL",
    "ADA CHOUTAAL": "ADA_CHOUTAAL",
    "DHAMAAR": "DHAMAAR",
    "DHAMAR": "DHAMAR",
    "SULTAAL": "SULTAAL",
    "BRAHMATAAL": "BRAHMATAAL",
}

ORNAMENT_CHARS = set("@`~!#$%^_qweabcx<>/()[]{}0123456789;:,\"'")


def normalize_name(value: str | None) -> str:
    return (value or "").strip().upper().replace("-", "_").replace(" ", "_")


def normalize_tala(value: str | None) -> str:
    raw = (value or "TEENTAL").strip().upper().replace("-", " ")
    return TALA_ALIASES.get(raw, raw.replace(" ", "_"))


def content_text(element: ET.Element) -> str:
    return " ".join("".join(element.itertext()).split())


def extract_swaras(text: str, raga: str) -> list[str]:
    symbol_map = RAGA_SYMBOL_MAP.get(raga, {})
    swaras: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "-":
            index += 1
            continue

        if char in symbol_map:
            swara = symbol_map[char]
            index += 1
            octave = ""
            if index < len(text) and text[index] in {"u", "l"}:
                octave = "'" if text[index] == "u" else "."
                index += 1
            swaras.append(f"{swara}{octave}:1")
            continue

        if char in ORNAMENT_CHARS or char.isspace() or not char.isalpha():
            index += 1
            continue

        index += 1
    return swaras


def convert_xml(path: Path) -> str | None:
    root = ET.parse(path).getroot()
    raga = normalize_name(root.findtext("./RAAG/RAAG_NAME"))
    tala = normalize_tala(root.findtext("./TAAL/TAAL_NAME"))
    if raga not in RAGA_SYMBOL_MAP:
        return None

    swaras: list[str] = []
    for content in root.findall(".//CONTENT"):
        swaras.extend(extract_swaras(content_text(content), raga))

    if len(swaras) < 8:
        return None

    grouped: list[str] = []
    for index, swara in enumerate(swaras):
        if index and index % 4 == 0:
            grouped.append("|")
        grouped.append(swara)

    source = path.name.replace(" ", "_")
    return f"RAGA={raga} TALA={tala} TEMPO=84 LAYA=MADHYA SOURCE={source} :: {' '.join(grouped)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/external/Sangeet/Bhatkhande Dataset")
    parser.add_argument("--output", default="data/sargam/sangeet_sargam.txt")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    lines = []
    skipped = 0
    for path in sorted(input_dir.glob("*.xml")):
        line = convert_xml(path)
        if line is None:
            skipped += 1
            continue
        lines.append(line)

    if not lines:
        raise SystemExit(f"No usable XML files found in {input_dir}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} converted compositions to {output}")
    if skipped:
        print(f"skipped {skipped} files")


if __name__ == "__main__":
    main()
