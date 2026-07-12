"""Evaluate raga-conditioned sargam generation and plot simple research metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import math
import statistics

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is required. Run: python3 -m pip install -r requirements.txt") from exc

try:
    from .generate_sargam import generate_tokens, load_model, repair_raga
    from .sargam_tokens import read_sargam_source, sequence_metadata, tokens_to_sargam_events
    from .validate_sargam import validate_sargam_tokens
except ImportError:
    from generate_sargam import generate_tokens, load_model, repair_raga
    from sargam_tokens import read_sargam_source, sequence_metadata, tokens_to_sargam_events
    from validate_sargam import validate_sargam_tokens


@dataclass(frozen=True)
class TrainingProfile:
    raga: str
    swara_ngrams: Counter[tuple[str, ...]]


def swara_sequence(tokens: list[str]) -> list[str]:
    events, _ = tokens_to_sargam_events(tokens)
    return [event.swara for event in events]


def ngrams(items: list[str], n: int = 3) -> Counter[tuple[str, ...]]:
    if len(items) < n:
        return Counter()
    return Counter(tuple(items[index : index + n]) for index in range(len(items) - n + 1))


def jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def build_profiles(source_path: str | Path) -> list[TrainingProfile]:
    profiles = []
    for tokens in read_sargam_source(source_path):
        metadata = sequence_metadata(tokens)
        raga = str(metadata.get("raga", "UNKNOWN"))
        profiles.append(TrainingProfile(raga=raga, swara_ngrams=ngrams(swara_sequence(tokens))))
    return profiles


def nearest_similarity(generated: list[str], profiles: list[TrainingProfile], raga: str) -> float:
    generated_set = set(ngrams(generated))
    candidates = [profile for profile in profiles if profile.raga == raga] or profiles
    if not candidates:
        return 0.0
    return max(jaccard(generated_set, set(profile.swara_ngrams)) for profile in candidates)


def repetition_rate(swaras: list[str]) -> float:
    grams = ngrams(swaras, n=3)
    total = sum(grams.values())
    if total == 0:
        return 0.0
    repeated = sum(count - 1 for count in grams.values() if count > 1)
    return repeated / total


def difficulty_score(tokens: list[str]) -> float:
    events, _ = tokens_to_sargam_events(tokens)
    if not events:
        return 0.0
    unique_swaras = len({event.swara for event in events})
    octave_jumps = sum(1 for left, right in zip(events, events[1:]) if abs(right.octave - left.octave) > 0)
    density = len(events) / max(1, len(tokens) / 16)
    return min(100.0, unique_swaras * 8 + octave_jumps * 2 + density * 4)


def row_for_tokens(
    tokens: list[str],
    raw_tokens: list[str],
    raga: str,
    tala: str,
    temperature: float,
    sample_index: int,
    profiles: list[TrainingProfile],
) -> dict[str, object]:
    validation = validate_sargam_tokens(tokens)
    raw_validation = validate_sargam_tokens(raw_tokens)
    swaras = swara_sequence(tokens)
    nearest = nearest_similarity(swaras, profiles, raga)
    return {
        "raga": raga,
        "tala": tala,
        "temperature": temperature,
        "sample": sample_index,
        "raw_valid": raw_validation["valid"],
        "raw_score": raw_validation["score"],
        "valid": validation["valid"],
        "score": validation["score"],
        "notes": validation["notes"],
        "unique_swaras": validation["unique_swaras"],
        "nearest_similarity": round(nearest, 4),
        "novelty": round(1.0 - nearest, 4),
        "repetition_rate": round(repetition_rate(swaras), 4),
        "difficulty": round(difficulty_score(tokens), 2),
        "errors": "; ".join(validation["errors"]),
        "warnings": "; ".join(validation["warnings"]),
    }


def write_csv(rows: list[dict[str, object]], path: str | Path) -> None:
    if not rows:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_mean(rows: list[dict[str, object]], key: str, value: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(float(row[value]))
    return {item: statistics.mean(values) for item, values in grouped.items()}


def write_summary(rows: list[dict[str, object]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temperatures = sorted({float(row["temperature"]) for row in rows})
    ragas = sorted({str(row["raga"]) for row in rows})
    lines = [
        "# Generation Evaluation Summary",
        "",
        f"Samples: {len(rows)}",
        f"Ragas: {', '.join(ragas)}",
        f"Temperatures: {', '.join(str(value) for value in temperatures)}",
        "",
        "## Overall",
        "",
        f"Raw validity: {statistics.mean(1.0 if row['raw_valid'] else 0.0 for row in rows):.3f}",
        f"Repaired validity: {statistics.mean(1.0 if row['valid'] else 0.0 for row in rows):.3f}",
        f"Mean novelty: {statistics.mean(float(row['novelty']) for row in rows):.3f}",
        f"Mean repetition: {statistics.mean(float(row['repetition_rate']) for row in rows):.3f}",
        f"Mean difficulty: {statistics.mean(float(row['difficulty']) for row in rows):.2f}",
        "",
        "## Mean Novelty By Temperature",
        "",
    ]
    novelty_by_temp = grouped_mean(rows, "temperature", "novelty")
    for temperature in sorted(novelty_by_temp, key=float):
        lines.append(f"- {temperature}: {novelty_by_temp[temperature]:.3f}")
    lines.extend(["", "## Raw Validity By Raga", ""])
    for raga in ragas:
        subset = [row for row in rows if row["raga"] == raga]
        lines.append(f"- {raga}: {statistics.mean(1.0 if row['raw_valid'] else 0.0 for row in subset):.3f}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(rows: list[dict[str, object]], output_dir: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots")
        return

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    temperatures = sorted({float(row["temperature"]) for row in rows})
    metrics = ["novelty", "repetition_rate", "difficulty", "raw_score"]

    for metric in metrics:
        values = []
        for temperature in temperatures:
            subset = [row for row in rows if float(row["temperature"]) == temperature]
            values.append(statistics.mean(float(row[metric]) for row in subset))
        plt.figure(figsize=(7, 4))
        plt.plot(temperatures, values, marker="o")
        plt.xlabel("temperature")
        plt.ylabel(metric)
        plt.title(f"{metric} vs temperature")
        plt.tight_layout()
        plt.savefig(output / f"{metric}_by_temperature.png", dpi=160)
        plt.close()

    ragas = sorted({str(row["raga"]) for row in rows})
    raw_validity = []
    repaired_validity = []
    for raga in ragas:
        subset = [row for row in rows if row["raga"] == raga]
        raw_validity.append(statistics.mean(1.0 if row["raw_valid"] else 0.0 for row in subset))
        repaired_validity.append(statistics.mean(1.0 if row["valid"] else 0.0 for row in subset))
    x = range(len(ragas))
    plt.figure(figsize=(7, 4))
    plt.bar([item - 0.18 for item in x], raw_validity, width=0.36, label="raw")
    plt.bar([item + 0.18 for item in x], repaired_validity, width=0.36, label="repaired")
    plt.xticks(list(x), ragas)
    plt.ylim(0, 1.05)
    plt.ylabel("validity rate")
    plt.title("raga validity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "validity_by_raga.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/sangeet_sargam_llm.pt")
    parser.add_argument("--training-source", default="data/sargam/sangeet_sargam.txt")
    parser.add_argument("--ragas", nargs="+", default=["BHAIRAV", "TODI", "POORVI"])
    parser.add_argument("--tala", default="TRITAAL")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.35, 0.55, 0.75, 0.95])
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--csv-out", default="generated/eval/generation_metrics.csv")
    parser.add_argument("--summary-out", default="generated/eval/summary.md")
    parser.add_argument("--plot-dir", default="generated/eval/plots")
    args = parser.parse_args()

    profiles = build_profiles(args.training_source)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stoi, itos = load_model(args.checkpoint, device)
    rows = []

    for raga in args.ragas:
        for temperature in args.temperatures:
            for sample_index in range(args.samples):
                torch.manual_seed(args.seed + sample_index + int(temperature * 1000) + len(rows))
                prompt = f"BOS RAGA_{raga} TALA_{args.tala} LAYA_MADHYA TEMPO_84 BAR"
                raw_tokens = generate_tokens(
                    model,
                    stoi,
                    itos,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=temperature,
                    top_k=args.top_k,
                    repair=False,
                    device=device,
                )
                repaired_tokens = repair_raga(raw_tokens)
                rows.append(row_for_tokens(repaired_tokens, raw_tokens, raga, args.tala, temperature, sample_index, profiles))

    write_csv(rows, args.csv_out)
    write_summary(rows, args.summary_out)
    write_plots(rows, args.plot_dir)
    print(f"wrote metrics to {args.csv_out}")
    print(f"wrote summary to {args.summary_out}")
    print(f"wrote plots to {args.plot_dir}")


if __name__ == "__main__":
    main()
