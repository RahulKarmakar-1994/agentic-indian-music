"""Generate raga-aware sargam tokens with a trained checkpoint and render MIDI."""

from __future__ import annotations

from pathlib import Path
import argparse

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is required for generation. Run: python3 -m pip install -r requirements.txt") from exc

try:
    from .explain_sargam import explain_sargam_tokens
    from .render_sargam_audio import render_tokens_to_wav
    from .sargam_tokens import RAGA_DEFINITIONS, SWARA_INTERVALS, sequence_metadata, tokens_to_midi_notes
    from .simple_midi import write_midi
    from .train_music_llm import ModelConfig, MusicLanguageModel
    from .validate_sargam import validate_sargam_tokens
except ImportError:
    from explain_sargam import explain_sargam_tokens
    from render_sargam_audio import render_tokens_to_wav
    from sargam_tokens import RAGA_DEFINITIONS, SWARA_INTERVALS, sequence_metadata, tokens_to_midi_notes
    from simple_midi import write_midi
    from train_music_llm import ModelConfig, MusicLanguageModel
    from validate_sargam import validate_sargam_tokens


def load_model(path: str | Path, device: str):
    checkpoint = torch.load(path, map_location=device)
    config = ModelConfig(**checkpoint["config"])
    model = MusicLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stoi = checkpoint["stoi"]
    itos = {int(key): value for key, value in checkpoint["itos"].items()}
    return model, stoi, itos


def nearest_allowed_swara(swara: str, allowed: set[str]) -> str:
    if swara in allowed or swara not in SWARA_INTERVALS:
        return swara
    source = SWARA_INTERVALS[swara]
    return min(allowed, key=lambda item: abs(SWARA_INTERVALS[item] - source))


def repair_raga(tokens: list[str]) -> list[str]:
    metadata = sequence_metadata(tokens)
    raga = str(metadata.get("raga", "UNKNOWN"))
    definition = RAGA_DEFINITIONS.get(raga)
    if not definition:
        return tokens
    allowed = set(definition["allowed"])
    repaired = []
    for token in tokens:
        if token.startswith("SWARA_"):
            swara = token.removeprefix("SWARA_")
            repaired.append(f"SWARA_{nearest_allowed_swara(swara, allowed)}")
        else:
            repaired.append(token)
    return repaired


def raga_from_prompt(prompt: str) -> str:
    for token in prompt.split():
        if token.startswith("RAGA_"):
            return token.removeprefix("RAGA_")
    return "UNKNOWN"


def invalid_swara_ids(stoi: dict[str, int], raga: str) -> list[int]:
    definition = RAGA_DEFINITIONS.get(raga)
    if not definition:
        return []
    allowed = set(definition["allowed"])
    invalid = []
    for token, token_id in stoi.items():
        if token.startswith("SWARA_") and token.removeprefix("SWARA_") not in allowed:
            invalid.append(token_id)
    return invalid


@torch.no_grad()
def sample_tokens(
    model: MusicLanguageModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    masked_token_ids: list[int] | None = None,
) -> torch.Tensor:
    masked_token_ids = masked_token_ids or []
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(0.05, temperature)
        if masked_token_ids:
            logits[:, masked_token_ids] = -float("inf")
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits[logits < values[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_tokens(
    model: MusicLanguageModel,
    stoi: dict[str, int],
    itos: dict[int, str],
    prompt: str,
    max_new_tokens: int = 160,
    temperature: float = 0.75,
    top_k: int | None = 18,
    repair: bool = False,
    constrain_raga: bool = False,
    device: str = "cpu",
) -> list[str]:
    prompt_tokens = prompt.split()
    missing = [token for token in prompt_tokens if token not in stoi]
    if missing:
        raise ValueError(f"Prompt contains tokens missing from vocabulary: {missing}")

    idx = torch.tensor([[stoi[token] for token in prompt_tokens]], dtype=torch.long, device=device)
    masked_token_ids = invalid_swara_ids(stoi, raga_from_prompt(prompt)) if constrain_raga else []
    generated = sample_tokens(model, idx, max_new_tokens, temperature, top_k, masked_token_ids)[0].tolist()
    tokens = [itos[index] for index in generated]
    if "EOS" in tokens:
        tokens = tokens[: tokens.index("EOS") + 1]
    else:
        tokens.append("EOS")
    if repair:
        tokens = repair_raga(tokens)
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/sargam_llm.pt")
    parser.add_argument("--prompt", default="BOS RAGA_YAMAN TALA_TEENTAL LAYA_MADHYA TEMPO_84 BAR")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=18)
    parser.add_argument("--sa", default="C4")
    parser.add_argument("--constrain-raga", action="store_true", help="Block out-of-raga SWARA tokens during sampling")
    parser.add_argument("--repair-raga", action="store_true", help="Map out-of-raga swaras to nearest allowed swara before rendering")
    parser.add_argument("--tokens-out", default="generated/generated_sargam.tokens")
    parser.add_argument("--midi-out", default="generated/generated_sargam.mid")
    parser.add_argument("--wav-out", help="Optional WAV audio output path")
    parser.add_argument("--lesson-out", help="Optional teaching-note output path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stoi, itos = load_model(args.checkpoint, device)
    try:
        tokens = generate_tokens(
            model,
            stoi,
            itos,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repair=args.repair_raga,
            constrain_raga=args.constrain_raga,
            device=device,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    Path(args.tokens_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.tokens_out).write_text(" ".join(tokens) + "\n", encoding="utf-8")
    notes, tempo_us = tokens_to_midi_notes(tokens, sa=args.sa)
    write_midi(args.midi_out, notes, tempo_us_per_beat=tempo_us)
    if args.wav_out:
        render_result = render_tokens_to_wav(tokens, args.wav_out, sa=args.sa)
        print(f"wrote WAV to {render_result['path']} ({render_result['seconds']} sec)")
    if args.lesson_out:
        Path(args.lesson_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.lesson_out).write_text(explain_sargam_tokens(tokens) + "\n", encoding="utf-8")
        print(f"wrote lesson to {args.lesson_out}")

    print(f"wrote tokens to {args.tokens_out}")
    print(f"wrote MIDI to {args.midi_out}")
    print(f"validation: {validate_sargam_tokens(tokens)}")


if __name__ == "__main__":
    main()
