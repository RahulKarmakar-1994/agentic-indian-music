"""Generate music tokens with your trained music LLM and render MIDI."""

from __future__ import annotations

from pathlib import Path
import argparse

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is required for generation. Run: python3 -m pip install -r requirements.txt") from exc

try:
    from .music_tokens import tokens_to_events
    from .simple_midi import write_midi
    from .train_music_llm import ModelConfig, MusicLanguageModel
    from .validate_music import validate_tokens
except ImportError:
    from music_tokens import tokens_to_events
    from simple_midi import write_midi
    from train_music_llm import ModelConfig, MusicLanguageModel
    from validate_music import validate_tokens


def load_model(path: str | Path, device: str):
    checkpoint = torch.load(path, map_location=device)
    config = ModelConfig(**checkpoint["config"])
    model = MusicLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stoi = checkpoint["stoi"]
    itos = {int(key): value for key, value in checkpoint["itos"].items()}
    return model, stoi, itos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/music_llm.pt")
    parser.add_argument("--prompt", default="BOS TEMPO_92 TS_4_4 BAR")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--tokens-out", default="generated/generated.tokens")
    parser.add_argument("--midi-out", default="generated/generated.mid")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stoi, itos = load_model(args.checkpoint, device)
    prompt_tokens = args.prompt.split()
    missing = [token for token in prompt_tokens if token not in stoi]
    if missing:
        raise SystemExit(f"Prompt contains tokens missing from vocabulary: {missing}")

    idx = torch.tensor([[stoi[token] for token in prompt_tokens]], dtype=torch.long, device=device)
    generated = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k)[0].tolist()
    tokens = [itos[index] for index in generated]
    if "EOS" in tokens:
        tokens = tokens[: tokens.index("EOS") + 1]
    else:
        tokens.append("EOS")

    Path(args.tokens_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.tokens_out).write_text(" ".join(tokens) + "\n", encoding="utf-8")
    notes, tempo = tokens_to_events(tokens)
    write_midi(args.midi_out, notes, tempo_us_per_beat=tempo)

    result = validate_tokens(tokens)
    print(f"wrote tokens to {args.tokens_out}")
    print(f"wrote MIDI to {args.midi_out}")
    print(f"validation: {result}")


if __name__ == "__main__":
    main()

