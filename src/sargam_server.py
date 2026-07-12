"""Small local server for the Indian music tutor UI."""

from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import argparse
import json
import time
import traceback

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is required. Run inside spyder-env or install requirements.txt") from exc

try:
    from .explain_sargam import explain_sargam_tokens
    from .generate_sargam import generate_tokens, load_model
    from .render_sargam_audio import render_tokens_to_wav
    from .sargam_tokens import RAGA_DEFINITIONS, TALA_MATRAS, sequence_metadata, tokens_to_midi_notes
    from .simple_midi import write_midi
    from .validate_sargam import validate_sargam_tokens
except ImportError:
    from explain_sargam import explain_sargam_tokens
    from generate_sargam import generate_tokens, load_model
    from render_sargam_audio import render_tokens_to_wav
    from sargam_tokens import RAGA_DEFINITIONS, TALA_MATRAS, sequence_metadata, tokens_to_midi_notes
    from simple_midi import write_midi
    from validate_sargam import validate_sargam_tokens


ROOT = Path(__file__).resolve().parents[1]


def safe_name(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_") or "sargam"


def build_prompt(payload: dict[str, object]) -> str:
    raga = str(payload.get("raga", "BHAIRAV")).upper()
    tala = str(payload.get("tala", "TRITAAL")).upper()
    laya = str(payload.get("laya", "MADHYA")).upper()
    tempo = int(payload.get("tempo", 84))
    if raga not in RAGA_DEFINITIONS:
        raise ValueError(f"Unsupported raga: {raga}")
    if tala not in TALA_MATRAS:
        raise ValueError(f"Unsupported tala: {tala}")
    return f"BOS RAGA_{raga} TALA_{tala} LAYA_{laya} TEMPO_{tempo} BAR"


class TutorState:
    def __init__(self, checkpoint: Path):
        self.checkpoint = checkpoint
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.stoi, self.itos = load_model(checkpoint, self.device)


class TutorHandler(SimpleHTTPRequestHandler):
    state: TutorState

    def __init__(self, *args, directory: str | None = None, **kwargs):
        super().__init__(*args, directory=str(ROOT if directory is None else directory), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.path = "/indian_tutor.html"
        elif parsed.path == "/api/options":
            self.send_json({
                "ragas": sorted(RAGA_DEFINITIONS),
                "talas": sorted(TALA_MATRAS),
                "checkpoint": str(self.state.checkpoint.relative_to(ROOT)),
                "device": self.state.device,
            })
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/generate":
            self.send_error(404, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            result = self.generate(payload)
            self.send_json(result)
        except Exception as exc:
            traceback.print_exc()
            self.send_json({"error": str(exc)}, status=500)

    def generate(self, payload: dict[str, object]) -> dict[str, object]:
        prompt = build_prompt(payload)
        raga = str(payload.get("raga", "BHAIRAV")).upper()
        sa = str(payload.get("sa", "C4"))
        max_new_tokens = int(payload.get("max_new_tokens", 140))
        temperature = float(payload.get("temperature", 0.55))
        top_k = int(payload.get("top_k", 10))
        repair = bool(payload.get("repair", True))
        constrain = bool(payload.get("constrain", True))

        tokens = generate_tokens(
            self.state.model,
            self.state.stoi,
            self.state.itos,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repair=repair,
            constrain_raga=constrain,
            device=self.state.device,
        )
        validation = validate_sargam_tokens(tokens)
        lesson = explain_sargam_tokens(tokens)
        stamp = int(time.time() * 1000)
        base = ROOT / "generated" / f"ui_{safe_name(raga)}_{stamp}"
        tokens_path = base.with_suffix(".tokens")
        midi_path = base.with_suffix(".mid")
        wav_path = base.with_suffix(".wav")
        lesson_path = base.with_name(base.name + "_lesson.md")

        tokens_path.parent.mkdir(parents=True, exist_ok=True)
        tokens_path.write_text(" ".join(tokens) + "\n", encoding="utf-8")
        notes, tempo_us = tokens_to_midi_notes(tokens, sa=sa)
        write_midi(midi_path, notes, tempo_us_per_beat=tempo_us)
        audio = render_tokens_to_wav(tokens, wav_path, sa=sa)
        lesson_path.write_text(lesson + "\n", encoding="utf-8")
        metadata = sequence_metadata(tokens)

        return {
            "prompt": prompt,
            "tokens": tokens,
            "metadata": metadata,
            "validation": validation,
            "lesson": lesson,
            "audio": audio,
            "files": {
                "tokens": "/" + str(tokens_path.relative_to(ROOT)),
                "midi": "/" + str(midi_path.relative_to(ROOT)),
                "wav": "/" + str(wav_path.relative_to(ROOT)),
                "lesson": "/" + str(lesson_path.relative_to(ROOT)),
            },
        }

    def send_json(self, payload: dict[str, object], status: int = 200) -> None:
        raw = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/sangeet_sargam_llm.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    checkpoint = ROOT / args.checkpoint
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    TutorHandler.state = TutorState(checkpoint)
    server = ThreadingHTTPServer((args.host, args.port), TutorHandler)
    print(f"Indian tutor UI: http://{args.host}:{args.port}/indian_tutor.html")
    print(f"Using checkpoint: {checkpoint}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
