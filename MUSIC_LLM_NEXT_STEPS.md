# Own Music LLM: Next Step

This repo now has the first version of a symbolic music language pipeline. It is designed to evolve your SMILES Transformer idea into a music-token Transformer.

## What Was Added

- `src/simple_midi.py`: dependency-free MIDI reader/writer for melody-first experiments.
- `src/music_tokens.py`: converts MIDI files to token sequences and token sequences back to MIDI.
- `src/validate_music.py`: validates generated token sequences.
- `src/train_music_llm.py`: small GPT-style PyTorch model for your own music LLM.
- `src/generate_music.py`: generates music tokens from a trained checkpoint and exports MIDI.
- `data/tokenized/seed_motifs.txt`: tiny starter corpus for smoke testing.
- `requirements.txt`: dependencies for training.

## Music Token Format

Example:

```text
BOS TEMPO_92 TS_4_4 BAR POS_0 NOTE_60 DUR_2 VEL_80 POS_2 NOTE_62 DUR_2 VEL_80 EOS
```

Meaning:

- `BOS` / `EOS`: sequence boundary.
- `TEMPO_92`: tempo in BPM.
- `TS_4_4`: time signature marker.
- `BAR`: new bar.
- `POS_0`: sixteenth-note position inside the bar.
- `NOTE_60`: MIDI pitch.
- `DUR_2`: duration in sixteenth-note steps.
- `VEL_80`: velocity.

## First Real Training Run

1. Add MIDI files to:

```text
data/raw_midi/
```

2. Tokenize them:

```bash
python3 src/music_tokens.py encode-midi \
  --input-dir data/raw_midi \
  --output data/tokenized/corpus.txt \
  --vocab data/tokenized/vocab.json \
  --max-bars 16
```

3. Install training dependencies:

```bash
python3 -m pip install -r requirements.txt
```

4. Train your own music LLM:

```bash
python3 src/train_music_llm.py \
  --corpus data/tokenized/corpus.txt \
  --out models/music_llm.pt \
  --block-size 128 \
  --max-iters 1200
```

5. Generate a new motif:

```bash
python3 src/generate_music.py \
  --checkpoint models/music_llm.pt \
  --prompt "BOS TEMPO_92 TS_4_4 BAR" \
  --tokens-out generated/generated.tokens \
  --midi-out generated/generated.mid
```

## Recommended Dataset Choice

Start with monophonic or melody-heavy MIDI. Do not begin with full orchestral or dense piano MIDI. The first goal is not beautiful production; it is valid symbolic generation.

Good first datasets:

- Your own short melodies.
- Public-domain folk/classical melodies.
- Small curated raga phrases, if your learning goal includes Indian music.
- User-owned MIDI exports from a DAW.

## How This Connects To The Interface

The current browser app uses a handcrafted motif generator. The next integration step is:

1. Train `models/music_llm.pt`.
2. Create a small local API that calls `src/generate_music.py`.
3. Replace the browser generator in `app.js` with model-generated token sequences.
4. Render those tokens in the existing piano-roll and agent panels.

