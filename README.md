# Agentic Musics

Symbolic Indian music generation and tutoring experiments.

This repo explores a SMILES-style next-token generation idea for music: instead
of predicting molecular tokens, a small causal Transformer predicts sargam
tokens conditioned on raga, tala, laya, and tempo.

## What Works

- Convert Indian sargam notation into token sequences.
- Import the SANGEET XML Hindustani notation dataset into the local sargam
  format.
- Train a small local Transformer on symbolic music tokens.
- Generate raga-conditioned sargam phrases.
- Validate generated phrases against raga/tala rules.
- Render generated phrases to MIDI and simple WAV audio.
- Produce basic practice/lesson notes.
- Run a small local browser UI for generation and listening.

## Local Tutor UI

Train or place a checkpoint at:

```text
models/sangeet_sargam_llm.pt
```

Then run:

```bash
conda run -n spyder-env python src/sargam_server.py \
  --checkpoint models/sangeet_sargam_llm.pt \
  --host 127.0.0.1 \
  --port 8765
```

Open:

```text
http://127.0.0.1:8765/indian_tutor.html
```

## Data

The repo includes small hand-written seed examples. Larger downloaded datasets
and derived corpora are ignored by Git.

The SANGEET dataset can be downloaded separately:

```bash
mkdir -p data/external
git clone --depth 1 https://github.com/cmisra/Sangeet.git data/external/Sangeet
```

Then convert/tokenize/train using the commands in
`INDIAN_MUSIC_TUTOR_NEXT_STEPS.md`.

## Model Notes

The trained checkpoints are ignored by default:

```text
models/
```

For sharing checkpoints, prefer GitHub Releases, Git LFS, or a model registry.
Keeping binary checkpoints in normal Git is possible, but it makes repository
history bulky.

## Status

This is a research/prototype repo, not a polished product yet. The current
audio renderer is a simple dependency-free synth for listening tests. Product
quality audio should later use SoundFont rendering or a sampler.

## Next Steps

- Add constrained decoding instead of post-generation raga repair.
- Add novelty/similarity metrics against the training corpus.
- Preserve more SANGEET ornamentation and timing details.
- Add saved generation history in the UI.
- Add better audio rendering.
- Evaluate outputs with musicians/teachers.
