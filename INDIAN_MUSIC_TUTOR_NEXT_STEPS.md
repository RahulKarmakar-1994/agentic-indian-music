# Indian Music Tutor Pipeline

This repo now has a symbolic Indian music path for training on sargam, rendering
MIDI, validating raga constraints, and building tutor-style product features.

## Source Format

Write one phrase per line in `data/sargam/seed_sargam.txt`:

```text
RAGA=YAMAN TALA=TEENTAL TEMPO=84 LAYA=MADHYA :: S:1 R:1 G:1 M^:1 | P:1 D:1 N:1 S':2
```

- `S R G M P D N` are shuddha swaras.
- `r g d n` are komal swaras.
- `M^` is tivra Ma.
- `S'` is upper Sa; `S.` is lower Sa.
- Durations after `:` are in matras.
- Vibhag/phrase divisions are separated with `|`; a new tala cycle is inserted
  automatically when the matra count passes the tala length.

## Tokenize Sargam

```bash
conda run -n spyder-env python src/sargam_tokens.py encode-text \
  --input data/sargam/seed_sargam.txt \
  --output data/tokenized/seed_sargam.txt \
  --vocab data/tokenized/seed_sargam_vocab.json
```

## Validate Raga Rules

```bash
conda run -n spyder-env python src/validate_sargam.py \
  data/tokenized/seed_sargam.txt
```

## Render A Phrase To MIDI

```bash
conda run -n spyder-env python src/sargam_tokens.py decode-midi \
  --tokens data/tokenized/seed_sargam.txt \
  --sequence-index 0 \
  --sa C4 \
  --output generated/yaman_seed.mid
```

## Generate A Teaching Note

```bash
conda run -n spyder-env python src/explain_sargam.py \
  data/tokenized/seed_sargam.txt \
  --sequence-index 0 \
  --out generated/yaman_seed_lesson.md
```

## Train A Sargam Model

```bash
conda run -n spyder-env python src/train_music_llm.py \
  --corpus data/tokenized/seed_sargam.txt \
  --out models/sargam_llm.pt \
  --block-size 64 \
  --batch-size 16 \
  --max-iters 1200
```

## Generate Sargam

```bash
conda run -n spyder-env python src/generate_sargam.py \
  --checkpoint models/sargam_llm.pt \
  --prompt "BOS RAGA_YAMAN TALA_TEENTAL LAYA_MADHYA TEMPO_84 BAR" \
  --sa C4 \
  --tokens-out generated/generated_sargam.tokens \
  --midi-out generated/generated_sargam.mid
```

## Product Direction

The useful product loop is:

```text
Choose raga/tala/level -> generate sargam -> validate -> play -> explain -> practice -> mutate
```

Recommended teaching panels:

- swaras used in the raga
- komal/tivra notes
- vadi and samvadi notes
- aroha/avaroha practice
- tala counting with sam emphasis
- call-and-response exercises
- common mistakes, such as using a swara outside the raga

Avoid training on copyrighted film songs unless you have rights. Start with
teacher-authored exercises, public-domain bandishes, and self-created raga
phrases.

## Import The SANGEET XML Dataset

This repo includes a converter for the SANGEET Hindustani XML dataset. The
converter extracts recognizable Bhatkhande swara symbols and writes them into
the same sargam format used by the local model.

Download:

```bash
mkdir -p data/external
git clone --depth 1 https://github.com/cmisra/Sangeet.git data/external/Sangeet
```

Convert XML to friendly sargam:

```bash
conda run -n spyder-env python src/import_sangeet.py \
  --input-dir "data/external/Sangeet/Bhatkhande Dataset" \
  --output data/sargam/sangeet_sargam.txt
```

Tokenize:

```bash
conda run -n spyder-env python src/sargam_tokens.py encode-text \
  --input data/sargam/sangeet_sargam.txt \
  --output data/tokenized/sangeet_sargam.txt \
  --vocab data/tokenized/sangeet_sargam_vocab.json
```

Validate:

```bash
conda run -n spyder-env python src/validate_sargam.py \
  data/tokenized/sangeet_sargam.txt
```

Train:

```bash
conda run -n spyder-env python src/train_music_llm.py \
  --corpus data/tokenized/sangeet_sargam.txt \
  --out models/sangeet_sargam_llm.pt \
  --block-size 128 \
  --batch-size 32 \
  --max-iters 1200 \
  --n-embd 128 \
  --n-head 4 \
  --n-layer 4
```

Generate with raga repair:

```bash
conda run -n spyder-env python src/generate_sargam.py \
  --checkpoint models/sangeet_sargam_llm.pt \
  --prompt "BOS RAGA_BHAIRAV TALA_TRITAAL LAYA_MADHYA TEMPO_84 BAR" \
  --constrain-raga \
  --repair-raga \
  --sa C4 \
  --tokens-out generated/sangeet_bhairav_repaired.tokens \
  --midi-out generated/sangeet_bhairav_repaired.mid \
  --wav-out generated/sangeet_bhairav_repaired.wav \
  --lesson-out generated/sangeet_bhairav_repaired_lesson.md
```

The generated files, downloaded datasets, and trained checkpoints are ignored by
`.gitignore` so GitHub can stay focused on source code, seed examples, and docs.

`--constrain-raga` changes decoding only; it does not require retraining or a
new checkpoint. It masks out `SWARA_*` tokens that are not allowed in the chosen
raga before sampling.

## Run The Local Tutor UI

After cloning, install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Start the model-backed local server:

```bash
python3 src/sargam_server.py \
  --checkpoint models/sangeet_sargam_llm.pt \
  --host 127.0.0.1 \
  --port 8765
```

Open:

```text
http://127.0.0.1:8765/indian_tutor.html
```

The UI calls `/api/generate`, creates tokens, MIDI, WAV audio, and a lesson file
under `generated/`.

The WAV renderer includes a light Sa/Pa drone by default. For renderer ablation
or plain melody output:

```bash
python3 src/render_sargam_audio.py generated/demo.tokens \
  --output generated/demo_no_drone.wav \
  --sa C4 \
  --no-drone
```

## GitHub Notes

The small `models/sangeet_sargam_llm.pt` demo checkpoint is allowed in Git so a
new user can launch the UI after cloning. For larger future checkpoints, prefer:

- GitHub Release asset
- Git LFS for `models/*.pt`
- Hugging Face model repo

Large binary model files make normal Git history clunky.

## Research Next Steps

1. Add constrained decoding so the model cannot sample swaras outside the chosen
   raga, instead of repairing after generation.
2. Add novelty metrics: n-gram overlap, swara transition distance, and phrase
   similarity against SANGEET compositions.
3. Compare prompt-conditioned generation for Bhairav, Todi, and Poorvi.
4. Evaluate generated phrases with validator score, raga accuracy, and teacher
   judgment.
5. Preserve ornament/kan/meend information from SANGEET XML instead of flattening
   it to equal-matra notes.

## Run A Generation Evaluation

Use this to generate many samples and study validity, novelty, repetition, and
difficulty across ragas and temperatures:

```bash
python3 src/evaluate_generation.py \
  --checkpoint models/sangeet_sargam_llm.pt \
  --training-source data/sargam/sangeet_sargam.txt \
  --ragas BHAIRAV TODI POORVI \
  --temperatures 0.35 0.55 0.75 0.95 \
  --samples 8 \
  --max-new-tokens 110 \
  --csv-out generated/eval/generation_metrics.csv \
  --summary-out generated/eval/summary.md \
  --plot-dir generated/eval/plots
```

First local result from 96 samples:

- Raw raga validity was much lower for Bhairav than Todi/Poorvi.
- Raga repair lifted validity to 100%, but this is a post-processing fix.
- Higher temperature increased novelty at 0.95, while raw validity decreased.
- Repetition decreased as temperature increased.

This suggests the next research priority is constrained decoding and better
balanced phrase-level training data.

After adding constrained decoding, a small 24-sample check showed:

- Raw validity: 0.583
- Constrained validity: 1.000
- Repaired validity: 1.000
- Constrained novelty stayed close to repaired novelty.

This means constrained decoding is a better product default than repair alone,
while repair can remain as a safety fallback.

## Product Next Steps

1. Make the tutor UI support saved sessions and generated phrase history.
2. Add slow practice mode, loop cells, and metronome/sam emphasis.
3. Add user-uploaded teacher exercises in the same sargam format.
4. Add a better sound engine using SoundFont or a sampler.
5. Add export buttons for MIDI, WAV, lesson notes, and token data.

## Optional Ollama Tutor Layer

Ollama can be used as a local LLM layer for conversational explanations. It does
not replace the sargam model; it only rewrites/explains the generated lesson.

Install Ollama:

```text
https://ollama.com
```

Pull a small local model:

```bash
ollama pull llama3.2:1b
```

Run Ollama if needed:

```bash
ollama serve
```

Then start this repo's UI server and enable **Ollama tutor** in the browser.

Important: the local Ollama API still needs a model downloaded on the machine.
To avoid large downloads, start with `llama3.2:1b` or another small model from
the Ollama library.
