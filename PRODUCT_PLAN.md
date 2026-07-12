# Agentic Music Platform Plan

## Is It A Good Idea?

Yes, but the strongest version is not "another prompt-to-song generator." Suno, Udio, Stable Audio, Soundraw, Mubert, AIVA, and similar tools already cover fast generation. Your research angle can be different: an agentic music learning and discovery platform where the system generates symbolic music, critiques it, teaches the theory, mutates it, and helps the user develop musical taste.

The analogy to your SMILES work is strong:

- SMILES tokens -> music tokens such as pitch, duration, chord, velocity, articulation, section, motif.
- RDKit validity -> music validity such as playable range, rhythmic consistency, harmonic fit, phrase balance, and novelty.
- Canonicalization/uniqueness -> motif fingerprints, interval contours, rhythmic hashes, and training-set similarity checks.
- Property optimization -> target emotion, tension curve, learnability, genre fit, technical difficulty, or raga/scale compliance.

## Product Thesis

Build a "music molecule lab": users grow new motifs, not just songs. The platform should make every generated idea inspectable, playable, editable, explainable, and learnable.

## Differentiated Product Concept

Working name: MotifLab Agents

Core promise: "Generate original musical ideas and learn why they work."

The market has many AI music generators. The gap is a product that teaches composition while generating:

- A Composer Agent creates motifs, chords, grooves, and variations.
- A Theory Tutor explains scale degrees, tension, rhythm, and arrangement choices.
- A Critic Agent scores novelty, singability, balance, and training-set similarity.
- A Mutation Agent creates controlled variations: invert, stretch, reharmonize, syncopate, simplify.
- A Curator Agent stores a lineage tree so users can see how ideas evolved.

## Recommended Technical Path

### Phase 1: Symbolic Music MVP

Do not start with raw audio. Start with symbolic music:

- Data formats: MIDI, MusicXML, ABC notation, Humdrum, or tokenized event streams.
- Token examples: `NOTE_C4`, `DUR_1_8`, `VEL_72`, `CHORD_ii`, `BAR`, `REST`, `SECTION_A`.
- Model: causal Transformer similar to your SMILES model, first trained on monophonic melodies, then chords, then multi-track MIDI.
- Validity checks: timing sums per bar, scale adherence, instrument range, no impossible note density.
- Novelty checks: n-gram overlap, interval-contour similarity, rhythm hash similarity.

### Phase 2: Agentic Layer

Use a larger LLM above the music model, not necessarily as the music generator itself:

- Converts user intent into constraints.
- Chooses dataset/model/style.
- Calls symbolic generator many times.
- Filters using music validators.
- Explains results in natural language.
- Suggests practice exercises and next variations.

### Phase 3: Audio Layer

Add sound after the symbolic layer works:

- Render MIDI through SoundFonts or synthesis.
- Add stems: melody, bass, chord, drums.
- Later connect to models such as MusicGen-style audio generation for timbre and arrangement.

## Data Strategy

Start legally and cleanly:

- Public-domain classical MIDI.
- Lakh MIDI Dataset with careful license review.
- User-owned MIDI uploads.
- Your own generated/annotated phrase dataset.
- For Indian music learning, build a small curated raga phrase corpus with metadata.

Avoid training on copyrighted commercial songs unless you have rights.

## Research Milestones

1. Train a char/token model on monophonic MIDI melodies.
2. Generate valid 4-bar motifs.
3. Add BPE-like token merges for recurring music phrases.
4. Add validators and novelty scoring.
5. Add controlled conditioning: scale, BPM, chord progression, difficulty.
6. Add multi-agent generation loop.
7. Add browser interface with playback and visual piano roll.
8. Add export: MIDI, WAV render, lesson notes.

## First Product Prototype

The included `index.html`, `styles.css`, and `app.js` implement a browser prototype. It generates short symbolic motifs, plays them with Web Audio, visualizes a piano roll, and presents agent-style critique/learning panels.

