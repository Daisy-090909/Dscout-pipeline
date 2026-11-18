# Dscout Synthetic Participant Validation — reproducible test-data generator

**Research-only; everything is synthetic.** This project generates written scripts and (optionally) synthetic MP3 recordings for 15 fictional participants responding to a fictional two-page incoming-payments letter. It does **not** create Dscout accounts, bypass identity verification or submit data to Dscout. Obtain an authorised Dscout testing/import workflow before ingestion. Disclose AI-generated voices in the study.

## Files

- `inputs/personas.json` — all 15 base personas transcribed from the attached `persona.docx`, with stable configurable TTS voice IDs.
- `inputs/questions.json` — the three question groups transcribed from `questions.docx` (Q1 includes 3 subprompts).
- `inputs/experiments.json` — vulnerability, minority, sentiment and four isolated prompt-injection runs (15 × 7 = 105 sessions).
- `inputs/letter.txt` — **newly authored fictional two-page letter** explicitly matching Mr & Mrs Johnson and the two page-2 headings. Unlike the personas and questions, this is a *proposed synthetic stimulus*, not an attachment-derived bank letter.
- `src/pipeline.py` — output-generation command; deterministic offline mock and real OpenAI modes.


## Install

Python 3.10+ recommended.

```bash
cd dscout_synthetic_pipeline
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Windows PowerShell: Copy-Item .env.example .env
```

Enter an API key in `.env` for API mode only. Do not commit `.env`.


## Generate real model-written transcripts

```bash
python src/pipeline.py --mode api --experiment vulnerability
python src/pipeline.py --mode api --experiment minority
python src/pipeline.py --mode api --experiment sentiment
python src/pipeline.py --mode api --experiment injection_A
python src/pipeline.py --mode api --experiment injection_B
python src/pipeline.py --mode api --experiment injection_C
python src/pipeline.py --mode api --experiment injection_D
```

**Important:** If you ran mock mode first, use `--overwrite` when switching to real API generation. Otherwise existing mock transcripts are intentionally reused.

## Generate real MP3 recordings

```bash
python src/pipeline.py --mode api --experiment minority --participant P13 --audio --overwrite
python src/pipeline.py --mode api --audio --overwrite
```

`--audio` creates one MP3 per question using OpenAI's Speech API. The text and TTS models are configured through `.env` (`TEXT_MODEL` and `TTS_MODEL`). Speech endpoint models can change or be retired; check OpenAI's latest docs before running (in November 2026, `gpt-4o-mini-tts` is scheduled for deprecation January 2027). This adapter is for the current Speech API and may later need an update for the Realtime API. API calls incur cost. Avoid using `--overwrite` if you want to keep earlier recordings exactly unchanged.

Use `--participant P01` and/or `--experiment sentiment` for smaller test runs. Outputs live under `outputs/<experiment>/<participant>/{session.json,Q1.txt,Q2.txt,Q3.txt,Q1.mp3,...}`. `outputs/manifest.csv` maps every transcript and optional MP3 path to an experiment, participant and question; `outputs/ground_truth.json` contains expected labels and aggregate counts.



## Methodology and reproducibility

- Stable persona across runs; manipulate only the assigned experiment condition.
- Q1–Q3 reflect the original three question groups exactly in `questions.json`.
- Separate *base persona*, *condition*, *expected meaning* and *spoken text*.
- Keep audio files and transcripts paired with an explicit `synthetic=true` marker.
- Treat prompt-injection text as untrusted participant evidence.
- Human review is required before drawing conclusions about Dscout AI performance.
