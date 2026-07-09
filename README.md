# ResearchGap AI — Engine

Real six-stage pipeline. No static data, no hardcoded results — every run
fetches live literature and computes gaps from it at that moment.

## Run it (Mac)

```bash
cd researchgap-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then edit .env:
                            #   OPENALEX_MAILTO = your real email (free, no signup)
                            #   ANTHROPIC_API_KEY = optional (better agent + report)
                            #   S2_API_KEY = optional (higher rate limits)

python main.py "federated learning in healthcare"
```

First run downloads the Sentence-BERT model (~80 MB) once; after that
embedding is fully local. A full run takes roughly 1–3 minutes depending
on API rate limits.

## What runs where

| Stage | Module | Live? |
|-------|--------|-------|
| 1 Input | `main.py` | user input at runtime |
| 2 Collection | `engine/fetch.py` | live Semantic Scholar + OpenAlex HTTP |
| 3 NLP | `engine/nlp.py` | local SBERT + BERTopic on fetched papers |
| 4 Gap detection | `engine/gaps.py` | pure math on the live embeddings |
| 5 Agent | `engine/agent.py` | live OpenAlex re-verification per gap |
| 6 Synthesis | `engine/synth.py` | LLM verbalizes computed results (or plain fallback) |

Works with **zero API keys** (rate-limited + plain-text report).
Add `ANTHROPIC_API_KEY` for LLM-generated verification queries and the
polished report.

## The examiner test

Ask anyone to type ANY topic. Different topic -> different papers ->
different clusters -> different gaps. Watch stage 5 reject gaps live —
that is the proof nothing is canned.
