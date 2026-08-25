# legal-ai-mvp-classifier

Document-kind classifier for the `legal-ai-mvp` synthetic corpus.

**Setting up from a fresh clone: see [AGENT_SETUP.md](AGENT_SETUP.md)** —
step-by-step, routed by goal, with verification output and failure modes.

## Built on Harvey LAB

Every document classified here comes from
**[Harvey LAB (Legal Agent Benchmark)](https://github.com/harveyai/harvey-labs)**
— MIT licensed, Copyright (c) 2026 Harvey AI — and this project would not exist
without it. There is no other freely available legal corpus at this scale and
realism; Harvey licensing the dataset under plain MIT, with no separate or more
restrictive data terms, is what made all of this possible. Thank you.

The classifications in `out/` are derived data keyed to Harvey documents and
carry no Harvey document text; you need the corpus itself to use them. **[NOTICE](NOTICE)**
carries the full attribution and licence, and must travel with copies of that
derived data. It also records Harvey's own honest caveat about the corpus:
these are synthetic documents generated under lawyer review, not per-document
sign-off — realistic demonstration material, never a model of correct legal
drafting, and never legal advice.

Standalone, one-off pipeline that assigns a real document type to each of the
**8,617 distinct documents** in the `legal-ai-mvp` synthetic corpus.

Deliberately isolated. It lives entirely in this directory, is not part of the
pnpm workspace, and:

- **writes nothing to the `legal-ai-mvp` repo** — it only reads
  `synthetic-data/documents/*.txt` and the `DEEPSEEK_*` values from `.env`;
- **never connects to Postgres or Supabase**, local or hosted;
- produces reviewable files in `out/`, and stops there. Deciding whether and
  how these land in `mock_dms.documents.kind` is a separate call to make after
  reading them.

## How it works

1. **`taxonomy.json`** — 29 document kinds, hand-authored from legal domain
   knowledge and then checked against what the corpus actually contains.
2. **`build_index.py`** — one row per distinct document. The corpus stores
   versions as `<stem>.v1.txt`, `.v2.txt`, …; versions of one document are the
   same kind, so they collapse to the highest version. That keeps a document's
   v1 out of training while its v3 sits in validation, which would otherwise
   inflate the score.
3. **`label_sample.py`** — the only step that costs money or leaves the machine.
   DeepSeek labels a random sample using the head-700 + tail-500 character
   prompt shape already measured as accurate in the repo's
   `packages/indexing/src/classify-document.ts`.
4. **`topup.py`** — adds targeted examples for kinds the random sample left
   too thin to learn. Filename keywords find *candidates* only; DeepSeek still
   reads the text and assigns every label.
5. **`train.py`** — fine-tunes `nlpaueb/legal-bert-base-uncased` locally on
   those labels, trained on the same head+tail view the teacher judged from.
   Prints a **per-class** precision/recall/F1 report and the top confusions.
6. **`classify_all.py`** — runs the fine-tuned model over all 8,617 documents
   locally. No API calls.
7. **`calibrate.py`** — measures what the confidence score is worth, per band,
   on the held-out split.
8. **`label_all.py`** — labels the whole corpus with DeepSeek. One-off, and it
   supersedes the distillation for this corpus. Resumable; carries over labels
   already paid for rather than re-buying them.
9. **`build_outputs.py`** — writes the deliverables from the full-corpus labels.

`finalize.py` belongs to the superseded distil-only path and is kept only so
that route stays reproducible.

`eval_merge.py` is diagnostic only: it re-scores the trained model under merged
taxonomies to tell a data problem apart from a taxonomy problem. It was used to
decide *not* to merge.

## Outputs

- `out/classifications.csv` — one row per document: `documentId`, `matterId`,
  `file`, `versionCount`, `kind`, `kindLabel`, `source`, `modelPrediction`,
  `modelConfidence`, `isEmail`, `deepseekLabel`, `agreement`, `needsReview`,
  `reviewReason`.
- `out/classifications.json` — the same rows plus a summary block: both
  distributions, the calibration table, and which kinds are weakest.
- `out/needs-review.csv` — just the 381 flagged rows.

Flagged means one of two things, both recorded in `reviewReason`: DeepSeek
returned no label (35), or a *confident* model (>= 0.90, the band where it is
right 93.7% of the time) disagreed with DeepSeek (346). The second is the only
signal available for finding DeepSeek's own mistakes.

## Results

29 kinds, 8,617 documents, every one labelled by DeepSeek directly.

**The distillation step turned out to be unnecessary for a corpus this size.**
CLASSIFIER.md exists to avoid "an unbounded DeepSeek bill (8,612 documents x
per-call cost)". Measured, that bill is **$0.44** and 12 minutes: 281 fresh
input tokens + 1,408 prefix-cached + 5 output per call, at off-peak
deepseek-v4-flash rates. Distilling to a local model to dodge a 44-cent cost
meant accepting ~19% imitation loss on 60% of the corpus. So the full corpus was
labelled directly, and the fine-tuned model was kept for what it is actually
good for: classifying new documents as they arrive, locally and free.

### The student beats the teacher on emails

615 documents ship as `.eml`, which is a fact about their form that neither
classifier saw — the only independent ground truth available here. Against it:

| | calls real emails `correspondence` |
|---|---|
| DeepSeek | 482 / 615 (78.4%) |
| fine-tuned model | **538 / 615 (87.5%)** |

DeepSeek is pulled by subject matter (31 emails called
`legal_research_memorandum`, 15 `engagement_letter`, 8
`securities_offering_document`); Legal-BERT reads the From/To/Subject headers as
form. So `kind` takes the model's label on `.eml` documents and DeepSeek's
everywhere else. `deepseekLabel` always records what DeepSeek said.

This is worth dwelling on: every accuracy figure in this project measures
agreement with DeepSeek, so DeepSeek's own systematic errors are invisible to
all of them. The disagreement between two classifiers was the only thing that
exposed this one.

### Fine-tuned model (kept for ongoing classification)

Trained on 3,466 DeepSeek-labelled documents.

| | value |
|---|---|
| Validation accuracy | 0.806 |
| Macro F1 | 0.780 |
| Kinds at zero recall | 0 |
| Teacher/student agreement, full corpus | 0.815 |
| Flagged for review | 381 (4.4%) |

Corpus-wide agreement of 0.815 against a held-out accuracy of 0.806 is a useful
check: the validation estimate was not an artefact of the split.

Confidence is well calibrated, which is what makes the split useful:

| band | share | accuracy |
|---|---|---|
| >= 0.90 | 70.9% | 0.937 |
| 0.70-0.90 | 18.2% | 0.563 |
| 0.50-0.70 | 6.7% | 0.413 |
| < 0.50 | 4.2% | 0.276 |

Weakest kinds: `settlement_agreement` (F1 0.483), `ip_or_technology_agreement`
(0.545), `case_assessment_memo` (0.581), `regulatory_filing` (0.600).
`closing_document_index`, `ip_or_technology_agreement` and
`confidentiality_agreement` each rest on 5-6 validation documents — treat those
scores as unmeasured rather than good or bad.

`kind` is the value to use. `source` says where it came from: `deepseek` for
documents in the labeled sample (the stronger labeller, kept in preference to
the model's own guess), `legal-bert` for the rest. `modelPrediction` is always
recorded, so teacher/student agreement is measurable.

## Running it

```
python3 build_index.py                          # stdlib only
python3 label_sample.py --n 400                 # needs DEEPSEEK_API_KEY
python3 topup.py --target 100 --oversample 1.6

python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python train.py --epochs 6 --class-weights   # --subset all for a better model
./.venv/bin/python classify_all.py --batch 16
./.venv/bin/python calibrate.py

python3 label_all.py                            # the full pass: 12 min, $0.44
./.venv/bin/python build_outputs.py
```

Total: about 18 minutes of DeepSeek calls ($0.44 measured, from the API's own
usage blocks) and roughly four hours of CPU. No GPU.

## What is and is not committed

The fine-tuned model (`model/`, 418MB) is **not** in git — it is four times
GitHub's per-file limit, and it is a slow function of data that is committed.
Rebuild it with `train.py`, or take `document-kind-classifier.tar.gz` from a
Release. What *is* committed is `data/labeled-all.jsonl`: 8,617 DeepSeek labels
at 1.8MB, and the only artifact here that cost money to produce.

`dataset.py` reads document text from the `legal-ai-mvp` corpus on demand rather
than storing a copy, so **this repo needs that checkout to retrain**. Point
`LEGAL_AI_REPO` at it if it is not at `~/development/legal-ai-mvp`. Classifying
new documents needs no such thing — that is what the bundle is for.

`train.py --subset sample` (the default) reproduces the shipped model from the
3,466 documents it was trained on. `--subset all` trains on all 8,617 and should
produce a better model, but its validation split is not comparable to the
figures published here.

## Things to know before trusting the numbers

- **The training set is not distributed like the corpus.** `topup.py`
  deliberately enriches rare kinds. Read corpus proportions off
  `out/classifications.json`'s distribution, never off the training labels.
- **Validation accuracy measures agreement with DeepSeek, not with a lawyer.**
  DeepSeek is the ground truth here. Where it is systematically wrong, the
  fine-tuned model learns to be wrong the same way, and the validation report
  cannot see it. Only reading documents catches that.
- **Two kinds were changed during the run, both from evidence:**
  `engagement_letter`'s definition was too broad and was swallowing every
  client letter (22 of 34 labels wrong on the first pass, fixed);
  `financial_analysis` was dropped entirely — the corpus's `*-analysis`
  documents are legal analyses in memo form, not financial models, so the kind
  had one real example in 1,020.
