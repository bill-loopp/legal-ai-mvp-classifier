# Setup from a fresh clone — agent instructions

Audience: an automated agent. Follow the section matching your goal. Do not
improvise around a failure; each one has a documented cause below.

## 0. Decide which goal you have

| Goal | Section | Needs network | Needs GPU | Wall time |
|---|---|---|---|---|
| Read existing classifications for the 8,617 corpus documents | **A** | no | no | seconds |
| Classify new/unseen documents | **B** | pip install only | no | ~10 min setup |
| Rebuild the fine-tuned model from committed labels | **C** | pip install only | no | ~4 h |
| Re-label the corpus with DeepSeek from scratch | **D** | yes, paid API | no | ~12 min |

Most tasks are A or B. Do not run C or D unless the task explicitly requires
retraining or re-labelling — C costs four hours and D costs money.

## Preconditions common to all sections

- Python 3.10 or newer: `python3 --version`
- On Debian/Ubuntu, `python3 -m venv` needs the `python3-venv` package. If you
  see `ensurepip is not available`, run `apt install python3-venv`. Do not work
  around this by installing into system Python.
- Sections B (rebuild path), C and D additionally need a checkout of the
  `legal-ai-mvp` repo, because this repo stores labels that reference corpus
  documents by filename instead of copying their text. Default location is
  `~/development/legal-ai-mvp`; override with:
  `export LEGAL_AI_REPO=/path/to/legal-ai-mvp`
  Verify: `ls "$LEGAL_AI_REPO/synthetic-data/documents" | head -3`
  Expect filenames like `hv-1001-00001-acquiring-person-hsr-form.v1.txt`.

## What the clone does and does not contain

Present: all `*.py`, `taxonomy.json`, `data/index.json`,
`data/labeled-all.jsonl`, `out/classifications.{csv,json}`,
`out/needs-review.csv`.

**Absent by design** (see `.gitignore`): `model/` (418 MB, exceeds GitHub's
100 MB file limit), `.venv/`, `data/labeled.jsonl` and its two predecessors,
`bundle/`, and any `*.tar.gz`. If your task needs `model/`, you must either run
section C or obtain `document-kind-classifier.tar.gz` from a GitHub Release.

---

## A. Read the existing classifications

No installation. The results are committed.

```
python3 -c "
import csv
rows=list(csv.DictReader(open('out/classifications.csv')))
print(len(rows), 'documents')
print(rows[0])
"
```

Expect `8617 documents`. Columns: `documentId`, `matterId`, `file`,
`versionCount`, `kind`, `kindLabel`, `isEmail`, `deepseekLabel`, `source`,
`modelPrediction`, `modelConfidence`, `agreement`, `needsReview`,
`reviewReason`.

Use `kind`. It is one of the 29 keys in `taxonomy.json`. `documentId` is the
filename stem and joins one-to-one to `mock_dms.documents.document_id`.

`out/classifications.json` carries the same rows plus a `summary` block with
the distribution and calibration table. `out/needs-review.csv` is the 381 rows
worth human attention.

Do not recompute these files to answer a question about them.

---

## B. Classify new documents

Preferred: use the prebuilt bundle, which is self-contained and needs no corpus
checkout.

```
tar xzf document-kind-classifier.tar.gz
cd document-kind-classifier
python3 -m venv .venv
./.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
./.venv/bin/pip install transformers
./.venv/bin/python classify.py /path/to/docs -o kinds.csv
```

Use the CPU index URL. A plain `pip install torch` fetches the CUDA build
(~2.5 GB) which is useless without an NVIDIA GPU.

If the tarball is not present (it is gitignored), complete section C first, then:

```
./.venv/bin/python classify_all.py --batch 16
```

Input must be plain text. Extract text from `.docx`/`.eml` before classifying —
the model was trained on plain-text renderings, not markup.

**Do not change** the truncation constants (`HEAD_CHARS=700`, `TAIL_CHARS=500`,
`max_length=352`). They match training. Changing them degrades accuracy with no
error raised.

---

## C. Rebuild the fine-tuned model

Requires `LEGAL_AI_REPO` (see preconditions). Roughly four hours on CPU; it uses
every core. Run it detached, not in a foreground call that may time out.

```
python3 -m venv .venv
./.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python train.py --epochs 6 --class-weights
```

Verify the data layer before committing four hours to it:

```
./.venv/bin/python -c "
from dataset import load_labeled
r=load_labeled('sample'); print(len(r),'rows'); assert r[0]['text']
"
```

Expect `3451 rows`. If it raises `corpus not found`, `LEGAL_AI_REPO` is wrong.

`--subset sample` (default) reproduces the shipped model from the 3,466
documents it was trained on, and keeps `calibrate.py`'s published figures
meaningful. `--subset all` trains on all 8,617 labels and should produce a
better model, but its validation split is not comparable to published numbers.
Use `sample` unless the task says otherwise.

Expected final output: validation accuracy ~0.806, macro F1 ~0.780, no class at
zero recall, model written to `model/`. Accuracy materially below 0.75, or any
class at 0.000 recall, means something is wrong — do not proceed to inference;
report it.

---

## D. Re-label the corpus with DeepSeek

**This spends money (~$0.44) and calls an external API. Do not run it without
explicit instruction.**

Needs `DEEPSEEK_API_KEY`, read from the environment or from
`$LEGAL_AI_REPO/.env`. Never copy that key into this repo.

```
python3 build_index.py          # rebuilds data/index.json from the corpus
python3 label_all.py            # ~12 min, resumable
./.venv/bin/python build_outputs.py
```

`label_all.py` appends each result as it completes and skips documents already
present in `data/labeled-all.jsonl`, so an interrupted run resumes rather than
restarting or double-charging. To force a genuine re-label, delete that file
first — and note that doing so discards the labels that are this repo's most
expensive artifact.

Re-labelling is only correct if `taxonomy.json` changed. Labels are meaningless
against a different taxonomy than the one that produced them.

---

## Failure modes

| Symptom | Cause | Action |
|---|---|---|
| `ensurepip is not available` | `python3-venv` missing | `apt install python3-venv` |
| `corpus not found at ...` | `LEGAL_AI_REPO` unset or wrong | export it; verify with the `ls` above |
| `no fine-tuned model found` | `model/` is gitignored, absent from clones | run section C, or extract the Release tarball |
| `DEEPSEEK_API_KEY not found` | key absent from env and `.env` | only section D needs it; do not fabricate one |
| Downloads ~2.5 GB installing torch | CUDA build | use the CPU `--index-url` |
| Training accuracy far below 0.806 | wrong subset, corpus mismatch, or truncated data | stop; report; do not ship the model |
| `classify.py` output all one class | wrong `labels.json`, or altered truncation | confirm `model/labels.json` has 29 keys matching `taxonomy.json` |

## Invariants — do not violate

1. **This project never writes to any database.** Not Postgres, not Supabase.
   It produces files. Loading results into `mock_dms.documents` is a separate,
   deliberate decision made elsewhere.
2. **It never writes to the `legal-ai-mvp` repo.** It reads
   `synthetic-data/documents/*.txt` and `DEEPSEEK_*` from `.env`. Nothing else.
3. **Truncation constants are fixed** at head 700 / tail 500 characters and 352
   tokens, in `classify.py`, `train.py`, `classify_all.py`, `calibrate.py` and
   `dataset.py` consumers. They must stay identical across all of them.
4. **`taxonomy.json` and `model/labels.json` must agree.** `labels.json` fixes
   the output index order. Editing the taxonomy invalidates the model and every
   label in `data/labeled-all.jsonl`.

## Corpus provenance an agent should carry

The documents classified here are from
[Harvey LAB](https://github.com/harveyai/harvey-labs) (MIT, Copyright (c) 2026
Harvey AI). See `NOTICE` — it is required to travel with copies of anything in
`data/` or `out/`, which are derived data keyed to Harvey documents.

Harvey describes the corpus as synthetically generated under the guidance and
review of human lawyers, containing imperfections, and not a perfect reflection
of documents drafted by a practising lawyer. It is not a per-document sign-off,
and known defects exist. Never present these documents, or classifications of
them, as correct legal drafting or as legal advice.

## Accuracy context an agent should carry

Reported accuracy (0.806) measures **agreement with DeepSeek**, which labelled
the training data — not agreement with a lawyer. Where DeepSeek is
systematically wrong, the model reproduces the error and no metric here detects
it. One such error was found and corrected during development (DeepSeek
misclassifies emails by their subject matter; the fine-tuned model is more
accurate on `.eml` documents, 87.5% against 78.4%, which is why `kind` takes the
model's label for those). Assume others remain. Do not present these
classifications as verified ground truth.
