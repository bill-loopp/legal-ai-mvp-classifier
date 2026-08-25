"""Loads labelled training data without needing the document text inlined.

`data/labeled.jsonl` stored each document's full text beside its label, which
made it 212MB — a redundant copy of a corpus that already exists on disk. This
reads `data/labeled-all.jsonl` (1.8MB, labels only) and fetches text through
each row's `file` field instead. Same data, same model, 1% of the bytes.

`subset` selects which labels to train on:
  "sample" — the 3,466 documents the shipped model was trained on. Reproduces
             it exactly, and keeps calibrate.py's held-out numbers meaningful.
  "all"    — all 8,617. Trains a better model than the shipped one, but its
             validation split is not comparable to the published figures.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LEGAL_AI_REPO", os.path.expanduser("~/development/legal-ai-mvp"))
DOCS = os.path.join(REPO, "synthetic-data", "documents")
LEAN = os.path.join(HERE, "data", "labeled-all.jsonl")
FAT = os.path.join(HERE, "data", "labeled.jsonl")


def load_labeled(subset: str = "sample") -> list[dict]:
    """Returns rows of {documentId, file, label, text}. Unlabelled rows dropped."""
    if os.path.exists(LEAN):
        rows = [json.loads(l) for l in open(LEAN, encoding="utf-8")]
        if subset == "sample":
            rows = [r for r in rows if r.get("pass") == "sample"]
        rows = [r for r in rows if r.get("label")]
        if not os.path.isdir(DOCS):
            raise SystemExit(
                f"corpus not found at {DOCS}.\n"
                "Set LEGAL_AI_REPO to the legal-ai-mvp checkout — the labels here "
                "reference its documents by filename rather than copying their text.")
        for r in rows:
            r["text"] = open(os.path.join(DOCS, r["file"]), encoding="utf-8", errors="replace").read()
        return rows
    # Fallback for a working tree that still has the superseded fat file.
    rows = [json.loads(l) for l in open(FAT, encoding="utf-8")]
    return [r for r in rows if r.get("label")]
