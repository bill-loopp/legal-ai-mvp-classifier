"""Classify every document in the corpus with the fine-tuned model. Fully local.

No API calls and no database: reads the .txt files, runs batched inference,
writes the review deliverable. Documents that were in the DeepSeek training
sample keep that label and are marked `deepseek` in the `source` column —
their model prediction is also recorded so teacher/student agreement can be
measured on data the student was fitted to, which is a sanity check, not an
accuracy estimate.

Outputs (both, same data):
  out/classifications.csv   — one row per document, for eyeballing in a sheet
  out/classifications.json  — same, plus the full per-kind distribution summary
"""
import argparse, csv, json, os, sys
from collections import Counter

import torch
torch.set_num_threads(os.cpu_count() or 4)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LEGAL_AI_REPO", os.path.expanduser("~/development/legal-ai-mvp"))
DOCS = os.path.join(REPO, "synthetic-data", "documents")
HEAD_CHARS, TAIL_CHARS = 700, 500


def head_tail(t: str) -> str:
    return t if len(t) <= HEAD_CHARS + TAIL_CHARS else f"{t[:HEAD_CHARS]}\n...\n{t[-TAIL_CHARS:]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    mdir = os.path.join(HERE, "model")
    if not os.path.exists(os.path.join(mdir, "labels.json")):
        sys.exit("no fine-tuned model found — run train.py first")
    keys = json.load(open(os.path.join(mdir, "labels.json"), encoding="utf-8"))
    taxonomy = {t["key"]: t["label"] for t in json.load(open(os.path.join(HERE, "taxonomy.json"), encoding="utf-8"))}
    index = json.load(open(os.path.join(HERE, "data", "index.json"), encoding="utf-8"))
    teacher = {r["documentId"]: r["label"]
               for r in map(json.loads, open(os.path.join(HERE, "data", "labeled.jsonl"), encoding="utf-8"))}

    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    print(f"device: {dev}; {len(index)} documents")

    rows = []
    for start in range(0, len(index), args.batch):
        chunk = index[start:start + args.batch]
        texts = [head_tail(open(os.path.join(DOCS, r["file"]), encoding="utf-8", errors="replace").read())
                 for r in chunk]
        enc = tok(texts, truncation=True, padding=True, max_length=352, return_tensors="pt").to(dev)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        conf, idx = probs.max(-1)
        for r, c, i in zip(chunk, conf.cpu().tolist(), idx.cpu().tolist()):
            pred = keys[i]
            t = teacher.get(r["documentId"])
            rows.append({
                "documentId": r["documentId"],
                "matterId": r["matterId"] or "",
                "file": r["file"],
                "versionCount": r["versionCount"],
                # `kind` is the value to actually use: the DeepSeek label where we
                # have one (it is the better labeller), the model's otherwise.
                "kind": t if t else pred,
                "kindLabel": taxonomy.get(t if t else pred, ""),
                "source": "deepseek" if t else "legal-bert",
                "modelPrediction": pred,
                "modelConfidence": round(c, 4),
            })
        if (start // args.batch) % 40 == 0:
            print(f"  {min(start+args.batch,len(index))}/{len(index)}", flush=True)

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    csv_path = os.path.join(HERE, "out", "classifications.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    dist = Counter(r["kind"] for r in rows)
    agree = [r for r in rows if r["source"] == "deepseek"]
    n_agree = sum(1 for r in agree if r["kind"] == r["modelPrediction"])
    low = sum(1 for r in rows if r["modelConfidence"] < 0.5)

    summary = {
        "documents": len(rows),
        "kinds": len(keys),
        "labelledByDeepSeek": len(agree),
        "labelledByModel": len(rows) - len(agree),
        "teacherStudentAgreementOnTrainingSet": round(n_agree / len(agree), 4) if agree else None,
        "modelPredictionsBelow0.5Confidence": low,
        "distribution": [{"kind": k, "label": taxonomy.get(k, ""), "count": v,
                          "share": round(v / len(rows), 4)} for k, v in dist.most_common()],
    }
    json_path = os.path.join(HERE, "out", "classifications.json")
    json.dump({"summary": summary, "documents": rows}, open(json_path, "w", encoding="utf-8"), indent=1)

    print(f"\n{len(rows)} documents -> {csv_path}")
    print(f"                    -> {json_path}")
    print(f"  teacher/student agreement on the training set: {summary['teacherStudentAgreementOnTrainingSet']}")
    print(f"  model predictions below 0.5 confidence: {low} ({low/len(rows):.1%})")
    print("\ndistribution across the corpus:")
    for d in summary["distribution"]:
        print(f"   {d['count']:5d}  {d['share']:6.1%}  {d['kind']}")


if __name__ == "__main__":
    main()
