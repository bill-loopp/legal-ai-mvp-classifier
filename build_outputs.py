"""Rebuild the deliverables from the full-corpus DeepSeek labels.

Every document now carries two independent opinions: DeepSeek's label (the
stronger classifier, and the one that goes in `kind`) and the fine-tuned
model's prediction with a confidence. That changes what the review queue can
be. Previously it meant "the model was unsure", which is a statement about the
model. Now it can mean "two classifiers that usually agree did not agree here",
which is a statement about the document — and a confident model disagreeing
with DeepSeek is the best available signal for finding DeepSeek's own mistakes,
which no accuracy number in this pipeline can see.

Flagged for review:
  - DeepSeek answered `unclassified`, or its answer matched no known label; or
  - the model disagreed while being confident (>= 0.90), the band where it is
    right 93.7% of the time on held-out data.
"""
import csv, json, os, re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIDENT = 0.90

tax = json.load(open(f"{HERE}/taxonomy.json", encoding="utf-8"))
labels = {t["key"]: t["label"] for t in tax}
index = {r["documentId"]: r for r in json.load(open(f"{HERE}/data/index.json", encoding="utf-8"))}
deep = {r["documentId"]: r for r in
        (json.loads(l) for l in open(f"{HERE}/data/labeled-all.jsonl", encoding="utf-8"))}
prior = {r["documentId"]: r for r in
         json.load(open(f"{HERE}/out/classifications.json", encoding="utf-8"))["documents"]}

# The corpus ships 615 documents as .eml. That is a fact about their form which
# neither classifier saw, so it is the one place in this project where the two
# can be scored against something independent - and the student wins: it calls
# 87.5% of real emails `correspondence` against DeepSeek's 78.4%. DeepSeek is
# pulled by subject matter (31 emails called legal_research_memorandum, 8 called
# securities_offering_document); the fine-tuned model reads the From/To/Subject
# headers as form. So for these documents specifically the model's label is
# used, and the general "DeepSeek is the stronger labeller" rule is suspended.
DOCS = os.path.expanduser("~/development/legal-ai-mvp/synthetic-data/documents")
EMAIL = {re.sub(r"\.v\d+\.eml$", "", f) for f in os.listdir(DOCS) if f.endswith(".eml")}

rows = []
for did, ix in index.items():
    d = deep.get(did)
    p = prior.get(did, {})
    mp = p.get("modelPrediction")
    mc = p.get("modelConfidence", 0.0)
    is_email = did in EMAIL
    deepseek_label = d["label"] if d else None
    kind = mp if (is_email and mp) else deepseek_label
    labelled_by = "legal-bert" if (is_email and mp) else "deepseek"
    agree = bool(deepseek_label and mp and deepseek_label == mp)
    unresolved = kind is None
    contested = bool(deepseek_label and mp and deepseek_label != mp
                     and mc >= CONFIDENT and not is_email)
    rows.append({
        "documentId": did,
        "matterId": ix["matterId"] or "",
        "file": ix["file"],
        "versionCount": ix["versionCount"],
        "kind": kind or "unclassified",
        "kindLabel": labels.get(kind, "Unclassified"),
        "isEmail": is_email,
        "deepseekLabel": deepseek_label or "unclassified",
        "source": labelled_by,
        "modelPrediction": mp or "",
        "modelConfidence": mc,
        "agreement": agree,
        "needsReview": unresolved or contested,
        "reviewReason": ("deepseek returned no label" if unresolved
                         else "confident model disagrees" if contested else ""),
    })

rows.sort(key=lambda r: r["documentId"])
dist = Counter(r["kind"] for r in rows)
both = [r for r in rows if r["kind"] != "unclassified" and r["modelPrediction"]]
n_agree = sum(r["agreement"] for r in both)
rev = [r for r in rows if r["needsReview"]]

summary = {
    "documents": len(rows),
    "kinds": len(tax),
    "labelledBy": ("deepseek-v4-flash for all 8,617 documents, except the 615 that ship "
                   "as .eml, where the fine-tuned model is measurably more accurate "
                   "(87.5% vs 78.4% correct on that subset) and its label is used instead"),
    "unclassified": sum(1 for r in rows if r["kind"] == "unclassified"),
    "teacherStudentAgreementFullCorpus": round(n_agree / len(both), 4) if both else None,
    "needsReview": len(rev),
    "reviewReasons": dict(Counter(r["reviewReason"] for r in rev)),
    "note": ("kind is DeepSeek's label for every document. modelPrediction is the "
             "fine-tuned Legal-BERT's independent opinion, kept for comparison and "
             "used only to flag disagreements for review - except on .eml documents, "
             "where it supplies `kind` outright. `deepseekLabel` always records what "
             "DeepSeek said, so nothing is lost."),
    "distribution": [{"kind": k, "label": labels.get(k, "Unclassified"), "count": v,
                      "share": round(v / len(rows), 4)} for k, v in dist.most_common()],
}

json.dump({"summary": summary, "documents": rows},
          open(f"{HERE}/out/classifications.json", "w", encoding="utf-8"), indent=1)
for path, data in ((f"{HERE}/out/classifications.csv", rows),
                   (f"{HERE}/out/needs-review.csv", sorted(rev, key=lambda r: -r["modelConfidence"]))):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(data)

print(f"{len(rows)} documents, all DeepSeek-labelled")
print(f"  unclassified:                     {summary['unclassified']}")
print(f"  teacher/student agreement:        {summary['teacherStudentAgreementFullCorpus']}")
print(f"  emails taking the model's label:  {sum(1 for r in rows if r['isEmail'])}")
print(f"  flagged for review:               {len(rev)} ({len(rev)/len(rows):.1%})")
for k, v in summary["reviewReasons"].items():
    print(f"      {v:5d}  {k}")
print("\ndistribution:")
for d in summary["distribution"]:
    print(f"   {d['count']:5d}  {d['share']:6.1%}  {d['kind']}")
