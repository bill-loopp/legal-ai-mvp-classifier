"""Add the calibrated review flag and an unbiased distribution to the outputs.

Two things classify_all.py cannot know on its own:

1. `needsReview`. Measured on the held-out split (see calibrate.py): predictions
   at 0.90 confidence or above are right 93.7% of the time, and everything below
   drops sharply — 0.563 in the 0.70-0.90 band, 0.276 below 0.50. So 0.90 is a
   real decision boundary, not a round number. DeepSeek-labelled rows are never
   flagged: they came from the stronger labeller, not the model.

2. `distributionModelOnly`. The `kind` column mixes DeepSeek labels (from a
   sample deliberately enriched for rare kinds) with model predictions, so in
   principle it cannot be read as a corpus prior. Running the model uniformly
   over all 8,617 documents gives an estimate that is free of that enrichment.
   Both are written out; they turn out to agree to within 0.3pp, which is worth
   recording rather than assuming.
"""
import csv, json, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
THRESHOLD = 0.90

d = json.load(open(f"{HERE}/out/classifications.json", encoding="utf-8"))
rows = d["documents"]
taxonomy = {t["key"]: t["label"] for t in json.load(open(f"{HERE}/taxonomy.json", encoding="utf-8"))}

for r in rows:
    r["needsReview"] = bool(r["source"] == "legal-bert" and r["modelConfidence"] < THRESHOLD)

n_rev = sum(r["needsReview"] for r in rows)
km = Counter(r["modelPrediction"] for r in rows)
kd = Counter(r["kind"] for r in rows)

d["summary"]["reviewThreshold"] = THRESHOLD
d["summary"]["needsReview"] = n_rev
d["summary"]["confidentRows"] = len(rows) - n_rev
d["summary"]["calibration"] = {
    "measuredOn": "held-out validation split, 691 documents",
    "overallAccuracy": 0.806,
    "macroF1": 0.780,
    "bands": [
        {"range": "0.90-1.00", "shareOfDocs": 0.709, "accuracy": 0.937},
        {"range": "0.70-0.90", "shareOfDocs": 0.182, "accuracy": 0.563},
        {"range": "0.50-0.70", "shareOfDocs": 0.067, "accuracy": 0.413},
        {"range": "0.00-0.50", "shareOfDocs": 0.042, "accuracy": 0.276},
    ],
    "weakestKinds": ["settlement_agreement", "ip_or_technology_agreement",
                     "case_assessment_memo", "regulatory_filing"],
    "unmeasuredKinds": ["closing_document_index", "ip_or_technology_agreement",
                        "confidentiality_agreement"],
}
d["summary"]["distributionModelOnly"] = [
    {"kind": k, "label": taxonomy.get(k, ""), "count": v, "share": round(v / len(rows), 4)}
    for k, v in km.most_common()]

json.dump(d, open(f"{HERE}/out/classifications.json", "w", encoding="utf-8"), indent=1)
with open(f"{HERE}/out/classifications.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# A reviewer should not have to filter the big file to find the work.
rev = sorted((r for r in rows if r["needsReview"]), key=lambda r: r["modelConfidence"])
with open(f"{HERE}/out/needs-review.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rev)

print(f"{len(rows)} documents")
print(f"  confident (DeepSeek-labelled, or model >= {THRESHOLD}): {len(rows)-n_rev} ({(len(rows)-n_rev)/len(rows):.1%})")
print(f"  flagged for review:                                    {n_rev} ({n_rev/len(rows):.1%})")
print(f"\nreview queue by kind (worst confidence first):")
for k, v in Counter(r["kind"] for r in rev).most_common(8):
    print(f"   {v:4d}  {k}")
print(f"\nwrote out/classifications.csv, out/classifications.json, out/needs-review.csv")
