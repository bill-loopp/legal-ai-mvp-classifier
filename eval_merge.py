"""Re-score the trained model on the same validation split under merged taxonomies.

No retraining. The point is to separate two very different failure causes that
a single accuracy number cannot: is the model failing because it has too little
data, or because some of these 29 kinds are not distinguishable from the head
and tail of a document at all? If collapsing the confusable families recovers
most of the lost accuracy, the taxonomy is the problem. If it does not, data is.
"""
import json, os
import numpy as np, torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dataset import load_labeled

HERE = os.path.dirname(os.path.abspath(__file__)); SEED = 20260824
HEAD, TAIL = 700, 500
ht = lambda t: t if len(t) <= HEAD+TAIL else f"{t[:HEAD]}\n...\n{t[-TAIL:]}"

keys = json.load(open(f"{HERE}/model/labels.json")); k2i = {k:i for i,k in enumerate(keys)}
rows = [r for r in load_labeled("sample") if r["label"] in k2i]
X = [ht(r["text"]) for r in rows]; y = [k2i[r["label"]] for r in rows]
_, Xva, _, yva = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

tok = AutoTokenizer.from_pretrained(f"{HERE}/model")
model = AutoModelForSequenceClassification.from_pretrained(f"{HERE}/model").eval()
preds = []
for i in range(0, len(Xva), 16):
    enc = tok(Xva[i:i+16], truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        preds += model(**enc).logits.argmax(-1).tolist()
    print(f"  {min(i+16,len(Xva))}/{len(Xva)}", flush=True)

base = sum(p==g for p,g in zip(preds,yva))/len(yva)
print(f"\nbaseline (29 kinds, as trained): {base:.3f}")

SCENARIOS = {
 "A: merge the memo family (4->1)": {
   "attorney_memorandum": ["legal_research_memorandum","matter_administration_memo",
                           "due_diligence_document","case_assessment_memo"]},
 "B: merge litigation filings (4->1)": {
   "litigation_filing": ["pleading","motion_or_brief","court_order","discovery_document"]},
 "C: both A and B": {
   "attorney_memorandum": ["legal_research_memorandum","matter_administration_memo",
                           "due_diligence_document","case_assessment_memo"],
   "litigation_filing": ["pleading","motion_or_brief","court_order","discovery_document"]},
 "D: C plus fold hold-notice + regulatory into their sinks": {
   "attorney_memorandum": ["legal_research_memorandum","matter_administration_memo",
                           "due_diligence_document","case_assessment_memo","regulatory_filing"],
   "litigation_filing": ["pleading","motion_or_brief","court_order","discovery_document"],
   "correspondence": ["correspondence","litigation_hold_notice"]},
}
for name, groups in SCENARIOS.items():
    m = {}
    for tgt, members in groups.items():
        for s in members: m[s] = tgt
    remap = lambda i: m.get(keys[i], keys[i])
    g2 = [remap(i) for i in yva]; p2 = [remap(i) for i in preds]
    acc = sum(a==b for a,b in zip(g2,p2))/len(g2)
    n = len(set(keys) - set(sum(groups.values(), [])) | set(groups))
    print(f"{name}: {n} kinds -> accuracy {acc:.3f}  (+{acc-base:.3f})")

print("\n--- scenario C detail ---")
groups = SCENARIOS["C: both A and B"]
m = {s:t for t,ms in groups.items() for s in ms}
g2=[m.get(keys[i],keys[i]) for i in yva]; p2=[m.get(keys[i],keys[i]) for i in preds]
print(classification_report(g2,p2,zero_division=0,digits=3))
