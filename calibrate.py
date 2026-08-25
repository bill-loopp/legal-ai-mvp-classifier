"""Measure what the model's confidence score is actually worth.

`modelConfidence` is a softmax maximum, which is not automatically a
probability of being right. This runs the trained model over the held-out
validation split and reports accuracy within each confidence band, so the
review threshold in the output is a measured number rather than a guess.
"""
import json, os, torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dataset import load_labeled

HERE=os.path.dirname(os.path.abspath(__file__)); SEED=20260824
torch.set_num_threads(os.cpu_count() or 4)
HEAD,TAIL=700,500
ht=lambda t: t if len(t)<=HEAD+TAIL else f"{t[:HEAD]}\n...\n{t[-TAIL:]}"

keys=json.load(open(f"{HERE}/model/labels.json")); k2i={k:i for i,k in enumerate(keys)}
rows=[r for r in load_labeled("sample") if r["label"] in k2i]
X=[ht(r["text"]) for r in rows]; y=[k2i[r["label"]] for r in rows]
_,Xva,_,yva=train_test_split(X,y,test_size=0.2,random_state=SEED,stratify=y)

tok=AutoTokenizer.from_pretrained(f"{HERE}/model")
model=AutoModelForSequenceClassification.from_pretrained(f"{HERE}/model").eval()
conf,pred=[],[]
for i in range(0,len(Xva),16):
    enc=tok(Xva[i:i+16],truncation=True,padding=True,max_length=352,return_tensors="pt")
    with torch.no_grad(): p=torch.softmax(model(**enc).logits,-1)
    c,ix=p.max(-1); conf+=c.tolist(); pred+=ix.tolist()

BANDS=[(0.0,0.5),(0.5,0.7),(0.7,0.9),(0.9,1.01)]
print(f"{'confidence band':<18} {'n':>5} {'share':>7} {'accuracy':>9}")
for lo,hi in BANDS:
    sel=[(p,g) for p,g,c in zip(pred,yva,conf) if lo<=c<hi]
    if not sel: continue
    acc=sum(p==g for p,g in sel)/len(sel)
    print(f"  {lo:.2f} - {hi if hi<=1 else 1.0:.2f}      {len(sel):>5} {len(sel)/len(yva):>6.1%} {acc:>9.3f}")

print("\naccuracy if rows below a threshold are set aside for review:")
for t in (0.5,0.6,0.7,0.8,0.9):
    kept=[(p,g) for p,g,c in zip(pred,yva,conf) if c>=t]
    acc=sum(p==g for p,g in kept)/len(kept)
    print(f"  keep conf >= {t:.1f}:  {len(kept)/len(yva):>5.1%} of documents kept, {acc:.3f} accuracy on them")
