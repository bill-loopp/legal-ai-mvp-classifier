"""Fine-tune Legal-BERT on the DeepSeek-labeled sample. Fully local.

Input view matches the teacher's on purpose: DeepSeek assigned every label
from head-700 + tail-500 characters, so the student is trained on exactly
that view. Showing the student the full document instead would let it learn
from text the label was never derived from — the middle of a 40-page
agreement cannot support a judgement made from its first and last page.

A plain PyTorch loop rather than `Trainer`: forty lines, and immune to the
Trainer API moving under a pinned-by-nothing transformers install.

Usage: python3 train.py [--epochs 5] [--batch 8] [--class-weights]
"""
import argparse, json, os, random, sys
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from dataset import load_labeled

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "nlpaueb/legal-bert-base-uncased"
HEAD_CHARS, TAIL_CHARS = 700, 500
SEED = 20260824


def head_tail(t: str) -> str:
    return t if len(t) <= HEAD_CHARS + TAIL_CHARS else f"{t[:HEAD_CHARS]}\n...\n{t[-TAIL_CHARS:]}"


class DS(Dataset):
    """Unpadded; the collator pads each batch to its own longest member."""
    def __init__(self, enc, y):
        self.enc, self.y = enc, y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        d = {k: v[i] for k, v in self.enc.items()}
        d["labels"] = self.y[i]
        return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--subset", choices=["sample", "all"], default="sample",
                    help="sample reproduces the shipped model; all trains on every label")
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    taxonomy = json.load(open(os.path.join(HERE, "taxonomy.json"), encoding="utf-8"))
    keys = [t["key"] for t in taxonomy]
    k2i = {k: i for i, k in enumerate(keys)}

    rows = [r for r in load_labeled(args.subset) if r["label"] in k2i]
    X = [head_tail(r["text"]) for r in rows]
    y = [k2i[r["label"]] for r in rows]
    print(f"{len(rows)} labeled documents, {len(keys)} kinds")

    # Stratify so every kind appears in both splits; the rarest kind has ~13
    # examples, so a random split could otherwise leave it untested.
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"train {len(Xtr)}  val {len(Xva)}")

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=len(keys))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    print(f"device: {dev}")

    # 352, not 512: head-700 + tail-500 characters of legal prose tokenizes to
    # roughly 350 word-pieces, so the extra 160 positions were pure padding.
    enc = lambda xs: tok(xs, truncation=True, max_length=352)
    coll = DataCollatorWithPadding(tok)
    dl_tr = DataLoader(DS(enc(Xtr), ytr), batch_size=args.batch, shuffle=True, collate_fn=coll)
    dl_va = DataLoader(DS(enc(Xva), yva), batch_size=args.batch, collate_fn=coll)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total = len(dl_tr) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1)

    w = None
    if args.class_weights:
        c = Counter(ytr)
        w = torch.tensor([len(ytr) / (len(keys) * max(1, c[i])) for i in range(len(keys))],
                         dtype=torch.float, device=dev)
    lossf = torch.nn.CrossEntropyLoss(weight=w)

    step = 0
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for b in dl_tr:
            b = {k: v.to(dev) for k, v in b.items()}
            labels = b.pop("labels")
            out = model(**b)
            loss = lossf(out.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            run += loss.item(); step += 1
            if step % 20 == 0:
                print(f"  epoch {ep+1} step {step}/{total} loss {run/20:.4f}", flush=True); run = 0.0

        model.eval()
        preds, gold = [], []
        with torch.no_grad():
            for b in dl_va:
                b = {k: v.to(dev) for k, v in b.items()}
                labels = b.pop("labels")
                preds += model(**b).logits.argmax(-1).cpu().tolist()
                gold += labels.cpu().tolist()
        acc = sum(p == g for p, g in zip(preds, gold)) / len(gold)
        print(f"epoch {ep+1}: val accuracy {acc:.3f}", flush=True)

    print("\n" + "=" * 72)
    print("PER-CLASS VALIDATION REPORT — read this, not just the accuracy line")
    print("=" * 72)
    present = sorted(set(gold) | set(preds))
    print(classification_report(gold, preds, labels=present,
                                target_names=[keys[i] for i in present], zero_division=0, digits=3))

    cm = confusion_matrix(gold, preds, labels=present)
    print("Most frequent confusions (true -> predicted):")
    conf = [(cm[i][j], keys[present[i]], keys[present[j]])
            for i in range(len(present)) for j in range(len(present)) if i != j and cm[i][j] > 0]
    for n, a, b in sorted(conf, reverse=True)[:12]:
        print(f"   {n:3d}  {a}  ->  {b}")

    out = os.path.join(HERE, "model")
    model.save_pretrained(out); tok.save_pretrained(out)
    json.dump(keys, open(os.path.join(out, "labels.json"), "w"), indent=1)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
