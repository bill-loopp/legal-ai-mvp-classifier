"""Label a random sample of documents with DeepSeek, to train from.

This is the only step that costs money and the only step that leaves the
machine. It replicates the prompt shape already measured as accurate in
`packages/indexing/src/classify-document.ts` (head 700 chars + tail 500,
one label per line, an explicit `unclassified` escape) rather than importing
it, because this project is deliberately outside the repo's Node workspace.

One deviation from that file, made on purpose: each label carries a one-line
hint. The repo's taxonomy is 30 narrow titles ("Motion to Dismiss") where the
label alone is unambiguous; this taxonomy is 30 deliberately broader buckets
("Motion or Brief"), and without the hint the model has to guess each
bucket's boundary — exactly the judgement the hint encodes.

Usage: python3 label_sample.py --n 300 [--seed 20260824] [--limit-check]
"""
import argparse, json, os, random, re, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LEGAL_AI_REPO", os.path.expanduser("~/development/legal-ai-mvp"))
DOCS = os.path.join(REPO, "synthetic-data", "documents")

HEAD_CHARS, TAIL_CHARS = 700, 500
UNCLASSIFIED = "unclassified"


def load_env_key() -> tuple[str, str, str]:
    """DEEPSEEK_* from the repo .env — read only, never echoed."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    base = os.environ.get("DEEPSEEK_BASE_URL")
    model = os.environ.get("DEEPSEEK_MODEL")
    envp = os.path.join(REPO, ".env")
    if os.path.exists(envp):
        for line in open(envp, encoding="utf-8"):
            m = re.match(r"^\s*(DEEPSEEK_[A-Z_]+)\s*=\s*(.*?)\s*$", line)
            if m:
                v = m.group(2).strip().strip('"').strip("'")
                if m.group(1) == "DEEPSEEK_API_KEY" and not key: key = v
                if m.group(1) == "DEEPSEEK_BASE_URL" and not base: base = v
                if m.group(1) == "DEEPSEEK_MODEL" and not model: model = v
    if not key:
        sys.exit("DEEPSEEK_API_KEY not found in environment or repo .env")
    return key, base or "https://api.deepseek.com/anthropic", model or "deepseek-v4-flash"


def head_tail(text: str) -> str:
    if len(text) <= HEAD_CHARS + TAIL_CHARS:
        return text
    return f"{text[:HEAD_CHARS]}\n...\n{text[-TAIL_CHARS:]}"


def system_prompt(taxonomy: list[dict]) -> str:
    lines = "\n".join(f"- {t['label']}: {t['hint']}" for t in taxonomy)
    return (
        "You classify legal documents into exactly one of the listed types, "
        "based only on the document text provided.\n\n"
        "Classify by what the document IS — its form and function — not by a word "
        "that happens to appear in its title or subject line. Choose the single "
        "closest type even when the fit is imperfect; answer "
        f"{UNCLASSIFIED} only when genuinely none of the types apply.\n\n"
        "Respond with ONLY the exact type name from the list (the text before the "
        "colon), nothing else.\n\nTypes:\n" + lines
    )


class DeepSeek:
    def __init__(self, key: str, base: str, model: str):
        self.key, self.base, self.model = key, base, model

    def classify(self, system: str, text: str, timeout: int = 60) -> str:
        return self.classify_with_usage(system, text, timeout)[0]

    def classify_with_usage(self, system: str, text: str, timeout: int = 60):
        """Same call, but returns the API's usage block too — it was always
        being sent back and thrown away, which is why the first cost estimate
        for this pipeline was guesswork."""
        body = json.dumps({
            "model": self.model,
            "system": system,
            "max_tokens": 40,
            # deepseek-v4-flash defaults to extended-thinking, which would eat
            # all 40 output tokens on a reasoning block and return empty text.
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
        }).encode()
        req = urllib.request.Request(
            f"{self.base}/v1/messages", data=body, method="POST",
            headers={"content-type": "application/json", "x-api-key": self.key,
                     "anthropic-version": "2023-06-01"})
        last = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    payload = json.load(r)
                usage = payload.get("usage", {})
                for block in payload.get("content", []):
                    if block.get("type") == "text":
                        return block.get("text", "").strip(), usage
                return "", usage
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code not in (429, 500, 502, 503, 529):
                    raise
            except Exception as e:  # timeouts, connection resets
                last = type(e).__name__
            time.sleep(2 ** attempt)
        raise RuntimeError(f"DeepSeek failed after 4 attempts: {last}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0, help="classify N docs, print them, write nothing")
    args = ap.parse_args()

    taxonomy = json.load(open(os.path.join(HERE, "taxonomy.json"), encoding="utf-8"))
    index = json.load(open(os.path.join(HERE, "data", "index.json"), encoding="utf-8"))
    by_label = {t["label"].lower(): t["key"] for t in taxonomy}
    system = system_prompt(taxonomy)

    rng = random.Random(args.seed)
    n = args.smoke or args.n
    sample = rng.sample(index, min(n, len(index)))

    ds = DeepSeek(*load_env_key())
    lock, done = threading.Lock(), [0]

    def work(row: dict) -> dict | None:
        text = open(os.path.join(DOCS, row["file"]), encoding="utf-8", errors="replace").read()
        raw = ds.classify(system, head_tail(text))
        # A response that matches no known label is a parse miss, not evidence
        # the document is unclassifiable — both land as None and are dropped
        # from training rather than becoming a junk 31st class.
        cleaned = raw.strip().lstrip("-*\u2022 ").strip().strip('"').strip("'")
        cleaned = cleaned.split(":")[0].strip()
        key = None if cleaned.lower() == UNCLASSIFIED else by_label.get(cleaned.lower())
        with lock:
            done[0] += 1
            if done[0] % 25 == 0 or args.smoke:
                print(f"  {done[0]}/{len(sample)}", file=sys.stderr)
        return {"documentId": row["documentId"], "file": row["file"],
                "label": key, "raw": raw, "text": text}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(work, sample))

    if args.smoke:
        for r in results:
            print(f"\n{r['documentId']}\n  -> raw={r['raw']!r} key={r['label']}")
        return

    out = os.path.join(HERE, "data", "labeled.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({k: r[k] for k in ("documentId", "file", "label", "raw", "text")}) + "\n")

    from collections import Counter
    c = Counter(r["label"] for r in results)
    print(f"\n{len(results)} labeled in {time.time()-t0:.0f}s -> {out}")
    print(f"  unlabeled (unclassified / no label match): {c.get(None,0)}")
    print(f"  distinct labels used: {len([k for k in c if k])} of {len(taxonomy)}")
    for k, v in c.most_common():
        print(f"    {v:4d}  {k}")


if __name__ == "__main__":
    main()
