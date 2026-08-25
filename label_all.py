"""Label every document in the corpus with DeepSeek. One-off.

Supersedes the sample-then-distil approach for this corpus. Measured cost is
~$0.64 for all 8,617 documents (281 fresh input tokens + 1,408 prefix-cached +
5 output per call, at off-peak deepseek-v4-flash rates), which is cheaper than
accepting the fine-tuned model's ~19% error on the 60% of documents it labelled.

Deliberately lean rows: no document text. `labeled.jsonl` stores full text and
is 213MB at 3,466 rows; the text is already on disk and reachable through
`file`, and the only view that ever mattered is the head+tail one. Labels
already paid for are carried over rather than re-purchased.

Resumable: every result is appended as it completes, so an interrupted run
picks up where it stopped. Re-running only calls for documents not yet present.
"""
import json, os, sys, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_sample import DeepSeek, head_tail, load_env_key, system_prompt, DOCS, HERE, UNCLASSIFIED

OUT = os.path.join(HERE, "data", "labeled-all.jsonl")
WORKERS = 8


def main() -> None:
    taxonomy = json.load(open(os.path.join(HERE, "taxonomy.json"), encoding="utf-8"))
    by_label = {t["label"].lower(): t["key"] for t in taxonomy}
    system = system_prompt(taxonomy)
    index = json.load(open(os.path.join(HERE, "data", "index.json"), encoding="utf-8"))

    done: dict[str, dict] = {}
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            r = json.loads(line)
            done[r["documentId"]] = r
        print(f"resuming: {len(done)} already labeled in {os.path.basename(OUT)}")

    # Carry over the sample labels rather than paying for them twice. They came
    # from the same prompt and the same taxonomy, so they are the same product.
    prior = os.path.join(HERE, "data", "labeled.jsonl")
    if os.path.exists(prior) and not done:
        with open(OUT, "w", encoding="utf-8") as f:
            for line in open(prior, encoding="utf-8"):
                r = json.loads(line)
                row = {"documentId": r["documentId"], "file": r["file"],
                       "label": r["label"], "raw": r["raw"], "pass": "sample"}
                done[r["documentId"]] = row
                f.write(json.dumps(row) + "\n")
        print(f"carried over {len(done)} labels already paid for")

    todo = [r for r in index if r["documentId"] not in done]
    print(f"{len(todo)} documents to label ({len(done)} done, {len(index)} total)")
    if not todo:
        print("nothing to do")
        return

    ds = DeepSeek(*load_env_key())
    lock, state = threading.Lock(), {"n": 0, "in": 0, "cache": 0, "out": 0, "err": 0}
    t0 = time.time()
    fh = open(OUT, "a", encoding="utf-8")

    def work(row):
        try:
            text = open(os.path.join(DOCS, row["file"]), encoding="utf-8", errors="replace").read()
            raw, usage = ds.classify_with_usage(system, head_tail(text))
        except Exception as e:
            with lock:
                state["err"] += 1
            return {"documentId": row["documentId"], "file": row["file"],
                    "label": None, "raw": f"ERROR: {type(e).__name__}", "pass": "full"}
        cleaned = raw.strip().lstrip("-*• ").strip().strip('"').strip("'").split(":")[0].strip()
        key = None if cleaned.lower() == UNCLASSIFIED else by_label.get(cleaned.lower())
        out = {"documentId": row["documentId"], "file": row["file"],
               "label": key, "raw": raw, "pass": "full"}
        with lock:
            state["n"] += 1
            state["in"] += usage.get("input_tokens", 0)
            state["cache"] += usage.get("cache_read_input_tokens", 0)
            state["out"] += usage.get("output_tokens", 0)
            fh.write(json.dumps(out) + "\n")
            if state["n"] % 250 == 0:
                el = time.time() - t0
                rate = state["n"] / el
                print(f"  {state['n']}/{len(todo)}  {rate:.1f}/s  eta {(len(todo)-state['n'])/rate/60:.1f}min",
                      flush=True)
        return out

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(work, todo))
    fh.close()

    cost = state["in"]/1e6*0.22 + state["cache"]/1e6*0.007 + state["out"]/1e6*0.66
    print(f"\n{state['n']} labeled in {(time.time()-t0)/60:.1f} min ({state['err']} errors)")
    print(f"  tokens: fresh_in={state['in']/1e6:.2f}M cache_read={state['cache']/1e6:.2f}M out={state['out']/1e6:.3f}M")
    print(f"  measured cost at off-peak rates: ${cost:.2f}")

    allrows = list(done.values()) + results
    c = Counter(r["label"] for r in allrows)
    print(f"\n{len(allrows)} documents labeled, {c.get(None,0)} unclassified")
    for k, v in c.most_common():
        if k:
            print(f"   {v:5d}  {k}")


if __name__ == "__main__":
    main()
