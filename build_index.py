"""Build a document index from synthetic-data/documents/*.txt.

Read-only against the legal-ai-mvp repo. Never writes there, never touches
Postgres. One row per *distinct document* — the corpus stores several
versions of the same document as `<stem>.v1.txt`, `<stem>.v2.txt`, ... and
versions of one document are the same kind by definition. Collapsing them
here does two jobs at once: it stops the DeepSeek sample from paying twice
for near-identical text, and it keeps a document's v1 out of training while
its v3 sits in the held-out split, which would inflate validation accuracy.

The highest-numbered version is the representative text: it is the document
as it finally stood.
"""
import json, os, re, sys

REPO = os.environ.get("LEGAL_AI_REPO", os.path.expanduser("~/development/legal-ai-mvp"))
DOCS = os.path.join(REPO, "synthetic-data", "documents")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "index.json")

VERSION_RE = re.compile(r"^(?P<stem>.+)\.v(?P<version>\d+)\.txt$")

def main() -> None:
    if not os.path.isdir(DOCS):
        sys.exit(f"corpus not found: {DOCS}")

    docs: dict[str, dict] = {}
    for name in sorted(os.listdir(DOCS)):
        m = VERSION_RE.match(name)
        if not m:
            continue  # README.md, .docx/.eml siblings — the .txt is the text of record
        stem, version = m.group("stem"), int(m.group("version"))
        d = docs.setdefault(stem, {"documentId": stem, "versions": [], "latestVersion": 0, "file": ""})
        d["versions"].append(version)
        if version > d["latestVersion"]:
            d["latestVersion"] = version
            d["file"] = name

    rows = []
    for stem, d in docs.items():
        path = os.path.join(DOCS, d["file"])
        size = os.path.getsize(path)
        matter = None
        mm = re.match(r"^(hv-\d+-\d+)-", stem)
        if mm:
            matter = mm.group(1)
        rows.append({
            "documentId": stem,
            "matterId": matter,
            "file": d["file"],
            "versionCount": len(d["versions"]),
            "latestVersion": d["latestVersion"],
            "bytes": size,
        })

    rows.sort(key=lambda r: r["documentId"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)

    tiny = sum(1 for r in rows if r["bytes"] < 200)
    print(f"{len(rows)} distinct documents -> {OUT}")
    print(f"  {sum(r['versionCount'] for r in rows)} version files collapsed")
    print(f"  {sum(1 for r in rows if r['matterId'] is None)} without an hv- matter id")
    print(f"  {tiny} shorter than 200 bytes")

if __name__ == "__main__":
    main()
