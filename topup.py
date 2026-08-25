"""Top up under-represented kinds in the training set.

A uniform random sample tracks the corpus's real distribution, which is what
makes it honest — and also what leaves `financial_analysis` with zero examples
and six other kinds with three or fewer. A model cannot learn a class it never
sees, so those kinds would score zero recall no matter how good the approach is.

This adds targeted examples. Filename keywords are used ONLY to find candidate
documents; DeepSeek still reads the text and assigns every label, exactly as in
the random pass. A filename is a hint about where to look, never a label — if
the keyword search turns up a document that is really something else, DeepSeek
labels it something else, and that is a correct outcome, not a leak.

Consequence, deliberate and recorded: the training set after this step is NOT
distributed like the corpus. It is enriched for rare kinds. Nothing downstream
may read corpus proportions off it — the final prediction distribution over all
8,617 documents is the only place to look for that.

Usage: python3 topup.py [--target 15]
"""
import argparse, json, os, random, re, sys, threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_sample import DeepSeek, head_tail, load_env_key, system_prompt, DOCS, HERE, UNCLASSIFIED

# Where to look for each kind. Deliberately loose: over-broad patterns cost a
# few wasted DeepSeek calls, over-narrow ones leave the class empty.
CANDIDATES = {
    "financial_analysis": r"analysis|valuation|budget|damages|benchmark|forecast|sources-and-uses|model|synergy|projection|quantum",
    "conflict_check_memorandum": r"conflict",
    "closing_document_index": r"closing-checklist|binder-index|closing-binder|signature-page|checklist|deal-calendar|closing-agenda",
    "real_property_agreement": r"lease|deed|easement|estoppel|real-property|snda|title|premises|landlord|tenant",
    "ip_or_technology_agreement": r"ip-assign|patent|trademark|copyright|licen[cs]e|source-code|open-source|technology-transfer|joint-development",
    "confidentiality_agreement": r"\bnda\b|non-disclosure|nondisclosure|confidentiality|clean-team|common-interest|joint-defense",
    "litigation_hold_notice": r"litigation-hold|preservation|hold-notice|hold-reminder|hold-release",
    "pleading": r"complaint|answer-and-affirmative|petition|counterclaim|cross-claim|notice-of-removal|demand-for-arbitration",
    "organizational_document": r"bylaws|certificate-of-incorporation|charter|operating-agreement|llc-agreement|stockholders-agreement|partnership-agreement|investors-rights|indemnification-agreement",
    "court_order": r"-order|judgment|dismissal|award|stipulation",
    "settlement_agreement": r"settlement|release|tolling|consent-decree",
    "compliance_policy": r"policy|code-of-conduct|procedure|-sop|training|protocol|retention-schedule",
    "purchase_agreement": r"purchase-agreement|merger-agreement|disclosure-schedule|-apa|-spa|share-purchase",
    "employment_agreement": r"employment|offer-letter|severance|separation|retention-agreement|compensation|equity-award|restrictive-covenant",
    "case_assessment_memo": r"case-assessment|settlement-authority|exposure|mediation|coverage|merits",
    "commercial_agreement": r"services-agreement|-msa|statement-of-work|-sow|transition-services|supply|vendor|distribution|reseller|consulting",
    "term_sheet_or_loi": r"term-sheet|letter-of-intent|-loi|-mou|commitment-letter|exclusivity|indication-of-interest",
    "due_diligence_document": r"diligence",
    "legal_opinion_letter": r"opinion",
    "closing_certificate": r"certificate|firpta|solvency|incumbency|secretarys|officers-cert|perfection",
    "securities_offering_document": r"offering|prospectus|underwriting|registration-rights|comfort-letter|placement",
    "regulatory_filing": r"\bhsr\b|filing|application|submission|cfius|notification|regulator|permit",
    "board_or_stockholder_consent": r"resolution|written-consent|minutes|board-",
    "discovery_document": r"deposition|interrogator|request-for-production|privilege-log|subpoena|\besi\b|declaration|affidavit|expert-|witness",
    "motion_or_brief": r"motion|brief|memorandum-of-law|opposition|reply-in-support",
    "matter_administration_memo": r"matter-opening|matter-closing|new-matter|intake|staffing|closing-memo",
    "engagement_letter": r"engagement-letter|retention-letter|retainer|disengagement",
    "credit_or_financing_agreement": r"credit-agreement|indenture|promissory|guarantee|security-agreement|pledge|escrow|intercreditor|subordination|mortgage",
    "legal_research_memorandum": r"memo",
    "correspondence": r"email|letter",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=15, help="minimum labeled examples per kind")
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--oversample", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    taxonomy = json.load(open(os.path.join(HERE, "taxonomy.json"), encoding="utf-8"))
    index = json.load(open(os.path.join(HERE, "data", "index.json"), encoding="utf-8"))
    by_label = {t["label"].lower(): t["key"] for t in taxonomy}
    system = system_prompt(taxonomy)

    existing = [json.loads(l) for l in open(os.path.join(HERE, "data", "labeled.jsonl"), encoding="utf-8")]
    have = Counter(r["label"] for r in existing if r["label"])
    seen = {r["documentId"] for r in existing}

    rng = random.Random(args.seed + 1)
    picks, per_kind = [], {}
    for t in taxonomy:
        k = t["key"]
        need = max(0, args.target - have.get(k, 0))
        if not need:
            continue
        pat = CANDIDATES.get(k)
        if not pat:
            continue
        pool = [r for r in index if r["documentId"] not in seen and re.search(pat, r["documentId"])]
        # Over-sample: DeepSeek will reject some candidates into other kinds, so
        # asking for exactly `need` reliably lands short.
        take = rng.sample(pool, min(len(pool), int(need * args.oversample)))
        per_kind[k] = (need, len(pool), len(take))
        for r in take:
            seen.add(r["documentId"])
            picks.append(r)

    print(f"{len(picks)} candidate documents for {len(per_kind)} under-represented kinds")
    for k, (need, pool, take) in sorted(per_kind.items(), key=lambda x: -x[1][0]):
        print(f"   need {need:2d}  pool {pool:5d}  sampling {take:3d}   {k}")
    if not picks:
        print("nothing to top up")
        return

    ds = DeepSeek(*load_env_key())
    lock, done = threading.Lock(), [0]

    def work(row):
        text = open(os.path.join(DOCS, row["file"]), encoding="utf-8", errors="replace").read()
        raw = ds.classify(system, head_tail(text))
        cleaned = raw.strip().lstrip("-*• ").strip().strip('"').strip("'").split(":")[0].strip()
        key = None if cleaned.lower() == UNCLASSIFIED else by_label.get(cleaned.lower())
        with lock:
            done[0] += 1
            if done[0] % 50 == 0:
                print(f"  {done[0]}/{len(picks)}", file=sys.stderr)
        return {"documentId": row["documentId"], "file": row["file"], "label": key, "raw": raw, "text": text}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(work, picks))

    out = os.path.join(HERE, "data", "labeled.jsonl")
    with open(out, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    final = Counter(r["label"] for r in existing + results)
    print(f"\ntraining set now {len(existing)+len(results)} documents -> {out}")
    print(f"  unlabeled: {final.get(None,0)}")
    thin = [(k, v) for k, v in final.items() if k and v < 10]
    for k, v in sorted(final.items(), key=lambda x: -x[1]):
        if k:
            print(f"    {v:4d}  {k}")
    print(f"\n  kinds still under 10 examples: {len(thin)}")


if __name__ == "__main__":
    main()
