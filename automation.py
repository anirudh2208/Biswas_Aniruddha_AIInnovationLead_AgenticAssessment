#!/usr/bin/env python3
"""
Property Management Tenant Inquiry Automation
=============================================

Pipeline:
  Stage 1: CLASSIFY  -> LangChain + ChatGroq structured output
  Stage 2: RETRIEVE  -> ChromaDB semantic search (classification shapes query)
  Stage 3: REASON    -> LLM agent decides: respond / escalate / request info
  Stage 4: DRAFT     -> LLM generates response conditioned on agent decision

"""

import sys
import datetime
import argparse

from config import init_llm, init_embeddings, GROQ_MODEL, INQUIRY_FILE, LEASE_FILE
from utils import load_inquiries, load_lease_clauses
from stages import (
    index_lease_clauses, build_classification_chain, build_agent_chain,
    build_response_chain, classify_inquiry, retrieve_lease_clauses,
    agent_reasoning, resolve_escalation, draft_response,
)
from outputs import write_outputs


def process_inquiry(inquiry, classify_chain, agent_chain, response_chain, vectorstore) -> dict:
    """
    Run the 4-stage pipeline for one inquiry.
    """
    iid = inquiry["inquiry_id"]
    print(f"\n{'='*70}")
    print(f"  {iid} — {inquiry['subject']}")
    print(f"  Tenant: {inquiry['tenant_name']} ({inquiry['unit']})")
    print(f"{'='*70}")

    # ── Stage 1: CLASSIFY ─────────────────────────────────────────────
    print("\n  [Stage 1 — CLASSIFY] LangChain + ChatGroq")
    classification = classify_inquiry(classify_chain, inquiry)
    print(f"    {classification['category']} / {classification['urgency']}")
    print(f"    {classification['summary']}")
    print(f"    Topics: {', '.join(classification.get('key_topics', []))}")

    # ── Stage 2: RETRIEVE (shaped by classification) ──────────────────
    print("\n  [Stage 2 — RETRIEVE] ChromaDB semantic search")
    clauses = retrieve_lease_clauses(vectorstore, inquiry, classification)
    for c in clauses:
        flag = " [ESC]" if c["escalation_required"] else ""
        print(f"    -> {c['id']} {c['title']} [score: {c['relevance_score']}]{flag}")

    # ── Stage 3: REASON (uses classification + clauses as evidence) ───
    print("\n  [Stage 3 — REASON] Agentic decision")
    decision = agent_reasoning(agent_chain, inquiry, classification, clauses)
    print(f"    Action: {decision['action']} ({decision['confidence']})")
    print(f"    Why: {decision['reasoning']}")

    escalation = resolve_escalation(decision)
    if escalation:
        print(f"    ESCALATED -> {escalation['route_to']} (SLA: {escalation['sla_hours']}h)")

    # ── Stage 4: DRAFT (conditioned on agent decision) ────────────────
    print("\n  [Stage 4 — DRAFT] Response generation")
    response = draft_response(response_chain, inquiry, classification, clauses, decision, escalation)
    print(f"    Drafted ({len(response)} chars)")

    return {
        "inquiry_id": iid,
        "inquiry": {k: inquiry[k] for k in ("tenant_name", "unit", "email", "timestamp", "subject", "body")},
        "classification": classification,
        "matched_clauses": [
            {"id": c["id"], "section": c["section"], "title": c["title"],
             "relevance_score": c["relevance_score"], "escalation_required": c["escalation_required"]}
            for c in clauses
        ],
        "agent_decision": decision,
        "escalation": escalation,
        "drafted_response": response,
        "processed_at": datetime.datetime.now().isoformat(),
        "pipeline_mode": f"langchain + groq ({GROQ_MODEL}) + chromadb",
    }


def run_index_only(force_reindex=False):
    """
    Index lease clauses into ChromaDB and exit. No Groq API needed.
    """
    print("=" * 70)
    print("  LEASE CLAUSE INDEXING")
    print(f"  Time:  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    embeddings = init_embeddings()
    lease_data = load_lease_clauses(LEASE_FILE)
    vectorstore = index_lease_clauses(lease_data, embeddings, force_reindex)
    count = vectorstore._collection.count()
    print(f"\n  [OK] {count} lease clauses indexed and persisted to ChromaDB")
    print(f"  Ready to process inquiries.")
    print("=" * 70)


def run_pipeline(inquiry_filter=None, force_reindex=False):
    print("=" * 70)
    print("  PROPERTY MANAGEMENT TENANT INQUIRY AUTOMATION")
    print(f"  Stack: LangChain + ChatGroq ({GROQ_MODEL}) + ChromaDB")
    print(f"  Time:  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Init
    llm = init_llm()
    embeddings = init_embeddings()
    lease_data = load_lease_clauses(LEASE_FILE)
    vectorstore = index_lease_clauses(lease_data, embeddings, force_reindex)

    classify_chain = build_classification_chain(llm)
    agent_chain = build_agent_chain(llm)
    response_chain = build_response_chain(llm)

    inquiries = load_inquiries(INQUIRY_FILE)

    if inquiry_filter:
        inquiries = [i for i in inquiries if i["inquiry_id"] == inquiry_filter]
        if not inquiries:
            sys.exit(f"ERROR: {inquiry_filter} not found")

    print(f"\n  Processing {len(inquiries)} inquiries against {len(lease_data['clauses'])} lease clauses")

    # Process
    results, errors = [], []
    for inq in inquiries:
        try:
            results.append(process_inquiry(inq, classify_chain, agent_chain, response_chain, vectorstore))
        except Exception as e:
            print(f"\n  [FAIL] {inq['inquiry_id']}: {e}")
            errors.append(inq["inquiry_id"])

    if not results:
        sys.exit("No inquiries processed.")

    escalated = write_outputs(results)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {len(results)} processed, {len(errors)} errors, {len(escalated)} escalated")
    actions = {}
    for r in results:
        a = r["agent_decision"]["action"]
        actions[a] = actions.get(a, 0) + 1
    print(f"  Actions: {', '.join(f'{k}={v}' for k,v in sorted(actions.items()))}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tenant Inquiry Automation")
    parser.add_argument("--inquiry-id", help="Process single inquiry (e.g., INQ-006)")
    parser.add_argument("--reindex", action="store_true", help="Force re-index lease clauses")
    parser.add_argument("--index-only", action="store_true", help="Index lease clauses and exit (no Groq API needed)")
    args = parser.parse_args()

    if args.index_only:
        run_index_only(force_reindex=True)
    else:
        run_pipeline(inquiry_filter=args.inquiry_id, force_reindex=args.reindex)
