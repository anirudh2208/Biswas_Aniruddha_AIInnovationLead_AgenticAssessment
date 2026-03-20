"""
outputs.py — Write pipeline results to timestamped run directories.

Output structure:
  sample_io/
    2026-02-14_153042/
      results_153042.json
      summary_153042.csv
      escalation_log_153042.json
      info_requests_153042.json
      response_INQ-001_153042.txt
      response_INQ-006_153042.txt
      ...

Each run gets its own directory. Enables comparing results across runs.
"""

import csv
import json
import datetime
from pathlib import Path
from config import OUTPUT_DIR


def create_run_dir() -> tuple[Path, str]:
    """
    Create a timestamped directory for this run's outputs.
    """
    now = datetime.datetime.now()
    date_dir = now.strftime("%Y-%m-%d_%H%M%S")
    timestamp = now.strftime("%H%M%S")
    run_dir = OUTPUT_DIR / date_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, timestamp


def write_outputs(all_results: list[dict]) -> list[dict]:
    """
    Write all pipeline outputs to a timestamped run directory.
    """
    run_dir, ts = create_run_dir()
    print(f"\n  Run directory: {run_dir}")

    # 1. Full pipeline results
    with open(run_dir / f"results_{ts}.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  [OK] results_{ts}.json")

    # 2. Processing summary CSV
    rows = [{
        "inquiry_id": r["inquiry_id"],
        "tenant": r["inquiry"]["tenant_name"],
        "unit": r["inquiry"]["unit"],
        "category": r["classification"]["category"],
        "urgency": r["classification"]["urgency"],
        "agent_action": r["agent_decision"]["action"],
        "confidence": r["agent_decision"]["confidence"],
        "escalated": "YES" if r["escalation"] else "no",
        "routed_to": r["escalation"]["route_to"] if r["escalation"] else "Standard",
        "clauses": ", ".join(c["id"] for c in r["matched_clauses"]),
    } for r in all_results]
    with open(run_dir / f"summary_{ts}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  [OK] summary_{ts}.csv")

    # 3. Individual response drafts
    for r in all_results:
        with open(run_dir / f"response_{r['inquiry_id']}_{ts}.txt", "w") as f:
            f.write(f"TO: {r['inquiry']['email']}\n")
            f.write(f"SUBJECT: RE: {r['inquiry']['subject']}\n")
            f.write(f"DATE: {r['processed_at']}\n")
            f.write(f"ACTION: {r['agent_decision']['action']} ({r['agent_decision']['confidence']})\n")
            if r["escalation"]:
                f.write(f"ESCALATION: {r['escalation']['route_to']} (SLA: {r['escalation']['sla_hours']}h)\n")
                f.write(f"CC: {', '.join(r['escalation']['notify'])}\n")
            f.write(f"\n{'—'*50}\n\n")
            f.write(r["drafted_response"])
    print(f"  [OK] {len(all_results)} response drafts")

    # 4. Escalation log
    escalated = [r for r in all_results if r["escalation"]]
    if escalated:
        with open(run_dir / f"escalation_log_{ts}.json", "w") as f:
            json.dump([{
                "inquiry_id": r["inquiry_id"],
                "tenant": r["inquiry"]["tenant_name"],
                "category": r["classification"]["category"],
                "urgency": r["classification"]["urgency"],
                "agent_reasoning": r["agent_decision"]["reasoning"],
                "route_to": r["escalation"]["route_to"],
                "sla_hours": r["escalation"]["sla_hours"],
                "notify": r["escalation"]["notify"],
            } for r in escalated], f, indent=2)
        print(f"  [OK] escalation_log_{ts}.json ({len(escalated)} escalated)")

    # 5. Info-request log
    info_reqs = [r for r in all_results if r["agent_decision"]["action"] == "request_more_info"]
    if info_reqs:
        with open(run_dir / f"info_requests_{ts}.json", "w") as f:
            json.dump([{
                "inquiry_id": r["inquiry_id"],
                "tenant": r["inquiry"]["tenant_name"],
                "missing_info": r["agent_decision"]["missing_info"],
            } for r in info_reqs], f, indent=2)
        print(f"  [OK] info_requests_{ts}.json ({len(info_reqs)} requests)")

    return escalated
