"""
stages.py — The four pipeline stages.

  Stage 1: CLASSIFY    — LLM classifies category + urgency
  Stage 2: RETRIEVE    — ChromaDB semantic search for lease clauses
  Stage 3: REASON      — LLM agent decides action (respond / escalate / request info)
  Stage 4: DRAFT       — LLM generates response conditioned on agent decision

Each stage's output meaningfully influences the next:
  - Classification steers the retrieval query (category + topics shape what clauses are found)
  - Retrieved clauses inform the agent's reasoning (escalation flags + clause text = evidence)
  - Agent decision controls the response style (direct answer vs. acknowledgment vs. clarification)
"""

from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma

from config import InquiryClassification, AgentDecision, TEAM_DIRECTORY, CHROMA_DIR
from utils import call_with_retry


# ===========================================================================
# STAGE 0: INDEX — Embed lease clauses into ChromaDB (one-time)
# ===========================================================================
def index_lease_clauses(lease_data: dict, embeddings, force_reindex=False) -> Chroma:
    if CHROMA_DIR.exists() and not force_reindex:
        vs = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings,
                     collection_name="lease_clauses",
                     collection_metadata={"hnsw:space": "cosine"})
        if vs._collection.count() > 0:
            print(f"  Loaded {vs._collection.count()} clause embeddings from ChromaDB")
            return vs

    docs = [
        Document(
            page_content=f"Section {c['section']} — {c['title']}\n{c['text']}",
            metadata={
                "clause_id": c["id"], "section": c["section"], "title": c["title"],
                "category": c.get("category", "general"),
                "escalation_required": str(c.get("escalation_required", False)),
                "escalation_reason": c.get("escalation_reason", ""),
                "full_text": c["text"],
            },
        )
        for c in lease_data["clauses"]
    ]
    print(f"  Indexing {len(docs)} lease clauses into ChromaDB (cosine similarity)...")
    return Chroma.from_documents(docs, embeddings, persist_directory=str(CHROMA_DIR),
                                  collection_name="lease_clauses",
                                  collection_metadata={"hnsw:space": "cosine"})


# ===========================================================================
# STAGE 1: CLASSIFY  [AI — LLM structured output]
# ===========================================================================
_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI classifier for a property management company.
Return ONLY valid JSON.

Categories: maintenance (repairs, HVAC, mold, leaks), billing (rent, fees, deposits),
legal (lease terms, eviction, subletting, termination), general (noise, pets, parking).

Urgency: critical (safety hazard, no heat, mold+children, wrongful eviction),
high (habitability risk, extreme weather), medium (non-emergency repair, lease question),
low (general info request, noise, pet policy)."""),
    ("human", """Classify this inquiry as JSON (category, urgency, summary, key_topics, requires_escalation, escalation_reason):

Tenant: {tenant_name} ({unit})
Subject: {subject}
Body: {body}"""),
])


def build_classification_chain(llm):
    return _CLASSIFY_PROMPT | llm | JsonOutputParser(pydantic_object=InquiryClassification)


def classify_inquiry(chain, inquiry: dict) -> dict:
    return call_with_retry(chain, {
        "tenant_name": inquiry["tenant_name"], "unit": inquiry["unit"],
        "subject": inquiry["subject"], "body": inquiry["body"],
    }, "Classification")


# ===========================================================================
# STAGE 2: RETRIEVE  [AI — embedding-based semantic search]
# ===========================================================================
def retrieve_lease_clauses(vectorstore: Chroma, inquiry: dict, classification: dict, k=5, min_score=0.1) -> list[dict]:
    # Classification output SHAPES the query — category + topics steer retrieval
    query = (
        f"{inquiry['subject']}. {inquiry['body']}. "
        f"Category: {classification['category']}. "
        f"Topics: {', '.join(classification.get('key_topics', []))}"
    )
    # Fetch extra candidates, then filter by relevance threshold
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    matched = []
    for doc, score in results:
        if score < min_score:
            continue
        matched.append({
            "id": doc.metadata["clause_id"],
            "section": doc.metadata["section"],
            "title": doc.metadata["title"],
            "category": doc.metadata["category"],
            "text": doc.metadata.get("full_text", doc.page_content),
            "escalation_required": doc.metadata["escalation_required"] == "True",
            "escalation_reason": doc.metadata.get("escalation_reason", ""),
            "relevance_score": round(score, 3),
        })
    return matched[:3]  # Return top 3 above threshold


# ===========================================================================
# STAGE 3: REASON  [AI — agentic decision-making + routing]
# ===========================================================================
_REASON_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI agent for a property management company. DECIDE what action to take AND where to route it.

Actions:
1. "respond_directly" — Clear inquiry, lease clauses cover it, no specialist needed.
2. "escalate_and_respond" — Legal risk, safety concern, or complex matter needing specialist review. You MUST choose the right team and SLA.
3. "request_more_info" — Inquiry too vague to act on safely.

Available teams for escalation:
- "Legal Team" (SLA: 4h) — lease disputes, eviction, subletting, liability, any legal risk
- "Emergency Maintenance" (SLA: 2h) — safety hazards, no heat/water, mold, flooding, gas leaks
- "Maintenance Queue" (SLA: 48h) — non-urgent repairs, appliance issues, general maintenance
- "Accounts Team" (SLA: 24h) — billing disputes, payment arrangements, deposit questions
- "Front Desk" (SLA: 24h) — general inquiries, noise complaints, amenity questions

Return ONLY JSON: action, reasoning, confidence (number between 0.0 and 1.0), escalation_target, sla_hours, missing_info, response_guidance"""),
    ("human", """INQUIRY: {tenant_name} ({unit}) — {subject}
Message: {body}

CLASSIFICATION: {category} / {urgency} — Escalation flag: {requires_escalation}

MATCHED LEASE CLAUSES:
{clause_text}

Decide: respond_directly, escalate_and_respond, or request_more_info?
If escalating, which team and what SLA?""")])


def build_agent_chain(llm):
    return _REASON_PROMPT | llm | JsonOutputParser(pydantic_object=AgentDecision)


def agent_reasoning(chain, inquiry: dict, classification: dict, matched_clauses: list) -> dict:
    # Retrieved clauses ARE the evidence — escalation flags + text inform the decision
    clause_text = "\n".join(
        f"  [{c['id']}] {c['title']} (relevance: {c['relevance_score']}, "
        f"escalation_required: {c['escalation_required']}): {c['text'][:150]}..."
        for c in matched_clauses
    ) or "  No relevant clauses found."

    return call_with_retry(chain, {
        "tenant_name": inquiry["tenant_name"], "unit": inquiry["unit"],
        "subject": inquiry["subject"], "body": inquiry["body"],
        "category": classification["category"], "urgency": classification["urgency"],
        "requires_escalation": str(classification.get("requires_escalation", False)),
        "clause_text": clause_text,
    }, "Agent Reasoning")


def resolve_escalation(agent_decision: dict) -> Optional[dict]:
    """Build escalation object from the agent's AI-driven routing decision.
    The agent chooses the team; we look up contact info from the directory."""
    if agent_decision.get("action") != "escalate_and_respond":
        return None

    target = agent_decision.get("escalation_target", "")
    team_info = TEAM_DIRECTORY.get(target, {"sla_hours": 24, "notify": ["manager@sunriseapts.com"]})

    return {
        "route_to": target or "Manager (fallback)",
        "sla_hours": agent_decision.get("sla_hours", team_info["sla_hours"]),
        "notify": team_info["notify"],
    }


# ===========================================================================
# STAGE 4: DRAFT  [AI — conditioned on agent decision]
# ===========================================================================
_DRAFT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a professional property manager drafting a tenant response.
Be warm, professional, use their first name, reference lease sections.
Follow the ACTION and GUIDANCE strictly. 150-250 words. Email body only."""),
    ("human", """ACTION: {action}
GUIDANCE: {response_guidance}
{extra_context}

Tenant: {tenant_name} ({unit}) — Subject: {subject}
Message: {body}

Classification: {category} / {urgency}

Lease clauses:
{clause_text}

Escalation: {escalation_status}"""),
])


def build_response_chain(llm):
    return _DRAFT_PROMPT | llm


def draft_response(chain, inquiry, classification, matched_clauses, agent_decision, escalation) -> str:
    clause_text = "\n".join(
        f"[{c['id']}] Section {c['section']} — {c['title']}:\n{c['text']}"
        for c in matched_clauses
    ) or "No clauses matched."

    esc_status = (f"ESCALATED to {escalation['route_to']} — SLA: {escalation['sla_hours']}h"
                  if escalation else "NOT ESCALATED")

    action = agent_decision.get("action", "respond_directly")
    # Agent decision CONTROLS response style — different action = different email
    extra = ""
    if action == "request_more_info":
        extra = f"IMPORTANT: Ask tenant for: {agent_decision.get('missing_info', 'additional details')}"
    elif action == "escalate_and_respond":
        team = escalation['route_to'] if escalation else 'a specialist'
        sla = escalation['sla_hours'] if escalation else 24
        extra = f"IMPORTANT: Interim acknowledgment. {team} will follow up within {sla} hours."

    return call_with_retry(chain, {
        "tenant_name": inquiry["tenant_name"], "unit": inquiry["unit"],
        "subject": inquiry["subject"], "body": inquiry["body"],
        "category": classification["category"], "urgency": classification["urgency"],
        "clause_text": clause_text, "escalation_status": esc_status,
        "action": action, "response_guidance": agent_decision.get("response_guidance", ""),
        "extra_context": extra,
    }, "Response Drafting")
