"""
config.py — Configuration, Pydantic models, and component initialization.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# Load .env file (looks in project root)
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# API & Paths
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY and "--index-only" not in sys.argv:
    print("ERROR: GROQ_API_KEY not found.")
    print("  Option 1: Add it to .env file ->  GROQ_API_KEY=gsk_your_key")
    print("  Option 2: Export it directly  ->  export GROQ_API_KEY=gsk_your_key")
    print("  Get a free key at https://console.groq.com")
    print("  (Not needed for --index-only)")
    sys.exit(1)

GROQ_MODEL = "llama-3.3-70b-versatile"
#GROQ_MODEL = "llama-3.1-8b-instant"
BASE_DIR = Path(__file__).parent
INQUIRY_FILE = BASE_DIR / "tenant_inquiries.csv"
LEASE_FILE = BASE_DIR / "lease_clauses.json"
CHROMA_DIR = BASE_DIR / "chroma_db"
OUTPUT_DIR = BASE_DIR / "sample_io"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# ---------------------------------------------------------------------------
# Team directory — contact info & SLAs (used by AI agent for routing)
# The agent DECIDES which team; this table provides operational details.
# ---------------------------------------------------------------------------
TEAM_DIRECTORY = {
    "Legal Team":           {"sla_hours": 4,  "notify": ["legal@sunriseapts.com", "manager@sunriseapts.com"]},
    "Emergency Maintenance":{"sla_hours": 2,  "notify": ["maintenance-urgent@sunriseapts.com", "manager@sunriseapts.com"]},
    "Maintenance Queue":    {"sla_hours": 48, "notify": ["maintenance@sunriseapts.com"]},
    "Accounts Team":        {"sla_hours": 24, "notify": ["accounts@sunriseapts.com"]},
    "Front Desk":           {"sla_hours": 24, "notify": ["frontdesk@sunriseapts.com"]},
}

# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------
class InquiryClassification(BaseModel):
    category: str = Field(description="One of: maintenance, billing, legal, general")
    urgency: str = Field(description="One of: critical, high, medium, low")
    summary: str = Field(description="One-sentence summary")
    key_topics: list[str] = Field(description="Relevant topics")
    requires_escalation: bool = Field(description="Whether escalation is needed")
    escalation_reason: str = Field(default="", description="Reason if escalation needed")


class AgentDecision(BaseModel):
    action: str = Field(description="One of: respond_directly, escalate_and_respond, request_more_info")
    reasoning: str = Field(description="Why the agent chose this action")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    escalation_target: str = Field(default="", description="Team to route to: Legal Team, Emergency Maintenance, Maintenance Queue, Accounts Team, or Front Desk")
    sla_hours: int = Field(default=24, description="Hours within which escalation team must respond")
    missing_info: str = Field(default="", description="What info is missing")
    response_guidance: str = Field(description="Key points the response must address")


# ---------------------------------------------------------------------------
# Suppress noisy third-party logs
# ---------------------------------------------------------------------------
import logging
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["SAFETENSORS_FAST_GPU"] = "0"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Component initializers
# ---------------------------------------------------------------------------
def init_llm():
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.1, max_tokens=1000)

def init_embeddings():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
