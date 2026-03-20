# Property Management Tenant Inquiry Automation

---

## Quick Start (Docker — recommended)

```bash
# 1. Add your free Groq API key to .env (https://console.groq.com)
echo "GROQ_API_KEY=gsk_your_key" > .env

# 2. Build the image (installs all dependencies)
docker compose build --no-cache

# 3. Index lease clauses into ChromaDB (one-time, no Groq API needed)
#    Downloads embedding model (~90MB) and indexes 35 clauses
docker compose run --rm automation --index-only

# 4. Run tests
docker compose run --rm --entrypoint python3 automation -m pytest test_automation.py -v

# 5. Process a single inquiry
docker compose run --rm automation --inquiry-id INQ-006

# 6. Process all 20 inquiries
docker compose run --rm automation
```

### Clean rebuild (removes images, volumes, and cached data)

```bash
docker compose down --rmi all --volumes
rm -rf chroma_db/ sample_io/
docker compose build --no-cache
```

## Quick Start (Local)

```bash
# 1. Add your Groq API key to .env
echo "GROQ_API_KEY=gsk_your_key" > .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Index lease clauses (one-time)
python3 automation.py --index-only

# 4. Run tests
python3 -m pytest test_automation.py -v

# 5. Process a single inquiry
python3 automation.py --inquiry-id INQ-006

# 6. Process all 20 inquiries
python3 automation.py
```

### CLI options

| Flag | What it does |
|---|---|
| `--index-only` | Index lease clauses into ChromaDB and exit (no Groq API needed) |
| `--inquiry-id INQ-006` | Process a single inquiry by ID |
| `--reindex` | Force re-index lease clauses (use after editing lease_clauses.json) |

## What This Does

4-stage agentic pipeline for tenant inquiry intake:

```
Inquiry -> [1.CLASSIFY] -> [2.RETRIEVE] -> [3.REASON] -> [4.DRAFT] -> Outputs
              LLM           ChromaDB       LLM agent      LLM
           category,      lease clauses   decide action   conditioned
           urgency        via cosine      + pick team     on agent
                          similarity      + set SLA       decision
```

Each stage's output meaningfully shapes the next — see `stages.py` header for details.

## Project Structure

```
automation.py          Entry point + orchestrator
config.py              Constants, Pydantic models, team directory
utils.py               Retry logic with backoff, data loading
stages.py              All 4 pipeline stages
outputs.py             Timestamped file output writers
test_automation.py     26 unit tests

.env                   Groq API key (not committed to version control)
Dockerfile             Container definition
docker-compose.yml     One-command run with persistent volumes
requirements.txt       Python dependencies

tenant_inquiries.csv   20 synthetic tenant inquiries (input)
lease_clauses.json     35 lease clauses with realistic legal language (knowledge base)
sample_io/             Pipeline outputs in timestamped directories
```

## Output Structure

Each pipeline run creates a timestamped directory:

```
sample_io/
  2026-02-14_153042/                    <- first run
    results_153042.json                 <- full pipeline data
    summary_153042.csv                  <- overview for dashboard
    escalation_log_153042.json          <- escalation queue
    info_requests_153042.json           <- clarification needed
    response_INQ-001_153042.txt         <- drafted email
    response_INQ-006_153042.txt
    ...
  2026-02-14_154210/                    <- second run
    ...
```

## Requirements Mapping

| Requirement | Implementation |
|---|---|
| 3+ connected stages | Classify -> Retrieve -> Reason -> Draft (each output feeds the next) |
| AI steps | LLM classification, embedding retrieval (cosine similarity), agentic reasoning + routing, LLM drafting |
| Integration / data handoff | Writes response drafts, escalation log, summary CSV, info-request log to timestamped directories |
| Conditional logic + error handling | 3-way agent branching (respond/escalate/request info) + retry with backoff + per-inquiry error isolation |

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Orchestration | LangChain | Chain composition, structured output via Pydantic |
| LLM | Groq (Llama 3.3 70B) | Free tier, fast inference, no credit card |
| Vector Store | ChromaDB (cosine similarity) | Persistent semantic retrieval, relevance scoring |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local, free, no API needed |
| Container | Docker + Compose | Reproducible execution, volume persistence |
| Tests | pytest | 26 tests covering data, logic, retries, models |
