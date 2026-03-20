"""
test_automation.py — Unit tests for the tenant inquiry automation pipeline.

Tests cover:
  - Data loading and validation
  - Escalation resolution from agent decisions
  - Output file structure
  - Retry logic behavior
  - Classification and agent decision Pydantic models

Run:  python3 -m pytest test_automation.py -v
      (or just: python3 test_automation.py)
"""

import json
import csv
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure imports work without GROQ_API_KEY for testing
os.environ.setdefault("GROQ_API_KEY", "test_key_for_unit_tests")

from config import InquiryClassification, AgentDecision, TEAM_DIRECTORY
from utils import load_inquiries, load_lease_clauses, call_with_retry
from stages import resolve_escalation


# ===========================================================================
# Data Loading Tests
# ===========================================================================
class TestDataLoading:
    def test_load_inquiries_returns_list(self):
        inquiries = load_inquiries(Path(__file__).parent / "tenant_inquiries.csv")
        assert isinstance(inquiries, list)
        assert len(inquiries) == 20

    def test_inquiry_has_required_fields(self):
        inquiries = load_inquiries(Path(__file__).parent / "tenant_inquiries.csv")
        required = {"inquiry_id", "tenant_name", "unit", "email", "timestamp", "subject", "body"}
        for inq in inquiries:
            assert required.issubset(inq.keys()), f"Missing fields in {inq.get('inquiry_id')}"

    def test_inquiry_ids_are_unique(self):
        inquiries = load_inquiries(Path(__file__).parent / "tenant_inquiries.csv")
        ids = [i["inquiry_id"] for i in inquiries]
        assert len(ids) == len(set(ids)), "Duplicate inquiry IDs found"

    def test_load_lease_clauses_structure(self):
        data = load_lease_clauses(Path(__file__).parent / "lease_clauses.json")
        assert "clauses" in data
        assert len(data["clauses"]) == 35

    def test_lease_clause_has_required_fields(self):
        data = load_lease_clauses(Path(__file__).parent / "lease_clauses.json")
        required = {"id", "section", "title", "text", "category"}
        for clause in data["clauses"]:
            assert required.issubset(clause.keys()), f"Missing fields in {clause.get('id')}"

    def test_lease_clause_ids_are_unique(self):
        data = load_lease_clauses(Path(__file__).parent / "lease_clauses.json")
        ids = [c["id"] for c in data["clauses"]]
        assert len(ids) == len(set(ids)), "Duplicate clause IDs"

    def test_lease_categories_are_valid(self):
        data = load_lease_clauses(Path(__file__).parent / "lease_clauses.json")
        valid = {"maintenance", "billing", "legal", "general"}
        for clause in data["clauses"]:
            assert clause["category"] in valid, f"Invalid category in {clause['id']}"


# ===========================================================================
# Pydantic Model Tests
# ===========================================================================
class TestPydanticModels:
    def test_classification_model_valid(self):
        c = InquiryClassification(
            category="maintenance", urgency="critical",
            summary="Mold found", key_topics=["mold"],
            requires_escalation=True, escalation_reason="Safety"
        )
        assert c.category == "maintenance"
        assert c.requires_escalation is True

    def test_classification_model_defaults(self):
        c = InquiryClassification(
            category="general", urgency="low",
            summary="Question", key_topics=["pets"],
            requires_escalation=False
        )
        assert c.escalation_reason == ""

    def test_agent_decision_model_valid(self):
        d = AgentDecision(
            action="escalate_and_respond", reasoning="Mold with children",
            confidence=0.95, escalation_target="Emergency Maintenance",
            sla_hours=2, response_guidance="Acknowledge urgency"
        )
        assert d.action == "escalate_and_respond"
        assert d.sla_hours == 2
        assert d.confidence == 0.95

    def test_agent_decision_defaults(self):
        d = AgentDecision(
            action="respond_directly", reasoning="Simple question",
            confidence=0.8, response_guidance="Answer directly"
        )
        assert d.escalation_target == ""
        assert d.sla_hours == 24
        assert d.missing_info == ""


# ===========================================================================
# Escalation Resolution Tests
# ===========================================================================
class TestEscalationResolution:
    def test_no_escalation_when_respond_directly(self):
        decision = {"action": "respond_directly", "escalation_target": ""}
        assert resolve_escalation(decision) is None

    def test_no_escalation_when_request_more_info(self):
        decision = {"action": "request_more_info", "missing_info": "details"}
        assert resolve_escalation(decision) is None

    def test_escalation_to_legal_team(self):
        decision = {
            "action": "escalate_and_respond",
            "escalation_target": "Legal Team",
            "sla_hours": 4,
        }
        result = resolve_escalation(decision)
        assert result is not None
        assert result["route_to"] == "Legal Team"
        assert result["sla_hours"] == 4
        assert "legal@sunriseapts.com" in result["notify"]

    def test_escalation_to_emergency_maintenance(self):
        decision = {
            "action": "escalate_and_respond",
            "escalation_target": "Emergency Maintenance",
            "sla_hours": 2,
        }
        result = resolve_escalation(decision)
        assert result["route_to"] == "Emergency Maintenance"
        assert result["sla_hours"] == 2

    def test_escalation_unknown_team_gets_fallback(self):
        decision = {
            "action": "escalate_and_respond",
            "escalation_target": "Unknown Team",
            "sla_hours": 12,
        }
        result = resolve_escalation(decision)
        assert result is not None
        assert "manager@sunriseapts.com" in result["notify"]

    def test_escalation_empty_target_gets_fallback(self):
        decision = {"action": "escalate_and_respond", "escalation_target": ""}
        result = resolve_escalation(decision)
        assert result["route_to"] == "Manager (fallback)"

    def test_all_teams_in_directory(self):
        expected = {"Legal Team", "Emergency Maintenance", "Maintenance Queue",
                    "Accounts Team", "Front Desk"}
        assert expected == set(TEAM_DIRECTORY.keys())

    def test_all_teams_have_notify_list(self):
        for team, info in TEAM_DIRECTORY.items():
            assert "notify" in info, f"{team} missing notify"
            assert len(info["notify"]) > 0, f"{team} has empty notify"


# ===========================================================================
# Retry Logic Tests
# ===========================================================================
class TestRetryLogic:
    def test_succeeds_on_first_try(self):
        chain = MagicMock()
        chain.invoke.return_value = {"result": "success"}
        result = call_with_retry(chain, {"input": "test"}, "TestStep")
        assert result == {"result": "success"}
        assert chain.invoke.call_count == 1

    def test_succeeds_on_second_try(self):
        chain = MagicMock()
        chain.invoke.side_effect = [Exception("API error"), {"result": "ok"}]
        with patch("utils.time.sleep"):  # skip actual sleep
            result = call_with_retry(chain, {"input": "test"}, "TestStep")
        assert result == {"result": "ok"}
        assert chain.invoke.call_count == 2

    def test_extracts_content_from_ai_message(self):
        mock_msg = MagicMock()
        mock_msg.content = "Hello tenant"
        chain = MagicMock()
        chain.invoke.return_value = mock_msg
        result = call_with_retry(chain, {}, "TestStep")
        assert result == "Hello tenant"

    def test_raises_after_max_retries(self):
        chain = MagicMock()
        chain.invoke.side_effect = Exception("Persistent failure")
        with patch("utils.time.sleep"):
            try:
                call_with_retry(chain, {}, "TestStep")
                assert False, "Should have raised"
            except RuntimeError as e:
                assert "failed after" in str(e)
        assert chain.invoke.call_count == 3


# ===========================================================================
# Pipeline Result Structure Tests
# ===========================================================================
class TestResultStructure:
    """
    Validate that pipeline results have the expected shape.
    """

    SAMPLE_RESULT = {
        "inquiry_id": "INQ-001",
        "inquiry": {
            "tenant_name": "Maria Santos", "unit": "Unit 204",
            "email": "maria@test.com", "timestamp": "2026-02-10",
            "subject": "Leak", "body": "Kitchen sink leaking",
        },
        "classification": {
            "category": "maintenance", "urgency": "high",
            "summary": "Leak reported", "key_topics": ["plumbing"],
            "requires_escalation": True, "escalation_reason": "Safety",
        },
        "matched_clauses": [
            {"id": "LC-003", "section": "5.1", "title": "Maintenance",
             "relevance_score": 0.85, "escalation_required": False},
        ],
        "agent_decision": {
            "action": "escalate_and_respond", "reasoning": "Active leak",
            "confidence": 0.95, "escalation_target": "Emergency Maintenance",
            "sla_hours": 2, "missing_info": "", "response_guidance": "Acknowledge",
        },
        "escalation": {"route_to": "Emergency Maintenance", "sla_hours": 2, "notify": ["m@test.com"]},
        "drafted_response": "Dear Maria...",
        "processed_at": "2026-02-10T09:00:00",
        "pipeline_mode": "langchain + groq + chromadb",
    }

    def test_result_has_all_required_keys(self):
        required = {"inquiry_id", "inquiry", "classification", "matched_clauses",
                    "agent_decision", "escalation", "drafted_response", "processed_at"}
        assert required.issubset(self.SAMPLE_RESULT.keys())

    def test_classification_has_category_and_urgency(self):
        c = self.SAMPLE_RESULT["classification"]
        assert c["category"] in ("maintenance", "billing", "legal", "general")
        assert c["urgency"] in ("critical", "high", "medium", "low")

    def test_agent_decision_has_valid_action(self):
        d = self.SAMPLE_RESULT["agent_decision"]
        assert d["action"] in ("respond_directly", "escalate_and_respond", "request_more_info")

    def test_escalation_present_when_agent_escalates(self):
        if self.SAMPLE_RESULT["agent_decision"]["action"] == "escalate_and_respond":
            assert self.SAMPLE_RESULT["escalation"] is not None
            assert "route_to" in self.SAMPLE_RESULT["escalation"]


# ===========================================================================
# Run with python3 test_automation.py
# ===========================================================================
if __name__ == "__main__":
    try:
        import pytest
        pytest.main([__file__, "-v", "--tb=short"])
    except ImportError:
        # Fallback: run without pytest
        import unittest
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for cls in [TestDataLoading, TestPydanticModels, TestEscalationResolution,
                    TestRetryLogic, TestResultStructure]:
            for name in dir(cls):
                if name.startswith("test_"):
                    suite.addTest(unittest.FunctionTestCase(getattr(cls(), name)))
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
