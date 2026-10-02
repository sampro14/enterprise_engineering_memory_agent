"""Unified LangGraph flow: plan (external execution) and outcome modes, and outcome write-back through validation."""
from conftest import fact
from memory_agent.answering import QueryAnalysis
from memory_agent.episodes import Reflection
from memory_agent.experience import LessonCheck, LessonDraft
from memory_agent.graph import build_agent_graph
from memory_agent.ingest import ingest_text
from memory_agent.models import ExtractionResult, SourceType
from memory_agent.task_agent import Plan

STEPS = ["run_tests", "build_image", "deploy_staging", "deploy_production"]


def make_handler(ctx_facts: dict, prompts: list):
    def h(prompt, schema):
        prompts.append((schema.__name__ if schema is not str else "str", prompt))
        if schema is QueryAnalysis:
            return QueryAnalysis(needs_memory=True, entities=["PaymentService"], relations=["primary_database"])
        if schema is Plan:
            return Plan(steps=list(STEPS))
        if schema is Reflection:
            return Reflection(failure_class="db_mismatch", conditions=["oracle"], summary="failed")
        if schema is ExtractionResult:
            for marker, facts in ctx_facts.items():
                if marker in prompt:
                    return ExtractionResult(facts=facts)
            return ExtractionResult()
        if schema is LessonDraft:
            return LessonDraft(pattern="p", lesson="l", conditions=[], recommended_action="a")
        if schema is LessonCheck:
            return LessonCheck(consistent=True)
        raise AssertionError(schema)
    return h


def seed_facts(ctx, script):
    script["SEED"] = [fact("PaymentService", "primary_database", "Oracle", "primary database is Oracle SEED")]
    ingest_text(ctx, "primary database is Oracle SEED", "seed.md", SourceType.OFFICIAL_DOCS, "2025-01-01")


def test_plan_mode_uses_facts_and_stops_before_execution(ctx, script):
    seed_facts(ctx, script)
    prompts: list = []
    ctx.llm.handler = make_handler({}, prompts)
    graph = build_agent_graph(ctx)
    st = graph.invoke({"mode": "plan", "task_id": "t1", "task": "Deploy PaymentService v2", "allowed_steps": STEPS,
                       "with_facts": True})
    assert st["plan"] == STEPS and "outcome" not in st  # plan only: an external system executes it
    plan_prompt = next(p for n, p in prompts if n == "Plan")
    assert "ORGANIZATIONAL FACTS" in plan_prompt and "Oracle" in plan_prompt
    assert ctx.store.fetch("SELECT * FROM episodes") == []  # nothing is logged until the outcome is reported


def test_plan_mode_without_facts_skips_analysis(ctx):
    prompts: list = []
    ctx.llm.handler = make_handler({}, prompts)
    st = build_agent_graph(ctx).invoke({"mode": "plan", "task_id": "t", "task": "Deploy X", "allowed_steps": STEPS})
    assert st["plan"] == STEPS and not any(n == "QueryAnalysis" for n, _ in prompts)


def test_outcome_mode_logs_episode_and_mines(ctx):
    prompts: list = []
    ctx.llm.handler = make_handler({}, prompts)
    graph = build_agent_graph(ctx)
    for i in range(3):
        st = graph.invoke({"mode": "outcome", "task_id": f"t{i}", "task": "Deploy on oracle", "plan": STEPS,
                           "outcome": "failure", "error": "ERR-1 mismatch"})
        assert st["episode_id"]
    assert len(ctx.store.fetch("SELECT * FROM episodes")) == 3
    assert len(ctx.store.fetch("SELECT * FROM experiences")) == 1  # three failures with one signature -> a lesson


def test_writeback_goes_through_validation_and_drops_unknown_relations(ctx, script):
    ctx_facts = {
        "Task outcome": [
            fact("PaymentService", "primary_database", "Oracle", "Task: Migrate PaymentService to Oracle."),  # high impact
            fact("PaymentService", "deploy_quality", "flaky", "Outcome: failure."),                          # not in ontology
        ]
    }
    ctx.llm.handler = make_handler(ctx_facts, [])
    st = build_agent_graph(ctx).invoke({
        "mode": "outcome", "task_id": "t9", "task": "Migrate PaymentService to Oracle.", "plan": STEPS,
        "outcome": "failure", "error": "ERR", "writeback": True, "use_experience": False})
    rep = st["writeback_report"]
    assert rep.get("Decision.REQUIRE_HUMAN_REVIEW", rep.get("REQUIRE_HUMAN_REVIEW")) == 1  # agent inference can't set a high-impact fact
    assert sum(v for k, v in rep.items() if "REJECT" in k) == 1  # unknown relation dropped, not queued
    assert ctx.store.memories("status='active'") == []          # nothing became trusted memory
    assert len(ctx.store.pending_reviews()) == 1


def test_defaults_wrapper_preserves_poc_call_signature(ctx):
    from memory_agent.graph import build_task_graph
    prompts: list = []
    ctx.llm.handler = make_handler({}, prompts)
    st = build_task_graph(ctx).invoke({"task_id": "t", "task": "Deploy X", "allowed_steps": STEPS,
                                       "executor": lambda steps: ("success", None)})
    assert st["outcome"] == "success" and not any(n == "QueryAnalysis" for n, _ in prompts)
    assert not any(n == "ExtractionResult" for n, _ in prompts)  # no write-back by default in the eval path


def test_task_facts_are_retrieved_even_if_analyzer_says_no_memory_needed(ctx, script):
    """Regression (found by the live e2e run): a task naming an entity must get that entity's facts."""
    seed_facts(ctx, script)
    prompts: list = []
    base = make_handler({}, prompts)

    def handler(prompt, schema):
        if schema is QueryAnalysis:
            return QueryAnalysis(needs_memory=False, entities=["PaymentService"])
        return base(prompt, schema)

    ctx.llm.handler = handler
    st = build_agent_graph(ctx).invoke({"mode": "plan", "task_id": "t", "task": "Deploy PaymentService v3",
                                        "allowed_steps": STEPS, "with_facts": True})
    assert "Oracle" in st["context_text"]
    assert "ORGANIZATIONAL FACTS" in next(p for n, p in prompts if n == "Plan")


def test_known_entity_names_are_found_even_if_the_llm_returns_none(ctx, script):
    seed_facts(ctx, script)
    from memory_agent.retrieval import mentioned_entities
    assert mentioned_entities(ctx, "Please deploy paymentservice v3 tonight") == ["PaymentService"]
    assert mentioned_entities(ctx, "Deploy PaymentServiceX now") == []  # whole-token matches only
    ctx.llm.handler = lambda prompt, schema: (QueryAnalysis(needs_memory=False, entities=[]) if schema is QueryAnalysis
                                              else make_handler({}, [])(prompt, schema))
    st = build_agent_graph(ctx).invoke({"mode": "plan", "task_id": "t", "task": "Deploy PaymentService v3",
                                        "allowed_steps": STEPS, "with_facts": True})
    assert "Oracle" in st["context_text"]
