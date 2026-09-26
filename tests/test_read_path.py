from conftest import fact
from memory_agent.answering import Answer, QueryAnalysis
from memory_agent.graph import ask
from memory_agent.ingest import ingest_text
from memory_agent.models import ExtractionResult, SourceType
from memory_agent.retrieval import build_context, retrieve


def _seed(ctx, script, clock):
    script["JAN"] = [fact("PaymentService", "payment_provider", "Stripe", "uses Stripe JAN", valid_from="2025-01-01"),
                     fact("PaymentService", "owned_by", "PaymentsTeam", "owned by PaymentsTeam JAN", valid_from="2025-01-01")]
    ingest_text(ctx, "uses Stripe owned by PaymentsTeam JAN", "jan.md", SourceType.OFFICIAL_DOCS, "2025-01-01")
    script["SEP"] = [fact("PaymentService", "payment_provider", "Razorpay", "uses Razorpay SEP", valid_from="2026-09-11")]
    clock.d = "2026-10-01"
    ingest_text(ctx, "uses Razorpay SEP", "sep.md", SourceType.OFFICIAL_DOCS, "2026-09-11")
    script["CHAT"] = [fact("PaymentService", "payment_provider", "Stripe", "still Stripe CHAT")]
    ingest_text(ctx, "still Stripe CHAT", "chat", SourceType.DEVELOPER, "2026-10-02")


def test_retrieval_labels_current_historical_and_conflicts(ctx, script, clock):
    _seed(ctx, script, clock)
    ret = retrieve(ctx, "Which payment provider does PaymentService use?", ["PaymentService"], ["payment_provider"])
    states = {(r.object, r.state) for r in ret.items if r.memory.relation == "payment_provider"}
    assert states == {("Razorpay", "current"), ("Stripe", "historical")}
    assert len(ret.conflicts) == 1
    text = build_context(ret)
    assert "CURRENT" in text and "HISTORICAL" in text and "UNRESOLVED CONFLICTS" in text
    assert "sep.md" in text


def test_as_of_marks_valid_at_date(ctx, script, clock):
    _seed(ctx, script, clock)
    ret = retrieve(ctx, "provider in March 2026?", ["PaymentService"], ["payment_provider"], as_of="2026-03-01")
    valid = [r.object for r in ret.items if r.state == "valid_at" and r.memory.relation == "payment_provider"]
    assert valid == ["Stripe"]


def test_two_hop_graph_expansion(ctx, script, clock):
    script["DB"] = [fact("PaymentService", "primary_database", "Oracle", "db is Oracle DB"),
                    fact("OrdersService", "primary_database", "Oracle", "orders db is Oracle DB"),
                    fact("OrdersService", "owned_by", "OrdersTeam", "orders owned by OrdersTeam DB")]
    ingest_text(ctx, "db is Oracle orders db is Oracle orders owned by OrdersTeam DB", "db", SourceType.OFFICIAL_DOCS)
    ret = retrieve(ctx, "Who owns services that share a database with PaymentService?", ["PaymentService"])
    assert any(r.object == "OrdersTeam" for r in ret.items)


def test_langgraph_flow(ctx, script, clock):
    _seed(ctx, script, clock)
    calls = {}

    def handler(prompt, schema):
        if schema is ExtractionResult:
            return ExtractionResult()
        if schema is QueryAnalysis:
            return QueryAnalysis(needs_memory=True, entities=["PaymentService"], relations=["payment_provider"])
        assert schema is Answer
        calls["ctx"] = prompt
        return Answer(answer="Razorpay", values=["Razorpay"])

    ctx.llm.handler = handler
    state = ask(ctx, "Which provider does PaymentService use now?")
    assert state["answer"].values == ["Razorpay"] and state["status"] == "done"
    assert "CURRENT" in calls["ctx"] and "Razorpay" in calls["ctx"]


def test_no_memory_needed_skips_retrieval(ctx):
    ctx.llm.handler = lambda p, s: (QueryAnalysis(needs_memory=False) if s is QueryAnalysis
                                    else Answer(answer="4", values=["4"]))
    state = ask(ctx, "What is 2+2?")
    assert "retrieval" not in state and state["answer"].values == ["4"]
