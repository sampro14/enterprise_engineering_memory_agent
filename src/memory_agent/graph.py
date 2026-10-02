"""One LangGraph workflow for the whole agent (design 27).

    mode "ask"     analyze -> [retrieve -> build_context] -> reason
    mode "plan"    [analyze -> retrieve -> build_context] -> retrieve_experience -> plan            (returns a plan; execution is external)
    mode "execute" ...same as plan..., then execute (in-process callback) -> reflect -> mine -> [writeback]
    mode "outcome" reflect -> mine -> [writeback]                                               (an external system reports the result)

`[...]` steps are conditional: facts are retrieved only if the question/task needs memory (`with_facts`), and outcome
write-back (extract -> validate -> consolidate, with source type `agent_inference`) only if `writeback` is set.

`build_query_graph` and `build_task_graph` are thin wrappers that keep the PoC's call signatures.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .answering import Answer, QueryAnalysis, analyze, answer_from_context
from .context import Context
from .episodes import log_episode
from .experience import attribute_failure, mine, retrieve_experiences
from .ingest import ingest_text
from .models import SourceType
from .retrieval import Retrieval, build_context, mentioned_entities, retrieve
from .task_agent import plan_deployment


class MemoryAgentState(TypedDict, total=False):
    mode: str  # ask | plan | execute | outcome
    task_id: str
    user_query: str
    analysis: QueryAnalysis
    retrieval: Retrieval
    context_text: str
    answer: Answer
    # task fields
    task: str
    allowed_steps: list[str]
    executor: Callable[[list[str]], tuple[str, str | None]]  # execute mode: steps -> (outcome, error)
    experiences: list[dict[str, Any]]
    plan: list[str]
    outcome: str
    error: str | None
    episode_id: str
    writeback_report: dict[str, int]
    # switches
    with_facts: bool  # task modes: also retrieve organizational facts relevant to the task
    use_experience: bool  # retrieve lessons and mine new ones
    writeback: bool  # extract candidate memories from the outcome
    status: str


def build_agent_graph(ctx: Context):
    def route_start(s: MemoryAgentState) -> str:
        mode = s.get("mode", "ask")
        if mode == "ask":
            return "analyze"
        if mode == "outcome":
            return "reflect"
        return "analyze" if s.get("with_facts") else "retrieve_experience"

    def n_analyze(s: MemoryAgentState):
        if s.get("mode", "ask") == "ask":
            return {"analysis": analyze(ctx, s["user_query"]), "status": "analyzed"}
        a = analyze(ctx, s["task"], kind="task")
        a.as_of = None  # a task has no "requested date"; facts are read as of today
        known = mentioned_entities(ctx, s["task"])  # deterministic fallback for names the LLM missed
        a.entities = [*a.entities, *[n for n in known if n.lower() not in {x.lower() for x in a.entities}]]
        return {"analysis": a, "status": "analyzed"}

    def after_analyze(s: MemoryAgentState) -> str:
        a = s["analysis"]
        if s.get("mode", "ask") == "ask":
            return "retrieve" if a.needs_memory else "reason"
        # Tasks are not questions, so the analyzer's "needs memory" flag is unreliable for them: whenever the task
        # names entities, look their facts up (the caller already opted in with with_facts).
        return "retrieve" if (a.entities or a.needs_memory) else "retrieve_experience"

    def n_retrieve(s: MemoryAgentState):
        a = s["analysis"]
        q = s["user_query"] if s.get("mode", "ask") == "ask" else s["task"]
        return {"retrieval": retrieve(ctx, q, a.entities, a.relations, a.as_of), "status": "retrieved"}

    def n_context(s: MemoryAgentState):
        exps = s.get("experiences") if s.get("mode", "ask") == "ask" else None
        return {"context_text": build_context(s["retrieval"], exps), "status": "context_built"}

    def after_context(s: MemoryAgentState) -> str:
        return "reason" if s.get("mode", "ask") == "ask" else "retrieve_experience"

    def n_reason(s: MemoryAgentState):
        return {"answer": answer_from_context(ctx, s["user_query"], s.get("context_text", "")), "status": "done"}

    def n_experience(s: MemoryAgentState):
        exps = retrieve_experiences(ctx, s["task"]) if s.get("use_experience", True) else []
        return {"experiences": exps, "status": "experience_retrieved"}

    def n_plan(s: MemoryAgentState):
        parts = []
        if s.get("context_text"):
            parts.append("ORGANIZATIONAL FACTS:\n" + s["context_text"])
        exps = s.get("experiences") or []
        if exps:
            parts.append("ORGANIZATIONAL LESSONS (learned from earlier tasks):\n" + "\n".join(f"- {e['text']}" for e in exps))
        plan = plan_deployment(ctx, s["task"], s["allowed_steps"], "\n\n".join(parts))
        return {"plan": plan.steps, "status": "planned"}

    def after_plan(s: MemoryAgentState) -> str:
        return "execute" if s.get("mode") == "execute" else END

    def n_execute(s: MemoryAgentState):
        outcome, error = s["executor"](s["plan"])
        return {"outcome": outcome, "error": error, "status": "executed"}

    def n_reflect(s: MemoryAgentState):
        ep = log_episode(ctx, s["task_id"], s["task"], s["plan"], s["outcome"], s["error"])
        if s["outcome"] == "failure" and s.get("experiences"):
            refuted = attribute_failure(ctx, s["experiences"], ep.signature)
            if refuted:
                ctx.store.update("episodes", ep.id, {"counter_of": refuted})
        return {"episode_id": ep.id, "status": "reflected"}

    def n_mine(s: MemoryAgentState):
        if s.get("use_experience", True):
            mine(ctx)
        return {"status": "mined"}

    def after_mine(s: MemoryAgentState) -> str:
        return "writeback" if s.get("writeback") else END

    def n_writeback(s: MemoryAgentState):
        text = (f"Task outcome. Task: {s['task']} Plan: {' > '.join(s['plan'])}. Outcome: {s['outcome']}."
                + (f" Error: {s['error']}" if s.get("error") else ""))
        rep = ingest_text(ctx, text, f"task:{s['task_id']}", SourceType.AGENT_INFERENCE, unknown_relation="drop")
        counts: dict[str, int] = {}
        for o in rep.outcomes:
            counts[str(o.decision)] = counts.get(str(o.decision), 0) + 1
        return {"writeback_report": counts, "status": "done"}

    g = StateGraph(MemoryAgentState)
    for name, fn in [("analyze", n_analyze), ("retrieve", n_retrieve), ("build_context", n_context),
                     ("reason", n_reason), ("retrieve_experience", n_experience), ("plan", n_plan),
                     ("execute", n_execute), ("reflect", n_reflect), ("mine", n_mine), ("writeback", n_writeback)]:
        g.add_node(name, fn)
    g.add_conditional_edges(START, route_start, {"analyze": "analyze", "reflect": "reflect",
                                                 "retrieve_experience": "retrieve_experience"})
    g.add_conditional_edges("analyze", after_analyze, {"retrieve": "retrieve", "reason": "reason",
                                                       "retrieve_experience": "retrieve_experience"})
    g.add_edge("retrieve", "build_context")
    g.add_conditional_edges("build_context", after_context, {"reason": "reason", "retrieve_experience": "retrieve_experience"})
    g.add_edge("reason", END)
    g.add_edge("retrieve_experience", "plan")
    g.add_conditional_edges("plan", after_plan, {"execute": "execute", END: END})
    g.add_edge("execute", "reflect")
    g.add_edge("reflect", "mine")
    g.add_conditional_edges("mine", after_mine, {"writeback": "writeback", END: END})
    g.add_edge("writeback", END)
    return g.compile()


class _WithDefaults:
    """Compiled graph + default state, so the PoC's `graph.invoke({...})` call sites keep working."""

    def __init__(self, graph, defaults: dict[str, Any]):
        self.graph, self.defaults = graph, defaults

    def invoke(self, state: dict[str, Any]):
        return self.graph.invoke({**self.defaults, **state})


def build_query_graph(ctx: Context):
    return _WithDefaults(build_agent_graph(ctx), {"mode": "ask"})


def build_task_graph(ctx: Context, use_experience: bool = True):
    """In-process task execution with experience learning (used by the simulator/eval)."""
    return _WithDefaults(build_agent_graph(ctx), {"mode": "execute", "use_experience": use_experience,
                                                  "with_facts": False, "writeback": False})


def ask(ctx: Context, question: str, graph=None) -> MemoryAgentState:
    graph = graph or build_query_graph(ctx)
    return graph.invoke({"task_id": "q", "user_query": question})


__all__ = ["MemoryAgentState", "ask", "build_agent_graph", "build_query_graph", "build_task_graph"]
