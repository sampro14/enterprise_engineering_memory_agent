"""LangGraph workflow (design 27): analyze -> retrieve -> build context -> reason."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .answering import Answer, QueryAnalysis, analyze, answer_from_context
from .context import Context
from .episodes import log_episode, reflect
from .experience import attribute_failure, mine, retrieve_experiences
from .retrieval import Retrieval, build_context, retrieve
from .task_agent import plan_deployment


class MemoryAgentState(TypedDict, total=False):
    task_id: str
    user_query: str
    analysis: QueryAnalysis
    retrieval: Retrieval
    experiences: list[str]
    context_text: str
    answer: Answer
    status: str


def build_query_graph(ctx: Context):
    def n_analyze(s: MemoryAgentState):
        return {"analysis": analyze(ctx, s["user_query"]), "status": "analyzed"}

    def route(s: MemoryAgentState):
        return "retrieve" if s["analysis"].needs_memory else "reason"

    def n_retrieve(s: MemoryAgentState):
        a = s["analysis"]
        ret = retrieve(ctx, s["user_query"], a.entities, a.relations, a.as_of)
        return {"retrieval": ret, "status": "retrieved"}

    def n_context(s: MemoryAgentState):
        return {"context_text": build_context(s["retrieval"], s.get("experiences")), "status": "context_built"}

    def n_reason(s: MemoryAgentState):
        ans = answer_from_context(ctx, s["user_query"], s.get("context_text", ""))
        return {"answer": ans, "status": "done"}

    g = StateGraph(MemoryAgentState)
    g.add_node("analyze", n_analyze)
    g.add_node("retrieve", n_retrieve)
    g.add_node("build_context", n_context)
    g.add_node("reason", n_reason)
    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", route, {"retrieve": "retrieve", "reason": "reason"})
    g.add_edge("retrieve", "build_context")
    g.add_edge("build_context", "reason")
    g.add_edge("reason", END)
    return g.compile()


def ask(ctx: Context, question: str, graph=None) -> MemoryAgentState:
    graph = graph or build_query_graph(ctx)
    return graph.invoke({"task_id": "q", "user_query": question})


class TaskState(TypedDict, total=False):
    task_id: str
    task: str
    allowed_steps: list[str]
    executor: Callable[[list[str]], tuple[str, str | None]]  # steps -> (outcome, error)
    experiences: list[dict[str, Any]]
    plan: list[str]
    outcome: str
    error: str | None
    episode_id: str
    status: str


def build_task_graph(ctx: Context, use_experience: bool = True):
    """Task workflow with experience learning: retrieve lessons -> plan -> execute -> reflect (log episode, mine)."""

    def n_experience(s: TaskState):
        exps = retrieve_experiences(ctx, s["task"]) if use_experience else []
        return {"experiences": exps, "status": "experience_retrieved"}

    def n_plan(s: TaskState):
        exps = s.get("experiences") or []
        mem = ("ORGANIZATIONAL LESSONS (learned from earlier tasks):\n" + "\n".join(f"- {e['text']}" for e in exps)) if exps else ""
        plan = plan_deployment(ctx, s["task"], s["allowed_steps"], mem)
        return {"plan": plan.steps, "status": "planned"}

    def n_execute(s: TaskState):
        outcome, error = s["executor"](s["plan"])
        return {"outcome": outcome, "error": error, "status": "executed"}

    def n_reflect(s: TaskState):
        ep = log_episode(ctx, s["task_id"], s["task"], s["plan"], s["outcome"], s["error"])
        if s["outcome"] == "failure" and s.get("experiences"):
            refuted = attribute_failure(ctx, s["experiences"], ep.signature)
            if refuted:
                ctx.store.update("episodes", ep.id, {"counter_of": refuted})
        if use_experience:
            mine(ctx)
        return {"episode_id": ep.id, "status": "done"}

    g = StateGraph(TaskState)
    for name, fn in [("retrieve_experience", n_experience), ("plan", n_plan), ("execute", n_execute), ("reflect", n_reflect)]:
        g.add_node(name, fn)
    g.add_edge(START, "retrieve_experience")
    g.add_edge("retrieve_experience", "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", "reflect")
    g.add_edge("reflect", END)
    return g.compile()


__all__ = ["MemoryAgentState", "TaskState", "ask", "build_query_graph", "build_task_graph", "reflect"]
