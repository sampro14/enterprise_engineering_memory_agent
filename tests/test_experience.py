import json

from memory_agent.episodes import Reflection, log_episode
from memory_agent.experience import (
    LessonCheck,
    LessonDraft,
    attribute_failure,
    mine,
    record_counter_example,
    retrieve_experiences,
)
from memory_agent.graph import build_task_graph
from memory_agent.task_agent import Plan


def handler(prompt, schema):
    if schema is Reflection:
        return Reflection(failure_class="cold_cache_outage", conditions=["edge tier"], summary="cache cold")
    if schema is LessonDraft:
        return LessonDraft(pattern="Deploying an edge tier service", lesson="Cold cache breaches the latency SLO.",
                           conditions=["edge tier"], recommended_action="Run warm_cache before deploy_production.")
    if schema is LessonCheck:
        return LessonCheck(consistent=True)
    if schema is Plan:
        steps = ["run_tests", "build_image", "deploy_staging", "deploy_production"]
        if "ORGANIZATIONAL LESSONS" in prompt:
            steps.insert(3, "warm_cache")
        return Plan(steps=steps)
    raise AssertionError(schema)


def fail(ctx, n, task="Release X on the edge tier"):
    return log_episode(ctx, f"t{n}", task, ["deploy_production"], "failure", "ERR-5310 cold cache")


def test_promotion_requires_min_evidence(ctx):
    ctx.llm.handler = handler
    fail(ctx, 1)
    fail(ctx, 2)
    assert mine(ctx) == []
    fail(ctx, 3)
    created = mine(ctx)
    assert len(created) == 1
    row = ctx.store.fetch("SELECT * FROM experiences")[0]
    assert row["evidence_count"] == 3 and abs(row["confidence"] - 0.875) < 1e-6
    assert json.loads(row["conditions"])["conditions"] == ["edge tier"]
    assert mine(ctx) == []  # idempotent: no new evidence, no rewrite
    assert len(ctx.store.fetch("SELECT * FROM experience_evidence")) == 3


def test_same_task_repeated_counts_once(ctx):
    ctx.llm.handler = handler
    for _ in range(3):
        log_episode(ctx, "same", "Release X", ["a"], "failure", "ERR")
    assert mine(ctx) == []


def test_counter_examples_demote(ctx):
    ctx.llm.handler = handler
    for i in range(3):
        fail(ctx, i)
    mine(ctx)
    exp = retrieve_experiences(ctx, "Release Y on the edge tier", min_sim=0.0)
    assert len(exp) == 1 and "warm_cache" in exp[0]["text"]
    for _ in range(4):
        record_counter_example(ctx, exp[0]["id"])
    assert retrieve_experiences(ctx, "Release Y on the edge tier", min_sim=0.0) == []
    assert ctx.store.fetch("SELECT status FROM experiences")[0]["status"] == "stale"


def test_attribute_failure_matches_signature(ctx):
    ctx.llm.handler = handler
    for i in range(3):
        fail(ctx, i)
    mine(ctx)
    exp = retrieve_experiences(ctx, "Release Y on the edge tier", min_sim=0.0)
    assert attribute_failure(ctx, exp, "cold_cache_outage") == exp[0]["id"]
    assert attribute_failure(ctx, exp, "totally_different_reason_xyz") is None


def test_task_graph_learns_across_tasks(ctx):
    ctx.llm.handler = handler
    graph = build_task_graph(ctx)

    def executor(steps):
        return ("success", None) if "warm_cache" in steps else ("failure", "ERR-5310 cold cache")

    outcomes = []
    for i in range(5):
        st = graph.invoke({"task_id": f"t{i}", "task": "Release on the edge tier", "allowed_steps": ["warm_cache"],
                           "executor": executor})
        outcomes.append(st["outcome"])
    # tasks 0-2 fail, the lesson is promoted after the third failure, tasks 3-4 succeed
    assert outcomes == ["failure"] * 3 + ["success"] * 2


def test_inconsistent_lesson_is_retried_then_kept_as_hypothesis(ctx):
    drafts = []

    def h(prompt, schema):
        if schema is LessonDraft:
            drafts.append("rejected by a verifier" in prompt)
            return LessonDraft(pattern="p", lesson="l", conditions=["c"], recommended_action="REVERSED")
        if schema is LessonCheck:
            return LessonCheck(consistent=False, critique="the order is reversed")
        return handler(prompt, schema)

    ctx.llm.handler = h
    for i in range(3):
        fail(ctx, i)
    mine(ctx)
    assert drafts == [False, True]  # second draft saw the critique
    assert ctx.store.fetch("SELECT status FROM experiences")[0]["status"] == "hypothesis"
    assert retrieve_experiences(ctx, "Release on the edge tier", min_sim=0.0) == []


def test_counter_example_failures_are_not_evidence(ctx):
    ctx.llm.handler = handler
    for i in range(3):
        fail(ctx, i)
    mine(ctx)
    exp_id = ctx.store.fetch("SELECT id FROM experiences")[0]["id"]
    for i in range(3, 6):  # failed while the lesson was applied
        ep = fail(ctx, i)
        ctx.store.update("episodes", ep.id, {"counter_of": exp_id})
    mine(ctx)
    assert ctx.store.fetch("SELECT evidence_count FROM experiences")[0]["evidence_count"] == 3
