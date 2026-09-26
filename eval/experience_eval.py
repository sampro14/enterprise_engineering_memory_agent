"""Repeated-mistake experiment (design 32): does experience memory reduce recurring failures?

    uv run python -m eval.experience_eval --seeds 3 --tasks 24 --systems A,B,C,F
Repeated Mistake Rate = tasks where a previously *reported* failure pattern recurs
                        / tasks in which a previously reported failure pattern applies.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
from dataclasses import dataclass, field

from eval.metrics import wilson
from eval.simulator import RULES, STEPS, execute, make_sequence
from memory_agent.config import Settings
from memory_agent.context import Context
from memory_agent.graph import build_task_graph
from memory_agent.llm.cache import DiskCache
from memory_agent.llm.gemini import GeminiEmbedder, GeminiLLM
from memory_agent.store.sqlite import MemoryStore
from memory_agent.task_agent import plan_deployment


def make_ctx(cache: DiskCache) -> Context:
    s = Settings(experience_min_evidence=int(os.getenv("EXP_MIN_EVIDENCE", "3")))
    return Context(MemoryStore(), GeminiLLM(s, cache), GeminiEmbedder(s, cache), s)


@dataclass
class Agent:
    name: str
    ctx: Context
    history: list[str] = field(default_factory=list)
    n_lessons_used: int = 0

    def run(self, task, tid: str):  # -> (plan, outcome, error, violated)
        raise NotImplementedError


class NoMem(Agent):
    def run(self, task, tid):
        plan = plan_deployment(self.ctx, task.text, STEPS).steps
        outcome, err, viol = execute(task, plan)
        return plan, outcome, err, viol


class HistoryAgent(NoMem):
    """B: the last N raw episodes as conversation history."""

    def run(self, task, tid, window: int = 8):
        mem = ("EARLIER TASKS (most recent last):\n" + "\n".join(self.history[-window:])) if self.history else ""
        plan = plan_deployment(self.ctx, task.text, STEPS, mem).steps
        outcome, err, viol = execute(task, plan)
        self.history.append(f"- Task: {task.text} | Plan: {' > '.join(plan)} | Result: {err or 'success'}")
        return plan, outcome, err, viol


class EpisodicRAG(NoMem):
    """C: top-k similar raw episodes by embedding."""

    def run(self, task, tid, k: int = 4):
        mem = ""
        if self.history:
            q = self.ctx.embedder.embed([task.text])[0]
            hits = self.ctx.store.search_vectors("ep", q, k=k)
            mem = "SIMILAR EARLIER TASKS:\n" + "\n".join(self.history[int(i)] for i, _ in hits)
        plan = plan_deployment(self.ctx, task.text, STEPS, mem).steps
        outcome, err, viol = execute(task, plan)
        line = f"- Task: {task.text} | Plan: {' > '.join(plan)} | Result: {err or 'success'}"
        self.ctx.store.put_vector("ep", str(len(self.history)), self.ctx.embedder.embed([line])[0])
        self.history.append(line)
        return plan, outcome, err, viol


class ExperienceAgent(Agent):
    """F: LangGraph task flow with experience mining."""

    def __init__(self, name, ctx):
        super().__init__(name, ctx)
        self.graph = build_task_graph(ctx)

    def run(self, task, tid):
        box = {}

        def executor(steps):
            outcome, err, viol = execute(task, steps)
            box["viol"] = viol
            return outcome, err

        st = self.graph.invoke({"task_id": tid, "task": task.text, "allowed_steps": STEPS, "executor": executor})
        self.n_lessons_used += bool(st.get("experiences"))
        return st["plan"], st["outcome"], st["error"], box["viol"]


def run_sequence(agent: Agent, tasks) -> list[dict]:
    reported: set[str] = set()  # failure patterns the system has been told about
    rows = []
    for i, t in enumerate(tasks):
        plan, outcome, err, viol = agent.run(t, t.id)
        applicable_known = [r for r in RULES if t.applies(r) and r in reported]
        repeated = [r for r in applicable_known if r in viol]
        rows.append({"i": i, "outcome": outcome, "violated": viol, "opportunity": bool(applicable_known),
                     "repeated": bool(repeated), "plan": plan, "error": err})
        if outcome == "failure" and viol and err and err.startswith(RULES[viol[0]][1][:8]):
            reported.add(viol[0])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--tasks", type=int, default=24)
    ap.add_argument("--systems", default="A,B,C,F")
    ap.add_argument("--out", default="eval/results")
    a = ap.parse_args()
    cache = DiskCache(Settings().cache_path)
    kinds = {"A": ("A: no memory", NoMem), "B": ("B: history (last 8 episodes)", HistoryAgent),
             "C": ("C: episodic RAG (top-4)", EpisodicRAG), "F": ("Full: experience memory", ExperienceAgent)}
    keys = a.systems.split(",")
    agg = {k: collections.Counter() for k in keys}
    per_task = {k: collections.defaultdict(list) for k in keys}
    lessons = collections.Counter()
    raw = []
    for seed in range(a.first_seed, a.first_seed + a.seeds):
        tasks = make_sequence(seed, a.tasks)
        for k in keys:
            name, cls = kinds[k]
            agent = cls(name, make_ctx(cache))
            rows = run_sequence(agent, tasks)
            for r in rows:
                agg[k]["tasks"] += 1
                agg[k]["success"] += r["outcome"] == "success"
                agg[k]["opportunities"] += r["opportunity"]
                agg[k]["repeated"] += r["repeated"]
                agg[k]["failures"] += r["outcome"] == "failure"
                per_task[k][r["i"] // 8].append(r["outcome"] == "success")
                raw.append({"seed": seed, "system": k, **r})
            lessons[k] += agent.n_lessons_used
            print(f"seed {seed} {name}: success {sum(r['outcome'] == 'success' for r in rows)}/{len(rows)}, "
                  f"repeated {sum(r['repeated'] for r in rows)}/{sum(r['opportunity'] for r in rows)}")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005
    L = [f"# Repeated-mistake results ({stamp})", "",
         f"Seeds: {a.seeds}, tasks per seed: {a.tasks}, min evidence: {os.getenv('EXP_MIN_EVIDENCE', '3')}", "",
         "| System | Success rate | Repeated Mistake Rate (lower is better) | Failures | Tasks that received lessons |",
         "|---|---|---|---|---|"]
    for k in keys:
        c = agg[k]
        sl, sh = wilson(c["success"], c["tasks"])
        rl, rh = wilson(c["repeated"], c["opportunities"])
        rmr = f"{100 * c['repeated'] / max(c['opportunities'], 1):.0f}% ({100 * rl:.0f}-{100 * rh:.0f}) n={c['opportunities']}"
        L.append(f"| {kinds[k][0]} | {100 * c['success'] / c['tasks']:.0f}% ({100 * sl:.0f}-{100 * sh:.0f}) n={c['tasks']} "
                 f"| {rmr} | {c['failures']} | {lessons[k]} |")
    L += ["", "Success rate by task-index block (0-7, 8-15, 16-23):"]
    for k in keys:
        L.append(f"- {kinds[k][0]}: " + ", ".join(f"{100 * sum(v) / len(v):.0f}%" for _, v in sorted(per_task[k].items())))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"experience_{stamp}.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(a.out, f"experience_{stamp}.jsonl"), "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    print("\n".join(L))
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
