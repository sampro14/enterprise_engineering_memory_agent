"""Run the temporal/contradiction benchmark: baselines A/B/C vs the full memory system.

    uv run python -m eval.run_eval --seeds 1 --systems A,B,C,F
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
from concurrent.futures import ThreadPoolExecutor

from eval.metrics import Graded, bootstrap_diff, grade, wilson
from eval.scenarios import generate
from eval.systems import FullMemory, HistoryOnly, NoMemory, VectorRAG
from memory_agent.config import Settings
from memory_agent.context import Context
from memory_agent.llm.cache import DiskCache
from memory_agent.llm.gemini import GeminiEmbedder, GeminiLLM
from memory_agent.store.sqlite import MemoryStore


class CountingLLM:
    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    def generate(self, *a, **k):
        self.calls += 1
        return self.inner.generate(*a, **k)

    def generate_json(self, *a, **k):
        self.calls += 1
        return self.inner.generate_json(*a, **k)


def make_ctx(cache: DiskCache) -> Context:
    s = Settings()
    return Context(MemoryStore(), CountingLLM(GeminiLLM(s, cache)), GeminiEmbedder(s, cache), s)


def build_systems(keys: list[str], cache: DiskCache):
    table = {
        "A": lambda c: NoMemory(c),
        "B": lambda c: HistoryOnly(c, window=10),
        "BF": lambda c: HistoryOnly(c, window=None),
        "C": lambda c: VectorRAG(c),
        "F": lambda c: FullMemory(c),
    }
    return [(k, table[k](make_ctx(cache))) for k in keys]


def run(seeds: list[int], keys: list[str], workers: int, out_dir: str):
    cache = DiskCache(Settings().cache_path)
    graded: dict[str, list[Graded]] = collections.defaultdict(list)
    raw: list[dict] = []
    calls: dict[str, int] = collections.defaultdict(int)
    ctxchars: dict[str, list[int]] = collections.defaultdict(list)
    names: dict[str, str] = {}
    extra_stats: dict[str, dict] = {}
    for seed in seeds:
        world = generate(seed)
        print(f"seed {seed}: {len(world.docs)} docs, {len(world.questions)} questions, asked_at {world.asked_at}")
        for key, sysm in build_systems(keys, cache):
            names[key] = sysm.name
            for i, d in enumerate(world.docs):
                sysm.ingest(d)
                if key == "F" and (i + 1) % 10 == 0:
                    print(f"  [{key}] ingested {i + 1}/{len(world.docs)}")

            def ask_one(q, sysm=sysm, asked_at=world.asked_at):
                r = sysm.ask(q.text, asked_at)
                return q, r

            with ThreadPoolExecutor(workers) as ex:
                for q, r in ex.map(ask_one, world.questions):
                    g = grade(q, r.answer.values, r.answer.unknown)
                    graded[key].append(g)
                    ctxchars[key].append(r.context_chars)
                    raw.append({"seed": seed, "system": key, "qid": q.id, "category": q.category, "kind": q.kind,
                                "question": q.text, "expected": q.expected, "forbidden": q.forbidden,
                                "values": r.answer.values, "answer": r.answer.answer, "correct": g.correct,
                                "stale": g.stale, "unknown": g.unknown})
            calls[key] += sysm.ctx.llm.calls
            if key == "F":
                st = sysm.ctx.store
                extra_stats[key] = {
                    "memories": len(st.memories("status='active' AND recorded_until IS NULL")),
                    "conflicted": len(st.memories("status='conflicted'")),
                    "pending_reviews": len(st.pending_reviews()),
                    "entities": len(st.list_entities()),
                }
            print(f"  {sysm.name}: {sum(g.correct for g in graded[key])}/{len(graded[key])} correct (cumulative)")
    report(graded, raw, calls, ctxchars, names, extra_stats, keys, seeds, out_dir)


def pct(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.0f}% ({100 * lo:.0f}-{100 * hi:.0f}) n={n}"


def report(graded, raw, calls, ctxchars, names, extra_stats, keys, seeds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005
    lines = [f"# Temporal benchmark results ({stamp})", "",
             f"Seeds: {seeds}. Questions per system: {len(graded[keys[0]])}. Cells: accuracy % (95% Wilson CI) n.", ""]
    groups = {
        "Current fact": lambda g: g.kind == "current" and g.category in ("change", "alias", "out_of_order"),
        "Historical (as-of + before)": lambda g: g.kind in ("as_of", "before"),
        "Change date (when)": lambda g: g.kind == "when",
        "Multi-valued recall": lambda g: g.kind == "multi",
        "Poisoning / stale claim": lambda g: g.category == "poison",
        "Transient / planned": lambda g: g.category == "transient",
        "Alias variants (all kinds)": lambda g: g.category == "alias",
        "Out-of-order ingestion": lambda g: g.category == "out_of_order",
        "ALL": lambda g: True,
    }
    lines.append("| Metric | " + " | ".join(names[k] for k in keys) + " |")
    lines.append("|---|" + "---|" * len(keys))
    for title, f in groups.items():
        row = []
        for k in keys:
            sel = [g for g in graded[k] if f(g)]
            row.append(pct(sum(g.correct for g in sel), len(sel)))
        lines.append(f"| {title} | " + " | ".join(row) + " |")
    row = []
    for k in keys:
        sel = [g for g in graded[k] if g.kind == "current" or g.category in ("poison", "transient")]
        row.append(pct(sum(g.stale for g in sel), len(sel)))
    lines.append("| **Stale/poisoned answer rate** (lower is better) | " + " | ".join(row) + " |")
    lines.append("| Unknown / abstain rate | " + " | ".join(
        pct(sum(g.unknown for g in graded[k]), len(graded[k])) for k in keys) + " |")
    lines.append("| Avg context chars | " + " | ".join(
        f"{sum(ctxchars[k]) / max(len(ctxchars[k]), 1):.0f}" for k in keys) + " |")
    lines.append("| LLM calls (ingest + answer) | " + " | ".join(str(calls[k]) for k in keys) + " |")
    if "F" in keys and "C" in keys:
        a = [g.correct for g in graded["F"]]
        b = [g.correct for g in graded["C"]]
        lo, hi = bootstrap_diff(a, b)
        diff = 100 * (sum(a) - sum(b)) / len(a)
        lines.append("")
        lines.append("Paired bootstrap, Full minus Vector RAG (ALL questions): "
                     f"{diff:+.1f} pts, 95% CI [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    if extra_stats:
        lines += ["", f"Memory store after ingestion (last seed): {extra_stats.get('F')}"]
    fails = [r for r in raw if r["system"] == "F" and not r["correct"]]
    if fails:
        lines += ["", "## Full-system failures", ""]
        for r in fails:
            lines.append(f"- **{r['qid']}** ({r['kind']}): {r['question']}  expected={r['expected']} "
                         f"forbidden={r['forbidden']} got={r['values']} -> {r['answer']!r}")
    path = os.path.join(out_dir, f"temporal_{stamp}.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, f"temporal_{stamp}.jsonl"), "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    print("\n".join(lines))
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--first-seed", type=int, default=0, help="seed 0 was used for development; use 1+ for held-out runs")
    ap.add_argument("--systems", default="A,B,C,F")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="eval/results")
    a = ap.parse_args()
    run(list(range(a.first_seed, a.first_seed + a.seeds)), a.systems.split(","), a.workers, a.out)
