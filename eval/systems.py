"""Systems under test. All answer with the same LLM and the same Answer schema/prompt."""
from __future__ import annotations

from dataclasses import dataclass, field

from eval.scenarios import Doc
from memory_agent.answering import Answer, answer_from_context
from memory_agent.context import Context
from memory_agent.graph import ask as agent_ask
from memory_agent.graph import build_query_graph
from memory_agent.ingest import ingest_text


def _mut_clock(ctx: Context, date: str) -> None:
    ctx.clock = lambda: date


def _fmt(d: Doc) -> str:
    return f"[document dated {d.doc_date}; source: {d.source_type.value}; {d.uri}]\n{d.text}"


@dataclass
class Result:
    answer: Answer
    context_chars: int = 0
    extra: dict = field(default_factory=dict)


class NoMemory:  # Baseline A
    name = "A: no memory"

    def __init__(self, ctx: Context):
        self.ctx = ctx

    def ingest(self, d: Doc) -> None: ...

    def ask(self, question: str, asked_at: str) -> Result:
        _mut_clock(self.ctx, asked_at)
        return Result(answer_from_context(self.ctx, question, ""))


class HistoryOnly:  # Baseline B: recent documents fed as conversation history
    def __init__(self, ctx: Context, window: int | None):
        self.ctx, self.window, self.docs = ctx, window, []
        self.name = f"B: history ({f'last {window} docs' if window else 'all docs'})"

    def ingest(self, d: Doc) -> None:
        self.docs.append(d)

    def ask(self, question: str, asked_at: str) -> Result:
        _mut_clock(self.ctx, asked_at)
        docs = self.docs[-self.window:] if self.window else self.docs
        text = "DOCUMENTS (in the order received):\n\n" + "\n\n".join(_fmt(d) for d in docs)
        return Result(answer_from_context(self.ctx, question, text), len(text))


class VectorRAG:  # Baseline C: chunk = document, top-k cosine
    name = "C: vector RAG"

    def __init__(self, ctx: Context, k: int = 5):
        self.ctx, self.k, self.docs = ctx, k, []

    def ingest(self, d: Doc) -> None:
        self.ctx.store.put_vector("doc", str(len(self.docs)), self.ctx.embedder.embed([_fmt(d)])[0])
        self.docs.append(d)

    def ask(self, question: str, asked_at: str) -> Result:
        _mut_clock(self.ctx, asked_at)
        q = self.ctx.embedder.embed([question])[0]
        hits = self.ctx.store.search_vectors("doc", q, k=self.k)
        text = "RETRIEVED DOCUMENTS (most similar first):\n\n" + "\n\n".join(_fmt(self.docs[int(i)]) for i, _ in hits)
        return Result(answer_from_context(self.ctx, question, text), len(text))


class FullMemory:  # Proposed system
    name = "Full: temporal+confidence memory"

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.graph = build_query_graph(ctx)
        self.reviews_opened = 0

    def ingest(self, d: Doc) -> None:
        _mut_clock(self.ctx, d.ingest_date)
        ingest_text(self.ctx, d.text, d.uri, d.source_type, d.doc_date)

    def ask(self, question: str, asked_at: str) -> Result:
        _mut_clock(self.ctx, asked_at)
        st = agent_ask(self.ctx, question, self.graph)
        return Result(st["answer"], len(st.get("context_text", "")),
                      {"context": st.get("context_text", ""), "analysis": st["analysis"].model_dump()})
