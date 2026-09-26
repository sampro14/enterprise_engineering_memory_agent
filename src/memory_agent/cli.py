"""Command-line interface: ingest, ask, timeline, reviews, demo."""
from __future__ import annotations

import os

import typer
from rich.console import Console
from rich.table import Table

from . import review as review_mod
from . import temporal
from .config import get_settings
from .context import Context
from .entities import normalize_key
from .graph import ask as agent_ask
from .ingest import ingest_text
from .models import UNKNOWN_START, SourceType
from .store.sqlite import MemoryStore

app = typer.Typer(help="Enterprise Engineering Memory Agent (PoC)", no_args_is_help=True)
console = Console()


def _ctx(db: str | None = None, clock=None) -> Context:
    from .llm.gemini import GeminiEmbedder, GeminiLLM  # deferred so --help works without a key

    s = get_settings()
    store = MemoryStore(db or s.db_path, s.tenant_id)
    ctx = Context(store, GeminiLLM(s), GeminiEmbedder(s), s)
    if clock:
        ctx.clock = clock
    return ctx


def _obj(ctx: Context, m) -> str:
    return m.object_value or (ctx.store.get_entity(m.object_id).name if m.object_id else "")


@app.command()
def ingest(
    file: str = typer.Argument(..., help="Text/markdown/yaml file to learn from"),
    source_type: SourceType = typer.Option(SourceType.DEVELOPER, help="Authority tier of the source"),
    date: str = typer.Option(None, help="Document date YYYY-MM-DD (default: today)"),
    db: str = typer.Option(None, help="SQLite path"),
):
    """Extract, validate and store memories from a document."""
    ctx = _ctx(db)
    with open(file) as f:
        text = f.read()
    rep = ingest_text(ctx, text, os.path.basename(file), source_type, date)
    if rep.skipped_duplicate:
        console.print("[yellow]already ingested (identical content)[/yellow]")
        return
    t = Table("decision", "subject", "relation", "object", "reason")
    for o in rep.outcomes:
        t.add_row(o.decision, o.subject, o.relation, o.object, o.reason)
    console.print(t)


@app.command()
def ask(question: str, db: str = typer.Option(None), show_context: bool = typer.Option(False)):
    """Answer a question from memory (current, historical and as-of aware)."""
    ctx = _ctx(db)
    st = agent_ask(ctx, question)
    a = st["answer"]
    console.print(f"[bold]{a.answer}[/bold]")
    if a.values:
        console.print(f"values: {a.values}")
    if a.unknown:
        console.print("[yellow]memory does not contain the answer[/yellow]")
    if show_context:
        console.print("\n[dim]" + st.get("context_text", "(no memory used)") + "[/dim]")


@app.command()
def timeline(entity: str, relation: str = typer.Option(None), db: str = typer.Option(None)):
    """Show the validity history of an entity's facts."""
    ctx = _ctx(db)
    e = ctx.store.entity_by_alias_key(normalize_key(entity))
    if not e:
        console.print(f"[red]unknown entity {entity!r}[/red]")
        raise typer.Exit(1)
    rels = [relation] if relation else sorted({m.relation for m in ctx.store.memories("subject_id=?", (e.id,))})
    t = Table("relation", "value", "valid from", "valid until", "confidence", "status", "sources", title=e.name)
    for rel in rels:
        for m in temporal.history(ctx, e.id, rel) + temporal.open_conflicts(ctx, e.id, rel):
            src = ", ".join(sorted({r["source_uri"] for r in ctx.store.evidence_of(m.id)}))
            t.add_row(rel, _obj(ctx, m), "unknown" if m.valid_from == UNKNOWN_START else m.valid_from,
                      m.valid_until or "now", f"{m.confidence:.2f}", m.status, src)
    console.print(t)


@app.command()
def reviews(db: str = typer.Option(None)):
    """List pending reviews (conflicts and uncertain candidates)."""
    ctx = _ctx(db)
    t = Table("id", "reason", "memory", "created")
    for r in ctx.store.pending_reviews():
        t.add_row(r["id"], r["reason"], r["memory_id"] or "-", r["created_at"])
    console.print(t)


@app.command()
def decide(review_id: str, approve: bool = typer.Option(..., "--approve/--reject"), db: str = typer.Option(None)):
    """Approve or reject a pending review."""
    ctx = _ctx(db)
    try:
        (review_mod.approve if approve else review_mod.reject)(ctx, review_id)
    except review_mod.ReviewError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    console.print("done")


@app.command()
def demo():
    """Replay the Stripe -> Razorpay story in a throwaway in-memory store (uses the LLM)."""
    from .llm.gemini import GeminiEmbedder, GeminiLLM

    s = get_settings()
    day = {"d": "2025-02-01"}
    ctx = Context(MemoryStore(":memory:"), GeminiLLM(s), GeminiEmbedder(s), s, lambda: day["d"])
    docs = [
        ("2025-02-01", SourceType.OFFICIAL_DOCS, "arch-2025.md",
         ("PaymentService uses Stripe as its payment provider. It is owned by the Payments Team "
          "and its primary database is PostgreSQL.")),
        ("2026-09-15", SourceType.ARCHITECTURE_REPO, "adr-042.md",
         ("ADR-042: On 2026-09-11 the payments-svc migration completed. We moved from Stripe to Razorpay because "
          "enterprise customers required a specific settlement workflow. The primary database was migrated from "
          "PostgreSQL to Oracle on the same date. Tomorrow I will be testing the service locally.")),
        ("2026-10-05", SourceType.DEVELOPER, "chat", "hey, I think the payment service still uses Stripe, right?"),
    ]
    for d, st, uri, text in docs:
        day["d"] = d
        console.rule(f"{d}  ingest {uri} [{st.value}]")
        rep = ingest_text(ctx, text, uri, st, d)
        for o in rep.outcomes:
            console.print(f"  {o.decision:22} {o.subject} {o.relation} = {o.object}  [dim]{o.reason}[/dim]")
    day["d"] = "2026-10-06"
    console.rule("questions (today = 2026-10-06)")
    for q in ["Which payment provider does PaymentService currently use?",
              "Which payment provider did PaymentService use in March 2026?",
              "When did PaymentService switch from Stripe to Razorpay, and why?",
              "What is the primary database of PaymentService?"]:
        st = agent_ask(ctx, q)
        console.print(f"[bold]Q:[/bold] {q}\n[bold green]A:[/bold green] {st['answer'].answer}\n")
    console.rule("pending reviews")
    for r in ctx.store.pending_reviews():
        console.print(f"  {r['id']}: {r['reason']}")
    console.print("\ndemo complete")
