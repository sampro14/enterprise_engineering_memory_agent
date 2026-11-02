// Vanilla JS, no build step. Talks to the same-origin /api/v1 REST API.
(() => {
  "use strict";

  const API = "/api/v1";
  const KEY_STORAGE = "memory_agent_api_key";

  const $ = (sel) => document.querySelector(sel);
  const el = (tag, cls, html) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html !== undefined) n.innerHTML = html;
    return n;
  };
  const escapeHtml = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function getKey() {
    return localStorage.getItem(KEY_STORAGE) || "";
  }
  function setKey(k) {
    localStorage.setItem(KEY_STORAGE, k);
  }
  function clearKey() {
    localStorage.removeItem(KEY_STORAGE);
  }

  async function api(path, opts = {}) {
    const res = await fetch(API + path, {
      ...opts,
      headers: {
        "content-type": "application/json",
        authorization: "Bearer " + getKey(),
        ...(opts.headers || {}),
      },
    });
    if (res.status === 401) throw new Error("unauthorized");
    if (res.status === 204) return null;
    const body = await res.json().catch(() => null);
    if (!res.ok) throw new Error((body && (body.detail || body.error)) || `request failed (${res.status})`);
    return body;
  }

  // ---------------------------------------------------------------- connect
  async function tryConnect(key) {
    setKey(key);
    try {
      await api("/reviews?status=pending");
    } catch (e) {
      clearKey();
      throw e;
    }
  }

  function showApp() {
    $("#connect-view").classList.add("hidden");
    $("#app-view").classList.remove("hidden");
    const tenantLabel = getKey().slice(0, 10) + "…";
    $("#tenant-badge").textContent = tenantLabel;
    refreshReviews();
  }

  function showConnect(message) {
    $("#app-view").classList.add("hidden");
    $("#connect-view").classList.remove("hidden");
    const errBox = $("#connect-error");
    if (message) {
      errBox.textContent = message;
      errBox.classList.remove("hidden");
    } else {
      errBox.classList.add("hidden");
    }
  }

  $("#connect-btn").addEventListener("click", async () => {
    const key = $("#api-key-input").value.trim();
    if (!key) return;
    $("#connect-btn").disabled = true;
    try {
      await tryConnect(key);
      showApp();
    } catch (e) {
      showConnect("Could not connect: " + e.message);
    } finally {
      $("#connect-btn").disabled = false;
    }
  });

  $("#disconnect-btn").addEventListener("click", () => {
    clearKey();
    showConnect();
  });

  // ---------------------------------------------------------------- tabs
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => activateTab(btn.dataset.tab));
  });

  function activateTab(name) {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    ["demo", "ask", "review"].forEach((t) => $("#tab-" + t).classList.toggle("hidden", t !== name));
    if (name === "review") refreshReviews();
  }

  // ---------------------------------------------------------------- ingest + job polling
  function outcomeRow(o) {
    const tone =
      o.decision === "ACCEPT" ? "border-emerald-200 bg-emerald-50 text-emerald-800" :
      o.decision === "MARK_CONFLICT" ? "border-amber-200 bg-amber-50 text-amber-800" :
      "border-slate-200 bg-slate-50 text-slate-600";
    return el(
      "div",
      `fade-in rounded-lg border px-3 py-2 text-sm ${tone}`,
      `<span class="font-mono text-xs uppercase tracking-wide">${escapeHtml(o.decision)}</span>
       &middot; <b>${escapeHtml(o.subject)}</b> ${escapeHtml(o.relation)} = <b>${escapeHtml(o.object)}</b>
       <div class="text-xs opacity-75 mt-0.5">${escapeHtml(o.reason)}</div>`
    );
  }

  async function pollJob(jobId, { timeoutMs = 30000, intervalMs = 700 } = {}) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      const job = await api(`/jobs/${jobId}`);
      if (job.status === "succeeded" || job.status === "failed") return job;
      await new Promise((r) => setTimeout(r, intervalMs));
    }
    throw new Error("job timed out");
  }

  async function runIngest(feedEl, { text, uri, source_type, doc_date }) {
    const pending = el("div", "fade-in text-sm text-slate-400 flex items-center gap-2", `
      <svg class="spin h-3.5 w-3.5" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" stroke-dasharray="40" stroke-linecap="round"/></svg>
      ingesting ${escapeHtml(uri)}…`);
    feedEl.appendChild(pending);
    try {
      const accepted = await api("/ingest", { method: "POST", body: JSON.stringify({ text, uri, source_type, doc_date }) });
      const job = await pollJob(accepted.job_id);
      pending.remove();
      if (job.status === "failed") {
        feedEl.appendChild(el("div", "fade-in text-sm text-red-600", `failed: ${escapeHtml(job.error || "unknown error")}`));
        return;
      }
      const result = job.result || {};
      if (result.skipped_duplicate) {
        feedEl.appendChild(el("div", "fade-in text-sm text-slate-400", "already known (duplicate)"));
      }
      (result.outcomes || []).forEach((o) => feedEl.appendChild(outcomeRow(o)));
    } catch (e) {
      pending.remove();
      feedEl.appendChild(el("div", "fade-in text-sm text-red-600", escapeHtml(e.message)));
    }
  }

  // ---------------------------------------------------------------- demo story (mirrors cli.py demo())
  const DEMO_DOCS = [
    {
      doc_date: "2025-02-01",
      source_type: "official_docs",
      uri: "arch-2025.md",
      text: "PaymentService uses Stripe as its payment provider. It is owned by the Payments Team and its primary database is PostgreSQL.",
    },
    {
      doc_date: "2026-09-15",
      source_type: "architecture_repo",
      uri: "adr-042.md",
      text: "ADR-042: On 2026-09-11 the payments-svc migration completed. We moved from Stripe to Razorpay because enterprise customers required a specific settlement workflow. The primary database was migrated from PostgreSQL to Oracle on the same date. Tomorrow I will be testing the service locally.",
    },
    {
      doc_date: "2026-10-05",
      source_type: "developer",
      uri: "chat",
      text: "hey, I think the payment service still uses Stripe, right?",
    },
  ];

  const DEMO_QUESTIONS = [
    "Which payment provider does PaymentService currently use?",
    "Which payment provider did PaymentService use in March 2026?",
    "When did PaymentService switch from Stripe to Razorpay, and why?",
    "What is the primary database of PaymentService?",
  ];

  $("#load-demo-btn").addEventListener("click", async () => {
    const btn = $("#load-demo-btn");
    const feed = $("#demo-feed");
    feed.innerHTML = "";
    btn.disabled = true;
    btn.textContent = "Loading…";
    for (const doc of DEMO_DOCS) {
      feed.appendChild(el("div", "fade-in text-xs font-semibold text-slate-400 uppercase tracking-wide pt-2", `${doc.doc_date} · ${doc.uri} [${doc.source_type}]`));
      await runIngest(feed, doc);
    }
    btn.disabled = false;
    btn.textContent = "▶ Load Stripe → Razorpay Demo";
    const chips = $("#demo-question-chips");
    chips.innerHTML = "";
    DEMO_QUESTIONS.forEach((q) => {
      const chip = el("button", "chip bg-slate-100 hover:bg-slate-200 rounded-full px-3 py-1.5 text-sm text-left", escapeHtml(q));
      chip.addEventListener("click", () => {
        activateTab("ask");
        $("#ask-input").value = q;
        $("#ask-btn").click();
      });
      chips.appendChild(chip);
    });
    $("#demo-questions").classList.remove("hidden");
  });

  $("#ingest-btn").addEventListener("click", async () => {
    const text = $("#ingest-text").value.trim();
    if (!text) return;
    const doc = {
      text,
      uri: $("#ingest-uri").value.trim() || "manual",
      source_type: $("#ingest-source").value,
      doc_date: $("#ingest-date").value || null,
    };
    await runIngest($("#ingest-feed"), doc);
    $("#ingest-text").value = "";
  });

  // ---------------------------------------------------------------- ask
  $("#ask-btn").addEventListener("click", async () => {
    const question = $("#ask-input").value.trim();
    if (!question) return;
    const resultBox = $("#ask-result");
    resultBox.innerHTML = `<div class="text-sm text-slate-400 flex items-center gap-2 py-4">
      <svg class="spin h-3.5 w-3.5" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" stroke-dasharray="40" stroke-linecap="round"/></svg>
      thinking…</div>`;
    try {
      const res = await api("/agent/ask", { method: "POST", body: JSON.stringify({ question }) });
      resultBox.innerHTML = "";
      resultBox.appendChild(renderAnswer(res));
    } catch (e) {
      resultBox.innerHTML = `<div class="text-sm text-red-600 py-2">${escapeHtml(e.message)}</div>`;
    }
  });
  $("#ask-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") $("#ask-btn").click();
  });

  function renderAnswer(res) {
    const card = el("div", "bg-white rounded-2xl border border-slate-200 p-6 fade-in space-y-3");
    const hasConflicts = (res.conflicts || []).length > 0;
    card.appendChild(el("div", "text-lg", escapeHtml(res.answer)));
    // `conflicts` covers every open conflict on the entities involved, not only the fact in this answer.
    if (res.used_memory) {
      card.appendChild(el("div", "inline-block text-xs font-medium bg-emerald-100 text-emerald-800 rounded-full px-2.5 py-1 mr-2", "✓ answered from memory"));
    }
    if (hasConflicts) {
      card.appendChild(el("div", "inline-block text-xs font-medium bg-amber-100 text-amber-800 rounded-full px-2.5 py-1", "⚠ open conflict on this entity"));
    }
    if ((res.citations || []).length) {
      card.appendChild(el("div", "text-sm text-slate-500", `<b class="text-slate-700">Citations:</b> ${res.citations.map(escapeHtml).join(", ")}`));
    }
    if (hasConflicts) {
      card.appendChild(el("div", "text-sm text-amber-700", `<b>Open conflicts:</b> ${res.conflicts.map(escapeHtml).join("; ")}`));
    }
    if (res.context) {
      const details = el("details", "text-sm");
      details.innerHTML = `<summary class="cursor-pointer text-slate-400 hover:text-slate-700">Show retrieved context</summary>
        <pre class="mt-2 whitespace-pre-wrap bg-slate-50 rounded-lg p-3 text-xs text-slate-600">${escapeHtml(res.context)}</pre>`;
      card.appendChild(details);
    }
    return card;
  }

  // ---------------------------------------------------------------- review queue
  $("#refresh-reviews-btn").addEventListener("click", refreshReviews);

  async function refreshReviews() {
    const list = $("#review-list");
    try {
      const reviews = await api("/reviews?status=pending");
      list.innerHTML = "";
      if (!reviews.length) {
        list.appendChild(el("div", "text-sm text-slate-400 py-6 text-center", "No pending reviews."));
        return;
      }
      reviews.forEach((r) => list.appendChild(renderReview(r)));
    } catch (e) {
      list.innerHTML = `<div class="text-sm text-red-600">${escapeHtml(e.message)}</div>`;
    }
  }

  function renderReview(r) {
    const card = el("div", "fade-in bg-white rounded-2xl border border-amber-200 p-5 flex items-start justify-between gap-4");
    const fact = (r.candidate && r.candidate.fact) || {};
    const left = el("div", "", `
      <div class="text-xs font-mono text-slate-400 mb-1">${escapeHtml(r.id)}</div>
      <div class="text-sm font-medium text-slate-800 mb-1">${escapeHtml(r.reason)}</div>
      ${fact.subject ? `<div class="text-sm text-slate-500"><b>${escapeHtml(fact.subject)}</b> ${escapeHtml(fact.relation || "")} = <b>${escapeHtml(fact.object || "")}</b></div>` : ""}
      ${r.candidate && r.candidate.source_uri ? `<div class="text-xs text-slate-400 mt-1">source: ${escapeHtml(r.candidate.source_uri)} (${escapeHtml(r.candidate.source_type || "")})</div>` : ""}
    `);
    const actions = el("div", "flex gap-2 shrink-0");
    const approveBtn = el("button", "bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-medium rounded-lg px-3 py-1.5", "Approve");
    const rejectBtn = el("button", "bg-white border border-slate-300 hover:bg-slate-50 text-sm font-medium rounded-lg px-3 py-1.5", "Reject");
    const decide = async (decision) => {
      approveBtn.disabled = rejectBtn.disabled = true;
      try {
        await api(`/reviews/${r.id}/decision`, { method: "POST", body: JSON.stringify({ decision }) });
        card.remove();
      } catch (e) {
        alert(e.message);
        approveBtn.disabled = rejectBtn.disabled = false;
      }
    };
    approveBtn.addEventListener("click", () => decide("approve"));
    rejectBtn.addEventListener("click", () => decide("reject"));
    actions.append(approveBtn, rejectBtn);
    card.append(left, actions);
    return card;
  }

  // ---------------------------------------------------------------- boot
  (async function boot() {
    const key = getKey();
    if (!key) return showConnect();
    try {
      await api("/reviews?status=pending");
      showApp();
    } catch {
      showConnect();
    }
  })();
})();
