// Hosted demo only: answers the UI's /api/v1 calls from responses recorded by scripts/record_demo.py, so the
// page works with no backend. Loaded before app.js by `make demo-site`; never served by the real API.
(() => {
  "use strict";

  const REPO = "https://github.com/sampro14/enterprise_engineering_memory_agent";
  const fixture = window.fetch("fixture.json").then((r) => r.json());
  const realFetch = window.fetch.bind(window);
  const state = { loaded: new Set(), review: "pending" };

  localStorage.setItem("memory_agent_api_key", "mem_demo-replay");

  const json = (status, body, delayMs) =>
    new Promise((resolve) =>
      setTimeout(() => resolve(new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } })), delayMs));
  const notRecorded = (what) =>
    json(422, { error: "not_recorded", detail: `${what} This page replays a recorded run; clone the repo to try your own input.` }, 300);

  async function replay(path, opts) {
    const f = await fixture;
    const body = opts.body ? JSON.parse(opts.body) : {};
    const demoLoaded = Object.keys(f.jobs).every((uri) => state.loaded.has(uri));

    if (path.startsWith("/reviews") && opts.method !== "POST") {
      return json(200, demoLoaded && state.review === "pending" ? f.reviews : [], 100);
    }
    if (path.endsWith("/decision")) {
      state.review = body.decision;
      return json(200, f.decisions[body.decision], 400);
    }
    if (path === "/ingest") {
      const job = f.jobs[body.uri];
      if (!job) return notRecorded("Custom ingest needs the live pipeline.");
      state.loaded.add(body.uri);
      if (Object.keys(f.jobs).every((uri) => state.loaded.has(uri))) state.review = "pending";
      return json(202, { job_id: job.id, status: "queued", status_url: `/api/v1/jobs/${job.id}` }, 200);
    }
    if (path.startsWith("/jobs/")) {
      const job = Object.values(f.jobs).find((j) => path === `/jobs/${j.id}`);
      return json(200, job, 1200);
    }
    if (path === "/agent/ask") {
      if (!demoLoaded) return json(409, { error: "conflict", detail: "Nothing in memory yet: run ▶ Load Stripe → Razorpay Demo on the Demo tab first." }, 300);
      const answer = f.answers[state.review][body.question];
      if (!answer) return notRecorded(`Only the four suggested questions were recorded (see the Demo tab).`);
      return json(200, answer, 1000);
    }
    return notRecorded("That route was not recorded.");
  }

  window.fetch = (url, opts = {}) => {
    const u = String(url);
    return u.startsWith("/api/v1") ? replay(u.slice("/api/v1".length), opts) : realFetch(url, opts);
  };

  document.addEventListener("DOMContentLoaded", async () => {
    const f = await fixture;
    const banner = document.createElement("div");
    banner.className = "bg-slate-900 text-slate-100 text-sm text-center px-4 py-2";
    banner.innerHTML = `Recorded replay: every response below came from a real run (Gemini 2.5 Flash, ${f.recorded_at}), sped up.
      No backend is running on this page. <a class="underline" href="${REPO}">Source and setup</a>`;
    document.body.prepend(banner);
  });
})();
