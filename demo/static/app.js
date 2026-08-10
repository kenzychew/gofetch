"use strict";

const form = document.getElementById("query-form");
const input = document.getElementById("query-input");
const askBtn = document.getElementById("ask-btn");
const statusEl = document.getElementById("status");
const answerSection = document.getElementById("answer-section");
const answerEl = document.getElementById("answer");
const sourcesSection = document.getElementById("sources-section");
const sourcesEl = document.getElementById("sources");
const latencySection = document.getElementById("latency-section");
const latencyEl = document.getElementById("latency");

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function renderAnswer(rawText) {
  const escaped = escapeHtml(rawText);
  answerEl.innerHTML = escaped.replace(
    /\[(\d+)\]/g,
    '<a href="#source-$1" class="citation">[$1]</a>'
  );
}

function renderSources(citations) {
  sourcesEl.innerHTML = "";
  for (const c of citations) {
    const li = document.createElement("li");
    li.id = `source-${c.index}`;
    const src = document.createElement("div");
    src.className = "source-name";
    src.textContent = `${c.source} (score: ${c.score.toFixed(4)})`;
    const text = document.createElement("div");
    text.className = "source-text";
    text.textContent = c.text;
    li.append(src, text);
    sourcesEl.appendChild(li);
  }
  sourcesSection.hidden = citations.length === 0;
}

function renderLatency(latencyMs, confidence, lowConfidence) {
  latencyEl.innerHTML = "";
  const rows = Object.entries(latencyMs);
  for (const [stage, ms] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = stage;
    const dd = document.createElement("dd");
    dd.textContent = `${ms.toFixed(1)} ms`;
    latencyEl.append(dt, dd);
  }
  const dt = document.createElement("dt");
  dt.textContent = "confidence";
  const dd = document.createElement("dd");
  dd.textContent = lowConfidence
    ? `${confidence.toFixed(4)} (low)`
    : confidence.toFixed(4);
  latencyEl.append(dt, dd);
  latencySection.hidden = rows.length === 0;
}

function setStatus(message) {
  if (!message) {
    statusEl.hidden = true;
    statusEl.textContent = "";
    return;
  }
  statusEl.hidden = false;
  statusEl.textContent = message;
}

async function streamQuery(query) {
  const response = await fetch(`/query?q=${encodeURIComponent(query)}`, {
    headers: { Accept: "text/event-stream" },
  });

  if (response.status === 429) {
    const retryAfter = response.headers.get("Retry-After");
    throw new Error(
      retryAfter
        ? `Rate limit exceeded. Try again in ${retryAfter}s.`
        : "Rate limit exceeded. Try again shortly."
    );
  }
  if (!response.ok || !response.body) {
    throw new Error(`Request failed (${response.status}).`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventType = "";
  let answer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    let newlineIndex;
    while ((newlineIndex = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);

      if (line.startsWith("event:")) {
        eventType = line.slice("event:".length).trim();
        continue;
      }
      if (!line.startsWith("data:")) continue;

      const data = line.slice("data:".length).trim();
      if (eventType === "token") {
        answer += data;
        renderAnswer(answer);
      } else if (eventType === "metadata") {
        const meta = JSON.parse(data);
        renderSources(meta.citations || []);
        renderLatency(meta.latency_ms || {}, meta.confidence || 0, meta.low_confidence);
      } else if (eventType === "error") {
        const err = JSON.parse(data);
        throw new Error(err.error || "Pipeline error.");
      }
    }
  }

  return answer;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (!query) return;

  askBtn.disabled = true;
  setStatus("Searching and generating...");
  answerSection.hidden = false;
  sourcesSection.hidden = true;
  latencySection.hidden = true;
  answerEl.textContent = "";
  sourcesEl.innerHTML = "";
  latencyEl.innerHTML = "";

  try {
    const answer = await streamQuery(query);
    setStatus(answer ? "" : "No answer returned.");
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  } finally {
    askBtn.disabled = false;
  }
});
