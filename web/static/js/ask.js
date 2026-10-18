/* The Ask page: send a question, show the answer and what it was built from.
   Answers take several seconds, so the waiting state matters as much as the
   answer does. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const form = $("ask-form");
  const input = $("ask-input");
  const send = $("ask-send");
  const out = $("ask-out");
  const answer = $("ask-answer");
  const priced = $("ask-priced");
  const sources = $("ask-sources");

  let inFlight = 0;

  /* ---------------- rendering the answer ---------------- */

  const escapeHtml = (s) =>
    s.replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  /* The model replies in light markdown. Rather than pull in a parser, escape
     everything first and then allow back exactly three things: bold, inline
     code and bullet lists. Anything the model invents beyond that shows as
     plain text, which is the safe way to fail. */
  function render(md) {
    const inline = (s) =>
      escapeHtml(s)
        .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
        .replace(/`([^`]+)`/g, "<code>$1</code>");

    const blocks = [];
    let list = null;

    for (const raw of md.split("\n")) {
      const line = raw.trim();
      const bullet = line.match(/^[-*•]\s+(.*)$/);
      if (bullet) {
        list = list || [];
        list.push(`<li>${inline(bullet[1])}</li>`);
        continue;
      }
      if (list) { blocks.push(`<ul>${list.join("")}</ul>`); list = null; }
      if (line) blocks.push(`<p>${inline(line)}</p>`);
    }
    if (list) blocks.push(`<ul>${list.join("")}</ul>`);
    return blocks.join("") || `<p>${inline(md)}</p>`;
  }

  function showThinking() {
    out.hidden = false;
    priced.hidden = true;
    sources.hidden = true;
    sources.open = false;
    answer.innerHTML =
      '<div class="ask-thinking"><span class="ask-thinking__dot"></span>' +
      "<span>Reading the project's records…</span></div>";
    const started = Date.now();
    return setInterval(() => {
      const s = Math.round((Date.now() - started) / 1000);
      const label = answer.querySelector(".ask-thinking span:last-child");
      if (label && s >= 4) label.textContent = `Writing the answer… ${s}s`;
    }, 1000);
  }

  function showPriced(est) {
    if (!est) { priced.hidden = true; return; }
    const money = (v) => `$${Number(v).toFixed(2)}`;
    const rows = [
      ["Route", `${est.pickup.zone} → ${est.dropoff.zone}`],
      ["When", `${est.day} ${String(est.hour).padStart(2, "0")}:00`],
      ["Predicted fare", money(est.fare)],
      ["Likely between", `${money(est.low)} and ${money(est.high)}`],
      ["Trip distance", `${est.miles} mi`],
      ["Uber vs Lyft", `${money(est.uber)} vs ${money(est.lyft)}`],
    ];
    $("ask-priced-kv").innerHTML = rows
      .map(([k, v]) => `<div><dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd></div>`)
      .join("");
    priced.hidden = false;
  }

  function showSources(list) {
    if (!list || !list.length) { sources.hidden = true; return; }
    $("ask-sources-list").innerHTML = list
      .map(
        (s) =>
          '<div class="ask-source"><div class="ask-source__top">' +
          `<span class="ask-source__title">${escapeHtml(s.title)}</span>` +
          `<span class="ask-source__topic">${escapeHtml(s.topic)}</span></div>` +
          `<p class="ask-source__text">${escapeHtml(s.text)}</p></div>`
      )
      .join("");
    sources.hidden = false;
  }

  /* ---------------- asking ---------------- */

  async function ask(question) {
    if (!question.trim()) return;
    const mine = ++inFlight;
    const ticker = showThinking();
    send.disabled = true;

    let data;
    try {
      const res = await fetch(`api/ask?q=${encodeURIComponent(question)}`);
      data = await res.json();
      if (!res.ok) throw new Error(data.detail || res.statusText);
    } catch (err) {
      clearInterval(ticker);
      if (mine !== inFlight) return;
      send.disabled = false;
      answer.innerHTML =
        "<p>The question could not be sent. Check that the site is still " +
        "running, then try again.</p>" +
        `<p class="ask-error">${escapeHtml(String(err.message || err))}</p>`;
      return;
    }

    clearInterval(ticker);
    if (mine !== inFlight) return;       // a newer question has replaced this one
    send.disabled = false;

    answer.innerHTML = render(data.answer || "");
    if (!data.ok && data.error) {
      answer.insertAdjacentHTML(
        "beforeend",
        `<p class="ask-error">${escapeHtml(data.error)}</p>`
      );
    }
    showPriced(data.priced);
    showSources(data.sources);
    if (!data.ok) sources.open = true;   // the facts are the answer in this case
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    ask(input.value);
  });

  $("ask-chips").addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    input.value = chip.textContent.trim();
    ask(input.value);
    input.focus();
  });

  /* how many facts there are, so the footnote is not a vague claim */
  fetch("api/meta")
    .then((r) => r.json())
    .then((m) => {
      if (m.knowledge_facts) $("ask-count").textContent = `${m.knowledge_facts}`;
    })
    .catch(() => { /* leave the wording as it is */ });

  /* ---------------- theme, same contract as the other pages ---------------- */

  const toggle = $("theme-toggle");
  let theme = document.documentElement.getAttribute("data-theme") === "light"
    ? "light" : "dark";

  function applyTheme(next) {
    theme = next;
    if (next === "light") document.documentElement.setAttribute("data-theme", "light");
    else document.documentElement.removeAttribute("data-theme");
    toggle.setAttribute("aria-label",
      next === "light" ? "Switch to dark theme" : "Switch to light theme");
    try { localStorage.setItem("nycride-theme", next); } catch (e) { /* private mode */ }
  }

  applyTheme(theme);
  toggle.addEventListener("click", () => applyTheme(theme === "light" ? "dark" : "light"));

  input.focus();
})();
