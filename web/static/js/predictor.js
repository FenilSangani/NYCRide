/* NYCRide — plan a trip.
   The map and the zone figures come from the exported data; every fare on this
   page comes from the trained model through the API in app/main.py. */

const STORAGE_KEY = "nycride-theme";

const TILES = {
  dark: "https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
  light: "https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
};

/* public demo server, no key; it is rate limited, so the straight line stays as a fallback */
const OSRM = "https://router.project-osrm.org/route/v1/driving";

const LAYERS = {
  trips: { legend: "Trips started per hour", get: (s) => s.trips_per_hour, fmt: (v) => Math.round(v).toLocaleString() },
  fare: { legend: "Median fare", get: (s) => s.median_fare, fmt: (v) => `$${v.toFixed(0)}` },
  wait: { legend: "Typical wait, minutes", get: (s) => s.mean_wait, fmt: (v) => v.toFixed(1) },
};

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

let layerKey = "trips";
let theme = document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";

const $ = (id) => document.getElementById(id);
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const ramp = () => [1, 2, 3, 4, 5].map((i) => cssVar(`--ramp-${i}`));
const money = (v) => `$${v.toFixed(2)}`;
const hourLabel = (h) => (h === 0 ? "midnight" : h === 12 ? "noon" : h < 12 ? `${h} am` : `${h - 12} pm`);
const compact = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}K` : String(n));

const fill = (key, value) =>
  document.querySelectorAll(`[data-fill="${key}"]`).forEach((el) => { el.textContent = value; });

/* Trip counts are wildly skewed - the airports sit far above everything else -
   so equal-width bins would paint almost the whole city one shade. */
function quantileBreaks(values, bands) {
  const sorted = [...values].sort((a, b) => a - b);
  const out = [];
  for (let i = 1; i < bands; i += 1) out.push(sorted[Math.floor((sorted.length * i) / bands)]);
  return out;
}

const bandFor = (v, breaks) => {
  let i = 0;
  while (i < breaks.length && v > breaks[i]) i += 1;
  return i;
};

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

async function main() {
  let stats;
  let zones;
  let meta;
  try {
    [stats, zones, meta] = await Promise.all([
      getJSON("data/zone_stats.json"),
      getJSON("data/nyc_zones.geojson"),
      getJSON("api/meta"),
    ]);
  } catch (err) {
    document.querySelector(".map__loading").textContent = "Could not reach the API";
    console.error(err);
    return;
  }

  const entries = Object.entries(stats);
  fill("trips_short", compact(meta.data.total_trips));
  fill("uber_share", `${meta.data.provider_share.Uber}%`);
  fill("lyft_share", `${meta.data.provider_share.Lyft}%`);

  const hourSel = $("pick-hour");
  for (let h = 0; h < 24; h += 1) {
    const o = document.createElement("option");
    o.value = String(h);
    o.textContent = `${String(h).padStart(2, "0")}:00 · ${hourLabel(h)}`;
    if (h === 17) o.selected = true;
    hourSel.appendChild(o);
  }

  const places = zones.features.map((f) => f.properties).filter((p) => stats[p.zone_id])
    .sort((a, b) => a.zone.localeCompare(b.zone));
  const placeById = Object.fromEntries(places.map((p) => [p.zone_id, p]));

  const state = { a: null, b: null, next: "a", shared: 0 };
  let token = 0;          // guards against a slow response overwriting a newer one

  /* ---------------- map ---------------- */

  document.querySelector(".map__loading")?.remove();

  const map = L.map("map-canvas", { zoomControl: true, scrollWheelZoom: false });
  let tiles = L.tileLayer(TILES[theme], { attribution: "Tiles &copy; Esri", maxZoom: 17 }).addTo(map);
  let colours = ramp();
  let breaks = [];

  const recomputeBreaks = () => {
    breaks = quantileBreaks(entries.map(([, s]) => LAYERS[layerKey].get(s)), colours.length);
  };

  function styleFor(feature) {
    const id = feature.properties.zone_id;
    const s = stats[id];
    const stroke = cssVar("--map-stroke");
    if (!s) return { fillColor: "transparent", fillOpacity: 0.05, color: stroke, weight: 0.6 };
    if (id === state.a || id === state.b) {
      /* keep the zone's own shade but ring it, so the selection reads against
         any step of the ramp rather than competing with the darkest one */
      return {
        fillColor: colours[bandFor(LAYERS[layerKey].get(s), breaks)],
        fillOpacity: 0.95,
        color: cssVar("--bar-hi"),
        weight: 3,
      };
    }
    return {
      fillColor: colours[bandFor(LAYERS[layerKey].get(s), breaks)],
      fillOpacity: 0.85, color: stroke, weight: 0.6,
    };
  }

  recomputeBreaks();

  /* the zones are polygons in the overlay pane, so a line added alongside them
     ends up underneath; give the route its own pane that sits above */
  map.createPane("routePane");
  map.getPane("routePane").style.zIndex = 450;

  const routeHalo = L.polyline([], { weight: 9, opacity: 0.3, pane: "routePane" }).addTo(map);
  const routeLine = L.polyline([], { weight: 4, opacity: 1, pane: "routePane" }).addTo(map);

  /* A and B as pins, not just a tinted zone: the busiest zones are already
     near-black on the Trips ramp, so a dark fill does not read as "selected" */
  const pins = { a: null, b: null };

  function setPin(end, place) {
    if (pins[end]) { map.removeLayer(pins[end]); pins[end] = null; }
    if (!place) return;
    pins[end] = L.marker([place.lat, place.lon], {
      icon: L.divIcon({
        className: "",
        html: `<span class="map-pin" data-end="${end}">${end.toUpperCase()}</span>`,
        iconSize: [28, 28],
        iconAnchor: [14, 14],
      }),
      keyboard: false,
      interactive: false,
    }).addTo(map);
  }

  const layer = L.geoJSON(zones, {
    style: styleFor,
    onEachFeature: (feature, lyr) => {
      const p = feature.properties;
      if (!stats[p.zone_id]) return;
      lyr.bindTooltip(`${p.zone} · ${p.borough}`, { sticky: true, direction: "auto", offset: [12, 0] });
      /* No bringToFront on hover: moving the node stops mouseout from firing. */
      lyr.on({
        mouseover: (e) => {
          if (p.zone_id === state.a || p.zone_id === state.b) return;
          e.target.setStyle({ weight: 2, color: cssVar("--map-hover"), fillOpacity: 1 });
        },
        mouseout: (e) => { layer.resetStyle(e.target); e.target.closeTooltip(); },
        click: () => { choose(p.zone_id); if (state.a && state.b) frameChoice(p.zone_id); },
      });
    },
  }).addTo(map);

  /* Frame the five boroughs. Newark Airport is a TLC zone too, but it sits far
     into New Jersey and dragged the whole city off to one side. */
  const cityBounds = L.latLngBounds([]);
  layer.eachLayer((l) => {
    const p = l.feature.properties;
    if (p.borough !== "EWR" && stats[p.zone_id]) cityBounds.extend(l.getBounds());
  });
  const home = cityBounds.isValid() ? cityBounds : layer.getBounds();
  map.fitBounds(home, { padding: [8, 8] });

  map.on("click", () => map.scrollWheelZoom.enable());
  map.on("mouseout", () => map.scrollWheelZoom.disable());

  const boundsOf = (id) => {
    let t = null;
    layer.eachLayer((l) => { if (l.feature.properties.zone_id === id) t = l; });
    return t ? t.getBounds() : null;
  };

  /* Once both ends are set, frame the pair - zooming hard into whichever zone
     was picked last hides the other half of the trip. */
  function frameChoice(justPicked) {
    if (state.a && state.b) {
      const bb = L.latLngBounds([]);
      [state.a, state.b].forEach((id) => { const b = boundsOf(id); if (b) bb.extend(b); });
      if (bb.isValid()) map.fitBounds(bb, { padding: [50, 50], maxZoom: 13 });
      return;
    }
    const one = boundsOf(justPicked);
    if (one) map.fitBounds(one, { padding: [40, 40], maxZoom: 13 });
  }

  /* ---------------- legend and layers ---------------- */

  function drawLegend() {
    const { fmt, legend } = LAYERS[layerKey];
    const lo = Math.min(...entries.map(([, s]) => LAYERS[layerKey].get(s)));
    const scale = $("legend-scale");
    scale.replaceChildren();
    colours.forEach((hex, i) => {
      const step = document.createElement("span");
      step.className = "legend__step";
      const chip = document.createElement("span");
      chip.className = "legend__chip";
      chip.style.background = hex;
      const cap = document.createElement("span");
      cap.className = "legend__cap";
      cap.textContent = i === 0 ? `${fmt(lo)}+` : `${fmt(breaks[i - 1])}+`;
      step.append(chip, cap);
      scale.appendChild(step);
    });
    $("legend-title").textContent = legend;
    $("layer-title").textContent = legend;
  }

  drawLegend();

  document.querySelectorAll(".chips button").forEach((btn) => {
    btn.addEventListener("click", () => {
      layerKey = btn.dataset.layer;
      document.querySelectorAll(".chips button").forEach((b) =>
        b.setAttribute("aria-pressed", String(b === btn)));
      recomputeBreaks();
      layer.setStyle(styleFor);
      drawLegend();
    });
  });

  /* ---------------- choosing the ends ---------------- */

  function choose(id) {
    if (state.a === id) { state.a = null; state.next = "a"; }
    else if (state.b === id) { state.b = null; state.next = "b"; }
    else if (state.next === "a") { state.a = id; state.next = "b"; }
    else { state.b = id; state.next = "a"; }
    refresh();
  }

  $("clear-a").addEventListener("click", () => { state.a = null; state.next = "a"; refresh(); });
  $("clear-b").addEventListener("click", () => { state.b = null; state.next = "b"; refresh(); });
  $("swap-ends").addEventListener("click", () => { [state.a, state.b] = [state.b, state.a]; refresh(); });
  $("reset-map").addEventListener("click", () => map.fitBounds(home, { padding: [8, 8] }));
  hourSel.addEventListener("change", refresh);
  $("pick-day").addEventListener("change", refresh);

  document.querySelectorAll(".segmented button").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.shared = Number(btn.dataset.shared);
      document.querySelectorAll(".segmented button").forEach((b) =>
        b.setAttribute("aria-pressed", String(b === btn)));
      refresh();
    });
  });

  /* ---------------- search ---------------- */

  const search = $("zone-search");
  const results = $("zone-results");

  const closeResults = () => {
    results.hidden = true;
    results.replaceChildren();
    search.setAttribute("aria-expanded", "false");
  };

  search.addEventListener("input", () => {
    const q = search.value.trim().toLowerCase();
    if (q.length < 2) return closeResults();
    const hits = places.filter((p) =>
      p.zone.toLowerCase().includes(q) || p.borough.toLowerCase().includes(q));
    results.replaceChildren();
    if (!hits.length) {
      const li = document.createElement("li");
      li.className = "results__empty";
      li.textContent = `No zone matches “${search.value.trim()}”`;
      results.appendChild(li);
    } else {
      hits.slice(0, 12).forEach((p) => {
        const li = document.createElement("li");
        const btn = document.createElement("button");
        btn.type = "button";
        btn.innerHTML = `<span>${p.zone}</span><em>${p.borough}</em>`;
        btn.addEventListener("click", () => {
          choose(p.zone_id);
          frameChoice(p.zone_id);
          search.value = "";
          closeResults();
        });
        li.appendChild(btn);
        results.appendChild(li);
      });
    }
    results.hidden = false;
    search.setAttribute("aria-expanded", "true");
  });

  search.addEventListener("keydown", (e) => { if (e.key === "Escape") closeResults(); });
  document.addEventListener("click", (e) => {
    if (!results.contains(e.target) && e.target !== search) closeResults();
  });

  /* ---------------- the road line ---------------- */

  async function drawRoute(a, b) {
    const straight = [[a.lat, a.lon], [b.lat, b.lon]];
    const paint = (pts, dashed) => {
      const c = cssVar("--bar-hi");
      routeLine.setLatLngs(pts).setStyle({ color: c, dashArray: dashed ? "8 8" : null });
      routeHalo.setLatLngs(pts).setStyle({ color: c });
    };
    paint(straight, true);   // show something immediately

    try {
      const url = `${OSRM}/${a.lon},${a.lat};${b.lon},${b.lat}?overview=full&geometries=geojson`;
      const r = await fetch(url, { signal: AbortSignal.timeout(6000) });
      const j = await r.json();
      const route = j.routes?.[0];
      if (!route) return;
      paint(route.geometry.coordinates.map(([lon, lat]) => [lat, lon]), false);
      $("route-road").textContent =
        `${(route.distance / 1609.34).toFixed(1)} mi by road · ${Math.round(route.duration / 60)} min drive`;
    } catch (e) {
      /* the public router is rate limited and sometimes just says no; the dashed
         straight line is already drawn, so say what it is and move on */
      $("route-road").textContent = "road route unavailable — straight line shown";
    }
  }

  /* ---------------- hourly chart ---------------- */

  const plot = $("plot");
  const tipText = $("tip-text");
  const RESTING = "Hover the line to read any hour";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const W = 960;
  const H = 280;
  const PAD = { l: 48, r: 14, t: 18, b: 30 };

  const el = (name, attrs = {}) => {
    const n = document.createElementNS(SVG_NS, name);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    return n;
  };

  /* A line, not bars: the hours run continuously, and the shape of the day is
     the point - where it dips, where it climbs. Bars invite you to compare
     24 separate things instead of reading one curve. */
  function drawHours(hours, chosen) {
    tipText.textContent = RESTING;
    plot.replaceChildren();
    plot.classList.remove("is-hover");

    const fares = hours.map((h) => h.fare);
    const lo = Math.min(...fares);
    const hi = Math.max(...fares);
    const pad = Math.max((hi - lo) * 0.18, 1);
    const yLo = Math.max(0, lo - pad);
    const yHi = hi + pad;

    const plotW = W - PAD.l - PAD.r;
    const plotH = H - PAD.t - PAD.b;
    const x = (h) => PAD.l + (h / 23) * plotW;
    const y = (v) => PAD.t + (1 - (v - yLo) / (yHi - yLo)) * plotH;

    const svg = el("svg", {
      viewBox: `0 0 ${W} ${H}`,
      role: "img",
      "aria-label": `Fare for this trip at each hour, from ${money(lo)} at ` +
        `${hourLabel(fares.indexOf(lo))} to ${money(hi)} at ${hourLabel(fares.indexOf(hi))}`,
    });

    /* y gridlines at four rounded values */
    for (let i = 0; i <= 3; i += 1) {
      const v = yLo + ((yHi - yLo) * i) / 3;
      const gy = y(v);
      svg.appendChild(el("line", { class: "plot__grid", x1: PAD.l, x2: W - PAD.r, y1: gy, y2: gy }));
      const t = el("text", { class: "plot__tick", x: PAD.l - 10, y: gy + 4, "text-anchor": "end" });
      t.textContent = `$${v.toFixed(0)}`;
      svg.appendChild(t);
    }

    /* x ticks every three hours */
    for (let h = 0; h <= 23; h += 3) {
      const t = el("text", { class: "plot__tick", x: x(h), y: H - 8, "text-anchor": "middle" });
      t.textContent = `${String(h).padStart(2, "0")}`;
      svg.appendChild(t);
    }
    svg.appendChild(el("line", {
      class: "plot__axis", x1: PAD.l, x2: W - PAD.r, y1: y(yLo), y2: y(yLo),
    }));

    const pts = hours.map((row, h) => [x(h), y(row.fare)]);
    const line = pts.map(([px, py]) => `${px.toFixed(1)},${py.toFixed(1)}`).join(" ");
    svg.appendChild(el("polygon", {
      class: "plot__area",
      points: `${PAD.l},${y(yLo)} ${line} ${W - PAD.r},${y(yLo)}`,
    }));
    svg.appendChild(el("polyline", { class: "plot__line", points: line }));

    /* mark the two ends of the day and the hour the rider picked */
    const marks = [
      { h: fares.indexOf(lo), label: `cheapest ${money(lo)}` },
      { h: fares.indexOf(hi), label: `dearest ${money(hi)}` },
    ];
    marks.forEach((m) => {
      svg.appendChild(el("circle", { class: "plot__dot", cx: x(m.h), cy: y(fares[m.h]), r: 5 }));
      const above = fares[m.h] >= (yLo + yHi) / 2;
      const t = el("text", {
        class: "plot__note",
        x: Math.min(Math.max(x(m.h), PAD.l + 44), W - PAD.r - 44),
        y: y(fares[m.h]) + (above ? -14 : 22),
        "text-anchor": "middle",
      });
      t.textContent = m.label;
      svg.appendChild(t);
    });
    svg.appendChild(el("circle", {
      class: "plot__dot plot__dot--chosen", cx: x(chosen), cy: y(fares[chosen]), r: 6,
    }));

    /* hover layer: a crosshair that snaps to the nearest hour */
    const cross = el("line", { class: "plot__cross", y1: PAD.t, y2: y(yLo) });
    const cursor = el("circle", { class: "plot__cursor", r: 6 });
    svg.appendChild(cross);
    svg.appendChild(cursor);

    const hit = el("rect", {
      class: "plot__hit", x: PAD.l, y: PAD.t, width: plotW, height: plotH,
    });

    const hourAt = (evt) => {
      const box = svg.getBoundingClientRect();
      const px = ((evt.clientX - box.left) / box.width) * W;
      return Math.min(23, Math.max(0, Math.round(((px - PAD.l) / plotW) * 23)));
    };

    hit.addEventListener("mousemove", (evt) => {
      const h = hourAt(evt);
      plot.classList.add("is-hover");
      cross.setAttribute("x1", x(h));
      cross.setAttribute("x2", x(h));
      cursor.setAttribute("cx", x(h));
      cursor.setAttribute("cy", y(fares[h]));
      tipText.innerHTML = `<b>${hourLabel(h)}</b><i>·</i> <b>${money(fares[h])}</b>` +
        `<i>·</i> ${money(hours[h].per_mile)} / mile`;
    });
    hit.addEventListener("mouseleave", () => {
      plot.classList.remove("is-hover");
      tipText.textContent = RESTING;
    });
    hit.addEventListener("click", (evt) => {
      hourSel.value = String(hourAt(evt));
      refresh();
    });
    svg.appendChild(hit);

    plot.appendChild(svg);
  }

  function emptyChart(message) {
    plot.replaceChildren();
    tipText.textContent = message;
    $("chart-notes").innerHTML = "";
  }

  /* ---------------- small vertical bar charts ---------------- */

  /* One helper for the provider pair, the week and the importance list: the
     same shape of question each time, so the same chart. */
  function barChart(host, rows, opts = {}) {
    const { fmt = (v) => money(v), zeroBased = true, height = 220 } = opts;
    const n = rows.length;
    /* scale the canvas to the number of bars: a fixed-width viewBox leaves two
       bars stranded as slivers in a wide box */
    const w = Math.max(300, n * 96);
    const pad = { t: 26, b: 34, x: 12 };
    const slot = (w - pad.x * 2) / n;
    const barW = Math.min(slot * 0.62, 86);
    const plotH = height - pad.t - pad.b;
    const top = Math.max(...rows.map((r) => r.value));
    const floor = zeroBased ? 0 : Math.min(...rows.map((r) => r.value)) * 0.86;
    const h = (v) => Math.max(3, ((v - floor) / (top - floor || 1)) * plotH);

    const svg = el("svg", {
      viewBox: `0 0 ${w} ${height}`,
      role: "img",
      "aria-label": rows.map((r) => `${r.name} ${fmt(r.value)}`).join(", "),
    });

    rows.forEach((r, i) => {
      const cx = pad.x + slot * i + slot / 2;
      const bh = h(r.value);
      const y = pad.t + plotH - bh;
      svg.appendChild(el("rect", {
        class: "bar-rect", x: cx - barW / 2, y, width: barW, height: bh, rx: 4,
        fill: r.on ? "var(--bar-hi)" : "var(--bar)",
      }));
      const v = el("text", { class: "bar-value", x: cx, y: y - 8, "text-anchor": "middle" });
      v.textContent = fmt(r.value);
      svg.appendChild(v);
      const nm = el("text", {
        class: `bar-name${r.on ? " bar-name--on" : ""}`,
        x: cx, y: height - 12, "text-anchor": "middle",
      });
      nm.textContent = r.name;
      svg.appendChild(nm);
      const t = el("title");
      t.textContent = `${r.name}: ${fmt(r.value)}`;
      svg.appendChild(t);
    });

    svg.appendChild(el("line", {
      class: "bar-base", x1: pad.x, x2: w - pad.x,
      y1: pad.t + plotH + 0.5, y2: pad.t + plotH + 0.5,
    }));

    host.replaceChildren(svg);
  }

  /* ---------------- render ---------------- */

  function renderLocal() {
    const a = state.a ? placeById[state.a] : null;
    const b = state.b ? placeById[state.b] : null;
    $("name-a").textContent = a ? a.zone : "Click a zone on the map";
    $("name-a").dataset.empty = a ? "false" : "true";
    $("name-b").textContent = b ? b.zone : "Then click another";
    $("name-b").dataset.empty = b ? "false" : "true";
    $("end-a").dataset.active = String(state.next === "a");
    $("end-b").dataset.active = String(state.next === "b");
    layer.setStyle(styleFor);
    setPin("a", a);
    setPin("b", b);
    if (!a || !b) {
      routeLine.setLatLngs([]);
      routeHalo.setLatLngs([]);
      $("route-road").textContent = "";
    }
  }

  function clearResults(message) {
    $("fare-value").textContent = "—";
    $("fare-range").textContent = message;
    ["kv-dist", "kv-wait", "kv-zonefare", "kv-ppm", "kv-zoneppm"].forEach((id) => {
      $(id).textContent = "—";
    });
    $("provider-chart").replaceChildren();
    emptyChart("Pick two zones to price the whole day");
    $("nearby-list").innerHTML =
      '<li><span class="nearby__name" style="color: var(--ink-3); font-weight: 400">' +
      "Choose both ends to compare nearby pick-ups</span></li>";
    $("week-chart").replaceChildren();
    $("scenarios-list").innerHTML = '<p class="hint">Pick two zones to compare the options.</p>';
  }

  async function refresh() {
    renderLocal();
    const a = state.a ? placeById[state.a] : null;
    const b = state.b ? placeById[state.b] : null;
    const hour = Number(hourSel.value);
    const dow = Number($("pick-day").value);

    $("route-text").textContent = a && b ? `${a.zone} → ${b.zone}`
      : a ? `${a.zone} → pick a drop-off` : "No route chosen yet";

    if (!a || !b) {
      $("route-dist").textContent = "—";
      clearResults("pick two zones to see a figure");
      return;
    }

    const mine = ++token;
    $("fare-range").textContent = "pricing…";
    drawRoute(a, b);

    const q = `pu=${a.zone_id}&do=${b.zone_id}&shared=${state.shared}`;
    let est;
    let hourly;
    let weekly;
    let near;
    let scen;
    try {
      [est, hourly, weekly, near, scen] = await Promise.all([
        getJSON(`api/estimate?${q}&hour=${hour}&dow=${dow}`),
        getJSON(`api/hourly?${q}&dow=${dow}`),
        getJSON(`api/weekly?${q}&hour=${hour}`),
        getJSON(`api/nearby?${q}&hour=${hour}&dow=${dow}`),
        getJSON(`api/scenarios?${q}&hour=${hour}&dow=${dow}`),
      ]);
    } catch (err) {
      console.error(err);
      clearResults("the API did not answer");
      return;
    }
    if (mine !== token) return;    // a newer request has already landed

    /* route bar */
    $("route-dist").textContent =
      `${est.miles} mi (${est.miles_source}) · ${DAYS[dow]} ${String(hour).padStart(2, "0")}:00` +
      `${state.shared ? " · shared" : ""}`;

    /* 01 fare */
    $("fare-value").textContent = money(est.fare);
    $("fare-range").textContent = `likely between ${money(est.low)} and ${money(est.high)}`;
    $("kv-dist").textContent = `${est.miles} mi`;
    $("kv-ppm").textContent = money(est.per_mile);
    $("kv-wait").textContent = `${est.pickup.mean_wait.toFixed(1)} min`;
    $("kv-zonefare").textContent = money(est.pickup.median_fare);
    $("kv-zoneppm").textContent = money(est.pickup.median_ppm);

    barChart($("provider-chart"), [
      { name: "Uber", value: est.uber, on: est.cheaper === "uber" },
      { name: "Lyft", value: est.lyft, on: est.cheaper === "lyft" },
    ], { height: 200 });

    /* 02 timing */
    drawHours(hourly.hours, hour);
    const save = hourly.hours[hour].fare - hourly.cheapest_fare;
    $("chart-notes").innerHTML =
      `<span><i aria-hidden="true"></i> Cheapest at <b>${hourLabel(hourly.cheapest_hour)}</b>` +
      ` — <b>${money(hourly.cheapest_fare)}</b></span>` +
      `<span><i aria-hidden="true"></i> Dearest at <b>${hourLabel(hourly.dearest_hour)}</b>` +
      ` — <b>${money(hourly.dearest_fare)}</b></span>` +
      `<span><i data-plain="true" aria-hidden="true"></i> ` +
      (save > 0.5
        ? `Leaving at ${hourLabel(hourly.cheapest_hour)} instead saves <b>${money(save)}</b>`
        : `You have already picked about the cheapest hour`) +
      `</span>`;

    /* 03 nearby */
    const list = $("nearby-list");
    list.replaceChildren();
    const worth = near.options.filter((o) => o.saving > 0);
    if (!worth.length) {
      list.innerHTML =
        '<li><span class="nearby__name" style="color: var(--ink-3); font-weight: 400">' +
        `Nothing within ${near.radius_km} km prices cheaper than ${a.zone}</span></li>`;
    } else {
      worth.forEach((o) => {
        const li = document.createElement("li");
        li.innerHTML =
          `<span class="nearby__name">${o.zone}</span>` +
          `<span class="nearby__walk">${o.walk_m} m · ${money(o.fare)}</span>` +
          `<span class="nearby__save">save ${money(o.saving)}</span>`;
        list.appendChild(li);
      });
    }

    /* 04 week — zero-based would squash seven similar bars into one block, so
       the axis starts just below the cheapest day and the floor is stated */
    barChart($("week-chart"), weekly.days.map((d) => ({
      name: d.day, value: d.fare, on: d.dow === dow,
    })), { zeroBased: false, height: 230 });

    /* 05 scenarios */
    const sl = $("scenarios-list");
    sl.replaceChildren();
    const worstFare = Math.max(...scen.options.map((o) => o.fare));
    const bestFare = Math.min(...scen.options.map((o) => o.fare));
    scen.options.forEach((o) => {
      const row = document.createElement("div");
      row.className = "scenario";
      row.dataset.baseline = String(o.baseline);
      const width = (o.fare / worstFare) * 100;
      const delta = o.baseline ? "your pick"
        : `${o.delta < 0 ? "−" : "+"}${money(Math.abs(o.delta)).slice(1)}`;
      row.innerHTML =
        `<span class="scenario__label">${o.label}</span>` +
        `<span class="scenario__detail">${o.detail}</span>` +
        `<span class="scenario__bar"><span style="width:${width}%" ` +
        `data-best="${o.fare === bestFare}"></span></span>` +
        `<span class="scenario__fare">${money(o.fare)}</span>` +
        `<span class="scenario__delta" data-good="${o.delta < -0.5}">${delta}</span>`;
      sl.appendChild(row);
    });
  }

  refresh();

  /* ---------------- model lab ---------------- */

  const fareRows = meta.fare.scores.map((s) =>
    `<tr><td>${s.model}</td><td>$${s.mae.toFixed(2)}</td><td>${s.r2.toFixed(3)}</td></tr>`).join("");
  $("fare-table").innerHTML = fareRows;
  const demandRows = meta.demand.scores.map((s) =>
    `<tr><td>${s.model}</td><td>${s.r2.toFixed(3)}</td></tr>`).join("");
  $("demand-table").innerHTML = demandRows;
  $("meta-note").innerHTML =
    `Every row is scored on <b>${meta.fare.test_rows.toLocaleString()}</b> trips from ` +
    `<b>${meta.fare.test_month}</b>, which no model here was trained on. The plain ` +
    `baselines are not padding: they are the bar the model had to clear, and the jump ` +
    `from a straight line through distance to the full model is the part worth claiming.`;

  barChart($("importance"), meta.fare.importance.slice(0, 6).map((r, i) => ({
    name: r.feature.replace(/_/g, " "), value: r.drop, on: i === 0,
  })), { fmt: (v) => v.toFixed(3), height: 230 });

  const bestScore = meta.fare.scores.reduce((a, s) => (s.mae < a.mae ? s : a));
  const naive = meta.fare.scores.find((s) => s.model === "mean fare");
  const labTiles = [
    ["Error on unseen trips", `$${bestScore.mae.toFixed(2)}`,
      `median fare is about $19, so roughly ${Math.round((bestScore.mae / 19) * 100)}% out`],
    ["Variance explained", bestScore.r2.toFixed(3),
      `against ${naive ? naive.r2.toFixed(3) : "0.000"} for guessing the average`],
    ["Range coverage", `${(meta.fare.range.coverage * 100).toFixed(0)}%`,
      "share of real fares inside the 10–90 band, aiming at 80%"],
    ["Trips trained on", meta.fare.train_rows.toLocaleString(),
      `tested on ${meta.fare.test_rows.toLocaleString()} from ${meta.fare.test_month}`],
  ];
  $("lab-top").replaceChildren(...labTiles.map(([k, v, s]) => {
    const d = document.createElement("dl");
    d.className = "lab-tile";
    d.innerHTML = `<dt>${k}</dt><dd>${v}<small>${s}</small></dd>`;
    return d;
  }));

  const facts = [
    ["Algorithm", "Gradient boosted trees"],
    ["Target", meta.fare.target],
    ["Features given", String(meta.fare.features.length)],
    ["Training months", meta.fare.train_months.join(", ")],
    ["Test month", meta.fare.test_month],
    ["Measured zone pairs", meta.distance.zone_pairs.toLocaleString()],
    ["Detour factor", `× ${meta.distance.detour_factor}`],
  ];
  $("lab-facts").replaceChildren(...facts.map(([k, v]) => {
    const d = document.createElement("div");
    d.innerHTML = `<dt>${k}</dt><dd>${v}</dd>`;
    return d;
  }));

  /* ---------------- theme ---------------- */

  const toggle = $("theme-toggle");

  function applyTheme(next) {
    theme = next;
    if (next === "light") document.documentElement.setAttribute("data-theme", "light");
    else document.documentElement.removeAttribute("data-theme");
    try { localStorage.setItem(STORAGE_KEY, next); } catch (e) { /* not persisted */ }
    tiles.remove();
    tiles = L.tileLayer(TILES[next], { attribution: "Tiles &copy; Esri", maxZoom: 17 }).addTo(map);
    colours = ramp();
    layer.setStyle(styleFor);
    const c = cssVar("--bar-hi");
    routeLine.setStyle({ color: c });
    routeHalo.setStyle({ color: c });
    drawLegend();
    toggle.setAttribute("aria-label", next === "light" ? "Switch to dark theme" : "Switch to light theme");
  }

  toggle.setAttribute("aria-label", theme === "light" ? "Switch to dark theme" : "Switch to light theme");
  toggle.addEventListener("click", () => applyTheme(theme === "light" ? "dark" : "light"));
}

main();
