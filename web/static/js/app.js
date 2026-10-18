/* NYCRide welcome page.
   Every figure on the page is read from the exported data; nothing is typed in. */

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

const TILES = {
  dark: "https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
  light: "https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
};

const STORAGE_KEY = "nycride-theme";

const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

/* The ramp lives in CSS so both themes are defined in one place. */
const ramp = () => [1, 2, 3, 4, 5].map((i) => cssVar(`--ramp-${i}`));

const currentTheme = () =>
  document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";

const LAYERS = {
  trips: {
    title: "Where the rides begin",
    legend: "Trips started per hour",
    get: (s) => s.trips_per_hour,
    fmt: (v) => v.toLocaleString(undefined, { maximumFractionDigits: 0 }),
  },
  fare: {
    title: "What a ride costs from here",
    legend: "Median fare",
    get: (s) => s.median_fare,
    fmt: (v) => `$${v.toFixed(0)}`,
  },
  wait: {
    title: "How long the car takes",
    legend: "Typical wait, minutes",
    get: (s) => s.mean_wait,
    fmt: (v) => v.toFixed(1),
  },
};

let layerKey = "trips";

const fill = (key, value) =>
  document.querySelectorAll(`[data-fill="${key}"]`).forEach((el) => { el.textContent = value; });

const hourLabel = (h) => (h === 0 ? "midnight" : h === 12 ? "noon" : h < 12 ? `${h} am` : `${h - 12} pm`);

const compact = (n) =>
  n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}K` : String(n);

const monthRange = (months) => {
  const f = (m) => {
    const [y, mm] = m.split("-");
    return `${MONTHS[Number(mm) - 1]} ${y}`;
  };
  return months.length === 1 ? f(months[0]) : `${f(months[0])} – ${f(months[months.length - 1])}`;
};

/* Trip counts are wildly skewed - the airports sit far above everything else -
   so equal-width bins would paint almost the whole city one shade. */
function quantileBreaks(values, bands) {
  const sorted = [...values].sort((a, b) => a - b);
  const out = [];
  for (let i = 1; i < bands; i += 1) out.push(sorted[Math.floor((sorted.length * i) / bands)]);
  return out;
}

function bandFor(value, breaks) {
  let i = 0;
  while (i < breaks.length && value > breaks[i]) i += 1;
  return i;
}

async function main() {
  let summary;
  let stats;
  let zones;
  try {
    [summary, stats, zones] = await Promise.all([
      fetch("data/summary.json").then((r) => r.json()),
      fetch("data/zone_stats.json").then((r) => r.json()),
      fetch("data/nyc_zones.geojson").then((r) => r.json()),
    ]);
  } catch (err) {
    const box = document.querySelector(".map__loading");
    if (box) box.textContent = "The trip data could not be loaded";
    console.error(err);
    return;
  }

  const entries = Object.entries(stats);
  const swing = ((summary.dearest_ppm / summary.cheapest_ppm) - 1) * 100;

  fill("trips_long", summary.total_trips.toLocaleString());
  fill("trips_short", `${compact(summary.total_trips)}+`);
  fill("zones", summary.zones);
  fill("month_list", monthRange(summary.months));
  fill("uber_share", `${summary.provider_share.Uber}%`);
  fill("lyft_share", `${summary.provider_share.Lyft}%`);
  fill("swing", `+${swing.toFixed(0)}%`);
  fill("cheap_hour", hourLabel(summary.cheapest_hour));
  fill("dear_hour", hourLabel(summary.dearest_hour));
  fill("busy_hour", hourLabel(summary.busiest_hour));
  fill("cheap_ppm", `$${summary.cheapest_ppm.toFixed(2)}`);
  fill("dear_ppm", `$${summary.dearest_ppm.toFixed(2)}`);
  fill("busy_trips", compact(summary.trips_by_hour[summary.busiest_hour]));

  fillModelScores();
  buildChart(summary);
  buildMap(summary, stats, zones, entries);
}


/* The two headline scores used to be typed into the page and went stale the
   first time the model was retrained. Read them from the API instead, and leave
   whatever the HTML already says if it cannot be reached. */
async function fillModelScores() {
  try {
    const meta = await fetch("api/meta").then((r) => r.json());
    const best = meta.fare.scores.reduce((a, s) => (s.mae < a.mae ? s : a));
    fill("model_mae", `$${best.mae.toFixed(2)}`);
    fill("model_r2", best.r2.toFixed(3));
  } catch (err) {
    console.warn("model scores unavailable", err);
  }
}

/* ---------------- hourly chart ---------------- */

function buildChart(summary) {
  const bars = document.getElementById("bars");
  const axis = document.getElementById("bar-axis");
  const tipText = document.getElementById("tip-text");
  const max = Math.max(...summary.trips_by_hour);
  const resting = "Hover any hour to read it";

  summary.trips_by_hour.forEach((n, h) => {
    const bar = document.createElement("button");
    bar.type = "button";
    bar.className = "bar";
    bar.style.height = `${(n / max) * 100}%`;
    if (h === summary.cheapest_hour || h === summary.dearest_hour) bar.dataset.mark = "true";
    bar.setAttribute("aria-label",
      `${hourLabel(h)}: ${n.toLocaleString()} trips, $${summary.ppm_by_hour[h].toFixed(2)} per mile`);

    const show = () => {
      tipText.innerHTML =
        `<b>${hourLabel(h)}</b><i>·</i> ${compact(n)} rides` +
        `<i>·</i> <b>$${summary.ppm_by_hour[h].toFixed(2)}</b>/mile` +
        `<i>·</i> avg fare <b>$${summary.fare_by_hour[h].toFixed(2)}</b>` +
        `<i>·</i> wait <b>${summary.wait_by_hour[h].toFixed(1)} min</b>`;
      bar.dataset.active = "true";
    };
    const hide = () => {
      tipText.textContent = resting;
      delete bar.dataset.active;
    };

    bar.addEventListener("mouseenter", show);
    bar.addEventListener("mouseleave", hide);
    bar.addEventListener("focus", show);
    bar.addEventListener("blur", hide);
    bars.appendChild(bar);

    const tick = document.createElement("span");
    tick.textContent = h % 6 === 0 ? String(h).padStart(2, "0") : "";
    axis.appendChild(tick);
  });
}

/* ---------------- map ---------------- */

function buildMap(summary, stats, zones, entries) {
  document.querySelector(".map__loading")?.remove();

  let theme = currentTheme();
  let colours = ramp();

  const map = L.map("map-canvas", { zoomControl: true, scrollWheelZoom: false });
  let tiles = L.tileLayer(TILES[theme], { attribution: "Tiles &copy; Esri", maxZoom: 16 }).addTo(map);

  let breaks = [];
  const recomputeBreaks = () => {
    breaks = quantileBreaks(entries.map(([, s]) => LAYERS[layerKey].get(s)), colours.length);
  };

  const styleFor = (feature) => {
    const s = stats[feature.properties.zone_id];
    const stroke = cssVar("--map-stroke");
    if (!s) return { fillColor: "transparent", fillOpacity: 0.05, color: stroke, weight: 0.6 };
    return {
      fillColor: colours[bandFor(LAYERS[layerKey].get(s), breaks)],
      fillOpacity: 0.88,
      color: stroke,
      weight: 0.6,
    };
  };

  recomputeBreaks();

  /* one shared readout instead of 261 tooltips: the numbers stay where the eye
     can return to them, and it still says something when nothing is hovered */
  const el = {
    name: document.getElementById("readout-name"),
    where: document.getElementById("readout-where"),
    trips: document.getElementById("readout-trips"),
    fare: document.getElementById("readout-fare"),
    wait: document.getElementById("readout-wait"),
  };

  const city = (() => {
    const total = entries.reduce((a, [, s]) => a + s.trips, 0);
    const hours = summary.months.length * 30 * 24;
    const mean = (f) => entries.reduce((a, [, s]) => a + f(s) * s.trips, 0) / total;
    return {
      name: "all zones",
      where: "· city-wide average",
      trips: Math.round(total / hours).toLocaleString(),
      fare: `$${mean((s) => s.median_fare).toFixed(2)}`,
      wait: `${mean((s) => s.mean_wait).toFixed(1)} min`,
    };
  })();

  const paint = (v) => {
    el.name.textContent = v.name;
    el.where.textContent = v.where;
    el.trips.textContent = v.trips;
    el.fare.textContent = v.fare;
    el.wait.textContent = v.wait;
  };

  const showCity = () => paint(city);
  showCity();

  const layer = L.geoJSON(zones, {
    style: styleFor,
    onEachFeature: (feature, lyr) => {
      const s = stats[feature.properties.zone_id];
      if (!s) return;
      /* No bringToFront on hover: moving the node stops mouseout from firing. */
      lyr.on({
        mouseover: (e) => {
          e.target.setStyle({ weight: 2, color: cssVar("--map-hover"), fillOpacity: 1 });
          paint({
            name: s.zone,
            where: `· ${s.borough}`,
            trips: s.trips_per_hour.toLocaleString(),
            fare: `$${s.median_fare.toFixed(2)}`,
            wait: `${s.mean_wait.toFixed(1)} min`,
          });
        },
        mouseout: (e) => {
          layer.resetStyle(e.target);
          showCity();
        },
      });
    },
  }).addTo(map);

  /* Frame the five boroughs. Newark Airport is a TLC zone too, but it sits far
     into New Jersey and dragged the whole city off to one side. */
  const cityBounds = L.latLngBounds([]);
  layer.eachLayer((lyr) => {
    const p = lyr.feature.properties;
    if (p.borough !== "EWR" && stats[p.zone_id]) cityBounds.extend(lyr.getBounds());
  });
  map.fitBounds(cityBounds.isValid() ? cityBounds : layer.getBounds(), { padding: [8, 8] });

  /* wheel zoom only after a click, so the page keeps scrolling past the map */
  map.on("click", () => map.scrollWheelZoom.enable());
  map.on("mouseout", () => map.scrollWheelZoom.disable());

  const legendScale = document.getElementById("legend-scale");

  function drawLegend() {
    const { fmt } = LAYERS[layerKey];
    const lo = Math.min(...entries.map(([, s]) => LAYERS[layerKey].get(s)));
    legendScale.replaceChildren();
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
      legendScale.appendChild(step);
    });
    document.getElementById("legend-title").textContent = LAYERS[layerKey].legend;
  }

  drawLegend();

  document.querySelectorAll(".chips button").forEach((btn) => {
    btn.addEventListener("click", () => {
      layerKey = btn.dataset.layer;
      document.querySelectorAll(".chips button").forEach((b) => {
        b.setAttribute("aria-pressed", String(b === btn));
      });
      recomputeBreaks();
      layer.setStyle(styleFor);
      drawLegend();
      fill("figure_title", LAYERS[layerKey].title);
    });
  });

  /* ---- theme ---- */

  const toggle = document.getElementById("theme-toggle");

  function applyTheme(next) {
    theme = next;
    if (next === "light") document.documentElement.setAttribute("data-theme", "light");
    else document.documentElement.removeAttribute("data-theme");
    try { localStorage.setItem(STORAGE_KEY, next); } catch (e) { /* not persisted */ }

    tiles.remove();
    tiles = L.tileLayer(TILES[next], { attribution: "Tiles &copy; Esri", maxZoom: 16 }).addTo(map);
    colours = ramp();
    layer.setStyle(styleFor);
    drawLegend();
    toggle.setAttribute("aria-label", next === "light" ? "Switch to dark theme" : "Switch to light theme");
  }

  toggle.setAttribute("aria-label", theme === "light" ? "Switch to dark theme" : "Switch to light theme");
  toggle.addEventListener("click", () => applyTheme(theme === "light" ? "dark" : "light"));
}

main();
