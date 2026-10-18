# NYCRide — the website

Two pages, both served by the FastAPI app in `app/`:

| Page | What it does |
|---|---|
| `index.html` | the welcome page: city-wide figures, the zone map, hourly price bars |
| `predictor.html` | pick two zones and a time, then seven sections of model output |

## Run it

From the project root:

```bash
.venv/Scripts/python.exe -m uvicorn app.main:app --port 5173
```

Then open <http://localhost:5173>. Opening the HTML straight from disk will **not**
work: both pages `fetch` their data, which browsers block on `file://`.

## Files

| Path | What it is |
|---|---|
| `index.html`, `predictor.html` | the two pages |
| `static/css/styles.css` | shared theme tokens and the welcome page |
| `static/css/predictor.css` | the predictor page |
| `static/js/app.js` | welcome page: map, hourly bars, headline figures |
| `static/js/predictor.js` | predictor page: zone picker, route map, every chart |
| `data/*.json`, `data/*.geojson` | written by `pipeline/export_web_data.py` |

Regenerate the data after re-running the pipeline:

```bash
cd pipeline && python export_web_data.py
```

## Nothing is typed into the pages

Every figure comes from the API or from `data/summary.json` at load time — trip counts,
the cheapest and dearest hour, and the model's own MAE and R². That is deliberate: the
two model scores on the welcome page used to be written into the HTML and went stale the
first time the model was retrained. They now read from `/api/meta`.

The `?v=N` on the stylesheet and script tags is a cache buster. Bump it when you change
either file, or browsers keep serving the old one.

## Themes

Dark is the default; the toggle in the header switches to light and the choice is kept in
`localStorage`. A small inline script in `<head>` applies it before first paint, so a
saved choice never flashes the other theme.

Light is a **selected** set of steps, not an inversion. Three things had to be re-chosen
rather than flipped:

- **The map ramp runs the other way.** On the dark page the quiet zones are dark and
  recede into the basemap; on the light page the pale step has to play that role.
- **The bar blues are darker in light mode.** `#6ea8fe` reads well on near-black and
  washes out on white, so light mode uses `#1f5fc0`.
- **The box hover lifts differently.** The card is already white in light mode, so the
  hover is a shadow; in dark mode it is a lighter fill.

`.btn--solid` paints its label in `--page`, so its hover shade has to move with `--ink`.
It was once hard-coded to a light grey, which meant that in light mode the dark button
turned pale on hover and the label vanished into it. It is now `--ink-hover`, defined per
theme (contrast measured at 10.9:1 light, 12.3:1 dark).

## The map

Leaflet 1.9.4 from cdnjs, with Esri's **World Dark Gray** / **World Light Gray** basemaps —
free and keyless, swapped with the theme. CartoDB's dark tiles were tried first but now
stamp "API KEY REQUIRED" across every tile.

Three layers share one set of polygons: trips per hour, median fare, typical wait. Breaks
are **quantiles, not equal width** — the airports have so many more pickups than anywhere
else that equal-width bins paint the whole city one shade. The legend prints the real
break values.

Hovering a zone updates **one shared readout** under the map rather than opening a
tooltip: no 263 floating boxes, the numbers stay in a fixed place, and at rest it shows
the city-wide figures.

Two things to know before touching the map code:

- **No `bringToFront()` on hover.** Moving the SVG node mid-hover stops `mouseout` from
  firing, which left highlights stuck across the map.
- **Newark Airport is excluded from the initial fit.** It is a real TLC zone but sits far
  into New Jersey and dragged the five boroughs off to one side of the frame.

The predictor page draws the actual road route between the two zones using the public
OSRM demo server. If that call fails the page falls back to a straight line, so a routing
outage degrades the map instead of breaking the page.

## Credits

Trip data: NYC Taxi & Limousine Commission High Volume FHV records. Basemap tiles: Esri.
Routing: OSRM demo server. Weather: Open-Meteo.
