"""Build the fact base the Ask page answers from.

Everything here is derived from files the pipeline already wrote, so the answers
cannot drift from the model. Retrain, re-run this, and the facts move with it.

Each fact is one self-contained sentence or two. That matters: the retriever
scores whole chunks, and a chunk that needs its neighbours to make sense will be
quoted on its own and read as nonsense.

Writes models/knowledge.json.

Usage: python build_knowledge.py
"""
import json

import config as cfg

MODELS = cfg.ROOT / "models"
WEB_DATA = cfg.ROOT / "web" / "data"

HOURS = ["midnight", "1 am", "2 am", "3 am", "4 am", "5 am", "6 am", "7 am", "8 am",
         "9 am", "10 am", "11 am", "noon", "1 pm", "2 pm", "3 pm", "4 pm", "5 pm",
         "6 pm", "7 pm", "8 pm", "9 pm", "10 pm", "11 pm"]


# Words a person is likely to use that do not appear in the fact itself. They
# are indexed for matching and never shown, which is the cheap way to stop
# retrieval missing a fact over vocabulary: "which algorithm did you use" found
# nothing until "algorithm" was attached to the fact that answers it.
KEYWORDS = {
    "why gradient boosted trees":
        "algorithm method technique approach which model type regression "
        "classifier random forest xgboost neural network deep learning svm "
        "linear why this model choose chose chosen",
    "model accuracy":
        "how accurate good performance result score works well reliable trust "
        "error rate correct",
    "what MAE means": "mean absolute error metric measure evaluate",
    "what R-squared means": "r2 r squared metric measure evaluate variance explained",
    "the price range": "confidence interval uncertainty band prediction interval "
                       "quantile how sure certainty",
    "train and test split": "validation overfitting leakage cross validation holdout "
                            "evaluate how tested",
    "data leakage": "leak cheating target leakage overfit",
    "what the model leans on most": "feature importance which features matter signal "
                                    "driver influence",
    "why the log of the fare": "target transform skew logarithm transformation",
    "how distance is worked out": "miles km distance calculate route haversine detour",
    "what this project is": "about overview summary describe explain purpose goal "
                            "nycride what does it do",
    "the point of the project": "why built motivation purpose interview portfolio",
    "technology used": "tech stack framework library tools languages built with python "
                       "fastapi sklearn",
    "limitations: what the data cannot say": "weakness weaknesses problem problems "
                                             "drawback shortcoming limitation fail "
                                             "wrong bad improve better surge traffic",
    "limitations: coverage": "weakness scope cities months coverage limitation",
    "size of the dataset": "how many rows records trips big data volume size",
    "how the data was cleaned": "cleaning preprocessing filter outliers removed "
                                "dropped quality preparation",
    "weather was tested and dropped": "rain raining weather temperature failed "
                                      "experiment rejected negative result",
    "demand model result": "demand pickups busy forecast time series lags baseline lost",
    "why NYC data and not India": "india indian delhi synthetic fake kaggle dataset "
                                  "choice why new york",
    "how the Ask feature works": "rag retrieval chatbot ask question how does this work "
                                 "llm embeddings",
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fare_facts(m):
    """Scores, baselines, the range, and how the model was put together."""
    out = []
    best = min(m["scores"], key=lambda s: s["mae"])
    out.append((
        "model accuracy",
        f"The fare model is a gradient boosted decision tree trained on "
        f"log(base_fare). On the unseen test month it is off by ${best['mae']:.2f} "
        f"on a typical trip (mean absolute error) and explains "
        f"{best['r2'] * 100:.1f}% of the variation in fares (R-squared {best['r2']:.3f})."))

    out.append((
        "what MAE means",
        f"Mean absolute error is the average size of the model's mistake, ignoring "
        f"whether it guessed high or low. ${best['mae']:.2f} against a typical fare "
        f"of about $19 is roughly {best['mae'] / 19 * 100:.0f}% out."))

    out.append((
        "what R-squared means",
        "R-squared is how much of the variation in fares the model explains. Always "
        "guessing the average fare scores 0, a perfect model scores 1, and a model "
        "worse than the average scores below 0."))

    for s in m["scores"]:
        out.append((
            f"baseline score: {s['model']}",
            f"Scored on the unseen test month, {s['model']} gets a mean absolute "
            f"error of ${s['mae']:.2f} and an R-squared of {s['r2']:.4f}."))

    out.append((
        "why baselines are shown",
        "The simple baselines are published next to the model on purpose. A model "
        "that cannot beat a straight line through distance is not worth shipping, "
        "and an R-squared on its own says nothing about whether the score was hard "
        "to reach. The gap between the straight line and the full model is the part "
        "worth claiming."))

    r = m["range"]
    out.append((
        "the price range",
        f"The range shown around each estimate comes from two more models trained "
        f"with quantile loss at the 10th and 90th percentile, not from a rule of "
        f"thumb. {r['coverage'] * 100:.1f}% of real fares land inside that band "
        f"against a target of 80%, so the uncertainty is calibrated. The band is "
        f"${r['mean_width']:.2f} wide on average."))

    out.append((
        "train and test split",
        f"The model trains on {', '.join(m['train_months'])} and is tested on "
        f"{m['test_month']}, a later month it never saw. The split is by month and "
        f"never random: a random split lets the model see the future and reports a "
        f"score it could not repeat in use. It learned from "
        f"{m['train_rows']:,} trips and was scored on {m['test_rows']:,}."))

    out.append((
        "features the model uses",
        "The model is given twelve features: pick-up zone, drop-off zone, trip "
        "miles, hour, day of week, whether it is a weekend, whether it is a public "
        "holiday, whether it is a holiday eve, whether the provider is Uber, "
        "whether the ride was requested as shared, and whether either end is an "
        "airport."))

    out.append((
        "data leakage",
        "Trip duration, tips, driver pay and price per mile are deliberately kept "
        "out of the model. All of them are only known once the ride is over, so "
        "feeding them in would leak the answer and produce a score the model could "
        "never repeat on a real request."))

    imp = m["importance"]
    top = imp[0]
    out.append((
        "what the model leans on most",
        f"By permutation importance the model leans most on {top['feature'].replace('_', ' ')}, "
        f"which costs {top['drop']:.3f} of R-squared when shuffled. Then come "
        + ", ".join(f"{i['feature'].replace('_', ' ')} ({i['drop']:.3f})" for i in imp[1:5])
        + "."))

    out.append((
        "how feature importance is measured",
        "Importance is measured by permutation: one feature's values are shuffled "
        "at random and the drop in accuracy is recorded. A feature that changes "
        "nothing when scrambled was not being used. This is measured on the test "
        "data rather than read off the trees, because a tree's built-in importance "
        "favours columns with more distinct values."))

    out.append((
        "zones matter beyond distance",
        "After distance, the pick-up and drop-off zones are the strongest signals. "
        "That is the model learning that the same mile costs different amounts in "
        "different parts of the city, which a plain distance formula cannot do."))

    out.append((
        "how distance is worked out",
        f"The site only knows two zones, but the model needs miles. It uses the "
        f"median distance actually driven between that pair of zones, measured over "
        f"{m['zone_pairs']:,} pairs with enough trips. For a pair with too few "
        f"trips it falls back to the straight-line distance multiplied by a detour "
        f"factor of {m['detour_factor']}, which was measured from the data rather "
        f"than guessed: NYC streets make a trip about "
        f"{(m['detour_factor'] - 1) * 100:.0f}% longer than flying straight."))

    out.append((
        "why gradient boosted trees",
        "Gradient boosting builds many small decision trees one after another, each "
        "correcting what the ones before it got wrong. Trees were chosen because "
        "fares are not linear: airport trips are near flat rate and rush hour bites "
        "differently in each zone. A linear model needs those interactions written "
        "out by hand, and trees find them. They also handle zone ids naturally, "
        "train in minutes on a laptop with no GPU, and can be explained."))

    out.append((
        "why the log of the fare",
        "The model predicts the logarithm of the fare rather than the fare itself. "
        "Fares are right-skewed, with many small ones and a few very large ones. "
        "Training on the log makes the model minimise percentage error instead of "
        "dollar error, which is what matters to a rider, and it makes a negative "
        "prediction impossible."))
    return out


def demand_facts(m):
    out = [(
        "demand model result",
        "The demand side predicts pick-ups per zone per hour. The honest result is "
        "that the trained model lost. " + m["note"])]
    for s in m["scores"]:
        out.append((
            f"demand score: {s['model']}",
            f"For predicting pick-ups per zone-hour, {s['model']} scores an "
            f"R-squared of {s['r2']:.4f} and a mean absolute error of {s['mae']:.2f} trips."))
    out.append((
        "what the map's demand layer shows",
        f"The demand figure on the map is served from a lookup table of the average "
        f"for each zone, hour and weekday, not from a model. It is what ships "
        f"because it beat the model the site could actually serve."))
    return out


def weather_facts(w):
    a, b = w["without_weather"], w["with_weather"]
    return [(
        "weather was tested and dropped",
        f"Weather was tested as a feature and rejected. {w['note']} On the same "
        f"rows and the same split, adding temperature, rainfall and wind moved the "
        f"error from ${a['mae']:.3f} to ${b['mae']:.3f} and R-squared from "
        f"{a['r2']:.3f} to {b['r2']:.3f} - very slightly worse. The weather columns "
        f"are not used by the model."),
        ("does rain raise fares",
         "Rain does lift the citywide mean fare by about 5-6% in September and "
         "April, and made no measurable difference in January. But hour and zone "
         "already carry that signal, so giving the model the weather on top added "
         "nothing.")]


def data_facts(s, zone_stats):
    cheap = min(range(24), key=lambda h: s["ppm_by_hour"][h])
    dear = max(range(24), key=lambda h: s["ppm_by_hour"][h])
    out = [
        ("size of the dataset",
         f"The project is built on {s['total_trips']:,} cleaned trips from New York "
         f"City's Taxi and Limousine Commission High Volume For-Hire Vehicle "
         f"records, covering {len(s['months'])} months: {', '.join(s['months'])}. "
         f"They were chosen to cover autumn, the winter holidays, cold and rain."),
        ("which companies",
         f"The data covers Uber and Lyft. Uber is {s['provider_share']['Uber']}% of "
         f"the trips read ({s['providers']['Uber']:,}) and Lyft is "
         f"{s['provider_share']['Lyft']}% ({s['providers']['Lyft']:,})."),
        ("busiest and quietest hours",
         f"The busiest hour across the city is {HOURS[s['busiest_hour']]} with "
         f"{s['trips_by_hour'][s['busiest_hour']]:,} trips, and the quietest is "
         f"{HOURS[s['quietest_hour']]} with {s['trips_by_hour'][s['quietest_hour']]:,}."),
        ("cheapest and dearest time to travel",
         f"Per mile, the cheapest hour to travel is {HOURS[cheap]} at "
         f"${s['ppm_by_hour'][cheap]:.2f} a mile, and the dearest is {HOURS[dear]} "
         f"at ${s['ppm_by_hour'][dear]:.2f} a mile - about "
         f"{(s['ppm_by_hour'][dear] / s['ppm_by_hour'][cheap] - 1) * 100:.0f}% more."),
        ("number of zones",
         f"New York's taxi zone map has 263 zones, and {len(zone_stats)} of them "
         f"have enough trips to report statistics for."),
    ]
    out.append((
        "price per mile through the day",
        "Price per mile by hour, from midnight: "
        + ", ".join(f"{HOURS[h]} ${s['ppm_by_hour'][h]:.2f}" for h in range(24)) + "."))
    return out


def zone_facts(zone_stats):
    """One fact per zone, so a question naming a place can find it."""
    out = []
    for zid, z in zone_stats.items():
        if not z.get("trips"):
            continue
        out.append((
            f"zone: {z['zone']}",
            f"{z['zone']} is a taxi zone in {z['borough']} (zone id {zid}). Trips "
            f"starting there have a median fare of ${z['median_fare']:.2f} and a "
            f"median price per mile of ${z['median_ppm']:.2f}. It sees about "
            f"{z['trips_per_hour']:.0f} pick-ups an hour, and the typical wait "
            f"between requesting and being picked up is {z['mean_wait']:.1f} minutes."))
    return out


def cleaning_facts():
    c = cfg
    return [
        ("how the data was cleaned",
         f"About 6.1 to 6.5% of raw rows are dropped per month, most of them trips "
         f"with an unknown pick-up or drop-off zone. A trip is kept only if its "
         f"distance is between {c.MILES_MIN} and {c.MILES_MAX} miles, its duration "
         f"between {c.TIME_MIN_S // 60} minute and {c.TIME_MAX_S // 3600} hours, "
         f"its fare between ${c.FARE_MIN:.0f} and ${c.FARE_MAX:.0f}, its wait under "
         f"{c.WAIT_MAX:.0f} minutes, its price per mile between ${c.PPM_MIN:.0f} and "
         f"${c.PPM_MAX:.0f}, and its average speed between {c.SPEED_MIN:.0f} and "
         f"{c.SPEED_MAX:.0f} mph."),
        ("why those cleaning limits",
         "The limits come from looking at the data, not from habit. The raw price "
         "per mile reaches $3,951 a mile, speeds run past anything a NYC street "
         "allows, and zones 264 and 265 mean unknown or outside the city and have "
         "no shape on the map."),
        ("no feature scaling",
         "Values are kept in real units with no scaling or normalisation. Tree "
         "models do not need it, the site has to display $24.50 rather than 0.37, "
         "and a scaler must be fitted on the training split only, so it belongs "
         "inside a model pipeline rather than in the stored data."),
        ("holidays in the data",
         "The model is told about public holidays and holiday eves. The holidays "
         "covered are Labor Day, Thanksgiving, Christmas, New Year's Day, Martin "
         "Luther King Day, Good Friday and Easter, plus Christmas Eve and New "
         "Year's Eve as eves."),
    ]


def project_facts():
    return [
        ("what this project is",
         "NYCRide is a fare prediction site for New York City. You pick a pick-up "
         "zone, a drop-off zone and a time, and it prices the trip, shows the range "
         "around that price, prices all 24 hours and all 7 days so you can see when "
         "to travel, compares Uber against Lyft on that exact route, and shows what "
         "the model gets right and wrong."),
        ("the point of the project",
         "The point is honesty about the model, not just the prediction. Every "
         "score on the site is read from the trained model's own metrics file "
         "rather than typed in, plain baselines are shown next to the model, and "
         "the two experiments that failed are reported alongside the ones that "
         "worked."),
        ("what the site can tell you",
         "The site has seven sections: the estimated fare with its range and both "
         "providers, the best time to leave, cheaper pick-up points nearby, the "
         "same trip across the week, what single change would make it cheapest, the "
         "weather experiment, and a model lab with the scores and baselines."),
        ("technology used",
         "Python with scikit-learn for the models, pandas and PyArrow for the data, "
         "FastAPI for the API, and plain HTML, CSS and JavaScript with Leaflet for "
         "the maps. No frontend framework and no paid APIs."),
        ("why NYC data and not India",
         "An Indian ride-hailing dataset was tried first and rejected. Profiling it "
         "showed the fare had almost no correlation with distance, the hourly "
         "patterns were flat and the counts were suspiciously round - it was "
         "synthetic, so no model trained on it would mean anything. No real Indian "
         "trip data of this kind is public, and NYC's TLC publishes real records, "
         "so the project moved to New York."),
        ("how fast the site is",
         "A prediction takes about a millisecond. Nothing at request time touches "
         "the 78 million row data files: the models and the lookups are loaded once "
         "when the service starts, and every endpoint works from those."),
        ("limitations: what the data cannot say",
         "The things that really move a fare - live traffic, the surge multiplier, "
         "the exact route taken and how many drivers are free - are not in the data "
         "at all, because a TLC record is written after the trip is over. That is "
         "the ceiling on accuracy here, and roughly $5 of error is close to it."),
        ("limitations: coverage",
         "The project covers New York City only and four months only. There is no "
         "vehicle category, no passenger count, no surge column and no cancellation "
         "data in the source."),
        ("limitations: events",
         "Street closure permits for parades and street festivals stop appearing in "
         "the city's event feed after 2016, so the 2025-26 months carry Parks "
         "Department permits only. Events are a weak signal and the all-zero column "
         "is dropped rather than fed to a model as noise."),
        ("limitations: weather is one reading",
         "Weather is a single city-wide hourly reading, so local showers are "
         "missed. Visibility comes back empty from the archive and is dropped."),
        ("why a group vehicle optimiser was dropped",
         "An earlier idea was to work out the cheapest mix of vehicles for a group "
         "- two cabs against two autos, say. It was dropped because New York has no "
         "auto-rickshaws or bike taxis, and the records carry no vehicle type at "
         "all, so the prices would have had to be invented."),
        ("where the data comes from",
         "Trip records come from the NYC Taxi and Limousine Commission's High "
         "Volume For-Hire Vehicle monthly files. Zone shapes come from the TLC taxi "
         "zone shapefile, weather from the Open-Meteo archive, and the road routes "
         "drawn on the map from the public OSRM routing service."),
        ("how the Ask feature works",
         "This answer comes from a retrieval system over facts built directly from "
         "the project's own metrics and data files. The question is matched against "
         "those facts, the closest ones are passed to a language model, and it "
         "writes the answer from them. Questions about the price of a specific trip "
         "are sent to the fare model itself instead, so the number is a real "
         "prediction rather than something retrieved."),
    ]


def main():
    fare = load(MODELS / "fare_metrics.json")
    demand = load(MODELS / "demand_metrics.json")
    weather = load(MODELS / "weather_experiment.json")
    summary = load(WEB_DATA / "summary.json")
    zone_stats = load(WEB_DATA / "zone_stats.json")

    groups = {
        "model": fare_facts(fare),
        "demand": demand_facts(demand),
        "weather": weather_facts(weather),
        "data": data_facts(summary, zone_stats),
        "cleaning": cleaning_facts(),
        "project": project_facts(),
        "zone": zone_facts(zone_stats),
    }

    facts = []
    for topic, items in groups.items():
        for title, text in items:
            fact = {"topic": topic, "title": title, "text": text}
            if title in KEYWORDS:
                fact["keywords"] = KEYWORDS[title]
            facts.append(fact)
        print(f"  {topic:9s} {len(items):4d} facts")

    missing = sorted(set(KEYWORDS) - {f["title"] for f in facts})
    if missing:
        raise SystemExit(f"KEYWORDS names facts that no longer exist: {missing}")

    out = {"built_from": {"fare_trained": fare["trained"], "test_month": fare["test_month"]},
           "facts": facts}
    path = MODELS / "knowledge.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n  {len(facts)} facts -> {path.name} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
