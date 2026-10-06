"""
Capstone builder: combines the outputs of Tasks 1-5 into one interactive report/dashboard.
Run after the earlier scripts have produced their CSVs, then open netflix_dashboard.html.
Change O to your project's output folder if needed.
"""
import json
from pathlib import Path

import pandas as pd

O = Path(".")
d = pd.read_csv("data/processed/netflix_cleaned.csv")
rel = pd.read_csv("trend_output/release_year_series.csv", index_col=0)
fc = pd.read_csv("trend_output/forecasts.csv")
bt = pd.read_csv("trend_output/backtest_results.csv")
clf = pd.read_csv("classification_output/audience_model_comparison.csv")
rep = pd.read_csv("classification_output/audience_classification_report.csv", index_col=0)
rec_eval = pd.read_csv("recommender_output/evaluation_results.csv")
recs = pd.read_csv("recommender_output/sample_recommendations.csv")

# ---- data for the interactive charts (Task 2 style analysis) ----
years = [{"y": f"{y}*" if y == 2021 else str(y), "M": int(rel.loc[y, "Movie"]), "V": int(rel.loc[y, "TV Show"])}
         for y in range(2008, 2022)]
g = d.assign(g=d.genres.str.split("|")).explode("g")
top = lambda s: [[k, int(v)] for k, v in s.g.value_counts().head(8).items()]
genres = {"All": top(g), "Movie": top(g[g.type == "Movie"]), "TV Show": top(g[g.type == "TV Show"])}


def aud(s):
    c = s.audience_group.value_counts()
    return {k: int(c.get(k, 0)) for k in ["Kids", "Teens", "Adults"]}


auds = {"All": aud(d), "Movie": aud(d[d.type == "Movie"]), "TV Show": aud(d[d.type == "TV Show"])}

fcd = {}
for s in ["Total", "Movie", "TV Show"]:
    sel = fc[(fc.series == s) & fc.selected].sort_values("year")
    fcd[s] = {"model": sel.model.iloc[0], "hist": [int(rel.loc[y, s]) for y in range(2008, 2021)],
              "pred": sel.forecast.round().astype(int).tolist(), "lo": sel.lower.round().astype(int).tolist(),
              "hi": sel.upper.round().astype(int).tolist(), "part": int(rel.loc[2021, s])}
bu = fc[fc.series == "Total (bottom-up)"].sort_values("year").forecast.round().astype(int).tolist()
fcd["Total"]["bu"] = bu

clf_rows = [[("Majority baseline" if m.startswith("Baseline") else m), round(a, 3), round(f, 3)]
            for m, a, f in zip(clf.model, clf.accuracy, clf.macro_f1)]
rec_rows = [[m.replace(" (full model)*", " (full)"), round(h, 3)] for m, h in zip(rec_eval.model, rec_eval.hit_rate_at_10)]
rec_dict = {q: [[r.title, r.type, int(r.release_year), float(r.similarity)] for r in grp.itertuples()]
            for q, grp in recs.sort_values(["query", "rank"], kind="stable").groupby("query", sort=False)}

data = {"years": years, "genres": genres, "aud": auds, "fy": list(range(2008, 2026)), "fc": fcd,
        "clf": clf_rows, "rec": rec_rows, "recs": rec_dict}

# ---- facts quoted in the report text (computed, never typed by hand) ----
best = clf[~clf.model.str.startswith("Baseline")].sort_values("macro_f1").iloc[-1]
base_acc = clf[clf.model.str.startswith("Baseline")].accuracy.iloc[0]
mv = rel.loc[2008:2020, "Movie"]
pk = int(mv.idxmax())
ad = d.groupby("year_added").size().drop(2021)
rated = d[d.audience_group != "Unrated"]
re_ = rec_eval.set_index("model").hit_rate_at_10
gi = g.g.value_counts()
sel = lambda s, y: int(fc[(fc.series == s) & fc.selected & (fc.year == y)].forecast.iloc[0])
selr = lambda s, y, c: int(fc[(fc.series == s) & fc.selected & (fc.year == y)][c].iloc[0])
F = {
    "n": f"{len(d):,}", "mp": f"{(d.type == 'Movie').mean() * 100:.0f}", "tp": f"{(d.type == 'TV Show').mean() * 100:.0f}",
    "tv08": f"{rel.loc[2008, 'tv_share_pct']:.0f}", "tv20": f"{rel.loc[2020, 'tv_share_pct']:.0f}",
    "mpk": str(pk), "mdrop": f"{abs((mv.loc[2020] / mv.loc[pk] - 1) * 100):.0f}",
    "apk": str(int(ad.idxmax())), "apkn": f"{int(ad.max()):,}",
    "kidm": f"{(rated[rated.type == 'Movie'].audience_group == 'Kids').mean() * 100:.0f}",
    "kidt": f"{(rated[rated.type == 'TV Show'].audience_group == 'Kids').mean() * 100:.0f}",
    "gim": f"{gi['International Movies']:,}", "git": f"{gi['International TV Shows']:,}",
    "tv25": f"{sel('TV Show', 2025):,}", "tvlo": f"{selr('TV Show', 2025, 'lower'):,}", "tvhi": f"{selr('TV Show', 2025, 'upper'):,}",
    "mv25": f"{sel('Movie', 2025):,}", "tot25": f"{sel('Total', 2025):,}", "bu25": f"{bu[-1]:,}",
    "tvmape": f"{bt[bt.series == 'TV Show'].MAPE_pct.min():.0f}",
    "model": best.model, "acc": f"{best.accuracy * 100:.0f}", "f1": f"{best.macro_f1:.2f}", "base": f"{base_acc * 100:.0f}",
    "teen": f"{rep.loc['Teens', 'recall'] * 100:.0f}",
    "hit": f"{re_['+ title words (full model)*'] * 100:.0f}", "hitmeta": f"{re_['+ country'] * 100:.0f}",
    "rnd": f"{re_['Random (same type)'] * 100:.1f}", "dirm": f"{d.director_missing.mean() * 100:.0f}",
}

html = (O / "dashboard_template.html").read_text(encoding="utf-8")
html = html.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
for k, v in F.items():
    html = html.replace("{{" + k + "}}", str(v))
assert "{{" not in html, "unreplaced placeholder"
out = O / "netflix_dashboard.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB)")
print(F)