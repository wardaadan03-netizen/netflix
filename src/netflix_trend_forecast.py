"""
Task 4 - Trend Prediction Analysis (Netflix dataset)
 
"""
import sys
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
 
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/netflix_cleaned.csv")
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("trend_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)
 
START_YEAR = 2008        # start of the modelling window (earlier years are a thin back-catalogue)
HORIZON = 5              # years to forecast
MAX_BT_H = 3             # longest horizon used in backtesting
RED, DARK, GREY = "#E50914", "#221F1F", "#B3B3B3"
SERIES = ["Total", "Movie", "TV Show"]
 
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110, "savefig.bbox": "tight"})
 
# ===========================================================================
# STEP 1: Prepare release-year data
# ===========================================================================
df = pd.read_csv(DATA_PATH, parse_dates=["date_added"])
data_end = df["date_added"].max()
partial_year = data_end.year if data_end < pd.Timestamp(data_end.year, 12, 25) else None
last_full = partial_year - 1 if partial_year else int(df["release_year"].max())
 
tbl = (df.groupby(["release_year", "type"]).size().unstack(fill_value=0)
         .reindex(range(int(df["release_year"].min()), int(df["release_year"].max()) + 1), fill_value=0))
tbl["Total"] = tbl["Movie"] + tbl["TV Show"]
tbl["tv_share_pct"] = (tbl["TV Show"] / tbl["Total"] * 100).round(1)
tbl["yoy_growth_pct"] = (tbl["Total"].pct_change() * 100).round(1)
tbl["is_partial"] = tbl.index == partial_year
tbl.to_csv(OUT_DIR / "release_year_series.csv")
 
pre = int((df["release_year"] < START_YEAR).sum())
print(f"Titles: {len(df):,} | release years {tbl.index.min()}-{tbl.index.max()} | additions end {data_end.date()}")
print(f"Modelling window: {START_YEAR}-{last_full} ({last_full - START_YEAR + 1} complete years). "
      f"{pre:,} titles ({pre / len(df):.0%}) released before {START_YEAR} are back-catalogue and are not modelled.")
if partial_year:
    print(f"{partial_year} is a partial year (data ends {data_end.date()}), so it is shown but excluded from model fitting.")
 
# Data-maturity check: titles keep arriving on Netflix for years after release, so the
# newest release years are probably under-counted.
d2 = df.assign(lag=(df["year_added"] - df["release_year"]).clip(lower=0))
cohorts = d2[d2["release_year"].between(last_full - 4, last_full)]
mat = (cohorts.groupby("release_year")["lag"].apply(
    lambda s: pd.Series({"added same year": (s == 0).mean() * 100,
                         "added 1 yr later": (s == 1).mean() * 100,
                         "added 2+ yrs later": (s >= 2).mean() * 100})).unstack().round(1))
print("\nWhen are titles added relative to their release year? (% of each release-year cohort)")
print(mat.to_string())
print("Titles keep arriving for years after release (the 2+ yrs column is still filling for the newest\n"
      "cohorts), so the latest release years are probably under-counted.")
 
# ===========================================================================
# STEP 2: Analyse yearly content trends
# ===========================================================================
win = tbl.loc[START_YEAR:last_full]
for s in SERIES:
    y = win[s]
    peak = int(y.idxmax())
    cagr = ((y.loc[peak] / y.loc[START_YEAR]) ** (1 / (peak - START_YEAR)) - 1) * 100
    after = ("still rising at the end of the window" if peak == last_full else
             f"{peak}->{last_full}: {(y.loc[last_full] / y.loc[peak] - 1) * 100:+.0f}% total")
    print(f"{s:8s}: peak {peak} ({int(y.loc[peak]):,} titles) | growth {START_YEAR}->{peak}: "
          f"{cagr:.1f}%/yr | {after}")
tv_cross = tbl.loc[START_YEAR:, :]
cross = tv_cross.index[(tv_cross["TV Show"] > tv_cross["Movie"])]
print(f"\nTV share of releases: {win['tv_share_pct'].loc[START_YEAR]:.0f}% in {START_YEAR} -> "
      f"{win['tv_share_pct'].loc[last_full]:.0f}% in {last_full}"
      + (f" -> {tbl.loc[partial_year, 'tv_share_pct']:.0f}% in partial {partial_year}" if partial_year else ""))
print("Years with more TV shows than movies:",
      [f"{int(c)}{' (partial year)' if c == partial_year else ''}" for c in cross] if len(cross) else "none")
print("\nYear-over-year growth of total releases (%):")
print(win["yoy_growth_pct"].dropna().loc[START_YEAR + 5:].to_string())
 
# Chart 1: history
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
h = tbl.loc[2000:]
colors = {"Movie": RED, "TV Show": DARK}
bottom = np.zeros(len(h))
for t in ["Movie", "TV Show"]:
    bars = axes[0].bar(h.index, h[t], bottom=bottom, color=colors[t], label=t, width=0.8)
    if partial_year:
        bars[-1].set_hatch("//")
        bars[-1].set_edgecolor("white")
    bottom += h[t].to_numpy()
axes[0].set_title("Titles by release year", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Number of titles")
axes[0].legend(frameon=False, loc="upper left")
if partial_year:
    axes[0].text(partial_year, bottom[-1] + 25, "partial", ha="center", fontsize=8.5, color="grey")
axes[1].plot(win.index, win["tv_share_pct"], color=DARK, lw=2.4, marker="o")
if partial_year:
    axes[1].plot([last_full, partial_year], [win["tv_share_pct"].iloc[-1], tbl.loc[partial_year, "tv_share_pct"]],
                 color=DARK, lw=1.6, ls=":", marker="o", mfc="white")
axes[1].axhline(50, color=GREY, ls="--", lw=1)
axes[1].set_title("TV Show share of releases (%)", fontsize=13, fontweight="bold")
axes[1].set_ylim(0, 65)
g = win["yoy_growth_pct"].dropna()
axes[2].bar(g.index, g.values, color=[RED if v >= 0 else DARK for v in g.values])
axes[2].axhline(0, color="grey", lw=0.8)
axes[2].set_title("Year-over-year growth, total releases (%)", fontsize=13, fontweight="bold")
for a in axes:
    a.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.tight_layout()
fig.savefig(OUT_DIR / "trend_01_history.png", dpi=150)
plt.close(fig)
 
# ===========================================================================
# STEP 3: Build forecasting models
# ===========================================================================
def naive(y, h):
    return np.repeat(y[-1], h).astype(float)
 
 
def _poly(y, h, deg, last=None):
    y = np.asarray(y, float)
    if last:
        y = y[-last:]
    t = np.arange(len(y))
    c = np.polyfit(t, y, deg)
    return np.maximum(np.polyval(c, np.arange(len(y), len(y) + h)), 0)
 
 
def linear(y, h):
    return _poly(y, h, 1)
 
 
def linear_recent(y, h):
    return _poly(y, h, 1, last=5)
 
 
def quadratic(y, h):
    return _poly(y, h, 2)
 
 
def log_linear(y, h):
    y = np.log(np.maximum(np.asarray(y, float), 1))
    c = np.polyfit(np.arange(len(y)), y, 1)
    return np.exp(np.polyval(c, np.arange(len(y), len(y) + h)))
 
 
def holt_damped(y, h):
    """Holt's additive damped-trend exponential smoothing, parameters chosen by grid search."""
    y = np.asarray(y, float)
    best = None
    for a in np.arange(0.1, 1.0, 0.1):
        for b in np.arange(0.05, 0.55, 0.05):
            for phi in (0.80, 0.85, 0.90, 0.95, 0.98):
                lvl, tr, sse = y[0], y[1] - y[0], 0.0
                for t in range(1, len(y)):
                    fc = lvl + phi * tr
                    sse += (y[t] - fc) ** 2
                    new = a * y[t] + (1 - a) * fc
                    tr = b * (new - lvl) + (1 - b) * phi * tr
                    lvl = new
                if best is None or sse < best[0]:
                    best = (sse, phi, lvl, tr)
    _, phi, lvl, tr = best
    return np.maximum(lvl + np.cumsum(phi ** np.arange(1, h + 1)) * tr, 0)
 
 
MODELS = {"Naive (last value)": naive, "Linear trend": linear, "Linear (last 5 yrs)": linear_recent,
          "Log-linear (exponential)": log_linear, "Quadratic trend": quadratic, "Holt damped trend": holt_damped}
 
print(f"Models: {', '.join(MODELS)}")
print(f"Backtest: rolling origin, train {START_YEAR}..T for T = {START_YEAR + 7}..{last_full - 1}, "
      f"forecast up to {MAX_BT_H} years ahead, compare with the actual counts.\n")
 
bt_rows, best_model, rmse1 = [], {}, {}
for s in SERIES:
    y = win[s].to_numpy(float)
    n = len(y)
    errs = []
    for name, f in MODELS.items():
        for T in range(7, n - 1):
            hmax = min(MAX_BT_H, n - 1 - T)
            pred = f(y[:T + 1], hmax)
            for hh in range(1, hmax + 1):
                errs.append((name, hh, y[T + hh], pred[hh - 1]))
    e = pd.DataFrame(errs, columns=["model", "h", "actual", "pred"])
    e["abs_err"] = (e["actual"] - e["pred"]).abs()
    e["pct_err"] = e["abs_err"] / e["actual"] * 100
    m = e.groupby("model").agg(MAPE_pct=("pct_err", "mean"), MAE=("abs_err", "mean"),
                               n_forecasts=("h", "size"))
    m["RMSE_1yr"] = e[e["h"] == 1].groupby("model").apply(
        lambda d: np.sqrt(((d["actual"] - d["pred"]) ** 2).mean()))
    m = m.sort_values("MAPE_pct")
    best_model[s] = m.index[0]
    rmse1[s] = m.loc[m.index[0], "RMSE_1yr"]
    print(f"--- {s} ---")
    print(m.round(1).to_string())
    print(f"    best by MAPE: {best_model[s]}\n")
    bt_rows.append(m.reset_index().assign(series=s))
bt = pd.concat(bt_rows)[["series", "model", "MAPE_pct", "MAE", "RMSE_1yr", "n_forecasts"]]
bt.round(2).to_csv(OUT_DIR / "backtest_results.csv", index=False)
 
# Final forecasts: refit every model on the full window
fc_years = list(range(last_full + 1, last_full + HORIZON + 1))
fc_rows = []
for s in SERIES:
    y = win[s].to_numpy(float)
    for name, f in MODELS.items():
        pred = f(y, HORIZON)
        for k, (yr, p) in enumerate(zip(fc_years, pred), start=1):
            row = {"series": s, "model": name, "year": yr, "forecast": round(float(p), 1),
                   "selected": name == best_model[s]}
            if name == best_model[s]:
                half = 1.96 * rmse1[s] * np.sqrt(k)      # approximate band from backtest error
                row["lower"] = round(max(p - half, 0), 1)
                row["upper"] = round(p + half, 1)
            fc_rows.append(row)
fc = pd.DataFrame(fc_rows)
# Bottom-up total: sum of the best Movie and best TV Show models (the separately chosen
# Total model does not have to agree with its parts, so both views are kept).
bu = (fc[fc["selected"] & fc["series"].isin(["Movie", "TV Show"])]
      .groupby("year")["forecast"].sum().round(1))
bottom_up = pd.DataFrame({"series": "Total (bottom-up)", "model": "Best Movie + best TV Show",
                          "year": bu.index, "forecast": bu.values, "selected": False})
pd.concat([fc, bottom_up], ignore_index=True).to_csv(OUT_DIR / "forecasts.csv", index=False)
 
# Chart 2: backtest accuracy
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for ax, s in zip(axes, SERIES):
    m = bt[bt["series"] == s].sort_values("MAPE_pct", ascending=False)
    ax.barh(m["model"], m["MAPE_pct"], color=[RED if v == best_model[s] else DARK for v in m["model"]])
    for yy, v in enumerate(m["MAPE_pct"]):
        ax.text(v + 0.8, yy, f"{v:.0f}%", va="center", fontsize=9)
    ax.set_title(f"{s}: backtest error (MAPE)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, bt["MAPE_pct"].max() * 1.15)
fig.tight_layout()
fig.savefig(OUT_DIR / "trend_02_backtest.png", dpi=150)
plt.close(fig)
 
# ===========================================================================
# STEP 4: Visualise future predictions
# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharex=True)
for ax, s in zip(axes, SERIES):
    hist = win[s]
    ax.plot(hist.index, hist.values, color=DARK, lw=2.6, marker="o", ms=4, label="Actual", zorder=5)
    if partial_year:
        ax.scatter([partial_year], [tbl.loc[partial_year, s]], facecolors="white", edgecolors=DARK,
                   s=55, zorder=6, label=f"{partial_year} so far (partial)")
    for name in MODELS:
        if name == best_model[s]:
            continue
        r = fc[(fc["series"] == s) & (fc["model"] == name)]
        ax.plot([last_full] + list(r["year"]), [hist.iloc[-1]] + list(r["forecast"]),
                color=GREY, lw=1.1, ls="--", zorder=2)
    r = fc[(fc["series"] == s) & (fc["selected"])]
    ax.fill_between(r["year"], r["lower"], r["upper"], color=RED, alpha=0.15, zorder=3)
    ax.plot([last_full] + list(r["year"]), [hist.iloc[-1]] + list(r["forecast"]),
            color=RED, lw=2.6, marker="o", ms=4, label=f"Best model: {best_model[s]}", zorder=4)
    if s == "Total":
        ax.plot([last_full] + list(bu.index), [hist.iloc[-1]] + list(bu.values), color=DARK, lw=1.8,
                ls=":", label="Movie + TV Show models (bottom-up)", zorder=4)
    ax.axvline(last_full + 0.5, color="grey", lw=0.8, ls=":")
    ax.set_title(s, fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(hist.max(), r["upper"].max()) * 1.3)     # runaway models may leave the frame
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
axes[0].set_ylabel("Titles released that year (in catalogue)")
fig.text(0.01, -0.03, "Exponential and quadratic models run off the top of some panels; "
         "all values are in forecasts.csv. The band is a rough range based on backtest error.",
         fontsize=8.5, color="grey")
fig.suptitle("Forecast of titles by release year (grey dashed = other models; band = approximate 95% range)",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "trend_03_forecast.png", dpi=150)
plt.close(fig)
 
for s in SERIES:
    r = fc[(fc["series"] == s) & (fc["selected"])]
    others = fc[(fc["series"] == s) & (~fc["selected"])].groupby("year")["forecast"].agg(["min", "max"])
    print(f"\n{s} - best model: {best_model[s]}")
    out = r[["year", "forecast", "lower", "upper"]].set_index("year").join(others.rename(
        columns={"min": "all_models_min", "max": "all_models_max"}))
    print(out.round(0).astype(int).to_string())
 
# ===========================================================================
# STEP 5: Interpret results
# ===========================================================================
tot = win["Total"]
peak = int(tot.idxmax())
sel = lambda s, yr: float(fc[(fc.series == s) & fc.selected & (fc.year == yr)]["forecast"].iloc[0])
end_year = fc_years[-1]
naive_better = [s for s in SERIES if best_model[s].startswith(("Naive", "Holt"))]
print(f"1. Releases grew fast to a {peak} peak of {int(tot.loc[peak]):,} titles, then fell to "
      f"{int(tot.loc[last_full]):,} by {last_full}.")
print(f"2. The mix shifted: TV shows rose from {win['tv_share_pct'].loc[START_YEAR]:.0f}% to "
      f"{win['tv_share_pct'].loc[last_full]:.0f}% of releases while movies fell from their peak.")
print(f"3. Best backtest models: " + "; ".join(f"{s} -> {best_model[s]}" for s in SERIES) + ".")
print(f"4. Forecast for {end_year}: movies ~{sel('Movie', end_year):,.0f}, TV shows ~{sel('TV Show', end_year):,.0f}. "
      f"Total is ~{sel('Total', end_year):,.0f} from its own best model (flat) but ~{bu.loc[end_year]:,.0f} when the two "
      f"parts are added up - the gap shows how uncertain the total is.")
worst_linear = bt[(bt.model == "Log-linear (exponential)")].set_index("series")["MAPE_pct"]
print(f"5. Pure growth models fit the boom years but fail after the peak "
      f"(log-linear backtest error: " + ", ".join(f"{s} {worst_linear[s]:.0f}%" for s in SERIES) + ").")
print("6. Treat these as scenarios, not predictions: 13 data points, a trend break around the peak,\n"
      "   and a catalogue snapshot in which recent years are still being filled in.")
print(f"\nFiles saved in: {OUT_DIR.resolve()}")
 





