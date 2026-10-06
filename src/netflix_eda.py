"""
Task 2 - Exploratory Data Analysis (Netflix dataset)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "data/processed/netflix_cleaned.csv"
)
FIG_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures")

FIG_DIR.mkdir(parents=True, exist_ok=True)

RED, DARK, GREY = "#E50914", "#221F1F", "#B3B3B3"

PALETTE = {
    "Movie": RED,
    "TV Show": DARK
}

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

DAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

sns.set_theme(style="whitegrid", font_scale=1.0)

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.bbox": "tight"
})


def save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    print(f"Saved: {name}")


# ---------------------------------------------------------------------------
# STEP 1: Load dataset and basic statistics
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH, parse_dates=["date_added"])

print("=" * 70)
print("NETFLIX EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(
    f"Date range: {df['date_added'].min().date()} → "
    f"{df['date_added'].max().date()}"
)

# ---------------------------------------------------------------------------
# STEP 2: Content distribution by type
# ---------------------------------------------------------------------------

type_counts = df["type"].value_counts()
type_pct = (type_counts / len(df) * 100).round(1)

print("\nContent distribution:")
for content_type in type_counts.index:
    print(
        f"  {content_type}: "
        f"{type_counts[content_type]:,} "
        f"({type_pct[content_type]}%)"
    )

by_year = df.groupby(["year_added", "type"]).size().unstack(fill_value=0)
by_year["total"] = by_year.sum(axis=1)

# Chart 1: Type distribution

fig, ax = plt.subplots(figsize=(5.5, 5.5))

ax.pie(
    type_counts,
    labels=[
        f"{t}\n{n:,} ({p}%)"
        for t, n, p in zip(type_counts.index, type_counts, type_pct)
    ],
    colors=[PALETTE[t] for t in type_counts.index],
    startangle=90,
    wedgeprops={"width": 0.42, "edgecolor": "white"},
    textprops={"fontsize": 11}
)

ax.set_title(
    "Movies vs TV Shows",
    fontsize=14,
    fontweight="bold"
)

save(fig, "01_type_distribution.png")


# Chart 2: Titles added per year

yr = by_year.loc[2014:, ["Movie", "TV Show"]]

fig, ax = plt.subplots(figsize=(9, 5))

yr.plot(
    kind="bar",
    stacked=True,
    color=[PALETTE["Movie"], PALETTE["TV Show"]],
    ax=ax,
    width=0.75
)

ax.set_title(
    "Titles Added to Netflix per Year",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("")
ax.set_ylabel("Number of titles")
ax.tick_params(axis="x", rotation=0)

ax.legend(title="", frameon=False)

ax.set_ylim(
    0,
    yr.sum(axis=1).max() * 1.12
)

ax.grid(axis="x", visible=False)

ax.text(
    len(yr) - 1,
    yr.iloc[-1].sum() + 40,
    "partial year\n(data ends Sep 25)",
    ha="center",
    va="bottom",
    fontsize=9,
    color="grey"
)

save(fig, "02_titles_added_per_year.png")


# ---------------------------------------------------------------------------
# STEP 3: Top countries and genres
# ---------------------------------------------------------------------------

known = df[df["country"] != "Unknown"]

top_countries = known["country"].value_counts().head(10)

print("\nTop 5 countries:")

for country, count in top_countries.head(5).items():
    print(f"  {country}: {count:,}")

genres = (
    df.assign(genre=df["genres"].str.split("|"))
      .explode("genre")
)

top_genres = genres["genre"].value_counts().head(10)

print("\nTop 5 genres:")

for genre, count in top_genres.head(5).items():
    print(f"  {genre}: {count:,}")


# Chart 3: Top countries

ct = (
    known[known["country"].isin(top_countries.index)]
    .groupby(["country", "type"])
    .size()
    .unstack(fill_value=0)
    .loc[top_countries.index[::-1]]
)

fig, ax = plt.subplots(figsize=(8.5, 5.5))

ct.plot(
    kind="barh",
    stacked=True,
    color=[PALETTE["Movie"], PALETTE["TV Show"]],
    ax=ax,
    width=0.75
)

ax.set_title(
    "Top 10 Content-Producing Countries",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Number of titles")
ax.set_ylabel("")

ax.legend(
    title="",
    frameon=False,
    loc="lower right"
)

ax.set_xlim(
    0,
    ct.sum(axis=1).max() * 1.1
)

for i, total in enumerate(ct.sum(axis=1)):
    ax.text(
        total + 20,
        i,
        f"{total:,}",
        va="center",
        fontsize=9
    )

save(fig, "03_top_countries.png")


# Chart 4: Top genres

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

for ax, content_type in zip(
    axes,
    ["Movie", "TV Show"]
):

    g = (
        genres[genres["type"] == content_type]["genre"]
        .value_counts()
        .head(10)[::-1]
    )

    ax.barh(
        g.index,
        g.values,
        color=PALETTE[content_type]
    )

    ax.set_title(
        f"Top 10 {content_type} Genres",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel("Number of titles")

    for i, value in enumerate(g.values):
        ax.text(
            value + g.max() * 0.01,
            i,
            f"{value:,}",
            va="center",
            fontsize=9
        )

fig.tight_layout()

save(fig, "04_top_genres.png")


# ---------------------------------------------------------------------------
# STEP 4: Ratings, duration and release gap
# ---------------------------------------------------------------------------

rating_counts = df["rating"].value_counts()

aud_by_type = (
    df.groupby("type")["audience_group"]
    .value_counts(normalize=True)
    .unstack()
    .mul(100)
    .round(1)
)

movies = df[df["type"] == "Movie"]
shows = df[df["type"] == "TV Show"]

print("\nKey metrics:")

print(
    f"  Median movie runtime: "
    f"{movies['duration_minutes'].median():.0f} minutes"
)

print(
    f"  TV shows with 1 season: "
    f"{(shows['seasons'] == 1).mean():.1%}"
)

print(
    f"  Median release-to-Netflix gap: "
    f"{df['years_to_netflix'].median():.0f} years"
)


# Chart 5: Ratings

aud_colors = {
    "Kids": "#46A3FF",
    "Teens": "#F5A623",
    "Adults": RED,
    "Unrated": GREY
}

rating_group = (
    df.groupby("rating")["audience_group"]
    .agg(lambda s: s.iloc[0])
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    rating_counts.index,
    rating_counts.values,
    color=[
        aud_colors[rating_group[r]]
        for r in rating_counts.index
    ]
)

ax.set_title(
    "Content Ratings",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel("Number of titles")

ax.tick_params(
    axis="x",
    rotation=45
)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=c)
    for c in aud_colors.values()
]

ax.legend(
    handles,
    aud_colors.keys(),
    title="Audience group",
    frameon=False
)

save(fig, "05_ratings.png")


# Chart 6: Movie runtime and TV seasons

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4.8)
)

axes[0].hist(
    movies["duration_minutes"],
    bins=40,
    color=RED,
    edgecolor="white"
)

axes[0].axvline(
    movies["duration_minutes"].median(),
    color=DARK,
    ls="--",
    lw=1.5
)

axes[0].set_title(
    "Movie Runtime",
    fontsize=13,
    fontweight="bold"
)

axes[0].set_xlabel("Minutes")
axes[0].set_ylabel("Number of movies")


sc = (
    shows["seasons"]
    .astype(int)
    .clip(upper=8)
    .value_counts()
    .sort_index()
)

labels = [
    str(i) if i < 8 else "8+"
    for i in sc.index
]

axes[1].bar(
    labels,
    sc.values,
    color=DARK
)

axes[1].set_title(
    "TV Show Seasons",
    fontsize=13,
    fontweight="bold"
)

axes[1].set_xlabel("Number of seasons")
axes[1].set_ylabel("Number of shows")

for i, value in enumerate(sc.values):
    axes[1].text(
        i,
        value + 15,
        f"{value:,}",
        ha="center",
        fontsize=9
    )

fig.tight_layout()

save(fig, "06_duration.png")


# ---------------------------------------------------------------------------
# STEP 5: Release year and Netflix arrival gap
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4.8)
)

rel = (
    df[df["release_year"] >= 1990]
    .groupby(["release_year", "type"])
    .size()
    .unstack(fill_value=0)
)

rel.plot(
    ax=axes[0],
    color=[PALETTE["Movie"], PALETTE["TV Show"]],
    lw=2.2
)

axes[0].set_title(
    "Titles by Release Year (1990+)",
    fontsize=13,
    fontweight="bold"
)

axes[0].set_xlabel("")
axes[0].set_ylabel("Number of titles")

axes[0].legend(
    title="",
    frameon=False
)


lag = (
    df["years_to_netflix"]
    .clip(0, 15)
    .value_counts()
    .sort_index()
)

axes[1].bar(
    lag.index.astype(str).str.replace("15", "15+"),
    lag.values,
    color=RED
)

axes[1].set_title(
    "Years Between Release and Netflix Arrival",
    fontsize=13,
    fontweight="bold"
)

axes[1].set_xlabel(
    "Years (negative values counted as 0)"
)

axes[1].set_ylabel("Number of titles")

fig.tight_layout()

save(fig, "07_release_vs_added.png")


# ---------------------------------------------------------------------------
# STEP 6: Addition timing
# ---------------------------------------------------------------------------

mcount = (
    df["month_name"]
    .value_counts()
    .reindex(MONTHS)
)

dcount = (
    df["day_of_week"]
    .value_counts()
    .reindex(DAYS)
)

print(
    f"\nBusiest month: {mcount.idxmax()} "
    f"({mcount.max():,} titles)"
)

print(
    f"Busiest weekday: {dcount.idxmax()} "
    f"({dcount.max() / len(df):.1%} of additions)"
)


# Chart 8: Month and weekday

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 4.8)
)

axes[0].bar(
    [m[:3] for m in MONTHS],
    mcount.values,
    color=[
        RED if m == mcount.idxmax() else GREY
        for m in MONTHS
    ]
)

axes[0].set_title(
    "Titles Added by Month",
    fontsize=13,
    fontweight="bold"
)

axes[0].set_ylabel("Number of titles")


axes[1].bar(
    [d[:3] for d in DAYS],
    dcount.values,
    color=[
        RED if d == dcount.idxmax() else GREY
        for d in DAYS
    ]
)

axes[1].set_title(
    "Titles Added by Weekday",
    fontsize=13,
    fontweight="bold"
)

fig.tight_layout()

save(fig, "08_added_month_weekday.png")


