"""
Task 3 - Recommendation System Analysis (Netflix dataset)

"""

import difflib
import re
import sys
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


# ===========================================================================
# STEP 0: Paths and settings
# ===========================================================================

DATA_PATH = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path("data/processed/netflix_cleaned.csv")
)

OUT_DIR = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else Path("recommender_output")
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_TITLES = [
    "Stranger Things",
    "Peaky Blinders",
    "Naruto",
    "Sacred Games",
    "Bird Box",
    "Dangal",
    "Our Planet",
    "Transformer Prime",
    "The Social Dilemma",
]

# Extra titles can be supplied after the output directory.
SELECTED_TITLES += sys.argv[3:]

TOP_N = 10
SEED = 42

RED = "#E50914"
DARK = "#221F1F"
GREY = "#B3B3B3"

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 48)


# ===========================================================================
# STEP 1: Load and prepare the data
# ===========================================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Input file not found: {DATA_PATH}\n"
        "Run Task 1 first so that netflix_cleaned.csv exists."
    )

raw = pd.read_csv(DATA_PATH)

# Make column names easier to work with.
raw.columns = [str(c).strip() for c in raw.columns]


def first_existing_column(frame, names, required=True):
    """Return the first matching column from a list of possible names."""
    for name in names:
        if name in frame.columns:
            return name

    if required:
        raise KeyError(
            f"None of these columns were found: {names}\n"
            f"Available columns are: {list(frame.columns)}"
        )

    return None


title_col = first_existing_column(raw, ["title"])
type_col = first_existing_column(raw, ["type"])
director_col = first_existing_column(raw, ["director"])
rating_col = first_existing_column(raw, ["rating"])
country_col = first_existing_column(raw, ["country"])
year_col = first_existing_column(raw, ["release_year"])

# Your cleaning task may call this column "listed_in" or "genres".
genre_col = first_existing_column(raw, ["genres", "listed_in", "geners"])

# audience_group was expected by the original script. If Task 1 did not
# create it, derive a simple audience group from the rating.
audience_col = first_existing_column(raw, ["audience_group"], required=False)

if audience_col is None:
    raw["audience_group"] = (
        raw[rating_col]
        .fillna("Unrated")
        .astype(str)
        .str.strip()
    )
    audience_col = "audience_group"

# Standardized internal names.
df = raw.rename(
    columns={
        title_col: "title",
        type_col: "type",
        director_col: "director",
        rating_col: "rating",
        country_col: "country",
        year_col: "release_year",
        genre_col: "genres",
        audience_col: "audience_group",
    }
).copy()

# Clean basic values.
df["title"] = df["title"].fillna("Unknown").astype(str).str.strip()
df["type"] = df["type"].fillna("Unknown").astype(str).str.strip()
df["director"] = df["director"].fillna("Unknown").astype(str).str.strip()
df["rating"] = df["rating"].fillna("Unrated").astype(str).str.strip()
df["country"] = df["country"].fillna("Unknown").astype(str).str.strip()
df["genres"] = df["genres"].fillna("Unknown").astype(str).str.strip()
df["audience_group"] = (
    df["audience_group"].fillna("Unrated").astype(str).str.strip()
)

df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
df["release_year"] = df["release_year"].fillna(df["release_year"].median())
df["release_year"] = df["release_year"].fillna(2000).astype(int)

# Normalized title key for exact/fuzzy matching.
def normalize_text(s):
    """Lower-case, strip accents/punctuation, and collapse spaces."""
    s = unicodedata.normalize("NFKD", str(s))
    s = s.encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


df["title_key"] = df["title"].map(normalize_text)

# Case/spacing variants of the same title should not be duplicated.
df = (
    df.drop_duplicates(
        subset=["title_key", "type", "release_year"]
    )
    .reset_index(drop=True)
)

print(
    f"Titles loaded: {len(raw):,} | "
    f"near-duplicate titles removed for the recommender: "
    f"{len(raw) - len(df):,} | "
    f"catalogue size: {len(df):,}"
)

print(
    "Features used: genres, director, rating + audience group, type, "
    "release decade, country, title words"
)
print(
    "Not available in this dataset: plot description, cast, viewing history "
    "or user ratings."
)


# Relative importance of each feature block.
DEFAULT_WEIGHTS = {
    "genres": 3.0,
    "director": 1.5,
    "maturity": 1.0,
    "type": 1.0,
    "era": 0.5,
    "country": 0.25,
    "title": 0.5,
}


# ===========================================================================
# STEP 2: Text preprocessing
# ===========================================================================

TITLE_STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "tv",
    "movie",
    "film",
    "series",
    "season",
    "part",
    "vol",
}


def to_token(s):
    """Convert multi-word values to one token."""
    return normalize_text(s).replace(" ", "_")


def title_tokens(title):
    words = normalize_text(title).split()
    return " ".join(
        w
        for w in words
        if w not in TITLE_STOPWORDS
        and len(w) >= 3
        and not w.isdigit()
    )


def era_token(year):
    if year < 1980:
        return "era_pre1980"
    return f"era_{int(year) // 10 * 10}s"


def split_genres(value):
    """Return individual genre labels from a pipe-separated genre string."""
    if not value or value == "Unknown":
        return []

    return [
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    ]


fields = pd.DataFrame(index=df.index)

fields["genres"] = df["genres"].map(
    lambda s: " ".join(to_token(g) for g in split_genres(s))
)

fields["director"] = df["director"].map(
    lambda s: ""
    if normalize_text(s) in {"", "unknown", "nan"}
    else " ".join(to_token(x) for x in s.split(","))
)

fields["maturity"] = (
    "rating_"
    + df["rating"].map(to_token)
    + " audience_"
    + df["audience_group"].map(to_token)
)

fields["type"] = df["type"].map(to_token)
fields["era"] = df["release_year"].map(era_token)

fields["country"] = df["country"].map(
    lambda s: ""
    if normalize_text(s) in {"", "unknown", "nan"}
    else " ".join(to_token(x) for x in s.split(","))
)

fields["title"] = df["title"].map(title_tokens)

print(
    "Preprocessing: accents stripped, lower-cased, punctuation removed, "
    "multi-word genres/names joined into single tokens, and title stop-words "
    "removed.\n"
)

demo = df.index[
    df["title"].isin(["Sacred Games", "Dangle"])
]

for i in demo:
    print(f"'{df.loc[i, 'title']}'")
    for col in fields.columns:
        print(f"{col:9s}: {fields.loc[i, col]}")


# ===========================================================================
# STEP 3: Calculate content similarity
# ===========================================================================

B = {}
P = {}

for name in DEFAULT_WEIGHTS:
    vec = TfidfVectorizer(
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None,
        binary=True,
        norm="l2",
        dtype=np.float32,
    )

    # Ensure there is at least one usable token.
    if not fields[name].str.strip().any():
        fields[name] = "unknown"

    M = vec.fit_transform(fields[name])

    # P = whether each title has data for this feature block.
    P[name] = (M.getnnz(axis=1) > 0).astype(np.float32)

    # Small vocabularies are faster as dense arrays.
    B[name] = (
        M.toarray()
        if M.shape[1] <= 128
        else M.tocsr()
    )

    print(
        f"   {name:9s}: {M.shape[1]:5d} features | "
        f"titles with data: {P[name].mean():.1%}"
    )

N = len(df)
YEARS = df["release_year"].to_numpy(dtype=np.float32)

# Genre membership matrix for diversity/consistency metrics.
genre_values = sorted(
    {
        genre
        for value in df["genres"]
        for genre in split_genres(value)
    }
)

genre_to_index = {genre: i for i, genre in enumerate(genre_values)}
GBIN = np.zeros((N, len(genre_values)), dtype=np.uint8)

for row_idx, value in enumerate(df["genres"]):
    for genre in split_genres(value):
        GBIN[row_idx, genre_to_index[genre]] = 1

ALL_PRESENT = {
    name: bool(P[name].all())
    for name in B
}


def similarity(q_blocks, q_present, weights):
    """
    Weighted-average cosine similarity of query rows against the catalogue.
    Missing feature blocks are excluded from the denominator.
    """
    num = 0.0
    den = 0.0

    for name, weight in weights.items():
        if weight <= 0:
            continue

        s = q_blocks[name] @ B[name].T

        if not isinstance(s, np.ndarray):
            s = s.toarray()

        num = num + weight * s

        if ALL_PRESENT[name] and q_present[name].all():
            den = den + weight
        else:
            den = den + weight * np.outer(
                q_present[name],
                P[name],
            )

    return num / np.maximum(den, 1e-9)


def query_parts(idx):
    return (
        {name: B[name][idx] for name in B},
        {name: P[name][idx] for name in B},
    )


def rank(similarity_scores, idx, k=TOP_N):
    """
    Top-k per row.

    Similarity is the main ranking score. A tiny year-based adjustment is
    used only to break very close ties in favour of similar release years.
    """
    scores = similarity_scores.copy()

    year_bonus = 1e-3 / (
        1.0
        + np.abs(
            YEARS[idx][:, None] - YEARS[None, :]
        )
    )

    scores = scores + year_bonus

    # Never recommend the title itself.
    scores[
        np.arange(len(idx)),
        idx
    ] = -np.inf

    k = min(k, N - 1)

    if k <= 0:
        return np.empty((len(idx), 0), dtype=int)

    top = np.argpartition(
        -scores,
        kth=k - 1,
        axis=1,
    )[:, :k]

    top_scores = np.take_along_axis(
        scores,
        top,
        axis=1,
    )

    order = np.argsort(
        -top_scores,
        axis=1,
    )

    return np.take_along_axis(top, order, axis=1)


# ===========================================================================
# STEP 4: Generate recommendations for selected titles
# ===========================================================================

def find_title(query):
    key = normalize_text(query)

    hits = df.index[
        df["title_key"] == key
    ]

    if len(hits):
        return int(hits[0])

    close = difflib.get_close_matches(
        key,
        df["title_key"].tolist(),
        n=3,
        cutoff=0.6,
    )

    suggestions = [
        df.loc[
            df["title_key"] == candidate,
            "title",
        ].iloc[0]
        for candidate in close
    ]

    raise KeyError(
        f"'{query}' not found. Did you mean: {suggestions}"
    )


def recommend(
    query,
    n=TOP_N,
    same_type_only=False,
    weights=DEFAULT_WEIGHTS,
):
    i = find_title(query)
    idx = np.array([i])

    sim = similarity(
        *query_parts(idx),
        weights,
    )

    ranking_scores = sim.copy()

    if same_type_only:
        different_type = (
            df["type"] != df.loc[i, "type"]
        ).to_numpy()

        ranking_scores[
            0,
            different_type
        ] = -np.inf

    rec = rank(
        ranking_scores,
        idx,
        k=n,
    )[0]

    out = df.loc[
        rec,
        [
            "title",
            "type",
            "release_year",
            "country",
            "rating",
            "genres",
        ],
    ].copy()

    out.insert(
        0,
        "rank",
        range(1, len(rec) + 1),
    )

    out["similarity"] = (
        sim[0, rec]
        .round(3)
    )

    out["genres"] = out["genres"].str.replace(
        "|",
        ", ",
        regex=False,
    )

    return (
        df.loc[i],
        out.reset_index(drop=True),
    )


rec_rows = []

for title in SELECTED_TITLES:
    try:
        query_row, recommendations = recommend(title)

    except KeyError as error:
        print(f"\n{error}")
        continue

    print(
        f"\n>>> {query_row['title']} "
        f"({query_row['type']}, "
        f"{query_row['country']}, "
        f"{query_row['rating']}) | "
        f"{query_row['genres'].replace('|', ', ')}"
    )

    print(
        recommendations[
            [
                "rank",
                "title",
                "type",
                "release_year",
                "genres",
                "similarity",
            ]
        ].to_string(index=False)
    )

    rec_rows.append(
        recommendations.assign(
            query=query_row["title"]
        )
    )

if rec_rows:
    pd.concat(rec_rows)[
        [
            "query",
            "rank",
            "title",
            "type",
            "release_year",
            "country",
            "rating",
            "genres",
            "similarity",
        ]
    ].to_csv(
        OUT_DIR / "sample_recommendations.csv",
        index=False,
    )
else:
    print(
        "\nNo selected titles were found. "
        "No sample_recommendations.csv was created."
    )


# ===========================================================================
# STEP 5: Evaluate recommendation quality
# ===========================================================================

print(
    "\nNo user ratings or watch histories exist, so there is no direct "
    "ground truth for 'would this viewer like it'. Quality is checked "
    "with a weak-label test, consistency metrics and manual inspection.\n"
)


# ---- 5a. Franchise recovery -----------------------------------------------

# Weak ground truth:
# titles sharing the text before ":" are treated as one franchise.
fkey = df["title"].map(
    lambda title: (
        normalize_text(title.split(":")[0])
        if ":" in title
        else None
    )
)

fsize = fkey.map(fkey.value_counts())

fr_idx = np.where(
    fkey.notna() & (fsize >= 2)
)[0]

members = {
    key: np.array(values)
    for key, values in df.groupby(fkey).groups.items()
}

siblings = [
    members[fkey.iloc[i]][
        members[fkey.iloc[i]] != i
    ]
    for i in fr_idx
]

print(
    f"5a. Franchise recovery: {len(fr_idx)} titles in "
    f"{fkey.iloc[fr_idx].nunique()} franchises "
    f"(prefix before ':'); each title should surface "
    f"its siblings in the top {TOP_N}."
)


def franchise_scores(weights, batch=500):
    hits = np.zeros(len(fr_idx))
    recall = np.zeros(len(fr_idx))

    for start in range(
        0,
        len(fr_idx),
        batch,
    ):
        idx = fr_idx[
            start:start + batch
        ]

        top = rank(
            similarity(
                *query_parts(idx),
                weights,
            ),
            idx,
        )

        for row_number in range(len(idx)):
            n_hit = np.isin(
                top[row_number],
                siblings[start + row_number],
            ).sum()

            hits[start + row_number] = (
                n_hit > 0
            )

            recall[start + row_number] = (
                n_hit
                / min(
                    len(siblings[start + row_number]),
                    TOP_N,
                )
            )

    return hits, recall


def random_expectation(same_type):
    """Expected hit rate / recall from random TOP_N selection."""
    types_ = df["type"].to_numpy()

    hits = []
    recall = []

    for i, sib in zip(fr_idx, siblings):
        if same_type:
            pool = (
                types_ == types_[i]
            ).sum() - 1

            s = int(
                (
                    types_[sib]
                    == types_[i]
                ).sum()
            )
        else:
            pool = N - 1
            s = len(sib)

        if pool <= 0:
            hits.append(0.0)
            recall.append(0.0)
            continue

        sample_size = min(TOP_N, pool)
        successes = min(s, sample_size)

        p_miss = 1.0

        for j in range(sample_size):
            remaining_pool = pool - j
            remaining_non_siblings = (
                pool - s - j
            )

            if remaining_non_siblings <= 0:
                p_miss = 0.0
                break

            p_miss *= (
                remaining_non_siblings
                / remaining_pool
            )

        hits.append(1 - p_miss)

        recall.append(
            sample_size
            * s
            / pool
            / max(
                min(len(sib), TOP_N),
                1,
            )
        )

    return (
        np.array(hits),
        np.array(recall),
    )


ladder = [
    (
        "Genres only",
        {
            "genres": DEFAULT_WEIGHTS["genres"]
        },
    )
]

labels = {
    "type": "+ type",
    "maturity": "+ rating",
    "director": "+ director",
    "era": "+ release era",
    "country": "+ country",
    "title": "+ title words (full model)*",
}

for name in [
    "type",
    "maturity",
    "director",
    "era",
    "country",
    "title",
]:
    ladder.append(
        (
            labels[name],
            {
                **ladder[-1][1],
                name: DEFAULT_WEIGHTS[name],
            },
        )
    )


scores = {
    "Random": random_expectation(False),
    "Random (same type)": random_expectation(True),
}

for label, weights in ladder:
    scores[label] = franchise_scores(weights)


rows = []

for label, (hit, recall) in scores.items():
    n = len(hit)

    if n > 1:
        hit_ci = (
            1.96
            * hit.std(ddof=1)
            / np.sqrt(n)
        )

        recall_ci = (
            1.96
            * recall.std(ddof=1)
            / np.sqrt(n)
        )
    else:
        hit_ci = 0.0
        recall_ci = 0.0

    rows.append(
        {
            "model": label,
            "hit_rate_at_10": hit.mean(),
            "hit_ci95": hit_ci,
            "recall_at_10": recall.mean(),
            "recall_ci95": recall_ci,
            "n_queries": n,
        }
    )


results = pd.DataFrame(rows)

results.round(4).to_csv(
    OUT_DIR / "evaluation_results.csv",
    index=False,
)

print(
    results.assign(
        hit_rate_at_10=lambda d:
            d.hit_rate_at_10.map(
                "{:.3f}".format
            ),
        hit_ci95=lambda d:
            d.hit_ci95.map(
                "±{:.3f}".format
            ),
        recall_at_10=lambda d:
            d.recall_at_10.map(
                "{:.3f}".format
            ),
        recall_ci95=lambda d:
            d.recall_ci95.map(
                "±{:.3f}".format
            ),
    ).to_string(index=False)
)

print(
    "   * uses title text, which is also how franchises were defined, "
    "so that row is partly circular.\n"
    "     The row above it (metadata only) is the fair test of the "
    "content features."
)


# ---- 5b. Consistency and beyond-accuracy metrics ---------------------------

all_top = np.zeros(
    (N, TOP_N),
    dtype=int,
)

all_sim = np.zeros(
    (N, TOP_N)
)

for start in range(
    0,
    N,
    1000,
):
    idx = np.arange(
        start,
        min(start + 1000, N),
    )

    sim = similarity(
        *query_parts(idx),
        DEFAULT_WEIGHTS,
    )

    top = rank(
        sim,
        idx,
    )

    all_top[idx] = top

    all_sim[idx] = np.take_along_axis(
        sim,
        top,
        axis=1,
    )


types = df["type"].to_numpy()

same_type = (
    types[all_top]
    == types[:, None]
).mean()

Gf = GBIN.astype(np.float32)

if GBIN.shape[1] > 0:
    inter = np.einsum(
        "nk,nrk->nr",
        Gf,
        Gf[all_top],
    )

    union = (
        Gf.sum(1)[:, None]
        + Gf[all_top].sum(2)
        - inter
    )

    jaccard = np.divide(
        inter,
        union,
        out=np.zeros_like(inter),
        where=union > 0,
    ).mean()

    rows_with_two_genres = (
        Gf.sum(1) >= 2
    )

    if rows_with_two_genres.any():
        share_2plus = (
            inter[rows_with_two_genres] >= 2
        ).mean()
    else:
        share_2plus = 0.0

    R = Gf[all_top]

    pair_inter = np.einsum(
        "nik,njk->nij",
        R,
        R,
    )

    pair_union = (
        R.sum(2)[:, :, None]
        + R.sum(2)[:, None, :]
        - pair_inter
    )

    pair_jaccard = np.divide(
        pair_inter,
        pair_union,
        out=np.zeros_like(pair_inter),
        where=pair_union > 0,
    )

    pair_mask = ~np.eye(
        TOP_N,
        dtype=bool,
    )

    diversity = (
        1 - pair_jaccard
    )[:, pair_mask].mean()

else:
    jaccard = 0.0
    share_2plus = 0.0
    diversity = 0.0


counts = np.bincount(
    all_top.ravel(),
    minlength=N,
)

coverage = (
    counts > 0
).mean()

top1pct_share = (
    np.sort(counts)[::-1][
        :max(1, N // 100)
    ].sum()
    / max(counts.sum(), 1)
)

never = (
    counts == 0
).sum()


print(
    "\n5b. Consistency and beyond-accuracy metrics "
    "(full model, all titles):"
)

print(
    f"   Same-type recommendation share: {same_type:.3f}"
)

print(
    f"   Mean genre Jaccard similarity:   {jaccard:.3f}"
)

print(
    f"   Recommendations sharing 2+ genres: {share_2plus:.3f}"
)

print(
    f"   Intra-list genre diversity:       {diversity:.3f}"
)

print(
    f"   Catalogue coverage:                {coverage:.3f}"
)

print(
    f"   Top 1% title recommendation share: {top1pct_share:.3f}"
)

print(
    f"   Titles never recommended:          {never:,}"
)


# ===========================================================================
# STEP 6: Charts
# ===========================================================================

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
    }
)

order = results["model"].tolist()[::-1]

colors = [
    GREY
    if model.startswith("Random")
    else (
        RED
        if model.endswith("*")
        else DARK
    )
    for model in order
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5.2),
    sharey=True,
)

for ax, (column, ci, chart_title) in zip(
    axes,
    [
        (
            "hit_rate_at_10",
            "hit_ci95",
            "Hit rate@10",
        ),
        (
            "recall_at_10",
            "recall_ci95",
            "Recall@10",
        ),
    ],
):
    result_plot = (
        results
        .set_index("model")
        .loc[order]
    )

    ax.barh(
        order,
        result_plot[column],
        xerr=result_plot[ci],
        color=colors,
        error_kw={
            "ecolor": "#888",
            "lw": 1,
        },
    )

    for y, (value, error) in enumerate(
        zip(
            result_plot[column],
            result_plot[ci],
        )
    ):
        ax.text(
            value + error + 0.012,
            y,
            f"{value:.2f}",
            va="center",
            fontsize=9,
        )

    ax.set_title(
        chart_title,
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xlim(
        0,
        max(
            1.0,
            result_plot[column].max() + 0.12,
        ),
    )

fig.suptitle(
    "Can the recommender find titles from the same franchise?",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)

fig.text(
    0.01,
    -0.03,
    "* includes title words, which also define the franchises "
    "(partly circular). Grey = random picks. Bars show 95% "
    "confidence intervals.",
    fontsize=8.5,
    color="grey",
)

fig.tight_layout()

fig.savefig(
    OUT_DIR / "rec_01_evaluation.png",
    dpi=150,
)

plt.close(fig)


# Sample recommendation chart.
show_titles = [
    title
    for title in [
        "Stranger Things",
        "Naruto",
        "Dangal",
        "Our Planet",
    ]
    if title in SELECTED_TITLES
]

show = []

for title in show_titles:
    try:
        show.append(find_title(title))
    except KeyError:
        pass


if show:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 8),
    )

    axes_flat = axes.ravel()

    for ax in axes_flat:
        ax.set_visible(False)

    for ax, i in zip(
        axes_flat,
        show,
    ):
        ax.set_visible(True)

        query_row, recommendations = recommend(
            df.loc[i, "title"]
        )

        labels_ = [
            title
            if len(title) <= 34
            else title[:32] + "…"
            for title in recommendations["title"]
        ][::-1]

        bar_colors = [
            RED
            if item == "Movie"
            else DARK
            for item in recommendations["type"][::-1]
        ]

        ax.barh(
            labels_,
            recommendations["similarity"][::-1],
            color=bar_colors,
        )

        ax.set_xlim(0, 1.0)

        ax.set_title(
            f"Because you watched: {query_row['title']}",
            fontsize=12,
            fontweight="bold",
        )

        ax.set_xlabel(
            "Similarity score"
        )

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=color,
        )
        for color in (RED, DARK)
    ]

    fig.legend(
        handles,
        ["Movie", "TV Show"],
        loc="lower center",
        ncol=2,
        frameon=False,
    )

    fig.tight_layout(
        rect=(0, 0.03, 1, 1)
    )

    fig.savefig(
        OUT_DIR / "rec_02_sample_recommendations.png",
        dpi=150,
    )

    plt.close(fig)


print(
    f"\nDone. Outputs saved to: {OUT_DIR.resolve()}"
)
