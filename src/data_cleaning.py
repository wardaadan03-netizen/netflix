"""
Task 1 - Data Cleaning & Preprocessing (Netflix dataset)

"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = "data/raw/Dataset.csv"
OUT_PATH = Path("data/processed/netflix_cleaned.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_PATH)

# ---------------------------------------------------------------------------
# STEP 1: Import the dataset
# ---------------------------------------------------------------------------
df = pd.read_csv(r"data\raw\Dataset.csv")
raw_shape = df.shape
print("STEP 1 | Loaded:", raw_shape)
print(df.dtypes, "\n")

# ---------------------------------------------------------------------------
# STEP 2: Identify missing and duplicate records
# ---------------------------------------------------------------------------
print("STEP 2 | Literal nulls per column:")
print(df.isna().sum(), "\n")

# Nulls can hide behind placeholder text, so look for those too
placeholders = {"Not Given", "Unknown", "N/A", "NA", "None", "", "-"}
for col in df.columns:
    if col == "release_year":
        continue
    n = df[col].astype(str).str.strip().isin(placeholders).sum()
    if n:
        print(f"  placeholder values in '{col}': {n}")
# (the single 'title' hit is the real 2011 film "Unknown" - a false positive)

print("\n  exact duplicate rows         :", df.duplicated().sum())
print("  duplicate show_id            :", df["show_id"].duplicated().sum())
content_cols = [c for c in df.columns if c != "show_id"]
dup_mask = df.duplicated(subset=content_cols, keep="first")
print("  duplicates ignoring show_id  :", dup_mask.sum())
print(df[df.duplicated(subset=content_cols, keep=False)]
      .sort_values("title")[["show_id", "title", "type", "release_year"]], "\n")

# ---------------------------------------------------------------------------
# STEP 3: Handle null values and inconsistent formats
# ---------------------------------------------------------------------------
# 3a. Remove duplicate records (same content, different show_id)
df = df.drop_duplicates(subset=content_cols, keep="first").copy()

# 3b. Trim stray whitespace (incl. non-breaking spaces) in every text column
for col in ["show_id", "type", "title", "director", "country", "rating",
            "duration", "listed_in", "date_added"]:
    df[col] = (df[col].astype(str)
                      .str.replace("\u00a0", " ", regex=False)
                      .str.strip())

# 3c. Placeholder "Not Given" -> proper null -> explicit "Unknown" label
for col in ["director", "country"]:
    df[col] = df[col].replace("Not Given", np.nan)
    df[col + "_missing"] = df[col].isna()        # keep a flag for analysis
    df[col] = df[col].fillna("Unknown")

# 3d. Standardise text casing / labels
df["type"] = df["type"].str.title().replace({"Tv Show": "TV Show"})
df["rating"] = df["rating"].str.upper()

# ---------------------------------------------------------------------------
# STEP 4: Transform categorical and date-related columns
# ---------------------------------------------------------------------------
# --- Dates -----------------------------------------------------------------
df["date_added"] = pd.to_datetime(df["date_added"], format="%m/%d/%Y")
df["year_added"] = df["date_added"].dt.year
df["month_added"] = df["date_added"].dt.month
df["month_name"] = df["date_added"].dt.month_name()
df["day_of_week"] = df["date_added"].dt.day_name()
df["years_to_netflix"] = df["year_added"] - df["release_year"]   # release -> added gap

# --- Duration: "90 min" / "2 Seasons" -> numeric columns --------------------
dur = df["duration"].str.extract(r"(?P<value>\d+)\s*(?P<unit>[A-Za-z]+)")
df["duration_value"] = dur["value"].astype(int)
df["duration_unit"] = dur["unit"].str.lower().replace({"seasons": "season"})
df["duration_minutes"] = np.where(df["type"] == "Movie", df["duration_value"], np.nan)
df["seasons"] = np.where(df["type"] == "TV Show", df["duration_value"], np.nan)

# --- Multi-valued categorical columns --------------------------------------
df["genres_list"] = df["listed_in"].str.split(",").apply(lambda xs: [x.strip() for x in xs])
df["primary_genre"] = df["genres_list"].str[0]
df["num_genres"] = df["genres_list"].str.len()
df["num_directors"] = np.where(df["director"] == "Unknown", 0,
                               df["director"].str.count(",") + 1)

# --- Rating -> audience group ----------------------------------------------
rating_map = {
    "TV-Y": "Kids", "TV-Y7": "Kids", "TV-Y7-FV": "Kids", "TV-G": "Kids", "G": "Kids",
    "TV-PG": "Teens", "PG": "Teens", "TV-14": "Teens", "PG-13": "Teens",
    "TV-MA": "Adults", "R": "Adults", "NC-17": "Adults",
    "NR": "Unrated", "UR": "Unrated",
}
df["audience_group"] = df["rating"].map(rating_map).fillna("Unrated")

# --- Memory-friendly categorical dtypes ------------------------------------
for col in ["type", "rating", "audience_group", "duration_unit",
            "primary_genre", "country", "month_name", "day_of_week"]:
    df[col] = df[col].astype("category")

# ---------------------------------------------------------------------------
# STEP 5: Create the clean dataset
# ---------------------------------------------------------------------------
# Sanity checks (fail loudly if something slipped through)
assert df["show_id"].is_unique
assert df.drop(columns=["genres_list"]).isna().sum().drop(["duration_minutes", "seasons"]).sum() == 0
assert df["date_added"].notna().all()

df = df.sort_values("date_added", ascending=False).reset_index(drop=True)
df["genres"] = df["genres_list"].str.join("|")       # CSV-friendly version of the list
df = df.drop(columns=["genres_list"])

ordered = ["show_id", "type", "title", "director", "num_directors", "director_missing",
           "country", "country_missing", "date_added", "year_added", "month_added",
           "month_name", "day_of_week", "release_year", "years_to_netflix",
           "rating", "audience_group", "duration", "duration_value", "duration_unit",
           "duration_minutes", "seasons", "listed_in", "genres", "primary_genre", "num_genres"]
df = df[ordered]
df.to_csv(OUT_PATH, index=False)

print("\nSTEP 5 | Raw shape  :", raw_shape)
print("         Clean shape:", df.shape)
print("         Saved ->", OUT_PATH)
print(df.head())
print(df.dtypes)