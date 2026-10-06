"""
Task 5 - Machine Learning Classification Model (Netflix dataset)
"""
import re
import sys
import time
import unicodedata
import warnings
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/netflix_cleaned.csv")
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("classification_output")
TARGET = sys.argv[3] if len(sys.argv) > 3 else "audience"
assert TARGET in ("audience", "type"), "target must be 'audience' or 'type'"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TOP_COUNTRIES, TOP_WORDS, N_BOOT = 20, 100, 1000
RED, DARK, GREY = "#E50914", "#221F1F", "#B3B3B3"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110, "savefig.bbox": "tight"})
rng = np.random.default_rng(SEED)
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)


def normalize_text(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", s.lower())).strip()


STOP = set(ENGLISH_STOP_WORDS) | {"tv", "movie", "film", "series", "season", "part", "vol"}


def title_words(t):
    return {w for w in normalize_text(t).split() if w not in STOP and len(w) >= 3 and not w.isdigit()}


def new_hgb(**kw):
    try:
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED, **kw)
    except TypeError:                       # older scikit-learn without class_weight
        return HistGradientBoostingClassifier(random_state=SEED, **kw)


# ===========================================================================
# STEP 1: Select and prepare features
# ===========================================================================
df = pd.read_csv(DATA_PATH)
df["title_key"] = df["title"].map(normalize_text)
n_raw = len(df)
df = df.drop_duplicates(subset=["title_key", "type", "release_year"]).reset_index(drop=True)
print(f"Rows: {n_raw:,} -> {len(df):,} after removing {n_raw - len(df)} near-duplicate titles "
      "(so the same title can't land in both train and test).")

if TARGET == "audience":
    n_unrated = int((df["audience_group"] == "Unrated").sum())
    df = df[df["audience_group"] != "Unrated"].reset_index(drop=True)
    y = df["audience_group"]
    print(f"Dropped {n_unrated} 'Unrated' titles (too few to learn). Classes: Kids / Teens / Adults.")
    print("Excluded on purpose: `rating` (the target is derived from it), title/show_id (identifiers),\n"
          "director names (~4,500 unique values), raw date_added (its parts are used instead).")
else:
    y = df["type"]
    print("Excluded on purpose: genres + num_genres (every one of the 42 genre labels belongs to exactly\n"
          "one type, so they give the answer away), duration/seasons (they define the type).")
majority = y.value_counts(normalize=True)
print("\nClass balance:\n" + (majority * 100).round(1).astype(str).add("%").to_string())
print(f"Majority-class baseline accuracy: {majority.max():.1%}")


def fit_vocab(d):
    cnt = Counter(w for t in d["title"] for w in title_words(t))
    return {"genres": sorted({g for s in d["genres"] for g in s.split("|")}),
            "countries": d.loc[d["country"] != "Unknown", "country"].value_counts().head(TOP_COUNTRIES).index.tolist(),
            "ratings": sorted(d["rating"].unique()),
            "words": [w for w, _ in cnt.most_common(TOP_WORDS)]}


def build_features(d, V):
    cols, groups = {}, {}

    def reg(group, frame):
        groups.setdefault(group, []).extend(frame.columns)
        for c in frame.columns:
            cols[c] = frame[c].to_numpy()

    reg("dates", d[["release_year", "year_added", "month_added", "years_to_netflix"]])
    reg("director info", pd.DataFrame({"director_missing": d["director_missing"].astype(int),
                                       "num_directors": d["num_directors"]}, index=d.index))
    ctry = d["country"].where(d["country"].isin(V["countries"]), "Other")
    reg("country", pd.DataFrame({**{f"country_{c}": (ctry == c).astype(int) for c in V["countries"] + ["Other"]},
                                 "country_missing": d["country_missing"].astype(int)}, index=d.index))
    tok = [title_words(t) for t in d["title"]]
    reg("title words", pd.DataFrame({f"word_{w}": [int(w in s) for s in tok] for w in V["words"]}, index=d.index))
    if TARGET == "audience":
        reg("type & duration", pd.DataFrame({"is_tv_show": (d["type"] == "TV Show").astype(int),
                                             "duration_minutes": d["duration_minutes"].fillna(0),
                                             "seasons": d["seasons"].fillna(0)}, index=d.index))
        gl = d["genres"].str.split("|")
        reg("genres", pd.DataFrame({**{f"genre_{g}": gl.map(lambda l, g=g: int(g in l)) for g in V["genres"]},
                                    "num_genres": d["num_genres"]}, index=d.index))
    else:
        reg("rating", pd.DataFrame({f"rating_{r}": (d["rating"] == r).astype(int) for r in V["ratings"]}, index=d.index))
    return pd.DataFrame(cols, index=d.index).astype(float), groups


# ===========================================================================
# STEP 2: Split data into training and testing sets
# ===========================================================================

tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED)
d_tr, d_te = df.iloc[tr_idx], df.iloc[te_idx]
ytr, yte = y.iloc[tr_idx].to_numpy(), y.iloc[te_idx].to_numpy()
VOCAB = fit_vocab(d_tr)                              # vocabularies come from the training set only
Xtr, GROUPS = build_features(d_tr, VOCAB)
Xte, _ = build_features(d_te, VOCAB)
print(f"Stratified 80/20 split: train {len(Xtr):,} | test {len(Xte):,} | features {Xtr.shape[1]}")
print("Vocabularies (top countries, title words) are learned on the training set only; the test set\n"
      "is not touched until the final evaluation. Feature selection below uses training data only.")
print("\nFeature groups:")
for g, cs in GROUPS.items():
    print(f"   {g:16s}: {len(cs):3d} columns")

# ---- Feature selection: group ablation with 5-fold CV on the training set -----------------


def cv_group_score(group_names):
    cols = [c for g in group_names for c in GROUPS[g]]
    s = cross_val_score(new_hgb(max_iter=60, learning_rate=0.15, max_bins=64), Xtr[cols], ytr, cv=CV,
                        scoring="f1_macro", n_jobs=-1)
    return s.mean(), s.std(ddof=1)


names = list(GROUPS)
full_m, full_s = cv_group_score(names)
se_full = full_s / np.sqrt(5)
rows = [{"feature_set": "ALL groups", "macro_f1": full_m, "std": full_s, "delta_vs_all": 0.0}]
for g in names:
    m, s = cv_group_score([n for n in names if n != g])
    rows.append({"feature_set": f"drop '{g}'", "macro_f1": m, "std": s, "delta_vs_all": m - full_m})
for g in names:
    m, s = cv_group_score([g])
    rows.append({"feature_set": f"only '{g}'", "macro_f1": m, "std": s, "delta_vs_all": m - full_m})
abl = pd.DataFrame(rows)
abl.round(4).to_csv(OUT_DIR / f"{TARGET}_feature_group_ablation.csv", index=False)
print(abl.assign(macro_f1=abl.macro_f1.map("{:.3f}".format), std=abl["std"].map("±{:.3f}".format),
                 delta_vs_all=abl.delta_vs_all.map("{:+.3f}".format)).to_string(index=False))

drop_effect = {g: full_m - abl.set_index("feature_set").loc[f"drop '{g}'", "macro_f1"] for g in names}
selected = [g for g in names if drop_effect[g] > se_full]          # keep a group only if dropping it clearly hurts
if selected:
    sel_m, _ = cv_group_score(selected)
    if sel_m < full_m - se_full:
        selected = names
else:
    selected = names
dropped = [g for g in names if g not in selected]
print(f"\nRule: keep a group if removing it lowers CV macro-F1 by more than one standard error ({se_full:.4f}).")
print(f"Kept   : {selected}")
print(f"Removed: {dropped if dropped else 'none'}")
FEATS = [c for g in selected for c in GROUPS[g]]
Xtr_s, Xte_s = Xtr[FEATS], Xte[FEATS]
print(f"Final feature count: {len(FEATS)}")

if TARGET == "type":                 # leakage demonstration for transparency
    gl = df["genres"].str.split("|").iloc[tr_idx]
    leak = pd.DataFrame({f"genre_{g}": gl.map(lambda l, g=g: int(g in l)).to_numpy() for g in VOCAB["genres"]},
                        index=Xtr.index)
    s = cross_val_score(new_hgb(max_iter=60, learning_rate=0.15, max_bins=64), pd.concat([Xtr_s, leak], axis=1), ytr, cv=CV,
                        scoring="accuracy", n_jobs=-1)
    print(f"\nLeakage demo: adding the genre labels lifts CV accuracy to {s.mean():.1%} - "
          "the model would just be reading the answer from the labels.")

# ===========================================================================
# STEP 3: Train classification models
# ===========================================================================

SPECS = {
    "Logistic Regression": (Pipeline([("scale", StandardScaler()),
                                      ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
                            {"clf__C": [0.03, 0.1, 0.3, 1.0]}, -1),
    "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
                      {"max_depth": [4, 6, 8, 12], "min_samples_leaf": [5, 20]}, -1),
    "Random Forest": (RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                             n_jobs=-1, random_state=SEED),
                      {"max_depth": [12, None], "min_samples_leaf": [1, 3]}, 1),
    "Gradient Boosting": (new_hgb(max_bins=64), {"learning_rate": [0.1], "max_depth": [3, 6],
                                                "max_iter": [100, 200]}, 1),
}
fitted = {}
for name, (est, grid, nj) in SPECS.items():
    t0 = time.time()
    gs = GridSearchCV(est, grid, scoring="f1_macro", cv=CV, n_jobs=nj, refit=True).fit(Xtr_s, ytr)
    fitted[name] = gs
    print(f"   {name:20s} CV macro-F1 {gs.best_score_:.3f} | best params {gs.best_params_} | {time.time() - t0:.0f}s")

# ===========================================================================
# STEP 4: Evaluate model performance (held-out test set)
# ===========================================================================

classes = sorted(np.unique(ytr))
preds, probas = {}, {}
dummy = DummyClassifier(strategy="most_frequent").fit(Xtr_s, ytr)
preds["Baseline: majority class"] = dummy.predict(Xte_s)
if TARGET == "type":
    preds["Baseline: no director -> TV Show"] = np.where(d_te["director_missing"].to_numpy(), "TV Show", "Movie")
for name, gs in fitted.items():
    preds[name] = gs.predict(Xte_s)
    probas[name] = gs.predict_proba(Xte_s)


def auc(name):
    if name not in probas:
        return np.nan
    if len(classes) == 2:
        return roc_auc_score(yte == classes[1], probas[name][:, 1])
    return roc_auc_score(yte, probas[name], multi_class="ovr", average="macro", labels=classes)


boot_idx = rng.integers(0, len(yte), size=(N_BOOT, len(yte)))
rows = []
for name, p in preds.items():
    acc_b = (yte[boot_idx] == p[boot_idx]).mean(axis=1)
    rows.append({"model": name, "accuracy": accuracy_score(yte, p),
                 "acc_ci_low": np.percentile(acc_b, 2.5), "acc_ci_high": np.percentile(acc_b, 97.5),
                 "balanced_accuracy": balanced_accuracy_score(yte, p),
                 "macro_f1": f1_score(yte, p, average="macro"),
                 "weighted_f1": f1_score(yte, p, average="weighted"), "roc_auc": auc(name),
                 "cv_macro_f1": fitted[name].best_score_ if name in fitted else np.nan})
res = pd.DataFrame(rows)
res.round(4).to_csv(OUT_DIR / f"{TARGET}_model_comparison.csv", index=False)
show = res.assign(accuracy=res.accuracy.map("{:.3f}".format),
                  acc_95ci=[f"[{a:.3f}, {b:.3f}]" for a, b in zip(res.acc_ci_low, res.acc_ci_high)],
                  balanced_accuracy=res.balanced_accuracy.map("{:.3f}".format),
                  macro_f1=res.macro_f1.map("{:.3f}".format), weighted_f1=res.weighted_f1.map("{:.3f}".format),
                  roc_auc=res.roc_auc.map(lambda v: "-" if np.isnan(v) else f"{v:.3f}"),
                  cv_macro_f1=res.cv_macro_f1.map(lambda v: "-" if np.isnan(v) else f"{v:.3f}"))
print(show[["model", "accuracy", "acc_95ci", "balanced_accuracy", "macro_f1", "roc_auc", "cv_macro_f1"]]
      .to_string(index=False))

# Selected model = best CV macro-F1 (chosen on training data, so the test score stays unbiased)
ranked = sorted(fitted, key=lambda n: fitted[n].best_score_, reverse=True)
best, runner = ranked[0], ranked[1]
best_est = fitted[best].best_estimator_
print(f"\nSelected model (best CV macro-F1): {best}")
rep = classification_report(yte, preds[best], digits=3, output_dict=True)
print(classification_report(yte, preds[best], digits=3))
pd.DataFrame(rep).T.round(4).to_csv(OUT_DIR / f"{TARGET}_classification_report.csv")
cm = confusion_matrix(yte, preds[best], labels=classes)

# ===========================================================================
# STEP 5: Compare model accuracy
# ===========================================================================

b, r_ = boot_idx, None
f1_b = lambda p: np.array([f1_score(yte[i], p[i], average="macro") for i in b])
d_acc = ((yte[b] == preds[best][b]).mean(1) - (yte[b] == preds[runner][b]).mean(1))
d_f1 = f1_b(preds[best]) - f1_b(preds[runner])
print(f"{best} vs {runner} (paired bootstrap, {N_BOOT} resamples of the test set):")
print(f"   accuracy difference  {np.mean(d_acc):+.3f}  95% CI [{np.percentile(d_acc, 2.5):+.3f}, {np.percentile(d_acc, 97.5):+.3f}]")
print(f"   macro-F1 difference  {np.mean(d_f1):+.3f}  95% CI [{np.percentile(d_f1, 2.5):+.3f}, {np.percentile(d_f1, 97.5):+.3f}]")
distinguishable = np.percentile(d_f1, 2.5) > 0

# Permutation importance of the selected model, by feature group and by single feature
base = f1_score(yte, preds[best], average="macro")
grp_imp = {}
for g in selected:
    drops = []
    for _ in range(5):
        Xp = Xte_s.copy()
        perm = rng.permutation(len(Xp))
        Xp[GROUPS[g]] = Xte_s[GROUPS[g]].to_numpy()[perm]
        drops.append(base - f1_score(yte, best_est.predict(Xp), average="macro"))
    grp_imp[g] = np.mean(drops)
feat_imp = {}
for c in FEATS:
    if Xte_s[c].nunique() < 2:
        continue
    drops = []
    for _ in range(3):
        Xp = Xte_s.copy()
        Xp[c] = rng.permutation(Xp[c].to_numpy())
        drops.append(base - f1_score(yte, best_est.predict(Xp), average="macro"))
    feat_imp[c] = np.mean(drops)
fi = pd.Series(feat_imp).sort_values(ascending=False)
fi.round(5).to_csv(OUT_DIR / f"{TARGET}_feature_importance.csv", header=["macro_f1_drop"])
print("\nPermutation importance by group (drop in test macro-F1 when the group is shuffled):")
for g, v in sorted(grp_imp.items(), key=lambda kv: -kv[1]):
    print(f"   {g:16s} {v:+.3f}")
print("Top 10 individual features:")
print(fi.head(10).round(4).to_string())

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
# 1: feature-group ablation
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
dd = abl[abl.feature_set.str.startswith("drop")].copy()
dd["g"] = dd.feature_set.str.extract(r"'(.*)'")[0]
dd = dd.sort_values("delta_vs_all", ascending=False)
axes[0].barh(dd["g"], dd["delta_vs_all"], color=[RED if v < -se_full else GREY for v in dd["delta_vs_all"]])
axes[0].axvline(0, color="grey", lw=0.8)
axes[0].axvline(-se_full, color="grey", lw=0.8, ls=":")
axes[0].set_title("Change in macro-F1 when a group is removed", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Δ macro-F1 vs all features (red = clearly hurts)")
ob = abl[abl.feature_set.str.startswith("only")].copy()
ob["g"] = ob.feature_set.str.extract(r"'(.*)'")[0]
ob = ob.sort_values("macro_f1")
axes[1].barh(ob["g"], ob["macro_f1"], color=DARK)
axes[1].axvline(full_m, color=RED, lw=1.5, ls="--")
axes[1].text(full_m, -0.7, " all features", color=RED, fontsize=9)
axes[1].set_title("Macro-F1 using one group alone", fontsize=12, fontweight="bold")
axes[1].set_xlabel("CV macro-F1")
fig.tight_layout()
fig.savefig(OUT_DIR / f"{TARGET}_01_feature_groups.png", dpi=150)
plt.close(fig)

# 2: model comparison
order = res["model"].tolist()[::-1]
rr = res.set_index("model").loc[order]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
colors = [GREY if m.startswith("Baseline") else (RED if m == best else DARK) for m in order]
axes[0].barh(order, rr["accuracy"], color=colors,
             xerr=[rr["accuracy"] - rr["acc_ci_low"], rr["acc_ci_high"] - rr["accuracy"]],
             error_kw={"ecolor": "#888", "lw": 1})
for y_, v in enumerate(rr["accuracy"]):
    axes[0].text(rr["acc_ci_high"].iloc[y_] + 0.01, y_, f"{v:.3f}", va="center", fontsize=9)
axes[0].set_xlim(0, 1.12)
axes[0].set_title("Test accuracy (95% bootstrap CI)", fontsize=12, fontweight="bold")
axes[1].barh(order, rr["macro_f1"], color=colors)
for y_, v in enumerate(rr["macro_f1"]):
    axes[1].text(v + 0.01, y_, f"{v:.3f}", va="center", fontsize=9)
axes[1].set_xlim(0, 1.12)
axes[1].set_title("Test macro-F1", fontsize=12, fontweight="bold")
fig.suptitle(f"Model comparison - target: {TARGET} (red = selected by CV)", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / f"{TARGET}_02_model_comparison.png", dpi=150)
plt.close(fig)

# 3: confusion matrix
cmn = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(5.8, 5))
im = ax.imshow(cmn, cmap="Reds", vmin=0, vmax=1)
ax.set_xticks(range(len(classes)), classes)
ax.set_yticks(range(len(classes)), classes)
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, f"{cmn[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center",
                color="white" if cmn[i, j] > 0.5 else "black", fontsize=10)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"{best}: confusion matrix (test set)", fontsize=12, fontweight="bold")
ax.grid(False)
fig.tight_layout()
fig.savefig(OUT_DIR / f"{TARGET}_03_confusion_matrix.png", dpi=150)
plt.close(fig)

# 4: importance
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
gi = pd.Series(grp_imp).sort_values()
axes[0].barh(gi.index, gi.values, color=DARK)
axes[0].set_title("Importance by feature group", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Drop in test macro-F1 when shuffled")
top = fi.head(12)[::-1]
axes[1].barh([c.replace("genre_", "genre: ").replace("word_", "title word: ").replace("country_", "country: ")
              for c in top.index], top.values, color=RED)
axes[1].set_title("Top 12 individual features", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Drop in test macro-F1 when shuffled")
fig.tight_layout()
fig.savefig(OUT_DIR / f"{TARGET}_04_importance.png", dpi=150)
plt.close(fig)

joblib.dump({"model": best_est, "features": FEATS, "classes": classes, "vocab": VOCAB, "target": TARGET},
            OUT_DIR / f"{TARGET}_best_model.joblib")

# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------
rb = res.set_index("model")
print(f"1. Baseline accuracy {majority.max():.1%} (always guess '{majority.idxmax()}'); "
      f"{best} reaches {rb.loc[best, 'accuracy']:.1%} accuracy and {rb.loc[best, 'macro_f1']:.3f} macro-F1 on the test set.")
print(f"2. Best vs runner-up ({runner}): "
      + ("the macro-F1 gap is statistically distinguishable." if distinguishable else
         "the gap is within noise - the top models are effectively tied."))
topg = sorted(grp_imp.items(), key=lambda kv: -kv[1])
print("3. Most important feature group(s): " + ", ".join(f"{g} ({v:+.3f})" for g, v in topg[:2]) + ".")
recalls = {c: rep[c]["recall"] for c in classes}
worst = min(recalls, key=recalls.get)
print(f"4. Per-class recall: " + ", ".join(f"{c} {v:.0%}" for c, v in recalls.items()) + f" - weakest: {worst}.")
if dropped:
    print(f"5. Feature selection removed: {dropped}.")
if TARGET == "type":
    rule = rb.loc["Baseline: no director -> TV Show", "accuracy"]
    print(f"5. A one-line rule (no director -> TV Show) already scores {rule:.1%}: the 'type' task is mostly "
          "a missing-data artifact, not a content signal.")
print(f"\nFiles saved in: {OUT_DIR.resolve()}")