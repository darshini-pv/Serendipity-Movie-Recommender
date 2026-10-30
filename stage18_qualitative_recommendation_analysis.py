
# ================================================================
# STAGE 18: QUALITATIVE RECOMMENDATION ANALYSIS
# Project: Serendipitous Novelty-Aware Movie Recommendation System
# ================================================================

import os
import numpy as np
import pandas as pd

print("=" * 75)
print("STAGE 18: QUALITATIVE RECOMMENDATION ANALYSIS")
print("=" * 75)

# ------------------------------------------------
# STEP 1: LOAD STAGE 17 OUTPUTS
# ------------------------------------------------

print("\nSTEP 1: Loading Stage 17 results...")

PROJECT_DIR = (
    r"C:\Users\manoj\OneDrive\Desktop"
    r"\Serendipitous Recommendation Engine"
)

metrics_path = os.path.join(
    PROJECT_DIR,
    "stage17_user_level_metrics.csv"
)

recommendations_path = os.path.join(
    PROJECT_DIR,
    "stage17_recommendation_details.csv"
)

if not os.path.exists(metrics_path):
    raise FileNotFoundError(
        "Missing stage17_user_level_metrics.csv. "
        "Run Stage 17 first."
    )

if not os.path.exists(recommendations_path):
    raise FileNotFoundError(
        "Missing stage17_recommendation_details.csv. "
        "Run Stage 17 first."
    )

user_metrics = pd.read_csv(metrics_path)
recommendations = pd.read_csv(recommendations_path)

print("User metrics shape:", user_metrics.shape)
print("Recommendation details shape:", recommendations.shape)

# ------------------------------------------------
# STEP 2: VALIDATE DATA
# ------------------------------------------------

print("\nSTEP 2: Validating recommendation records...")

required_columns = [
    "userId",
    "rank",
    "movieId",
    "title",
    "genres",
    "predicted_rating",
    "novelty",
    "unexpectedness",
    "heldout_relevant"
]

missing_columns = [
    column for column in required_columns
    if column not in recommendations.columns
]

if missing_columns:
    raise ValueError(
        f"Missing recommendation columns: {missing_columns}"
    )

recommendations = recommendations.drop_duplicates(
    subset=["userId", "rank"]
).copy()

recommendations["rank"] = pd.to_numeric(
    recommendations["rank"],
    errors="coerce"
)

recommendations["predicted_rating"] = pd.to_numeric(
    recommendations["predicted_rating"],
    errors="coerce"
)

recommendations["novelty"] = pd.to_numeric(
    recommendations["novelty"],
    errors="coerce"
)

recommendations["unexpectedness"] = pd.to_numeric(
    recommendations["unexpectedness"],
    errors="coerce"
)

recommendations["heldout_relevant"] = pd.to_numeric(
    recommendations["heldout_relevant"],
    errors="coerce"
).fillna(0).astype(int)

print("Valid recommendation records:", len(recommendations))
print("Unique users:", recommendations["userId"].nunique())
print("Unique movies:", recommendations["movieId"].nunique())

# ------------------------------------------------
# STEP 3: SELECT REPRESENTATIVE USERS
# ------------------------------------------------

print("\nSTEP 3: Selecting representative users...")

# User-level metrics already contain results from the same
# held-out evaluation performed in Stage 17.

user_metrics = user_metrics.sort_values(
    "userId"
).copy()

# Users with the strongest ranking quality.
best_user = user_metrics.loc[
    user_metrics["ndcg_at_10"].idxmax()
]

# Users with no relevant hits.
no_hit_users = user_metrics[
    user_metrics["relevant_hits"] == 0
]

# Users with at least one relevant hit.
hit_users = user_metrics[
    user_metrics["relevant_hits"] > 0
]

# A user whose NDCG is closest to the median.
median_ndcg = user_metrics["ndcg_at_10"].median()

typical_user = user_metrics.loc[
    (user_metrics["ndcg_at_10"] - median_ndcg).abs().idxmin()
]

selected_user_rows = [best_user, typical_user]

if not no_hit_users.empty:
    selected_user_rows.append(
        no_hit_users.iloc[0]
    )

# Remove duplicate user IDs while preserving selection order.
selected_users = []

for row in selected_user_rows:
    user_id = int(row["userId"])

    if user_id not in selected_users:
        selected_users.append(user_id)

print("Representative users selected:", selected_users)
print("Best NDCG@10:", round(float(best_user["ndcg_at_10"]), 5))
print("Median user NDCG@10:", round(float(median_ndcg), 5))
print("Users with zero relevant hits:", len(no_hit_users))

# ------------------------------------------------
# STEP 4: DISPLAY EXAMPLE RECOMMENDATIONS
# ------------------------------------------------

print("\nSTEP 4: Inspecting recommendations...")

example_rows = recommendations[
    recommendations["userId"].isin(selected_users)
].copy()

example_rows = example_rows.sort_values(
    ["userId", "rank"]
)

display_columns = [
    "userId",
    "rank",
    "title",
    "genres",
    "predicted_rating",
    "novelty",
    "unexpectedness",
    "heldout_relevant"
]

print("\nREPRESENTATIVE TOP-10 RECOMMENDATIONS")
print(
    example_rows[display_columns]
    .round({
        "predicted_rating": 3,
        "novelty": 4,
        "unexpectedness": 4
    })
    .to_string(index=False)
)

# ------------------------------------------------
# STEP 5: ANALYZE RELEVANT AND NON-RELEVANT MOVIES
# ------------------------------------------------

print("\nSTEP 5: Comparing relevant and non-relevant recommendations...")

relevant_recs = recommendations[
    recommendations["heldout_relevant"] == 1
]

non_relevant_recs = recommendations[
    recommendations["heldout_relevant"] == 0
]

comparison_rows = []

for label, subset in [
    ("Relevant", relevant_recs),
    ("Not relevant in held-out test", non_relevant_recs)
]:
    comparison_rows.append({
        "group": label,
        "recommendation_count": len(subset),
        "mean_predicted_rating": subset["predicted_rating"].mean(),
        "mean_novelty": subset["novelty"].mean(),
        "mean_unexpectedness": subset["unexpectedness"].mean()
    })

comparison_df = pd.DataFrame(comparison_rows)

print("\nRELEVANT VS NON-RELEVANT RECOMMENDATIONS")
print(
    comparison_df.round(4).to_string(index=False)
)

# Note: "Not relevant in held-out test" means the movie was
# not rated >=4 in the held-out ratings. It does not prove
# that the user dislikes the movie.

# ------------------------------------------------
# STEP 6: ANALYZE RANK POSITION
# ------------------------------------------------

print("\nSTEP 6: Analyzing relevance by rank...")

rank_analysis = (
    recommendations.groupby("rank")
    .agg(
        recommendations=("movieId", "count"),
        relevant_hits=("heldout_relevant", "sum"),
        mean_predicted_rating=("predicted_rating", "mean"),
        mean_novelty=("novelty", "mean"),
        mean_unexpectedness=("unexpectedness", "mean")
    )
    .reset_index()
)

rank_analysis["hit_rate"] = (
    rank_analysis["relevant_hits"]
    / rank_analysis["recommendations"]
)

print("\nMETRICS BY RECOMMENDATION RANK")
print(rank_analysis.round(4).to_string(index=False))

# ------------------------------------------------
# STEP 7: ANALYZE GENRE FREQUENCIES
# ------------------------------------------------

print("\nSTEP 7: Analyzing recommended genres...")

genre_counts = {}

for genre_string in recommendations["genres"].fillna(""):
    for genre in genre_string.split("|"):
        genre = genre.strip()

        if genre and genre != "(no genres listed)":
            genre_counts[genre] = (
                genre_counts.get(genre, 0) + 1
            )

genre_df = pd.DataFrame(
    [
        {
            "genre": genre,
            "recommendation_count": count,
            "percentage_of_recommendation_slots":
                100 * count / len(recommendations)
        }
        for genre, count in genre_counts.items()
    ]
)

genre_df = genre_df.sort_values(
    "recommendation_count",
    ascending=False
)

print("\nTOP 10 RECOMMENDED GENRES")
print(
    genre_df.head(10)
    .round(2)
    .to_string(index=False)
)

# ------------------------------------------------
# STEP 8: SUMMARIZE QUALITATIVE FINDINGS
# ------------------------------------------------

print("\nSTEP 8: Summarizing findings...")

total_recommendations = len(recommendations)
total_hits = int(recommendations["heldout_relevant"].sum())

overall_hit_rate = (
    total_hits / total_recommendations
    if total_recommendations else 0
)

print("\nQUALITATIVE ANALYSIS SUMMARY")
print("Total recommendations:", total_recommendations)
print("Held-out relevant hits:", total_hits)
print("Overall hit rate:", round(overall_hit_rate, 4))
print("Users analyzed:", user_metrics["userId"].nunique())
print("Distinct recommended movies:",
      recommendations["movieId"].nunique())

print("\nINTERPRETATION GUIDANCE")
print("1. Check whether relevant movies appear near the top.")
print("2. Review whether unexpected genres still produce relevant hits.")
print("3. Check whether recommendations concentrate on a few genres.")
print("4. Treat held-out ratings as relevance evidence, not direct user feedback.")
print("5. Do not label a movie truly serendipitous without stronger evidence.")

# ------------------------------------------------
# STEP 9: SAVE OUTPUT FILES
# ------------------------------------------------

print("\nSTEP 9: Saving Stage 18 outputs...")

example_rows[display_columns].to_csv(
    os.path.join(
        PROJECT_DIR,
        "stage18_representative_recommendations.csv"
    ),
    index=False
)

comparison_df.to_csv(
    os.path.join(
        PROJECT_DIR,
        "stage18_relevant_vs_nonrelevant.csv"
    ),
    index=False
)

rank_analysis.to_csv(
    os.path.join(
        PROJECT_DIR,
        "stage18_rank_analysis.csv"
    ),
    index=False
)

genre_df.to_csv(
    os.path.join(
        PROJECT_DIR,
        "stage18_genre_analysis.csv"
    ),
    index=False
)

print("Saved:")
print("stage18_representative_recommendations.csv")
print("stage18_relevant_vs_nonrelevant.csv")
print("stage18_rank_analysis.csv")
print("stage18_genre_analysis.csv")

print("\n" + "=" * 75)
print("STAGE 18 COMPLETED")
print("=" * 75)
