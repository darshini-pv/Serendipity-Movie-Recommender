
# ============================================================
# STAGE 16: FINAL MODEL VERIFICATION AND BASELINE COMPARISON
# ============================================================
# Project: Serendipitous Novelty-Aware Movie Recommendation
# Dataset: MovieLens 1M
#
# Purpose:
# 1. Reproduce the exact Stage 14 data split.
# 2. Reproduce the exact Stage 14 final-test user sample.
# 3. Compare baseline and selected model on identical users.
# 4. Investigate score ties and recommendation coverage.
# 5. Save final verification results.
#
# Baseline: lambda=0, novelty weight=0
# Selected: lambda=10, novelty weight=0.005
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = (
    r"C:\Users\manoj\OneDrive\Desktop"
    r"\Serendipitous Recommendation Engine"
)

DATA_DIR = os.path.join(PROJECT_DIR, "data")

RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
MOVIES_PATH = os.path.join(DATA_DIR, "movies.csv")

TOP_N = 10
MAX_NEIGHBORS = 50

N_TEST_USERS = 500

SPLIT_SEED = 42
HOLDOUT_SPLIT_SEED = 43
TEST_USER_SEED = 2027

SELECTED_LAMBDA = 10.0
SELECTED_NOVELTY_WEIGHT = 0.005

OUTPUT_PREFIX = os.path.join(
    PROJECT_DIR,
    "stage16"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 75)
print("STAGE 16: FINAL MODEL VERIFICATION")
print("=" * 75)

print("\nSTEP 1: Loading datasets...")

ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)

ratings = ratings[
    ["userId", "movieId", "rating"]
].copy()

movies = movies.drop_duplicates(
    subset="movieId"
).copy()

print("Ratings:", ratings.shape)
print("Movies:", movies.shape)


# ============================================================
# 3. REPRODUCE THE EXACT STAGE 14 SPLIT
# ============================================================

print("\nSTEP 2: Reproducing the exact Stage 14 split...")

train_df, holdout_df = train_test_split(
    ratings,
    test_size=0.20,
    random_state=SPLIT_SEED
)

validation_df, test_df = train_test_split(
    holdout_df,
    test_size=0.50,
    random_state=HOLDOUT_SPLIT_SEED
)

train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Training ratings:", len(train_df))
print("Validation ratings:", len(validation_df))
print("Final test ratings:", len(test_df))

train_pairs = set(
    zip(train_df["userId"], train_df["movieId"])
)

validation_pairs = set(
    zip(validation_df["userId"], validation_df["movieId"])
)

test_pairs = set(
    zip(test_df["userId"], test_df["movieId"])
)

train_validation_overlap = len(
    train_pairs & validation_pairs
)

train_test_overlap = len(
    train_pairs & test_pairs
)

validation_test_overlap = len(
    validation_pairs & test_pairs
)

print("Train-validation overlap:", train_validation_overlap)
print("Train-test overlap:", train_test_overlap)
print("Validation-test overlap:", validation_test_overlap)

if (
    train_validation_overlap
    or train_test_overlap
    or validation_test_overlap
):
    raise RuntimeError(
        "Overlapping user-movie rating pairs detected."
    )


# ============================================================
# 4. BUILD THE TRAINING MATRIX
# ============================================================

print("\nSTEP 3: Building the training matrix...")

rating_matrix = train_df.pivot_table(
    index="userId",
    columns="movieId",
    values="rating",
    aggfunc="mean"
)

user_means = rating_matrix.mean(axis=1)

centered_matrix = rating_matrix.sub(
    user_means,
    axis=0
)

centered_filled = centered_matrix.fillna(0)

user_ids = rating_matrix.index.to_numpy()
movie_ids = rating_matrix.columns.to_numpy()

rating_values = rating_matrix.to_numpy()
rated_mask = np.isfinite(rating_values)

user_to_index = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}

movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(movie_ids)
}

print("Matrix shape:", rating_matrix.shape)
print("Training users:", len(user_ids))
print("Training movies:", len(movie_ids))
print("Training mean:", round(train_df["rating"].mean(), 4))


# ============================================================
# 5. CALCULATE USER SIMILARITIES
# ============================================================

print("\nSTEP 4: Calculating user similarities...")

similarity_values = cosine_similarity(
    centered_filled.to_numpy()
)

np.fill_diagonal(similarity_values, 0.0)

similarity_df = pd.DataFrame(
    similarity_values,
    index=rating_matrix.index,
    columns=rating_matrix.index
)

print("Similarity matrix:", similarity_values.shape)


# ============================================================
# 6. CALCULATE NOVELTY USING TRAINING DATA ONLY
# ============================================================

print("\nSTEP 5: Calculating movie novelty...")

unique_raters = (
    train_df.groupby("movieId")["userId"]
    .nunique()
    .reindex(movie_ids)
    .fillna(0)
)

number_of_users = len(user_ids)

popularity = unique_raters / number_of_users

safe_popularity = popularity.clip(
    lower=1 / number_of_users,
    upper=1.0
)

novelty_series = (
    -np.log2(safe_popularity)
    / np.log2(number_of_users)
).clip(0, 1)

movie_novelty = pd.Series(
    novelty_series.to_numpy(),
    index=movie_ids
)

print("Mean novelty:", round(movie_novelty.mean(), 4))


# ============================================================
# 7. PREPARE RELEVANT TEST MOVIES
# ============================================================

print("\nSTEP 6: Preparing held-out relevant movies...")

relevant_df = test_df[
    (test_df["rating"] >= 4)
    & test_df["userId"].isin(rating_matrix.index)
    & test_df["movieId"].isin(rating_matrix.columns)
]

test_relevant = (
    relevant_df.groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

eligible_test_users = np.array(
    list(test_relevant.keys())
)

rng_test = np.random.default_rng(TEST_USER_SEED)

test_users = rng_test.choice(
    eligible_test_users,
    size=min(N_TEST_USERS, len(eligible_test_users)),
    replace=False
).tolist()

print("Eligible test users:", len(eligible_test_users))
print("Selected test users:", len(test_users))


# ============================================================
# 8. SELECT NEIGHBORS
# ============================================================

def get_neighbors(user_id):

    similarities = similarity_df.loc[user_id].drop(
        user_id
    )

    similarities = similarities[
        similarities > 0
    ]

    return similarities.sort_values(
        ascending=False,
        kind="mergesort"
    ).head(MAX_NEIGHBORS)


# ============================================================
# 9. PREDICT RATINGS FOR UNSEEN MOVIES
# ============================================================

def predict_all_movies(user_id, regularization):

    user_ratings = rating_matrix.loc[user_id]
    user_mean = float(user_means.loc[user_id])

    candidate_mask = user_ratings.isna()

    similarities = get_neighbors(user_id)

    if similarities.empty:

        predictions = pd.Series(
            user_mean,
            index=rating_matrix.columns,
            dtype=float
        )

    else:

        neighbor_ids = similarities.index
        weights = similarities.to_numpy()

        neighbor_ratings = rating_matrix.loc[
            neighbor_ids
        ]

        neighbor_centered = centered_matrix.loc[
            neighbor_ids
        ]

        observed_mask = (
            neighbor_ratings.notna().astype(float)
        )

        numerator = (
            neighbor_centered.fillna(0)
            .mul(weights, axis=0)
            .sum(axis=0)
        )

        denominator = (
            observed_mask
            .mul(weights, axis=0)
            .sum(axis=0)
        )

        if regularization > 0:

            deviations = numerator.div(
                denominator + regularization
            )

        else:

            deviations = numerator.div(
                denominator.replace(0, np.nan)
            )

        predictions = user_mean + deviations

        predictions = predictions.fillna(user_mean)

    predictions = predictions.clip(1, 5)

    return predictions.loc[candidate_mask]


# ============================================================
# 10. CREATE TOP-10 RECOMMENDATIONS
# ============================================================

def make_recommendations(
    user_id,
    regularization,
    novelty_weight
):

    predictions = predict_all_movies(
        user_id,
        regularization
    )

    recommendations = pd.DataFrame({
        "movieId": predictions.index.to_numpy(),
        "predicted_rating": predictions.to_numpy()
    })

    recommendations["novelty"] = (
        recommendations["movieId"]
        .map(movie_novelty)
        .fillna(0.0)
        .to_numpy()
    )

    recommendations["ranking_score"] = (
        (1 - novelty_weight)
        * ((recommendations["predicted_rating"] - 1) / 4)
        + novelty_weight * recommendations["novelty"]
    )

    recommendations = recommendations.sort_values(
        ["ranking_score", "movieId"],
        ascending=[False, True],
        kind="mergesort"
    ).reset_index(drop=True)

    return recommendations


# ============================================================
# 11. METRIC CALCULATION
# ============================================================

def calculate_metrics(recommendations, relevant_movies):

    top_recommendations = recommendations.head(TOP_N)

    recommended_ids = (
        top_recommendations["movieId"].tolist()
    )

    hits = [
        int(movie_id in relevant_movies)
        for movie_id in recommended_ids
    ]

    precision = sum(hits) / TOP_N

    recall = (
        sum(hits) / len(relevant_movies)
        if relevant_movies else 0.0
    )

    dcg = sum(
        hit / np.log2(position + 2)
        for position, hit in enumerate(hits)
    )

    ideal_count = min(
        len(relevant_movies),
        TOP_N
    )

    idcg = sum(
        1 / np.log2(position + 2)
        for position in range(ideal_count)
    )

    ndcg = dcg / idcg if idcg > 0 else 0.0

    average_novelty = (
        top_recommendations["novelty"].mean()
    )

    return {
        "precision_at_10": precision,
        "recall_at_10": recall,
        "ndcg_at_10": ndcg,
        "average_novelty": average_novelty
    }


# ============================================================
# 12. COMPARE BASELINE AND SELECTED MODEL
# ============================================================

print("\nSTEP 7: Evaluating both models on identical users...")

configurations = [
    {
        "model": "Mean-centered CF baseline",
        "regularization": 0.0,
        "novelty_weight": 0.0
    },
    {
        "model": "Selected regularized CF",
        "regularization": SELECTED_LAMBDA,
        "novelty_weight": SELECTED_NOVELTY_WEIGHT
    }
]

result_rows = []
user_metric_rows = []
recommendation_rows = []
tie_rows = []

for user_position, user_id in enumerate(
    test_users, start=1
):

    relevant_movies = test_relevant[user_id]

    if user_position % 50 == 0:
        print(
            "Evaluated users:",
            user_position,
            "of",
            len(test_users)
        )

    for config in configurations:

        recs = make_recommendations(
            user_id,
            config["regularization"],
            config["novelty_weight"]
        )

        metrics = calculate_metrics(
            recs,
            relevant_movies
        )

        top10 = recs.head(TOP_N)

        user_metric_rows.append({
            "userId": user_id,
            "model": config["model"],
            "regularization": config["regularization"],
            "novelty_weight": config["novelty_weight"],
            "number_of_relevant_test_movies": len(
                relevant_movies
            ),
            **metrics
        })

        tie_rows.append({
            "userId": user_id,
            "model": config["model"],
            "top10_unique_scores": int(
                top10["ranking_score"].round(8).nunique()
            ),
            "top10_tie_present": (
                top10["ranking_score"].round(8).nunique()
                < len(top10)
            ),
            "candidate_predictions_at_5_fraction": float(
                np.isclose(
                    recs["predicted_rating"],
                    5.0
                ).mean()
            ),
            "top10_score_range": float(
                top10["ranking_score"].max()
                - top10["ranking_score"].min()
            )
        })

        for rank, row in enumerate(
            top10.itertuples(index=False),
            start=1
        ):

            recommendation_rows.append({
                "userId": user_id,
                "model": config["model"],
                "rank": rank,
                "movieId": row.movieId,
                "predicted_rating": row.predicted_rating,
                "novelty": row.novelty,
                "ranking_score": row.ranking_score,
                "relevant_in_test": (
                    row.movieId in relevant_movies
                )
            })


# ============================================================
# 13. AGGREGATE RESULTS
# ============================================================

print("\nSTEP 8: Aggregating final test results...")

user_metrics_df = pd.DataFrame(user_metric_rows)

recommendations_df = pd.DataFrame(
    recommendation_rows
)

ties_df = pd.DataFrame(tie_rows)

summary_rows = []

for config in configurations:

    model_name = config["model"]

    model_metrics = user_metrics_df[
        user_metrics_df["model"] == model_name
    ]

    model_recs = recommendations_df[
        recommendations_df["model"] == model_name
    ]

    distinct_movies = model_recs["movieId"].nunique()

    summary_rows.append({
        "model": model_name,
        "regularization": config["regularization"],
        "novelty_weight": config["novelty_weight"],
        "users_evaluated": model_metrics["userId"].nunique(),
        "precision_at_10": model_metrics[
            "precision_at_10"
        ].mean(),
        "recall_at_10": model_metrics[
            "recall_at_10"
        ].mean(),
        "ndcg_at_10": model_metrics[
            "ndcg_at_10"
        ].mean(),
        "average_novelty": model_metrics[
            "average_novelty"
        ].mean(),
        "catalog_coverage": (
            distinct_movies / len(movie_ids)
        ),
        "distinct_recommended_movies": distinct_movies,
        "users_with_at_least_one_hit": int(
            (
                model_metrics["precision_at_10"] > 0
            ).sum()
        )
    })

summary_df = pd.DataFrame(summary_rows)

tie_summary_df = (
    ties_df.groupby("model")
    .agg(
        users_evaluated=("userId", "nunique"),
        users_with_top10_ties=("top10_tie_present", "sum"),
        mean_unique_top10_scores=("top10_unique_scores", "mean"),
        mean_top10_score_range=("top10_score_range", "mean"),
        mean_candidate_predictions_at_5=(
            "candidate_predictions_at_5_fraction",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# 14. PAIRED USER-LEVEL COMPARISON
# ============================================================

print("\nSTEP 9: Comparing the same users...")

paired = user_metrics_df.pivot(
    index="userId",
    columns="model",
    values="ndcg_at_10"
)

baseline_name = "Mean-centered CF baseline"
selected_name = "Selected regularized CF"

paired["ndcg_difference_selected_minus_baseline"] = (
    paired[selected_name] - paired[baseline_name]
)

paired_summary = pd.DataFrame([{
    "users_compared": len(paired),
    "mean_ndcg_difference": paired[
        "ndcg_difference_selected_minus_baseline"
    ].mean(),
    "users_selected_model_better": int(
        (
            paired["ndcg_difference_selected_minus_baseline"]
            > 0
        ).sum()
    ),
    "users_equal_ndcg": int(
        (
            paired["ndcg_difference_selected_minus_baseline"]
            == 0
        ).sum()
    ),
    "users_baseline_better": int(
        (
            paired["ndcg_difference_selected_minus_baseline"]
            < 0
        ).sum()
    )
}])


# ============================================================
# 15. SAVE OUTPUT FILES
# ============================================================

print("\nSTEP 10: Saving results...")

summary_df.to_csv(
    OUTPUT_PREFIX + "_model_comparison.csv",
    index=False
)

user_metrics_df.to_csv(
    OUTPUT_PREFIX + "_user_level_metrics.csv",
    index=False
)

recommendations_df.to_csv(
    OUTPUT_PREFIX + "_example_recommendations.csv",
    index=False
)

ties_df.to_csv(
    OUTPUT_PREFIX + "_score_tie_diagnostics.csv",
    index=False
)

tie_summary_df.to_csv(
    OUTPUT_PREFIX + "_score_tie_summary.csv",
    index=False
)

paired.to_csv(
    OUTPUT_PREFIX + "_paired_user_ndcg.csv"
)

paired_summary.to_csv(
    OUTPUT_PREFIX + "_paired_comparison_summary.csv",
    index=False
)


# ============================================================
# 16. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST MODEL COMPARISON")
print("=" * 75)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.5f}"
    )
)

print("\nSCORE-TIE SUMMARY")

print(
    tie_summary_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.5f}"
    )
)

print("\nPAIRED USER-LEVEL NDCG COMPARISON")

print(
    paired_summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.5f}"
    )
)

print("\nOUTPUT FILES:")
print(OUTPUT_PREFIX + "_model_comparison.csv")
print(OUTPUT_PREFIX + "_user_level_metrics.csv")
print(OUTPUT_PREFIX + "_example_recommendations.csv")
print(OUTPUT_PREFIX + "_score_tie_diagnostics.csv")
print(OUTPUT_PREFIX + "_score_tie_summary.csv")
print(OUTPUT_PREFIX + "_paired_user_ndcg.csv")
print(OUTPUT_PREFIX + "_paired_comparison_summary.csv")

print("\n" + "=" * 75)
print("STAGE 16 COMPLETED")
print("=" * 75)
