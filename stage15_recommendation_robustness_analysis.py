
# ================================================================
# STAGE 15: RECOMMENDATION QUALITY AND ROBUSTNESS ANALYSIS
# Project: Serendipitous Novelty-Aware Movie Recommendation System
# Dataset: MovieLens 1M
# ================================================================

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


# ================================================================
# 1. CONFIGURATION
# ================================================================

PROJECT_DIR = Path(
    r"C:\Users\manoj\OneDrive\Desktop\Serendipitous Recommendation Engine"
)

DATA_DIR = PROJECT_DIR / "data"

RATINGS_PATH = DATA_DIR / "ratings.csv"
MOVIES_PATH = DATA_DIR / "movies.csv"

OUTPUT_DIR = PROJECT_DIR

TOP_N = 10
NEIGHBOR_COUNT = 50

# Selected model from Stage 14
SELECTED_LAMBDA = 10.0
SELECTED_NOVELTY_WEIGHT = 0.005

# Stage 14 split settings
SPLIT_SEED = 42

# Number of users for the different investigations
TIE_DIAGNOSTIC_USERS = 100
EXAMPLE_USERS = 10
QUALITY_USERS = 500
ROBUSTNESS_USERS = 200


# ================================================================
# 2. LOAD DATA
# ================================================================

print("=" * 75)
print("STAGE 15: RECOMMENDATION QUALITY AND ROBUSTNESS ANALYSIS")
print("=" * 75)

print("\nSTEP 1: Loading datasets...")

ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)

ratings.columns = ratings.columns.str.strip()
movies.columns = movies.columns.str.strip()

required_rating_columns = {"userId", "movieId", "rating"}
required_movie_columns = {"movieId", "title"}

if not required_rating_columns.issubset(ratings.columns):
    raise ValueError(
        f"ratings.csv must contain: {required_rating_columns}"
    )

if not required_movie_columns.issubset(movies.columns):
    raise ValueError(
        f"movies.csv must contain: {required_movie_columns}"
    )

ratings["userId"] = pd.to_numeric(
    ratings["userId"], errors="raise"
).astype(int)

ratings["movieId"] = pd.to_numeric(
    ratings["movieId"], errors="raise"
).astype(int)

ratings["rating"] = pd.to_numeric(
    ratings["rating"], errors="raise"
)

movies["movieId"] = pd.to_numeric(
    movies["movieId"], errors="raise"
).astype(int)

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)
print("Unique users:", ratings["userId"].nunique())
print("Unique rated movies:", ratings["movieId"].nunique())


# ================================================================
# 3. RECREATE THE STAGE 14 TRAIN / VALIDATION / TEST SPLIT
# ================================================================

print("\nSTEP 2: Recreating the Stage 14 data split...")

train_data, temporary_data = train_test_split(
    ratings,
    test_size=0.20,
    random_state=SPLIT_SEED
)

validation_data, test_data = train_test_split(
    temporary_data,
    test_size=0.50,
    random_state=SPLIT_SEED
)

train_pairs = set(
    zip(train_data["userId"], train_data["movieId"])
)

validation_pairs = set(
    zip(validation_data["userId"], validation_data["movieId"])
)

test_pairs = set(
    zip(test_data["userId"], test_data["movieId"])
)

print("Training ratings:", len(train_data))
print("Validation ratings:", len(validation_data))
print("Final test ratings:", len(test_data))

print(
    "Train-validation pair overlap:",
    len(train_pairs.intersection(validation_pairs))
)

print(
    "Train-test pair overlap:",
    len(train_pairs.intersection(test_pairs))
)

print(
    "Validation-test pair overlap:",
    len(validation_pairs.intersection(test_pairs))
)


# ================================================================
# 4. CREATE THE USER-MOVIE MATRIX
# ================================================================

print("\nSTEP 3: Creating the training user-movie matrix...")

rating_matrix = train_data.pivot_table(
    index="userId",
    columns="movieId",
    values="rating",
    aggfunc="mean"
).sort_index().sort_index(axis=1)

rating_matrix = rating_matrix.astype(float)

user_ids = rating_matrix.index.to_numpy()
movie_ids = rating_matrix.columns.to_numpy()

rating_values = rating_matrix.to_numpy()

rated_mask = np.isfinite(rating_values)

# Calculate each user's average rating using training data only.
user_means = rating_matrix.mean(axis=1).to_numpy()

global_mean = float(train_data["rating"].mean())

# Missing ratings become zero deviations after centering.
centered_matrix = (
    rating_matrix.sub(rating_matrix.mean(axis=1), axis=0)
    .fillna(0.0)
    .to_numpy()
)

# The explicit rating mask is needed to distinguish missing ratings
# from genuine zero deviations.
rated_mask = np.isfinite(rating_matrix.to_numpy())

print("Matrix shape:", rating_matrix.shape)
print("Training global mean:", round(global_mean, 4))
print("Training users:", len(user_ids))
print("Training catalog movies:", len(movie_ids))


# ================================================================
# 5. CALCULATE USER SIMILARITIES
# ================================================================

print("\nSTEP 4: Calculating mean-centered user similarities...")

similarity_matrix = cosine_similarity(centered_matrix)

# Avoid treating a user as their own nearest neighbor.
np.fill_diagonal(similarity_matrix, 0.0)

print("Similarity matrix shape:", similarity_matrix.shape)
print("Similarity calculation completed.")


# ================================================================
# 6. CALCULATE MOVIE NOVELTY
# ================================================================

print("\nSTEP 5: Calculating movie novelty...")

number_of_training_users = len(user_ids)

unique_raters = rated_mask.sum(axis=0).astype(float)

# Avoid log(0) in case a movie has no training ratings.
safe_raters = np.maximum(unique_raters, 1.0)

novelty_values = (
    -np.log2(safe_raters / number_of_training_users)
    / np.log2(number_of_training_users)
)

novelty_values = np.clip(novelty_values, 0.0, 1.0)

novelty_by_movie = dict(zip(movie_ids, novelty_values))

print("Minimum novelty:", round(float(novelty_values.min()), 4))
print("Maximum novelty:", round(float(novelty_values.max()), 4))
print("Mean novelty:", round(float(novelty_values.mean()), 4))


# ================================================================
# 7. BUILD LOOKUP TABLES
# ================================================================

print("\nSTEP 6: Preparing recommendation lookups...")

user_to_index = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}

movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(movie_ids)
}

movie_title_lookup = (
    movies.drop_duplicates("movieId")
    .set_index("movieId")["title"]
    .to_dict()
)

# Relevant held-out movies: ratings of 4 or higher.
# Keep only users and movies represented in the training matrix.
test_relevant = (
    test_data.loc[test_data["rating"] >= 4]
    .loc[
        lambda frame:
        frame["userId"].isin(user_ids)
        & frame["movieId"].isin(movie_ids)
    ]
    .groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

eligible_test_users = np.array(
    [
        user_id
        for user_id, relevant_movies in test_relevant.items()
        if len(relevant_movies) > 0
        and user_id in user_to_index
    ],
    dtype=int
)

if len(eligible_test_users) == 0:
    raise ValueError(
        "No eligible test users found. Check the dataset split and columns."
    )

print("Eligible test users with relevant movies:", len(eligible_test_users))


# ================================================================
# 8. RECOMMENDATION FUNCTION
# ================================================================

def recommend_for_user(user_id, regularization, novelty_weight):
    """
    Generate top-N recommendations using mean-centered user CF.

    regularization:
        0   = unregularized collaborative filtering
        10  = Stage 14 selected regularization

    novelty_weight:
        0.005 = 0.5% novelty contribution to normalized ranking score
    """

    if user_id not in user_to_index:
        return pd.DataFrame()

    user_index = user_to_index[user_id]

    user_similarities = similarity_matrix[user_index].copy()

    # Use the top 50 positively similar neighbors.
    positive_indices = np.flatnonzero(user_similarities > 0)

    if len(positive_indices) == 0:
        return pd.DataFrame()

    neighbor_order = positive_indices[
        np.argsort(user_similarities[positive_indices])[::-1]
    ]

    neighbor_indices = neighbor_order[:NEIGHBOR_COUNT]

    neighbor_similarities = user_similarities[neighbor_indices]

    neighbor_ratings = rating_values[neighbor_indices]
    neighbor_masks = rated_mask[neighbor_indices]
    neighbor_centered = centered_matrix[neighbor_indices]

    # Calculate weighted deviations for every movie.
    numerator = np.sum(
        neighbor_similarities[:, None] * neighbor_centered,
        axis=0
    )

    # Only neighbors who rated a movie contribute to its denominator.
    denominator = np.sum(
        neighbor_similarities[:, None] * neighbor_masks,
        axis=0
    )

    if regularization == 0:
        deviation = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0
        )
    else:
        deviation = numerator / (denominator + regularization)

    prediction = np.clip(
        user_means[user_index] + deviation,
        1.0,
        5.0
    )

    # Recommend only movies the user did not rate in training.
    already_rated = rated_mask[user_index]
    candidate_indices = np.flatnonzero(~already_rated)

    if len(candidate_indices) == 0:
        return pd.DataFrame()

    candidate_movie_ids = movie_ids[candidate_indices]
    candidate_predictions = prediction[candidate_indices]
    candidate_novelty = novelty_values[candidate_indices]

    normalized_prediction = (
        candidate_predictions - 1.0
    ) / 4.0

    ranking_scores = (
        (1.0 - novelty_weight) * normalized_prediction
        + novelty_weight * candidate_novelty
    )

    recommendations = pd.DataFrame({
        "userId": user_id,
        "movieId": candidate_movie_ids,
        "title": [
            movie_title_lookup.get(movie_id, "Unknown title")
            for movie_id in candidate_movie_ids
        ],
        "predicted_rating": candidate_predictions,
        "novelty": candidate_novelty,
        "ranking_score": ranking_scores
    })

    # MovieId is only a deterministic secondary sort key.
    # Stage 15 explicitly investigates how much this tie-break matters.
    recommendations = recommendations.sort_values(
        ["ranking_score", "movieId"],
        ascending=[False, True],
        kind="mergesort"
    ).reset_index(drop=True)

    return recommendations


# ================================================================
# 9. METRIC CALCULATION
# ================================================================

def calculate_user_metrics(recommendation_movie_ids, relevant_movies):
    """Calculate Precision@10, Recall@10 and NDCG@10."""

    top_movies = list(recommendation_movie_ids[:TOP_N])

    if not relevant_movies:
        return {
            "precision_at_10": np.nan,
            "recall_at_10": np.nan,
            "ndcg_at_10": np.nan
        }

    hits = [
        1 if movie_id in relevant_movies else 0
        for movie_id in top_movies
    ]

    precision = sum(hits) / TOP_N
    recall = sum(hits) / len(relevant_movies)

    dcg = sum(
        hit / np.log2(rank + 2)
        for rank, hit in enumerate(hits)
    )

    ideal_hits_count = min(len(relevant_movies), TOP_N)

    ideal_dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank in range(ideal_hits_count)
    )

    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    return {
        "precision_at_10": precision,
        "recall_at_10": recall,
        "ndcg_at_10": ndcg
    }


# ================================================================
# 10. STAGE 15A: INVESTIGATE SCORE TIES
# ================================================================

print("\n" + "=" * 75)
print("STAGE 15A: PREDICTION SCORE TIE INVESTIGATION")
print("=" * 75)

rng_ties = np.random.RandomState(2030)

tie_users = rng_ties.choice(
    eligible_test_users,
    size=min(TIE_DIAGNOSTIC_USERS, len(eligible_test_users)),
    replace=False
)

tie_diagnostic_rows = []

for position, user_id in enumerate(tie_users, start=1):

    print(
        f"\rChecking score ties: {position}/{len(tie_users)}",
        end=""
    )

    for regularization in [0.0, SELECTED_LAMBDA]:

        recs = recommend_for_user(
            user_id,
            regularization=regularization,
            novelty_weight=0.0
        )

        if recs.empty:
            continue

        top10 = recs.head(TOP_N)
        top100 = recs.head(min(100, len(recs)))

        top10_scores = top10["ranking_score"].to_numpy()
        top100_scores = top100["ranking_score"].to_numpy()

        unique_top10 = len(np.unique(np.round(top10_scores, 8)))
        unique_top100 = len(np.unique(np.round(top100_scores, 8)))

        tie_diagnostic_rows.append({
            "userId": user_id,
            "regularization": regularization,
            "top10_unique_scores_rounded_8dp": unique_top10,
            "top10_score_tie_present": unique_top10 < len(top10_scores),
            "top10_score_range": (
                float(top10_scores.max() - top10_scores.min())
                if len(top10_scores) else np.nan
            ),
            "top100_unique_scores_rounded_8dp": unique_top100,
            "top100_score_tie_present": unique_top100 < len(top100_scores),
            "top100_score_range": (
                float(top100_scores.max() - top100_scores.min())
                if len(top100_scores) else np.nan
            ),
            "fraction_candidate_predictions_at_5": float(
                np.isclose(recs["predicted_rating"], 5.0).mean()
            ),
            "fraction_candidate_scores_at_max": float(
                np.isclose(
                    recs["ranking_score"],
                    recs["ranking_score"].max()
                ).mean()
            )
        })

print("\nTie diagnostics completed.")

tie_df = pd.DataFrame(tie_diagnostic_rows)

tie_df.to_csv(
    OUTPUT_DIR / "stage15a_score_tie_diagnostics.csv",
    index=False
)

if not tie_df.empty:
    tie_summary = (
        tie_df.groupby("regularization")
        .agg(
            users_checked=("userId", "nunique"),
            mean_unique_top10_scores=(
                "top10_unique_scores_rounded_8dp", "mean"
            ),
            users_with_top10_ties=(
                "top10_score_tie_present", "sum"
            ),
            mean_top10_score_range=("top10_score_range", "mean"),
            mean_unique_top100_scores=(
                "top100_unique_scores_rounded_8dp", "mean"
            ),
            mean_fraction_predictions_at_5=(
                "fraction_candidate_predictions_at_5", "mean"
            ),
            mean_fraction_candidates_at_max_score=(
                "fraction_candidate_scores_at_max", "mean"
            )
        )
        .reset_index()
    )

    print("\nScore tie summary:")
    print(tie_summary.to_string(index=False))

    tie_summary.to_csv(
        OUTPUT_DIR / "stage15a_score_tie_summary.csv",
        index=False
    )


# ================================================================
# 11. STAGE 15B: INSPECT ACTUAL RECOMMENDATIONS
# ================================================================

print("\n" + "=" * 75)
print("STAGE 15B: EXAMPLE RECOMMENDATION INSPECTION")
print("=" * 75)

example_users = np.random.RandomState(2031).choice(
    eligible_test_users,
    size=min(EXAMPLE_USERS, len(eligible_test_users)),
    replace=False
)

example_rows = []

for user_id in example_users:

    recs = recommend_for_user(
        user_id,
        regularization=SELECTED_LAMBDA,
        novelty_weight=SELECTED_NOVELTY_WEIGHT
    )

    if recs.empty:
        continue

    recs = recs.head(TOP_N).copy()
    recs["rank"] = np.arange(1, len(recs) + 1)

    relevant = test_relevant.get(user_id, set())

    recs["relevant_in_test"] = recs["movieId"].isin(relevant)
    recs["regularization"] = SELECTED_LAMBDA
    recs["novelty_weight"] = SELECTED_NOVELTY_WEIGHT

    example_rows.append(recs)

if example_rows:
    example_df = pd.concat(example_rows, ignore_index=True)

    example_df.to_csv(
        OUTPUT_DIR / "stage15b_example_recommendations.csv",
        index=False
    )

    for user_id in example_users:
        user_recs = example_df[
            example_df["userId"] == user_id
        ]

        if user_recs.empty:
            continue

        print(f"\nUser ID: {user_id}")
        print(
            user_recs[
                [
                    "rank",
                    "movieId",
                    "title",
                    "predicted_rating",
                    "novelty",
                    "ranking_score",
                    "relevant_in_test"
                ]
            ].to_string(index=False)
        )

print("\nExample recommendations saved.")


# ================================================================
# 12. STAGE 15C: USER-LEVEL RECOMMENDATION QUALITY
# ================================================================

print("\n" + "=" * 75)
print("STAGE 15C: USER-LEVEL QUALITY ANALYSIS")
print("=" * 75)

rng_quality = np.random.RandomState(2027)

quality_users = rng_quality.choice(
    eligible_test_users,
    size=min(QUALITY_USERS, len(eligible_test_users)),
    replace=False
)

quality_rows = []
recommendation_cache = {}

for position, user_id in enumerate(quality_users, start=1):

    print(
        f"\rEvaluating users: {position}/{len(quality_users)}",
        end=""
    )

    recs = recommend_for_user(
        user_id,
        regularization=SELECTED_LAMBDA,
        novelty_weight=SELECTED_NOVELTY_WEIGHT
    )

    recommendation_cache[user_id] = recs

    if recs.empty:
        continue

    relevant = test_relevant.get(user_id, set())

    metrics = calculate_user_metrics(
        recs["movieId"].head(TOP_N).tolist(),
        relevant
    )

    top_recs = recs.head(TOP_N)

    quality_rows.append({
        "userId": user_id,
        "number_of_relevant_test_movies": len(relevant),
        **metrics,
        "average_novelty_at_10": float(
            top_recs["novelty"].mean()
        ),
        "number_of_distinct_movies_at_10": int(
            top_recs["movieId"].nunique()
        ),
        "top10_unique_scores_rounded_8dp": int(
            top_recs["ranking_score"].round(8).nunique()
        )
    })

print("\nUser-level evaluation completed.")

quality_df = pd.DataFrame(quality_rows)

quality_df.to_csv(
    OUTPUT_DIR / "stage15c_user_level_quality.csv",
    index=False
)

if not quality_df.empty:

    metric_summary = quality_df[
        [
            "precision_at_10",
            "recall_at_10",
            "ndcg_at_10",
            "average_novelty_at_10",
            "top10_unique_scores_rounded_8dp"
        ]
    ].agg(["mean", "median", "std", "min", "max"]).T

    print("\nUser-level metric distribution:")
    print(metric_summary.to_string())

    metric_summary.to_csv(
        OUTPUT_DIR / "stage15c_user_level_metric_summary.csv"
    )

    print("\nUsers with at least one relevant hit in top 10:",
          int((quality_df["precision_at_10"] > 0).sum()))

    print("Users evaluated:", len(quality_df))


# ================================================================
# 13. STAGE 15D: ROBUSTNESS ACROSS DIFFERENT USER SAMPLES
# ================================================================

print("\n" + "=" * 75)
print("STAGE 15D: ROBUSTNESS ACROSS USER SAMPLES")
print("=" * 75)

robustness_rows = []

# Reuse recommendations already calculated in Stage 15C.
# Generate additional recommendations only for users not in the cache.
robustness_seeds = [2026, 2027, 2028]

for seed in robustness_seeds:

    rng = np.random.RandomState(seed)

    sampled_users = rng.choice(
        eligible_test_users,
        size=min(ROBUSTNESS_USERS, len(eligible_test_users)),
        replace=False
    )

    sample_metrics = []

    for position, user_id in enumerate(sampled_users, start=1):

        recs = recommendation_cache.get(user_id)

        if recs is None:
            recs = recommend_for_user(
                user_id,
                regularization=SELECTED_LAMBDA,
                novelty_weight=SELECTED_NOVELTY_WEIGHT
            )
            recommendation_cache[user_id] = recs

        if recs.empty:
            continue

        relevant = test_relevant.get(user_id, set())

        metrics = calculate_user_metrics(
            recs["movieId"].head(TOP_N).tolist(),
            relevant
        )

        top_recs = recs.head(TOP_N)

        sample_metrics.append({
            **metrics,
            "average_novelty_at_10": float(
                top_recs["novelty"].mean()
            )
        })

    sample_df = pd.DataFrame(sample_metrics)

    if sample_df.empty:
        continue

    robustness_rows.append({
        "sample_seed": seed,
        "users_evaluated": len(sample_df),
        "mean_precision_at_10": sample_df["precision_at_10"].mean(),
        "mean_recall_at_10": sample_df["recall_at_10"].mean(),
        "mean_ndcg_at_10": sample_df["ndcg_at_10"].mean(),
        "mean_novelty_at_10": sample_df["average_novelty_at_10"].mean()
    })

    print(
        f"\nSeed {seed}: "
        f"P@10={sample_df['precision_at_10'].mean():.5f}, "
        f"Recall@10={sample_df['recall_at_10'].mean():.5f}, "
        f"NDCG@10={sample_df['ndcg_at_10'].mean():.5f}, "
        f"Novelty={sample_df['average_novelty_at_10'].mean():.5f}"
    )

robustness_df = pd.DataFrame(robustness_rows)

robustness_df.to_csv(
    OUTPUT_DIR / "stage15d_sample_robustness.csv",
    index=False
)


# ================================================================
# 14. STAGE 15E: FINAL DIAGNOSTIC SUMMARY
# ================================================================

print("\n" + "=" * 75)
print("STAGE 15E: FINAL DIAGNOSTIC SUMMARY")
print("=" * 75)

summary_rows = [
    {
        "investigation": "Score tie diagnostics",
        "status": "Completed" if not tie_df.empty else "No results",
        "output_file": "stage15a_score_tie_diagnostics.csv"
    },
    {
        "investigation": "Example recommendations",
        "status": "Completed" if example_rows else "No results",
        "output_file": "stage15b_example_recommendations.csv"
    },
    {
        "investigation": "User-level quality",
        "status": "Completed" if not quality_df.empty else "No results",
        "output_file": "stage15c_user_level_quality.csv"
    },
    {
        "investigation": "Sample robustness",
        "status": "Completed" if not robustness_df.empty else "No results",
        "output_file": "stage15d_sample_robustness.csv"
    }
]

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUTPUT_DIR / "stage15_final_summary.csv",
    index=False
)

print(summary_df.to_string(index=False))

print("\n" + "=" * 75)
print("STAGE 15 COMPLETED")
print("=" * 75)

print("\nGenerated files:")
print("1. stage15a_score_tie_diagnostics.csv")
print("2. stage15a_score_tie_summary.csv")
print("3. stage15b_example_recommendations.csv")
print("4. stage15c_user_level_quality.csv")
print("5. stage15c_user_level_metric_summary.csv")
print("6. stage15d_sample_robustness.csv")
print("7. stage15_final_summary.csv")

print(
    "\nImportant: These results are diagnostic. "
    "They do not establish clinical or real-world effectiveness."
)
