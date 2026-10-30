
# ============================================================
# STAGE 12 (CORRECTED): RANKING REDESIGN COMPARISON
# ============================================================
# Project: Serendipitous Novelty-Aware Movie Recommendation
# Dataset: MovieLens 1M
#
# Corrections:
# 1. Popularity = unique users who rated each movie
# 2. Novelty = normalized self-information
# 3. Same 80/20 split and random seed
# 4. Same reproducible sample of 500 evaluation users
# 5. Compare weighted-score ranking with rank-based fusion
#
# Metrics:
# Precision@10, Recall@10, NDCG@10,
# Average Novelty, Catalog Coverage
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

OUTPUT_DIR = PROJECT_DIR

SEED = 42
TOP_N = 10
MAX_NEIGHBORS = 50
N_EVALUATION_USERS = 500

NOVELTY_WEIGHTS = [
    0.00, 0.01, 0.02, 0.05, 0.10, 0.15, 0.30
]


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 75)
print("STAGE 12 CORRECTED: NOVELTY AND RANKING COMPARISON")
print("=" * 75)

print("\nSTEP 1: Loading datasets...")

ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)

required_rating_columns = {"userId", "movieId", "rating"}
required_movie_columns = {"movieId", "title"}

if not required_rating_columns.issubset(ratings.columns):
    raise ValueError("ratings.csv is missing required columns.")

if not required_movie_columns.issubset(movies.columns):
    raise ValueError("movies.csv is missing required columns.")

ratings = ratings[["userId", "movieId", "rating"]].copy()
movies = movies.drop_duplicates(subset="movieId").copy()

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)
print("Unique users:", ratings["userId"].nunique())
print("Unique rated movies:", ratings["movieId"].nunique())


# ============================================================
# 3. REPRODUCIBLE TRAIN-TEST SPLIT
# ============================================================

print("\nSTEP 2: Creating train-test split...")

train_df, test_df = train_test_split(
    ratings,
    test_size=0.20,
    random_state=SEED
)

train_df = train_df.copy()
test_df = test_df.copy()

print("Training ratings:", len(train_df))
print("Testing ratings:", len(test_df))

pair_overlap = (
    set(zip(train_df["userId"], train_df["movieId"]))
    &
    set(zip(test_df["userId"], test_df["movieId"]))
)

print("Train-test pair overlap:", len(pair_overlap))

if pair_overlap:
    raise RuntimeError("Train-test pair overlap detected.")


# ============================================================
# 4. CREATE USER-MOVIE MATRIX
# ============================================================

print("\nSTEP 3: Creating user-movie matrix...")

rating_matrix = train_df.pivot_table(
    index="userId",
    columns="movieId",
    values="rating",
    aggfunc="mean"
)

user_means = rating_matrix.mean(axis=1)
global_mean = train_df["rating"].mean()

centered_matrix = rating_matrix.sub(user_means, axis=0)
centered_filled = centered_matrix.fillna(0)

catalog_movie_ids = rating_matrix.columns.to_numpy()

print("Rating matrix shape:", rating_matrix.shape)
print("Global training mean:", round(global_mean, 4))
print("Training catalog size:", len(catalog_movie_ids))


# ============================================================
# 5. CALCULATE USER SIMILARITIES
# ============================================================

print("\nSTEP 4: Calculating user similarities...")

similarity_array = cosine_similarity(centered_filled)

np.fill_diagonal(similarity_array, 0)

similarity_df = pd.DataFrame(
    similarity_array,
    index=rating_matrix.index,
    columns=rating_matrix.index
)

print("Mean-centered cosine similarity completed.")


# ============================================================
# 6. CORRECTED POPULARITY AND NOVELTY
# ============================================================

print("\nSTEP 5: Calculating corrected movie popularity...")

# Count distinct users who rated each movie in TRAINING data.
unique_raters = (
    train_df.groupby("movieId")["userId"]
    .nunique()
    .reindex(catalog_movie_ids)
    .fillna(0)
)

n_training_users = rating_matrix.shape[0]

# Popularity = fraction of training users who rated the movie.
movie_popularity = unique_raters / n_training_users

# Avoid log(0); movies in the training catalog have at least
# one training rating, but this is a numerical safety check.
safe_popularity = movie_popularity.clip(
    lower=1 / n_training_users,
    upper=1.0
)

# Normalized self-information:
# popular movies have lower novelty;
# less widely rated movies have higher novelty.
movie_novelty = (
    -np.log2(safe_popularity)
    / np.log2(n_training_users)
)

movie_novelty = movie_novelty.clip(0, 1)

novelty_table = pd.DataFrame({
    "movieId": catalog_movie_ids,
    "unique_training_users": unique_raters.to_numpy(),
    "training_user_popularity": movie_popularity.to_numpy(),
    "novelty": movie_novelty.to_numpy()
})

print("Training users:", n_training_users)
print("Movies with popularity information:", len(novelty_table))
print(
    "Novelty range:",
    round(movie_novelty.min(), 4),
    "to",
    round(movie_novelty.max(), 4)
)
print("Mean novelty:", round(movie_novelty.mean(), 4))
print("Median novelty:", round(movie_novelty.median(), 4))

novelty_path = os.path.join(
    OUTPUT_DIR,
    "stage12_corrected_movie_popularity_novelty.csv"
)

novelty_table.to_csv(novelty_path, index=False)


# ============================================================
# 7. PREDICT RATINGS USING MEAN-CENTERED USER-BASED CF
# ============================================================

print("\nSTEP 6: Preparing prediction function...")


def predict_user_ratings(user_id):

    if user_id not in rating_matrix.index:
        return pd.Series(dtype=float)

    user_ratings = rating_matrix.loc[user_id]
    candidate_mask = user_ratings.isna()

    similarities = similarity_df.loc[user_id].drop(user_id)

    # Only use positive-similarity neighbors.
    similarities = similarities[similarities > 0]

    similarities = similarities.sort_values(
        ascending=False,
        kind="mergesort"
    ).head(MAX_NEIGHBORS)

    if similarities.empty:
        predictions = pd.Series(
            user_means.loc[user_id],
            index=rating_matrix.columns,
            dtype=float
        )

    else:
        neighbor_ids = similarities.index
        weights = similarities.to_numpy()

        neighbor_ratings = rating_matrix.loc[neighbor_ids]
        neighbor_centered = centered_matrix.loc[neighbor_ids]

        observed_mask = neighbor_ratings.notna().astype(float)

        weighted_deviations = (
            neighbor_centered.fillna(0)
            .mul(weights, axis=0)
        )

        numerator = weighted_deviations.sum(axis=0)

        denominator = observed_mask.mul(
            weights, axis=0
        ).sum(axis=0)

        deviations = numerator.div(
            denominator.replace(0, np.nan)
        )

        predictions = (
            user_means.loc[user_id] + deviations
        )

        predictions = predictions.fillna(
            user_means.loc[user_id]
        )

    predictions = predictions.clip(1, 5)

    # Only return movies not rated by the user in training.
    return predictions.loc[candidate_mask]


# ============================================================
# 8. RECOMMENDATION RANKING METHODS
# ============================================================

print("Preparing ranking methods...")


def get_recommendations(
    user_id,
    novelty_weight,
    method,
    cached_predictions=None
):

    if cached_predictions is not None:
        predictions = cached_predictions
    else:
        predictions = predict_user_ratings(user_id)

    if predictions.empty:
        return pd.DataFrame(columns=[
            "movieId",
            "predicted_rating",
            "novelty",
            "recommendation_score"
        ])

    candidates = pd.DataFrame({
        "movieId": predictions.index.to_numpy(),
        "predicted_rating": predictions.to_numpy()
    })

    candidates["novelty"] = (
        candidates["movieId"]
        .map(movie_novelty)
        .fillna(0.0)
    )

    # Method A: weighted-score fusion.
    if method == "weighted":

        preference_score = (
            (candidates["predicted_rating"] - 1) / 4
        )

        candidates["recommendation_score"] = (
            (1 - novelty_weight) * preference_score
            + novelty_weight * candidates["novelty"]
        )

    # Method B: rank-based fusion.
    elif method == "rank":

        preference_rank = (
            candidates["predicted_rating"]
            .rank(method="average", ascending=True, pct=True)
        )

        novelty_rank = (
            candidates["novelty"]
            .rank(method="average", ascending=True, pct=True)
        )

        candidates["recommendation_score"] = (
            (1 - novelty_weight) * preference_rank
            + novelty_weight * novelty_rank
        )

    else:
        raise ValueError("Method must be 'weighted' or 'rank'.")

    candidates = candidates.sort_values(
        ["recommendation_score", "movieId"],
        ascending=[False, True],
        kind="mergesort"
    )

    return candidates.head(TOP_N).reset_index(drop=True)


# ============================================================
# 9. BUILD RELEVANT HELD-OUT MOVIES
# ============================================================

print("\nSTEP 7: Preparing evaluation users...")

relevant_test = test_df[test_df["rating"] >= 4]

training_pairs = set(
    zip(train_df["userId"], train_df["movieId"])
)

relevant_by_user = {}

for user_id, group in relevant_test.groupby("userId"):

    eligible_movies = [
        movie_id
        for movie_id in group["movieId"].tolist()
        if movie_id in rating_matrix.columns
        and (user_id, movie_id) not in training_pairs
    ]

    if eligible_movies:
        relevant_by_user[user_id] = set(eligible_movies)

eligible_users = np.array(list(relevant_by_user.keys()))

if len(eligible_users) == 0:
    raise RuntimeError("No eligible evaluation users found.")

evaluation_rng = np.random.default_rng(SEED)

evaluation_users = evaluation_rng.choice(
    eligible_users,
    size=min(N_EVALUATION_USERS, len(eligible_users)),
    replace=False
)

evaluation_users = list(evaluation_users)

print("Eligible users:", len(eligible_users))
print("Selected evaluation users:", len(evaluation_users))


# ============================================================
# 10. METRIC CALCULATION
# ============================================================

def calculate_metrics(recommendations, relevant_items):

    recommended_ids = recommendations["movieId"].tolist()

    hits = [
        1 if movie_id in relevant_items else 0
        for movie_id in recommended_ids
    ]

    precision = sum(hits) / TOP_N

    recall = (
        sum(hits) / len(relevant_items)
        if relevant_items else 0.0
    )

    dcg = sum(
        hit / np.log2(position + 2)
        for position, hit in enumerate(hits)
    )

    ideal_hits = min(len(relevant_items), TOP_N)

    idcg = sum(
        1 / np.log2(position + 2)
        for position in range(ideal_hits)
    )

    ndcg = dcg / idcg if idcg > 0 else 0.0

    average_novelty = (
        recommendations["novelty"].mean()
        if not recommendations.empty else 0.0
    )

    return {
        "precision_at_10": precision,
        "recall_at_10": recall,
        "ndcg_at_10": ndcg,
        "average_novelty": average_novelty
    }


# ============================================================
# 11. CACHE PREDICTIONS
# ============================================================

print("\nSTEP 8: Caching user predictions...")

prediction_cache = {}

for position, user_id in enumerate(evaluation_users, start=1):

    prediction_cache[user_id] = predict_user_ratings(user_id)

    if position % 100 == 0:
        print(
            "Cached predictions:",
            position,
            "/",
            len(evaluation_users)
        )

# Cache User 1 separately for the recommendation preview.
if 1 in rating_matrix.index and 1 not in prediction_cache:
    prediction_cache[1] = predict_user_ratings(1)


# ============================================================
# 12. EVALUATE BOTH METHODS
# ============================================================

print("\nSTEP 9: Comparing ranking methods...")

results = []
recommendation_detail = []

for weight in NOVELTY_WEIGHTS:

    print("\nNovelty weight:", f"{weight:.0%}")

    for method in ["weighted", "rank"]:

        precision_values = []
        recall_values = []
        ndcg_values = []
        novelty_values = []

        all_recommended_movies = set()

        for user_id in evaluation_users:

            recommendations = get_recommendations(
                user_id=user_id,
                novelty_weight=weight,
                method=method,
                cached_predictions=prediction_cache[user_id]
            )

            relevant_items = relevant_by_user[user_id]

            metrics = calculate_metrics(
                recommendations,
                relevant_items
            )

            precision_values.append(metrics["precision_at_10"])
            recall_values.append(metrics["recall_at_10"])
            ndcg_values.append(metrics["ndcg_at_10"])
            novelty_values.append(metrics["average_novelty"])

            all_recommended_movies.update(
                recommendations["movieId"].tolist()
            )

            for rank_position, row in recommendations.iterrows():

                recommendation_detail.append({
                    "userId": user_id,
                    "method": method,
                    "novelty_weight": weight,
                    "rank": rank_position + 1,
                    "movieId": row["movieId"],
                    "predicted_rating": row["predicted_rating"],
                    "novelty": row["novelty"],
                    "recommendation_score": row[
                        "recommendation_score"
                    ],
                    "is_relevant": (
                        row["movieId"] in relevant_items
                    )
                })

        catalog_coverage = (
            len(all_recommended_movies) / len(catalog_movie_ids)
        )

        results.append({
            "method": method,
            "novelty_weight": weight,
            "evaluation_users": len(evaluation_users),
            "precision_at_10": np.mean(precision_values),
            "recall_at_10": np.mean(recall_values),
            "ndcg_at_10": np.mean(ndcg_values),
            "average_novelty": np.mean(novelty_values),
            "catalog_coverage": catalog_coverage,
            "distinct_recommended_movies": len(
                all_recommended_movies
            )
        })

        print(
            method,
            "| Precision@10:", round(np.mean(precision_values), 5),
            "| Recall@10:", round(np.mean(recall_values), 5),
            "| NDCG@10:", round(np.mean(ndcg_values), 5),
            "| Novelty:", round(np.mean(novelty_values), 4),
            "| Coverage:", round(catalog_coverage * 100, 2), "%"
        )


# ============================================================
# 13. SAVE EVALUATION RESULTS
# ============================================================

print("\nSTEP 10: Saving evaluation files...")

results_df = pd.DataFrame(results)

results_path = os.path.join(
    OUTPUT_DIR,
    "stage12_ranking_method_comparison.csv"
)

results_df.to_csv(results_path, index=False)

detail_df = pd.DataFrame(recommendation_detail)

detail_path = os.path.join(
    OUTPUT_DIR,
    "stage12_recommendation_metrics_detail.csv"
)

detail_df.to_csv(detail_path, index=False)


# ============================================================
# 14. USER 1 RECOMMENDATION COMPARISON
# ============================================================

print("\nSTEP 11: Creating User 1 recommendations...")

user1_rows = []

if 1 in rating_matrix.index:

    for weight in NOVELTY_WEIGHTS:

        for method in ["weighted", "rank"]:

            recommendations = get_recommendations(
                user_id=1,
                novelty_weight=weight,
                method=method,
                cached_predictions=prediction_cache[1]
            )

            recommendations = recommendations.merge(
                movies[["movieId", "title"]],
                on="movieId",
                how="left"
            )

            for rank_position, row in recommendations.iterrows():

                user1_rows.append({
                    "userId": 1,
                    "method": method,
                    "novelty_weight": weight,
                    "rank": rank_position + 1,
                    "movieId": row["movieId"],
                    "title": row.get("title", ""),
                    "predicted_rating": row["predicted_rating"],
                    "novelty": row["novelty"],
                    "recommendation_score": row[
                        "recommendation_score"
                    ]
                })

    user1_df = pd.DataFrame(user1_rows)

    user1_path = os.path.join(
        OUTPUT_DIR,
        "stage12_user1_ranking_comparison.csv"
    )

    user1_df.to_csv(user1_path, index=False)

    print("\nUser 1 recommendations at 5% novelty weight:")

    preview = user1_df[
        np.isclose(user1_df["novelty_weight"], 0.05)
    ]

    for method in ["weighted", "rank"]:

        print("\nMethod:", method)

        print(
            preview[preview["method"] == method][
                [
                    "rank",
                    "title",
                    "predicted_rating",
                    "novelty"
                ]
            ].to_string(index=False)
        )

else:
    print("User 1 is not in the training matrix.")


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\nSTEP 12: FINAL RESULTS")

print(
    results_df[
        [
            "method",
            "novelty_weight",
            "precision_at_10",
            "recall_at_10",
            "ndcg_at_10",
            "average_novelty",
            "catalog_coverage"
        ]
    ].to_string(index=False)
)

print("\nFiles saved:")
print(results_path)
print(detail_path)
print(novelty_path)

if 1 in rating_matrix.index:
    print(user1_path)

print("\n" + "=" * 75)
print("STAGE 12 CORRECTED COMPLETED")
print("=" * 75)
