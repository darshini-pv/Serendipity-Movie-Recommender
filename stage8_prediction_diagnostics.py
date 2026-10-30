
# ================================================================
# STAGE 8: PREDICTION DIAGNOSTICS AND SCORING IMPROVEMENT
# Project: Serendipitous Novelty-Aware Movie Recommendation System
# Dataset: MovieLens 1M
# ================================================================

import os
import time
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ================================================================
# 1. SETTINGS AND FILE PATHS
# ================================================================

START_TIME = time.time()

PROJECT_PATH = (
    r"C:\Users\manoj\OneDrive\Desktop"
    r"\Serendipitous Recommendation Engine"
)

DATA_PATH = os.path.join(PROJECT_PATH, "data")

RATINGS_PATH = os.path.join(DATA_PATH, "ratings.csv")
MOVIES_PATH = os.path.join(DATA_PATH, "movies.csv")

OUTPUT_SUMMARY = os.path.join(
    PROJECT_PATH,
    "stage8_diagnostic_summary.csv"
)

OUTPUT_COMPARISON = os.path.join(
    PROJECT_PATH,
    "stage8_prediction_diagnostics.csv"
)

OUTPUT_RECOMMENDATIONS = os.path.join(
    PROJECT_PATH,
    "stage8_user1_recommendations.csv"
)

RANDOM_SEED = 42
EVALUATION_USER_LIMIT = 500
DIAGNOSTIC_RATING_LIMIT = 20000
TOP_K = 10
NEIGHBOR_LIMIT = 50

NOVELTY_WEIGHTS = [0.00, 0.05, 0.10, 0.15, 0.30]

print("=" * 72)
print("STAGE 8: PREDICTION DIAGNOSTICS")
print("=" * 72)


# ================================================================
# 2. LOAD AND VALIDATE DATA
# ================================================================

print("\nSTEP 1: Loading datasets...")

ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)

required_rating_columns = {
    "userId", "movieId", "rating"
}

required_movie_columns = {
    "movieId", "title", "genres"
}

if not required_rating_columns.issubset(ratings.columns):
    raise ValueError(
        "ratings.csv must contain userId, movieId and rating."
    )

if not required_movie_columns.issubset(movies.columns):
    raise ValueError(
        "movies.csv must contain movieId, title and genres."
    )

ratings = ratings[
    ["userId", "movieId", "rating"]
].copy()

movies = movies[
    ["movieId", "title", "genres"]
].copy()

ratings = ratings.dropna(
    subset=["userId", "movieId", "rating"]
)

ratings["userId"] = ratings["userId"].astype(int)
ratings["movieId"] = ratings["movieId"].astype(int)

ratings["rating"] = pd.to_numeric(
    ratings["rating"],
    errors="coerce"
)

ratings = ratings.dropna(subset=["rating"])

ratings = ratings.drop_duplicates(
    subset=["userId", "movieId"],
    keep="last"
)

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)
print("Unique users:", ratings["userId"].nunique())
print("Unique rated movies:", ratings["movieId"].nunique())


# ================================================================
# 3. CREATE REPRODUCIBLE TRAIN-TEST SPLIT
# ================================================================

print("\nSTEP 2: Creating train-test split...")

rng = np.random.default_rng(RANDOM_SEED)

permutation = rng.permutation(len(ratings))

split_point = int(len(ratings) * 0.80)

train = ratings.iloc[
    permutation[:split_point]
].copy()

test = ratings.iloc[
    permutation[split_point:]
].copy()

train_pairs = pd.MultiIndex.from_frame(
    train[["userId", "movieId"]]
)

test_pairs = pd.MultiIndex.from_frame(
    test[["userId", "movieId"]]
)

overlap = len(train_pairs.intersection(test_pairs))

print("Training ratings:", len(train))
print("Testing ratings:", len(test))
print("Train-test overlap:", overlap)

if overlap != 0:
    raise ValueError(
        "Train-test leakage detected."
    )


# ================================================================
# 4. BUILD USER-MOVIE MATRIX
# ================================================================

print("\nSTEP 3: Building user-movie matrix...")

matrix_df = train.pivot(
    index="userId",
    columns="movieId",
    values="rating"
).sort_index().sort_index(axis=1)

user_ids = matrix_df.index.to_numpy()
movie_ids = matrix_df.columns.to_numpy()

rating_matrix = matrix_df.fillna(0).to_numpy(
    dtype=np.float32
)

rated_mask = (
    rating_matrix > 0
).astype(np.float32)

user_to_index = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}

movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(movie_ids)
}

rating_counts = rated_mask.sum(axis=1)

user_means = np.divide(
    rating_matrix.sum(axis=1),
    rating_counts,
    out=np.zeros(
        len(user_ids),
        dtype=np.float32
    ),
    where=rating_counts > 0
)

global_mean = float(train["rating"].mean())

print("Rating matrix shape:", rating_matrix.shape)
print("Global training mean:", round(global_mean, 4))


# ================================================================
# 5. CALCULATE USER SIMILARITY
# ================================================================

print("\nSTEP 4: Calculating user similarity...")

similarity_matrix = cosine_similarity(
    rating_matrix
).astype(np.float32)

# Users cannot be their own neighbors.
np.fill_diagonal(similarity_matrix, 0)

print("Similarity matrix shape:", similarity_matrix.shape)
print("Similarity calculation completed.")


# ================================================================
# 6. CREATE MEAN-CENTERED RATINGS
# ================================================================

centered_matrix = np.where(
    rated_mask > 0,
    rating_matrix - user_means[:, None],
    0
).astype(np.float32)


# ================================================================
# 7. DEFINE RATING PREDICTION METHODS
# ================================================================

def predict_ratings(user_id, method="mean_centered"):

    if user_id not in user_to_index:
        return None

    user_index = user_to_index[user_id]

    similarities = similarity_matrix[
        user_index
    ].copy()

    # Only positive similarities are considered.
    similarities[similarities <= 0] = 0

    positive_indices = np.flatnonzero(
        similarities > 0
    )

    if len(positive_indices) > NEIGHBOR_LIMIT:

        strongest_positions = np.argpartition(
            similarities[positive_indices],
            -NEIGHBOR_LIMIT
        )[-NEIGHBOR_LIMIT:]

        neighbor_indices = positive_indices[
            strongest_positions
        ]

    else:
        neighbor_indices = positive_indices

    # Fallback when no positive-similarity neighbors exist.
    if len(neighbor_indices) == 0:

        return np.full(
            len(movie_ids),
            user_means[user_index],
            dtype=np.float32
        )

    weights = similarities[neighbor_indices]

    neighbor_ratings = rating_matrix[
        neighbor_indices
    ]

    neighbor_masks = rated_mask[
        neighbor_indices
    ]

    # ------------------------------------------------------------
    # METHOD A: RAW WEIGHTED RATINGS
    # ------------------------------------------------------------

    weighted_rating_sum = (
        weights @ neighbor_ratings
    )

    rating_weight_sum = (
        weights @ neighbor_masks
    )

    raw_predictions = np.divide(
        weighted_rating_sum,
        rating_weight_sum,
        out=np.full(
            len(movie_ids),
            user_means[user_index],
            dtype=np.float32
        ),
        where=rating_weight_sum > 0
    )

    # ------------------------------------------------------------
    # METHOD B: MEAN-CENTERED COLLABORATIVE FILTERING
    # ------------------------------------------------------------

    weighted_deviation_sum = (
        weights @ centered_matrix[neighbor_indices]
    )

    deviation_weight_sum = (
        weights @ neighbor_masks
    )

    adjustments = np.divide(
        weighted_deviation_sum,
        deviation_weight_sum,
        out=np.zeros(
            len(movie_ids),
            dtype=np.float32
        ),
        where=deviation_weight_sum > 0
    )

    centered_predictions = (
        user_means[user_index] + adjustments
    )

    if method == "raw":

        return np.clip(
            raw_predictions,
            1.0,
            5.0
        )

    elif method == "mean_centered":

        return np.clip(
            centered_predictions,
            1.0,
            5.0
        )

    else:
        raise ValueError(
            "Method must be 'raw' or 'mean_centered'."
        )


# ================================================================
# 8. SELECT TEST RATINGS FOR PREDICTION DIAGNOSTICS
# ================================================================

print("\nSTEP 5: Selecting test observations...")

test_candidates = test[
    test["userId"].isin(user_to_index)
    & test["movieId"].isin(movie_to_index)
].copy()

diagnostic_count = min(
    DIAGNOSTIC_RATING_LIMIT,
    len(test_candidates)
)

diagnostic_test = test_candidates.sample(
    n=diagnostic_count,
    random_state=RANDOM_SEED
).copy()

print("Test ratings available:", len(test_candidates))
print("Test ratings sampled:", len(diagnostic_test))


# ================================================================
# 9. COMPARE RAW AND MEAN-CENTERED PREDICTIONS
# ================================================================

print("\nSTEP 6: Comparing prediction methods...")

prediction_cache = {}

raw_actual = []
raw_predicted = []

centered_actual = []
centered_predicted = []

for row_number, row in enumerate(
    diagnostic_test.itertuples(index=False),
    start=1
):

    user_id = row.userId
    movie_id = row.movieId
    actual_rating = float(row.rating)

    if user_id not in prediction_cache:

        prediction_cache[user_id] = {
            "raw": predict_ratings(
                user_id,
                "raw"
            ),
            "mean_centered": predict_ratings(
                user_id,
                "mean_centered"
            )
        }

    movie_index = movie_to_index[movie_id]

    raw_prediction = prediction_cache[user_id][
        "raw"
    ][movie_index]

    centered_prediction = prediction_cache[user_id][
        "mean_centered"
    ][movie_index]

    raw_actual.append(actual_rating)
    raw_predicted.append(float(raw_prediction))

    centered_actual.append(actual_rating)
    centered_predicted.append(float(centered_prediction))

    if row_number % 5000 == 0:

        print(
            "Evaluated",
            row_number,
            "test ratings..."
        )


# ================================================================
# 10. CALCULATE PREDICTION ERROR METRICS
# ================================================================

def summarize_predictions(
    actual,
    predicted,
    method_name
):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "prediction_method": method_name,
        "observations": len(actual),
        "MAE": mean_absolute_error(
            actual,
            predicted
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        ),
        "mean_prediction": np.mean(predicted),
        "prediction_std": np.std(predicted),
        "percentage_at_5": (
            np.mean(predicted >= 4.999) * 100
        ),
        "percentage_at_1": (
            np.mean(predicted <= 1.001) * 100
        ),
        "unique_prediction_values_rounded": len(
            np.unique(np.round(predicted, 2))
        )
    }


summary_df = pd.DataFrame([
    summarize_predictions(
        raw_actual,
        raw_predicted,
        "Raw weighted ratings"
    ),
    summarize_predictions(
        centered_actual,
        centered_predicted,
        "Mean-centered CF"
    )
])

print("\n" + "=" * 72)
print("PREDICTION DIAGNOSTICS")
print("=" * 72)

print(
    summary_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.4f}".format,
            "RMSE": "{:.4f}".format,
            "mean_prediction": "{:.4f}".format,
            "prediction_std": "{:.4f}".format,
            "percentage_at_5": "{:.2f}%".format,
            "percentage_at_1": "{:.2f}%".format
        }
    )
)


# ================================================================
# 11. CALCULATE NOVELTY SCORES
# ================================================================

print("\nSTEP 7: Preparing novelty scores...")

movie_popularity = rated_mask.sum(axis=0)

popularity_probability = np.maximum(
    movie_popularity / len(user_ids),
    1 / len(user_ids)
)

novelty = (
    -np.log2(popularity_probability)
    / np.log2(len(user_ids))
).astype(np.float32)

print("Novelty scores calculated.")


# ================================================================
# 12. GENERATE RECOMMENDATIONS
# ================================================================

def recommend(
    user_id,
    predictions,
    novelty_weight
):

    user_index = user_to_index[user_id]

    # Exclude movies the user rated in training.
    candidate_indices = np.flatnonzero(
        rated_mask[user_index] == 0
    )

    if len(candidate_indices) == 0:
        return pd.DataFrame()

    candidate_predictions = predictions[
        candidate_indices
    ]

    preference_scores = (
        candidate_predictions - 1.0
    ) / 4.0

    candidate_novelty = novelty[
        candidate_indices
    ]

    final_scores = (
        (1.0 - novelty_weight) * preference_scores
        + novelty_weight * candidate_novelty
    )

    top_count = min(
        TOP_K,
        len(candidate_indices)
    )

    top_positions = np.argpartition(
        final_scores,
        -top_count
    )[-top_count:]

    top_positions = top_positions[
        np.argsort(
            final_scores[top_positions]
        )[::-1]
    ]

    selected_indices = candidate_indices[
        top_positions
    ]

    result = pd.DataFrame({
        "userId": user_id,
        "movieId": movie_ids[selected_indices],
        "predicted_rating": predictions[selected_indices],
        "novelty_score": novelty[selected_indices],
        "preference_score": (
            predictions[selected_indices] - 1.0
        ) / 4.0,
        "novelty_weight": novelty_weight,
        "final_score": final_scores[top_positions]
    })

    result = result.merge(
        movies,
        on="movieId",
        how="left"
    )

    return result


# ================================================================
# 13. PREPARE RELEVANT TEST ITEMS AND EVALUATION USERS
# ================================================================

print("\nSTEP 8: Preparing recommendation evaluation...")

# A test rating of 4 or 5 is considered relevant.
test_relevant = test[
    test["rating"] >= 4
].copy()

relevant_by_user = (
    test_relevant.groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

eligible_users = [
    user_id
    for user_id in relevant_by_user
    if user_id in user_to_index
    and len(relevant_by_user[user_id]) > 0
]

evaluation_rng = np.random.default_rng(
    RANDOM_SEED
)

evaluation_users = evaluation_rng.choice(
    eligible_users,
    size=min(
        EVALUATION_USER_LIMIT,
        len(eligible_users)
    ),
    replace=False
).tolist()

print("Eligible users:", len(eligible_users))
print("Evaluation users:", len(evaluation_users))


# ================================================================
# 14. CALCULATE RANKING METRICS
# ================================================================

def calculate_ndcg(
    recommended_ids,
    relevant_ids
):

    dcg = 0.0

    for rank, movie_id in enumerate(
        recommended_ids[:TOP_K],
        start=1
    ):

        if movie_id in relevant_ids:

            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(
        len(relevant_ids),
        TOP_K
    )

    if ideal_hits == 0:
        return 0.0

    ideal_dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    return dcg / ideal_dcg


# Cache predictions for the evaluation users.
for user_id in evaluation_users:

    if user_id not in prediction_cache:

        prediction_cache[user_id] = {
            "raw": predict_ratings(
                user_id,
                "raw"
            ),
            "mean_centered": predict_ratings(
                user_id,
                "mean_centered"
            )
        }


# ================================================================
# 15. COMPARE METHODS AND NOVELTY WEIGHTS
# ================================================================

print("\nSTEP 9: Evaluating recommendation settings...")

comparison_rows = []
recommendation_rows = []

for method_name in [
    "raw",
    "mean_centered"
]:

    for novelty_weight in NOVELTY_WEIGHTS:

        precisions = []
        recalls = []
        ndcgs = []
        novelty_values = []

        catalog = set()

        for user_id in evaluation_users:

            predictions = prediction_cache[user_id][
                method_name
            ]

            recommendations = recommend(
                user_id,
                predictions,
                novelty_weight
            )

            recommended_ids = (
                recommendations["movieId"].tolist()
            )

            relevant_ids = relevant_by_user.get(
                user_id,
                set()
            )

            hits = sum(
                movie_id in relevant_ids
                for movie_id in recommended_ids
            )

            precisions.append(
                hits / TOP_K
            )

            recalls.append(
                hits / len(relevant_ids)
                if relevant_ids
                else 0.0
            )

            ndcgs.append(
                calculate_ndcg(
                    recommended_ids,
                    relevant_ids
                )
            )

            novelty_values.append(
                recommendations["novelty_score"].mean()
                if not recommendations.empty
                else 0.0
            )

            catalog.update(recommended_ids)

            # Keep User 1's recommendations for every setting.
            if user_id == 1:

                user_recs = recommendations.copy()

                user_recs["prediction_method"] = (
                    method_name
                )

                recommendation_rows.append(
                    user_recs
                )

        comparison_rows.append({
            "prediction_method": method_name,
            "novelty_weight": novelty_weight,
            "preference_weight": 1.0 - novelty_weight,
            "precision_at_10": np.mean(precisions),
            "recall_at_10": np.mean(recalls),
            "ndcg_at_10": np.mean(ndcgs),
            "average_novelty": np.mean(novelty_values),
            "catalog_coverage": (
                len(catalog) / len(movie_ids)
            ),
            "distinct_movies_recommended": len(catalog),
            "evaluation_users": len(evaluation_users)
        })

        print(
            "Completed:",
            method_name,
            "| novelty weight:",
            f"{novelty_weight:.0%}"
        )


# ================================================================
# 16. SAVE RESULTS
# ================================================================

print("\nSTEP 10: Saving results...")

comparison_df = pd.DataFrame(
    comparison_rows
)

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False
)

comparison_df.to_csv(
    OUTPUT_COMPARISON,
    index=False
)

if recommendation_rows:

    all_user1_recommendations = pd.concat(
        recommendation_rows,
        ignore_index=True
    )

    all_user1_recommendations.to_csv(
        OUTPUT_RECOMMENDATIONS,
        index=False
    )

else:

    # Create an empty file with headers if User 1 is unavailable.
    pd.DataFrame(
        columns=[
            "userId",
            "movieId",
            "predicted_rating",
            "novelty_score",
            "preference_score",
            "novelty_weight",
            "final_score",
            "title",
            "genres",
            "prediction_method"
        ]
    ).to_csv(
        OUTPUT_RECOMMENDATIONS,
        index=False
    )


# ================================================================
# 17. DISPLAY MODEL COMPARISON
# ================================================================

print("\n" + "=" * 72)
print("STAGE 8: MODEL COMPARISON")
print("=" * 72)

print(
    comparison_df.to_string(
        index=False,
        formatters={
            "novelty_weight": "{:.0%}".format,
            "preference_weight": "{:.0%}".format,
            "precision_at_10": "{:.4f}".format,
            "recall_at_10": "{:.4f}".format,
            "ndcg_at_10": "{:.4f}".format,
            "average_novelty": "{:.4f}".format,
            "catalog_coverage": "{:.2%}".format
        }
    )
)


# ================================================================
# 18. DISPLAY USER 1'S RECOMMENDATIONS
# ================================================================

print("\n" + "=" * 72)
print("USER 1: RECOMMENDATION PREVIEW")
print("=" * 72)

if recommendation_rows:

    user1_results = pd.concat(
        recommendation_rows,
        ignore_index=True
    )

    preview = user1_results[
        user1_results["prediction_method"] == "mean_centered"
    ]

    preview = preview[
        preview["novelty_weight"] == 0.10
    ]

    if not preview.empty:

        print(
            preview[
                [
                    "movieId",
                    "title",
                    "genres",
                    "predicted_rating",
                    "novelty_score",
                    "final_score"
                ]
            ].to_string(index=False)
        )

    else:
        print("No recommendations available for the preview.")

else:
    print("User 1 was not available for recommendation preview.")


# ================================================================
# 19. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 72)
print("STAGE 8 COMPLETED SUCCESSFULLY")
print("=" * 72)

print("\nSaved files:")

print("1.", OUTPUT_SUMMARY)
print("2.", OUTPUT_COMPARISON)
print("3.", OUTPUT_RECOMMENDATIONS)

print(
    "\nTotal execution time:",
    round(time.time() - START_TIME, 2),
    "seconds"
)

print("\nPlease review the results before proceeding to Stage 9.")

# ================================================================
# STAGE 8: PREDICTION DIAGNOSTICS AND SCORING IMPROVEMENT
# Project: Serendipitous Novelty-Aware Movie Recommendation System
# Dataset: MovieLens 1M
# ================================================================

import os
import time
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ================================================================
# 1. SETTINGS AND FILE PATHS
# ================================================================

START_TIME = time.time()

PROJECT_PATH = (
    r"C:\Users\manoj\OneDrive\Desktop"
    r"\Serendipitous Recommendation Engine"
)

DATA_PATH = os.path.join(PROJECT_PATH, "data")

RATINGS_PATH = os.path.join(DATA_PATH, "ratings.csv")
MOVIES_PATH = os.path.join(DATA_PATH, "movies.csv")

OUTPUT_SUMMARY = os.path.join(
    PROJECT_PATH,
    "stage8_diagnostic_summary.csv"
)

OUTPUT_COMPARISON = os.path.join(
    PROJECT_PATH,
    "stage8_prediction_diagnostics.csv"
)

OUTPUT_RECOMMENDATIONS = os.path.join(
    PROJECT_PATH,
    "stage8_user1_recommendations.csv"
)

RANDOM_SEED = 42
EVALUATION_USER_LIMIT = 500
DIAGNOSTIC_RATING_LIMIT = 20000
TOP_K = 10
NEIGHBOR_LIMIT = 50

NOVELTY_WEIGHTS = [0.00, 0.05, 0.10, 0.15, 0.30]

print("=" * 72)
print("STAGE 8: PREDICTION DIAGNOSTICS")
print("=" * 72)


# ================================================================
# 2. LOAD AND VALIDATE DATA
# ================================================================

print("\nSTEP 1: Loading datasets...")

ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)

required_rating_columns = {
    "userId", "movieId", "rating"
}

required_movie_columns = {
    "movieId", "title", "genres"
}

if not required_rating_columns.issubset(ratings.columns):
    raise ValueError(
        "ratings.csv must contain userId, movieId and rating."
    )

if not required_movie_columns.issubset(movies.columns):
    raise ValueError(
        "movies.csv must contain movieId, title and genres."
    )

ratings = ratings[
    ["userId", "movieId", "rating"]
].copy()

movies = movies[
    ["movieId", "title", "genres"]
].copy()

ratings = ratings.dropna(
    subset=["userId", "movieId", "rating"]
)

ratings["userId"] = ratings["userId"].astype(int)
ratings["movieId"] = ratings["movieId"].astype(int)

ratings["rating"] = pd.to_numeric(
    ratings["rating"],
    errors="coerce"
)

ratings = ratings.dropna(subset=["rating"])

ratings = ratings.drop_duplicates(
    subset=["userId", "movieId"],
    keep="last"
)

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)
print("Unique users:", ratings["userId"].nunique())
print("Unique rated movies:", ratings["movieId"].nunique())


# ================================================================
# 3. CREATE REPRODUCIBLE TRAIN-TEST SPLIT
# ================================================================

print("\nSTEP 2: Creating train-test split...")

rng = np.random.default_rng(RANDOM_SEED)

permutation = rng.permutation(len(ratings))

split_point = int(len(ratings) * 0.80)

train = ratings.iloc[
    permutation[:split_point]
].copy()

test = ratings.iloc[
    permutation[split_point:]
].copy()

train_pairs = pd.MultiIndex.from_frame(
    train[["userId", "movieId"]]
)

test_pairs = pd.MultiIndex.from_frame(
    test[["userId", "movieId"]]
)

overlap = len(train_pairs.intersection(test_pairs))

print("Training ratings:", len(train))
print("Testing ratings:", len(test))
print("Train-test overlap:", overlap)

if overlap != 0:
    raise ValueError(
        "Train-test leakage detected."
    )


# ================================================================
# 4. BUILD USER-MOVIE MATRIX
# ================================================================

print("\nSTEP 3: Building user-movie matrix...")

matrix_df = train.pivot(
    index="userId",
    columns="movieId",
    values="rating"
).sort_index().sort_index(axis=1)

user_ids = matrix_df.index.to_numpy()
movie_ids = matrix_df.columns.to_numpy()

rating_matrix = matrix_df.fillna(0).to_numpy(
    dtype=np.float32
)

rated_mask = (
    rating_matrix > 0
).astype(np.float32)

user_to_index = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}

movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(movie_ids)
}

rating_counts = rated_mask.sum(axis=1)

user_means = np.divide(
    rating_matrix.sum(axis=1),
    rating_counts,
    out=np.zeros(
        len(user_ids),
        dtype=np.float32
    ),
    where=rating_counts > 0
)

global_mean = float(train["rating"].mean())

print("Rating matrix shape:", rating_matrix.shape)
print("Global training mean:", round(global_mean, 4))


# ================================================================
# 5. CALCULATE USER SIMILARITY
# ================================================================

print("\nSTEP 4: Calculating user similarity...")

similarity_matrix = cosine_similarity(
    rating_matrix
).astype(np.float32)

# Users cannot be their own neighbors.
np.fill_diagonal(similarity_matrix, 0)

print("Similarity matrix shape:", similarity_matrix.shape)
print("Similarity calculation completed.")


# ================================================================
# 6. CREATE MEAN-CENTERED RATINGS
# ================================================================

centered_matrix = np.where(
    rated_mask > 0,
    rating_matrix - user_means[:, None],
    0
).astype(np.float32)


# ================================================================
# 7. DEFINE RATING PREDICTION METHODS
# ================================================================

def predict_ratings(user_id, method="mean_centered"):

    if user_id not in user_to_index:
        return None

    user_index = user_to_index[user_id]

    similarities = similarity_matrix[
        user_index
    ].copy()

    # Only positive similarities are considered.
    similarities[similarities <= 0] = 0

    positive_indices = np.flatnonzero(
        similarities > 0
    )

    if len(positive_indices) > NEIGHBOR_LIMIT:

        strongest_positions = np.argpartition(
            similarities[positive_indices],
            -NEIGHBOR_LIMIT
        )[-NEIGHBOR_LIMIT:]

        neighbor_indices = positive_indices[
            strongest_positions
        ]

    else:
        neighbor_indices = positive_indices

    # Fallback when no positive-similarity neighbors exist.
    if len(neighbor_indices) == 0:

        return np.full(
            len(movie_ids),
            user_means[user_index],
            dtype=np.float32
        )

    weights = similarities[neighbor_indices]

    neighbor_ratings = rating_matrix[
        neighbor_indices
    ]

    neighbor_masks = rated_mask[
        neighbor_indices
    ]

    # ------------------------------------------------------------
    # METHOD A: RAW WEIGHTED RATINGS
    # ------------------------------------------------------------

    weighted_rating_sum = (
        weights @ neighbor_ratings
    )

    rating_weight_sum = (
        weights @ neighbor_masks
    )

    raw_predictions = np.divide(
        weighted_rating_sum,
        rating_weight_sum,
        out=np.full(
            len(movie_ids),
            user_means[user_index],
            dtype=np.float32
        ),
        where=rating_weight_sum > 0
    )

    # ------------------------------------------------------------
    # METHOD B: MEAN-CENTERED COLLABORATIVE FILTERING
    # ------------------------------------------------------------

    weighted_deviation_sum = (
        weights @ centered_matrix[neighbor_indices]
    )

    deviation_weight_sum = (
        weights @ neighbor_masks
    )

    adjustments = np.divide(
        weighted_deviation_sum,
        deviation_weight_sum,
        out=np.zeros(
            len(movie_ids),
            dtype=np.float32
        ),
        where=deviation_weight_sum > 0
    )

    centered_predictions = (
        user_means[user_index] + adjustments
    )

    if method == "raw":

        return np.clip(
            raw_predictions,
            1.0,
            5.0
        )

    elif method == "mean_centered":

        return np.clip(
            centered_predictions,
            1.0,
            5.0
        )

    else:
        raise ValueError(
            "Method must be 'raw' or 'mean_centered'."
        )


# ================================================================
# 8. SELECT TEST RATINGS FOR PREDICTION DIAGNOSTICS
# ================================================================

print("\nSTEP 5: Selecting test observations...")

test_candidates = test[
    test["userId"].isin(user_to_index)
    & test["movieId"].isin(movie_to_index)
].copy()

diagnostic_count = min(
    DIAGNOSTIC_RATING_LIMIT,
    len(test_candidates)
)

diagnostic_test = test_candidates.sample(
    n=diagnostic_count,
    random_state=RANDOM_SEED
).copy()

print("Test ratings available:", len(test_candidates))
print("Test ratings sampled:", len(diagnostic_test))


# ================================================================
# 9. COMPARE RAW AND MEAN-CENTERED PREDICTIONS
# ================================================================

print("\nSTEP 6: Comparing prediction methods...")

prediction_cache = {}

raw_actual = []
raw_predicted = []

centered_actual = []
centered_predicted = []

for row_number, row in enumerate(
    diagnostic_test.itertuples(index=False),
    start=1
):

    user_id = row.userId
    movie_id = row.movieId
    actual_rating = float(row.rating)

    if user_id not in prediction_cache:

        prediction_cache[user_id] = {
            "raw": predict_ratings(
                user_id,
                "raw"
            ),
            "mean_centered": predict_ratings(
                user_id,
                "mean_centered"
            )
        }

    movie_index = movie_to_index[movie_id]

    raw_prediction = prediction_cache[user_id][
        "raw"
    ][movie_index]

    centered_prediction = prediction_cache[user_id][
        "mean_centered"
    ][movie_index]

    raw_actual.append(actual_rating)
    raw_predicted.append(float(raw_prediction))

    centered_actual.append(actual_rating)
    centered_predicted.append(float(centered_prediction))

    if row_number % 5000 == 0:

        print(
            "Evaluated",
            row_number,
            "test ratings..."
        )


# ================================================================
# 10. CALCULATE PREDICTION ERROR METRICS
# ================================================================

def summarize_predictions(
    actual,
    predicted,
    method_name
):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "prediction_method": method_name,
        "observations": len(actual),
        "MAE": mean_absolute_error(
            actual,
            predicted
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        ),
        "mean_prediction": np.mean(predicted),
        "prediction_std": np.std(predicted),
        "percentage_at_5": (
            np.mean(predicted >= 4.999) * 100
        ),
        "percentage_at_1": (
            np.mean(predicted <= 1.001) * 100
        ),
        "unique_prediction_values_rounded": len(
            np.unique(np.round(predicted, 2))
        )
    }


summary_df = pd.DataFrame([
    summarize_predictions(
        raw_actual,
        raw_predicted,
        "Raw weighted ratings"
    ),
    summarize_predictions(
        centered_actual,
        centered_predicted,
        "Mean-centered CF"
    )
])

print("\n" + "=" * 72)
print("PREDICTION DIAGNOSTICS")
print("=" * 72)

print(
    summary_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.4f}".format,
            "RMSE": "{:.4f}".format,
            "mean_prediction": "{:.4f}".format,
            "prediction_std": "{:.4f}".format,
            "percentage_at_5": "{:.2f}%".format,
            "percentage_at_1": "{:.2f}%".format
        }
    )
)


# ================================================================
# 11. CALCULATE NOVELTY SCORES
# ================================================================

print("\nSTEP 7: Preparing novelty scores...")

movie_popularity = rated_mask.sum(axis=0)

popularity_probability = np.maximum(
    movie_popularity / len(user_ids),
    1 / len(user_ids)
)

novelty = (
    -np.log2(popularity_probability)
    / np.log2(len(user_ids))
).astype(np.float32)

print("Novelty scores calculated.")


# ================================================================
# 12. GENERATE RECOMMENDATIONS
# ================================================================

def recommend(
    user_id,
    predictions,
    novelty_weight
):

    user_index = user_to_index[user_id]

    # Exclude movies the user rated in training.
    candidate_indices = np.flatnonzero(
        rated_mask[user_index] == 0
    )

    if len(candidate_indices) == 0:
        return pd.DataFrame()

    candidate_predictions = predictions[
        candidate_indices
    ]

    preference_scores = (
        candidate_predictions - 1.0
    ) / 4.0

    candidate_novelty = novelty[
        candidate_indices
    ]

    final_scores = (
        (1.0 - novelty_weight) * preference_scores
        + novelty_weight * candidate_novelty
    )

    top_count = min(
        TOP_K,
        len(candidate_indices)
    )

    top_positions = np.argpartition(
        final_scores,
        -top_count
    )[-top_count:]

    top_positions = top_positions[
        np.argsort(
            final_scores[top_positions]
        )[::-1]
    ]

    selected_indices = candidate_indices[
        top_positions
    ]

    result = pd.DataFrame({
        "userId": user_id,
        "movieId": movie_ids[selected_indices],
        "predicted_rating": predictions[selected_indices],
        "novelty_score": novelty[selected_indices],
        "preference_score": (
            predictions[selected_indices] - 1.0
        ) / 4.0,
        "novelty_weight": novelty_weight,
        "final_score": final_scores[top_positions]
    })

    result = result.merge(
        movies,
        on="movieId",
        how="left"
    )

    return result


# ================================================================
# 13. PREPARE RELEVANT TEST ITEMS AND EVALUATION USERS
# ================================================================

print("\nSTEP 8: Preparing recommendation evaluation...")

# A test rating of 4 or 5 is considered relevant.
test_relevant = test[
    test["rating"] >= 4
].copy()

relevant_by_user = (
    test_relevant.groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

eligible_users = [
    user_id
    for user_id in relevant_by_user
    if user_id in user_to_index
    and len(relevant_by_user[user_id]) > 0
]

evaluation_rng = np.random.default_rng(
    RANDOM_SEED
)

evaluation_users = evaluation_rng.choice(
    eligible_users,
    size=min(
        EVALUATION_USER_LIMIT,
        len(eligible_users)
    ),
    replace=False
).tolist()

print("Eligible users:", len(eligible_users))
print("Evaluation users:", len(evaluation_users))


# ================================================================
# 14. CALCULATE RANKING METRICS
# ================================================================

def calculate_ndcg(
    recommended_ids,
    relevant_ids
):

    dcg = 0.0

    for rank, movie_id in enumerate(
        recommended_ids[:TOP_K],
        start=1
    ):

        if movie_id in relevant_ids:

            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(
        len(relevant_ids),
        TOP_K
    )

    if ideal_hits == 0:
        return 0.0

    ideal_dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    return dcg / ideal_dcg


# Cache predictions for the evaluation users.
for user_id in evaluation_users:

    if user_id not in prediction_cache:

        prediction_cache[user_id] = {
            "raw": predict_ratings(
                user_id,
                "raw"
            ),
            "mean_centered": predict_ratings(
                user_id,
                "mean_centered"
            )
        }


# ================================================================
# 15. COMPARE METHODS AND NOVELTY WEIGHTS
# ================================================================

print("\nSTEP 9: Evaluating recommendation settings...")

comparison_rows = []
recommendation_rows = []

for method_name in [
    "raw",
    "mean_centered"
]:

    for novelty_weight in NOVELTY_WEIGHTS:

        precisions = []
        recalls = []
        ndcgs = []
        novelty_values = []

        catalog = set()

        for user_id in evaluation_users:

            predictions = prediction_cache[user_id][
                method_name
            ]

            recommendations = recommend(
                user_id,
                predictions,
                novelty_weight
            )

            recommended_ids = (
                recommendations["movieId"].tolist()
            )

            relevant_ids = relevant_by_user.get(
                user_id,
                set()
            )

            hits = sum(
                movie_id in relevant_ids
                for movie_id in recommended_ids
            )

            precisions.append(
                hits / TOP_K
            )

            recalls.append(
                hits / len(relevant_ids)
                if relevant_ids
                else 0.0
            )

            ndcgs.append(
                calculate_ndcg(
                    recommended_ids,
                    relevant_ids
                )
            )

            novelty_values.append(
                recommendations["novelty_score"].mean()
                if not recommendations.empty
                else 0.0
            )

            catalog.update(recommended_ids)

            # Keep User 1's recommendations for every setting.
            if user_id == 1:

                user_recs = recommendations.copy()

                user_recs["prediction_method"] = (
                    method_name
                )

                recommendation_rows.append(
                    user_recs
                )

        comparison_rows.append({
            "prediction_method": method_name,
            "novelty_weight": novelty_weight,
            "preference_weight": 1.0 - novelty_weight,
            "precision_at_10": np.mean(precisions),
            "recall_at_10": np.mean(recalls),
            "ndcg_at_10": np.mean(ndcgs),
            "average_novelty": np.mean(novelty_values),
            "catalog_coverage": (
                len(catalog) / len(movie_ids)
            ),
            "distinct_movies_recommended": len(catalog),
            "evaluation_users": len(evaluation_users)
        })

        print(
            "Completed:",
            method_name,
            "| novelty weight:",
            f"{novelty_weight:.0%}"
        )


# ================================================================
# 16. SAVE RESULTS
# ================================================================

print("\nSTEP 10: Saving results...")

comparison_df = pd.DataFrame(
    comparison_rows
)

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False
)

comparison_df.to_csv(
    OUTPUT_COMPARISON,
    index=False
)

if recommendation_rows:

    all_user1_recommendations = pd.concat(
        recommendation_rows,
        ignore_index=True
    )

    all_user1_recommendations.to_csv(
        OUTPUT_RECOMMENDATIONS,
        index=False
    )

else:

    # Create an empty file with headers if User 1 is unavailable.
    pd.DataFrame(
        columns=[
            "userId",
            "movieId",
            "predicted_rating",
            "novelty_score",
            "preference_score",
            "novelty_weight",
            "final_score",
            "title",
            "genres",
            "prediction_method"
        ]
    ).to_csv(
        OUTPUT_RECOMMENDATIONS,
        index=False
    )


# ================================================================
# 17. DISPLAY MODEL COMPARISON
# ================================================================

print("\n" + "=" * 72)
print("STAGE 8: MODEL COMPARISON")
print("=" * 72)

print(
    comparison_df.to_string(
        index=False,
        formatters={
            "novelty_weight": "{:.0%}".format,
            "preference_weight": "{:.0%}".format,
            "precision_at_10": "{:.4f}".format,
            "recall_at_10": "{:.4f}".format,
            "ndcg_at_10": "{:.4f}".format,
            "average_novelty": "{:.4f}".format,
            "catalog_coverage": "{:.2%}".format
        }
    )
)


# ================================================================
# 18. DISPLAY USER 1'S RECOMMENDATIONS
# ================================================================

print("\n" + "=" * 72)
print("USER 1: RECOMMENDATION PREVIEW")
print("=" * 72)

if recommendation_rows:

    user1_results = pd.concat(
        recommendation_rows,
        ignore_index=True
    )

    preview = user1_results[
        user1_results["prediction_method"] == "mean_centered"
    ]

    preview = preview[
        preview["novelty_weight"] == 0.10
    ]

    if not preview.empty:

        print(
            preview[
                [
                    "movieId",
                    "title",
                    "genres",
                    "predicted_rating",
                    "novelty_score",
                    "final_score"
                ]
            ].to_string(index=False)
        )

    else:
        print("No recommendations available for the preview.")

else:
    print("User 1 was not available for recommendation preview.")


# ================================================================
# 19. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 72)
print("STAGE 8 COMPLETED SUCCESSFULLY")
print("=" * 72)

print("\nSaved files:")

print("1.", OUTPUT_SUMMARY)
print("2.", OUTPUT_COMPARISON)
print("3.", OUTPUT_RECOMMENDATIONS)

print(
    "\nTotal execution time:",
    round(time.time() - START_TIME, 2),
    "seconds"
)

print("\nPlease review the results before proceeding to Stage 9.")
