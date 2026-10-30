
# ================================================================
# STAGE 11: MEAN-CENTERED USER SIMILARITY COMPARISON
# Project: Serendipitous Novelty-Aware Movie Recommendation System
# Dataset: MovieLens 1M
# ================================================================

import os
import time
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


# ================================================================
# 1. SETTINGS
# ================================================================

START_TIME = time.time()

PROJECT_PATH = (
    r"C:\Users\manoj\OneDrive\Desktop"
    r"\Serendipitous Recommendation Engine"
)

DATA_PATH = os.path.join(PROJECT_PATH, "data")

RATINGS_PATH = os.path.join(DATA_PATH, "ratings.csv")
MOVIES_PATH = os.path.join(DATA_PATH, "movies.csv")

OUTPUT_METRICS = os.path.join(
    PROJECT_PATH,
    "stage11_prediction_comparison.csv"
)

OUTPUT_RANKING = os.path.join(
    PROJECT_PATH,
    "stage11_user1_ranking_comparison.csv"
)

OUTPUT_PREDICTIONS = os.path.join(
    PROJECT_PATH,
    "stage11_prediction_diagnostics.csv"
)

SEED = 42
SAMPLE_SIZE = 20000
NEIGHBOR_LIMIT = 50
TOP_K = 10

NOVELTY_WEIGHTS = [0.00, 0.01, 0.02, 0.05, 0.10]

print("=" * 72)
print("STAGE 11: MEAN-CENTERED SIMILARITY COMPARISON")
print("=" * 72)


# ================================================================
# 2. LOAD AND VALIDATE DATA
# ================================================================

print("\nSTEP 1: Loading datasets...")

ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)

ratings = ratings[
    ["userId", "movieId", "rating"]
].dropna().copy()

movies = movies[
    ["movieId", "title", "genres"]
].drop_duplicates("movieId").copy()

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


# ================================================================
# 3. RECREATE STAGE 9/10 TRAIN-TEST SPLIT
# ================================================================

print("\nSTEP 2: Creating the fixed train-test split...")

rng = np.random.default_rng(SEED)

permutation = rng.permutation(len(ratings))
split_point = int(0.80 * len(ratings))

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
print("Train-test pair overlap:", overlap)

if overlap != 0:
    raise ValueError("Train-test overlap detected.")


# ================================================================
# 4. BUILD THE TRAINING MATRIX
# ================================================================

print("\nSTEP 3: Building the training matrix...")

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

user_rating_counts = rated_mask.sum(axis=1)

user_means = np.divide(
    rating_matrix.sum(axis=1),
    user_rating_counts,
    out=np.zeros(
        len(user_ids),
        dtype=np.float32
    ),
    where=user_rating_counts > 0
)

global_mean = float(train["rating"].mean())

# Missing ratings remain zero in this matrix.
# Only observed ratings are centered.
centered_matrix = np.where(
    rated_mask > 0,
    rating_matrix - user_means[:, None],
    0
).astype(np.float32)

print("Matrix shape:", rating_matrix.shape)
print("Global training mean:", round(global_mean, 4))


# ================================================================
# 5. CALCULATE TWO SIMILARITY MATRICES
# ================================================================

print("\nSTEP 4: Calculating similarity matrices...")

# Model A: cosine similarity using raw ratings.
raw_similarity = cosine_similarity(
    rating_matrix
).astype(np.float32)

# Model B: cosine similarity using mean-centered ratings.
centered_similarity = cosine_similarity(
    centered_matrix
).astype(np.float32)

# Do not allow a user to be their own neighbor.
np.fill_diagonal(raw_similarity, 0)
np.fill_diagonal(centered_similarity, 0)

print("Raw-rating similarity completed.")
print("Mean-centered similarity completed.")


# ================================================================
# 6. DEFINE THE NEIGHBOR-BASED PREDICTOR
# ================================================================

def predict_user(user_id, similarity_matrix, use_centered):

    user_index = user_to_index[user_id]

    similarities = similarity_matrix[
        user_index
    ].copy()

    # Keep positive similarity values only.
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

    if len(neighbor_indices) == 0:

        # Fallback: target user's mean rating.
        base = user_means[user_index]

        return np.full(
            len(movie_ids),
            base,
            dtype=np.float32
        ), 0

    weights = similarities[neighbor_indices]

    neighbor_masks = rated_mask[
        neighbor_indices
    ]

    if use_centered:

        # Weighted deviations from neighbors' own means.
        neighbor_values = centered_matrix[
            neighbor_indices
        ]

        base = user_means[user_index]

    else:

        # Weighted raw ratings, retained for baseline comparison.
        neighbor_values = rating_matrix[
            neighbor_indices
        ]

        base = 0.0

    weighted_values = (
        weights @ neighbor_values
    )

    denominator = weights @ neighbor_masks

    neighbor_contribution = np.divide(
        weighted_values,
        denominator,
        out=np.zeros(
            len(movie_ids),
            dtype=np.float32
        ),
        where=denominator > 0
    )

    predictions = base + neighbor_contribution

    # For movies no selected neighbor rated, use a reasonable
    # fallback rather than treating missing ratings as zero.
    no_neighbor_rating = denominator <= 0

    if use_centered:

        predictions[no_neighbor_rating] = (
            user_means[user_index]
        )

    else:

        predictions[no_neighbor_rating] = (
            user_means[user_index]
        )

    return predictions.astype(np.float32), len(neighbor_indices)


# ================================================================
# 7. SAMPLE HELD-OUT RATINGS
# ================================================================

print("\nSTEP 5: Preparing held-out evaluation sample...")

sample_size = min(
    SAMPLE_SIZE,
    len(test)
)

sample = test.sample(
    n=sample_size,
    random_state=SEED
).copy()

sample = sample[
    sample["userId"].isin(user_to_index)
    & sample["movieId"].isin(movie_to_index)
].copy()

print("Evaluation rows:", len(sample))


# ================================================================
# 8. PREDICT USING BOTH METHODS
# ================================================================

print("\nSTEP 6: Generating predictions...")

prediction_cache_raw = {}
prediction_cache_centered = {}

prediction_rows = []

unique_sample_users = sample["userId"].unique()

for count, user_id in enumerate(
    unique_sample_users,
    start=1
):

    prediction_cache_raw[user_id] = predict_user(
        user_id,
        raw_similarity,
        use_centered=False
    )

    prediction_cache_centered[user_id] = predict_user(
        user_id,
        centered_similarity,
        use_centered=True
    )

    if count % 500 == 0:

        print(
            "Users processed:",
            count,
            "of",
            len(unique_sample_users)
        )

for row in sample.itertuples(index=False):

    user_id = row.userId
    movie_id = row.movieId
    actual = float(row.rating)

    movie_index = movie_to_index[movie_id]

    raw_predictions, raw_neighbors = (
        prediction_cache_raw[user_id]
    )

    centered_predictions, centered_neighbors = (
        prediction_cache_centered[user_id]
    )

    raw_prediction = float(
        raw_predictions[movie_index]
    )

    centered_prediction = float(
        centered_predictions[movie_index]
    )

    raw_clipped = float(
        np.clip(raw_prediction, 1, 5)
    )

    centered_clipped = float(
        np.clip(centered_prediction, 1, 5)
    )

    prediction_rows.append({
        "userId": user_id,
        "movieId": movie_id,
        "actual_rating": actual,

        "raw_prediction": raw_prediction,
        "raw_clipped_prediction": raw_clipped,
        "raw_absolute_error": abs(
            actual - raw_clipped
        ),

        "centered_prediction": centered_prediction,
        "centered_clipped_prediction": centered_clipped,
        "centered_absolute_error": abs(
            actual - centered_clipped
        ),

        "raw_neighbor_count": raw_neighbors,
        "centered_neighbor_count": centered_neighbors
    })

predictions_df = pd.DataFrame(prediction_rows)

predictions_df.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)

print("Saved prediction diagnostics.")


# ================================================================
# 9. CALCULATE PREDICTION METRICS
# ================================================================

print("\nSTEP 7: Comparing prediction accuracy...")

def prediction_metrics(actual, predicted):

    errors = actual - predicted

    return {
        "MAE": np.mean(np.abs(errors)),
        "RMSE": np.sqrt(np.mean(errors ** 2)),
        "prediction_mean": np.mean(predicted),
        "prediction_std": np.std(predicted),
        "fraction_at_1": np.mean(predicted <= 1.0),
        "fraction_at_5": np.mean(predicted >= 5.0),
        "unique_predictions_2dp": (
            pd.Series(predicted).round(2).nunique()
        )
    }


actual = predictions_df[
    "actual_rating"
].to_numpy()

raw_pred = predictions_df[
    "raw_clipped_prediction"
].to_numpy()

centered_pred = predictions_df[
    "centered_clipped_prediction"
].to_numpy()

raw_metrics = prediction_metrics(
    actual,
    raw_pred
)

centered_metrics = prediction_metrics(
    actual,
    centered_pred
)

metric_rows = []

for model_name, metrics in [
    ("Raw-rating cosine CF", raw_metrics),
    ("Mean-centered cosine CF", centered_metrics)
]:

    row = {
        "model": model_name,
        "evaluation_rows": len(actual)
    }

    row.update(metrics)

    metric_rows.append(row)

metrics_df = pd.DataFrame(metric_rows)

print("\nPREDICTION COMPARISON")

print(
    metrics_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.4f}".format,
            "RMSE": "{:.4f}".format,
            "prediction_mean": "{:.4f}".format,
            "prediction_std": "{:.4f}".format,
            "fraction_at_1": "{:.2%}".format,
            "fraction_at_5": "{:.2%}".format
        }
    )
)


# ================================================================
# 10. BUILD USER 1 RECOMMENDATION COMPARISON
# ================================================================

print("\nSTEP 8: Comparing User 1 recommendations...")

ranking_rows = []

if 1 in user_to_index:

    # Novelty is calculated from training data only.
    popularity = rated_mask.sum(axis=0)

    popularity_probability = np.maximum(
        popularity / len(user_ids),
        1 / len(user_ids)
    )

    novelty = (
        -np.log2(popularity_probability)
        / np.log2(len(user_ids))
    )

    raw_user1, raw_neighbor_count = predict_user(
        1,
        raw_similarity,
        use_centered=False
    )

    centered_user1, centered_neighbor_count = predict_user(
        1,
        centered_similarity,
        use_centered=True
    )

    user1_index = user_to_index[1]

    candidate_indices = np.flatnonzero(
        rated_mask[user1_index] == 0
    )

    for model_name, predictions, neighbor_count in [
        (
            "Raw-rating cosine CF",
            raw_user1,
            raw_neighbor_count
        ),
        (
            "Mean-centered cosine CF",
            centered_user1,
            centered_neighbor_count
        )
    ]:

        clipped = np.clip(
            predictions[candidate_indices],
            1,
            5
        )

        candidate_novelty = novelty[
            candidate_indices
        ]

        preference = (clipped - 1) / 4

        for weight in NOVELTY_WEIGHTS:

            final_score = (
                (1 - weight) * preference
                + weight * candidate_novelty
            )

            ranking_df = pd.DataFrame({
                "movieId": movie_ids[candidate_indices],
                "predicted_rating": clipped,
                "preference_score": preference,
                "novelty_score": candidate_novelty,
                "final_score": final_score
            })

            ranking_df = ranking_df.sort_values(
                ["final_score", "movieId"],
                ascending=[False, True],
                kind="mergesort"
            ).head(TOP_K)

            ranking_df.insert(
                0,
                "model",
                model_name
            )

            ranking_df["novelty_weight"] = weight

            ranking_df["neighbor_count"] = neighbor_count

            ranking_rows.append(
                ranking_df.merge(
                    movies,
                    on="movieId",
                    how="left"
                )
            )

    ranking_output = pd.concat(
        ranking_rows,
        ignore_index=True
    )

    ranking_output.to_csv(
        OUTPUT_RANKING,
        index=False
    )

    print(
        ranking_output[
            [
                "model",
                "novelty_weight",
                "movieId",
                "title",
                "predicted_rating",
                "novelty_score",
                "final_score"
            ]
        ].to_string(index=False)
    )

else:

    print("User 1 is not present in the training matrix.")

    pd.DataFrame().to_csv(
        OUTPUT_RANKING,
        index=False
    )


# ================================================================
# 11. SAVE SUMMARY
# ================================================================

metrics_df.to_csv(
    OUTPUT_METRICS,
    index=False
)


# ================================================================
# 12. INTERPRETATION
# ================================================================

print("\n" + "=" * 72)
print("STAGE 11 SUMMARY")
print("=" * 72)

mae_difference = (
    raw_metrics["MAE"] - centered_metrics["MAE"]
)

rmse_difference = (
    raw_metrics["RMSE"] - centered_metrics["RMSE"]
)

print(
    "MAE change (raw minus centered):",
    round(mae_difference, 4)
)

print(
    "RMSE change (raw minus centered):",
    round(rmse_difference, 4)
)

if mae_difference > 0 and rmse_difference > 0:

    print(
        "Mean-centered CF performed better on both MAE and RMSE "
        "for this evaluation sample."
    )

elif mae_difference < 0 and rmse_difference < 0:

    print(
        "Raw-rating CF performed better on both MAE and RMSE "
        "for this evaluation sample."
    )

else:

    print(
        "The methods have a mixed result. Compare both metrics "
        "before selecting the final model."
    )

print("\nSaved files:")

print("1.", OUTPUT_METRICS)
print("2.", OUTPUT_RANKING)
print("3.", OUTPUT_PREDICTIONS)

print(
    "\nExecution time:",
    round(time.time() - START_TIME, 2),
    "seconds"
)

print(
    "\nThese results are offline experiments and do not establish "
    "real-world recommendation quality."
)

print("\nSTAGE 11 COMPLETED")
