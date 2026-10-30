
# ================================================================
# STAGE 10: PREDICTION SATURATION AND RANKING DIAGNOSTICS
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

OUTPUT_PREDICTIONS = os.path.join(
    PROJECT_PATH,
    "stage10_prediction_diagnostics.csv"
)

OUTPUT_USER1 = os.path.join(
    PROJECT_PATH,
    "stage10_user1_score_diagnostics.csv"
)

OUTPUT_SUMMARY = os.path.join(
    PROJECT_PATH,
    "stage10_diagnostic_summary.csv"
)

SEED = 42
NEIGHBOR_LIMIT = 50
SAMPLE_SIZE = 20000

print("=" * 72)
print("STAGE 10: PREDICTION AND RANKING DIAGNOSTICS")
print("=" * 72)


# ================================================================
# 2. LOAD DATA
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
# 3. RECREATE THE SAME TRAIN-TEST SPLIT
# ================================================================

print("\nSTEP 2: Recreating the Stage 9 split...")

rng = np.random.default_rng(SEED)

permutation = rng.permutation(len(ratings))
split_point = int(0.80 * len(ratings))

train = ratings.iloc[
    permutation[:split_point]
].copy()

test = ratings.iloc[
    permutation[split_point:]
].copy()

print("Training ratings:", len(train))
print("Testing ratings:", len(test))


# ================================================================
# 4. BUILD TRAINING MATRIX
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

centered_matrix = np.where(
    rated_mask > 0,
    rating_matrix - user_means[:, None],
    0
).astype(np.float32)

print("Training matrix:", rating_matrix.shape)
print(
    "Global training mean:",
    round(float(train["rating"].mean()), 4)
)


# ================================================================
# 5. CALCULATE USER SIMILARITY
# ================================================================

print("\nSTEP 4: Calculating user similarity...")

similarity_matrix = cosine_similarity(
    rating_matrix
).astype(np.float32)

np.fill_diagonal(similarity_matrix, 0)

print("User similarity calculation completed.")


# ================================================================
# 6. DEFINE PREDICTION FUNCTION
# ================================================================

def predict_user(user_id):

    user_index = user_to_index[user_id]

    similarities = similarity_matrix[
        user_index
    ].copy()

    similarities[similarities <= 0] = 0

    positive_indices = np.flatnonzero(
        similarities > 0
    )

    if len(positive_indices) > NEIGHBOR_LIMIT:

        strongest = np.argpartition(
            similarities[positive_indices],
            -NEIGHBOR_LIMIT
        )[-NEIGHBOR_LIMIT:]

        neighbor_indices = positive_indices[strongest]

    else:

        neighbor_indices = positive_indices

    if len(neighbor_indices) == 0:

        raw_predictions = np.full(
            len(movie_ids),
            user_means[user_index],
            dtype=np.float32
        )

        return raw_predictions, raw_predictions.copy(), 0

    weights = similarities[neighbor_indices]

    neighbor_masks = rated_mask[neighbor_indices]

    weighted_deviations = (
        weights @ centered_matrix[neighbor_indices]
    )

    denominator = weights @ neighbor_masks

    adjustments = np.divide(
        weighted_deviations,
        denominator,
        out=np.zeros(
            len(movie_ids),
            dtype=np.float32
        ),
        where=denominator > 0
    )

    # Unclipped prediction for diagnosing saturation.
    raw_predictions = (
        user_means[user_index] + adjustments
    )

    # Actual predictions used in Stage 9.
    clipped_predictions = np.clip(
        raw_predictions,
        1.0,
        5.0
    )

    return (
        raw_predictions,
        clipped_predictions,
        len(neighbor_indices)
    )


# ================================================================
# 7. DIAGNOSE PREDICTIONS FOR A TEST SAMPLE
# ================================================================

print("\nSTEP 5: Sampling held-out ratings...")

sample_size = min(SAMPLE_SIZE, len(test))

sample = test.sample(
    n=sample_size,
    random_state=SEED
).copy()

prediction_rows = []

# Cache each user's predictions to avoid repeated calculations.
prediction_cache = {}

for count, row in enumerate(
    sample.itertuples(index=False),
    start=1
):

    user_id = row.userId
    movie_id = row.movieId
    actual_rating = float(row.rating)

    if user_id not in user_to_index:
        continue

    if movie_id not in movie_to_index:
        continue

    if user_id not in prediction_cache:

        prediction_cache[user_id] = predict_user(
            user_id
        )

    raw_predictions, clipped_predictions, neighbor_count = (
        prediction_cache[user_id]
    )

    movie_index = movie_to_index[movie_id]

    raw_prediction = float(
        raw_predictions[movie_index]
    )

    clipped_prediction = float(
        clipped_predictions[movie_index]
    )

    prediction_rows.append({
        "userId": user_id,
        "movieId": movie_id,
        "actual_rating": actual_rating,
        "raw_prediction": raw_prediction,
        "clipped_prediction": clipped_prediction,
        "absolute_error": abs(
            actual_rating - clipped_prediction
        ),
        "neighbor_count": neighbor_count
    })

    if count % 5000 == 0:

        print(
            "Sampled test rows processed:",
            count
        )

predictions_df = pd.DataFrame(prediction_rows)

if predictions_df.empty:
    raise ValueError(
        "No predictions could be generated."
    )

predictions_df.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)

print(
    "Saved prediction-level diagnostics:",
    len(predictions_df),
    "rows"
)


# ================================================================
# 8. ANALYSE PREDICTION SATURATION
# ================================================================

print("\nSTEP 6: Analysing prediction saturation...")

raw = predictions_df["raw_prediction"]
clipped = predictions_df["clipped_prediction"]

saturation_summary = {
    "sampled_predictions": len(predictions_df),

    "raw_prediction_mean": raw.mean(),
    "raw_prediction_std": raw.std(),

    "clipped_prediction_mean": clipped.mean(),
    "clipped_prediction_std": clipped.std(),

    "raw_predictions_below_1": (
        raw < 1.0
    ).mean(),

    "raw_predictions_above_5": (
        raw > 5.0
    ).mean(),

    "clipped_predictions_at_1": (
        clipped <= 1.0
    ).mean(),

    "clipped_predictions_at_5": (
        clipped >= 5.0
    ).mean(),

    "clipped_predictions_at_5_rounded": (
        clipped.round(2) == 5.0
    ).mean(),

    "unique_raw_predictions_2dp": (
        raw.round(2).nunique()
    ),

    "unique_clipped_predictions_2dp": (
        clipped.round(2).nunique()
    ),

    "mean_absolute_error": (
        predictions_df["absolute_error"].mean()
    ),

    "mean_neighbor_count": (
        predictions_df["neighbor_count"].mean()
    )
}

summary_df = pd.DataFrame(
    [
        {
            "diagnostic": key,
            "value": value
        }
        for key, value in saturation_summary.items()
    ]
)

print("\nPrediction diagnostics:")

for key, value in saturation_summary.items():

    if isinstance(value, (float, np.floating)):

        print(f"{key}: {value:.4f}")

    else:

        print(f"{key}: {value}")


# ================================================================
# 9. ANALYSE USER 1 CANDIDATE SCORES
# ================================================================

print("\nSTEP 7: Analysing User 1...")

user1_rows = []

if 1 in user_to_index:

    raw_predictions, clipped_predictions, neighbor_count = (
        predict_user(1)
    )

    user1_index = user_to_index[1]

    candidate_indices = np.flatnonzero(
        rated_mask[user1_index] == 0
    )

    candidate_predictions = clipped_predictions[
        candidate_indices
    ]

    # Calculate training-only novelty.
    popularity = rated_mask.sum(axis=0)

    popularity_probability = np.maximum(
        popularity / len(user_ids),
        1 / len(user_ids)
    )

    novelty = (
        -np.log2(popularity_probability)
        / np.log2(len(user_ids))
    )

    candidate_novelty = novelty[candidate_indices]

    candidate_table = pd.DataFrame({
        "movieId": movie_ids[candidate_indices],
        "raw_prediction": raw_predictions[candidate_indices],
        "predicted_rating": candidate_predictions,
        "novelty_score": candidate_novelty
    })

    candidate_table = candidate_table.merge(
        movies,
        on="movieId",
        how="left"
    )

    candidate_table["preference_score"] = (
        candidate_table["predicted_rating"] - 1
    ) / 4

    candidate_table["raw_preference_score"] = (
        candidate_table["raw_prediction"] - 1
    ) / 4

    candidate_table["final_score_weight_0"] = (
        candidate_table["preference_score"]
    )

    candidate_table["final_score_weight_001"] = (
        0.99 * candidate_table["preference_score"]
        + 0.01 * candidate_table["novelty_score"]
    )

    candidate_table["final_score_weight_005"] = (
        0.95 * candidate_table["preference_score"]
        + 0.05 * candidate_table["novelty_score"]
    )

    candidate_table["prediction_is_5"] = (
        candidate_table["predicted_rating"] >= 5.0
    )

    candidate_table["raw_prediction_above_5"] = (
        candidate_table["raw_prediction"] > 5.0
    )

    candidate_table["neighbor_count"] = neighbor_count

    # Rank each score independently.
    candidate_table["rank_weight_0"] = (
        candidate_table["final_score_weight_0"]
        .rank(method="min", ascending=False)
    )

    candidate_table["rank_weight_001"] = (
        candidate_table["final_score_weight_001"]
        .rank(method="min", ascending=False)
    )

    candidate_table["rank_weight_005"] = (
        candidate_table["final_score_weight_005"]
        .rank(method="min", ascending=False)
    )

    candidate_table = candidate_table.sort_values(
        ["final_score_weight_0", "movieId"],
        ascending=[False, True],
        kind="mergesort"
    )

    candidate_table.to_csv(
        OUTPUT_USER1,
        index=False
    )

    user1_rows = [
        {
            "diagnostic": "User 1 candidate movies",
            "value": len(candidate_table)
        },
        {
            "diagnostic": "User 1 neighbors used",
            "value": neighbor_count
        },
        {
            "diagnostic": "User 1 candidates predicted 5",
            "value": int(
                candidate_table["prediction_is_5"].sum()
            )
        },
        {
            "diagnostic": "User 1 proportion predicted 5",
            "value": candidate_table["prediction_is_5"].mean()
        },
        {
            "diagnostic": "User 1 raw predictions above 5",
            "value": int(
                candidate_table["raw_prediction_above_5"].sum()
            )
        },
        {
            "diagnostic": "User 1 unique clipped predictions 2dp",
            "value": candidate_table["predicted_rating"].round(2).nunique()
        },
        {
            "diagnostic": "User 1 unique baseline scores 4dp",
            "value": candidate_table["final_score_weight_0"].round(4).nunique()
        },
        {
            "diagnostic": "User 1 unique 1% novelty scores 4dp",
            "value": candidate_table["final_score_weight_001"].round(4).nunique()
        },
        {
            "diagnostic": "User 1 unique 5% novelty scores 4dp",
            "value": candidate_table["final_score_weight_005"].round(4).nunique()
        }
    ]

    print("User 1 training mean:", round(float(user_means[user1_index]), 4))
    print("User 1 neighbors used:", neighbor_count)
    print("User 1 candidate movies:", len(candidate_table))
    print(
        "User 1 candidate predictions at 5:",
        int(candidate_table["prediction_is_5"].sum())
    )
    print(
        "User 1 proportion at 5:",
        f"{candidate_table['prediction_is_5'].mean():.2%}"
    )
    print(
        "Unique clipped predictions (2 decimals):",
        candidate_table["predicted_rating"].round(2).nunique()
    )

    print("\nUser 1 top 10 by baseline preference:")

    print(
        candidate_table[
            [
                "movieId",
                "title",
                "raw_prediction",
                "predicted_rating",
                "novelty_score",
                "rank_weight_0",
                "rank_weight_001",
                "rank_weight_005"
            ]
        ].head(10).to_string(index=False)
    )

else:

    print("User 1 is not present in the training data.")


# ================================================================
# 10. SAVE SUMMARY
# ================================================================

print("\nSTEP 8: Saving diagnostic summaries...")

user1_summary_df = pd.DataFrame(user1_rows)

if not user1_summary_df.empty:

    summary_df = pd.concat(
        [
            summary_df,
            user1_summary_df
        ],
        ignore_index=True
    )

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ================================================================
# 11. FINAL OUTPUT
# ================================================================

print("\n" + "=" * 72)
print("STAGE 10 COMPLETED SUCCESSFULLY")
print("=" * 72)

print("\nFiles saved:")

print("1.", OUTPUT_PREDICTIONS)
print("2.", OUTPUT_USER1)
print("3.", OUTPUT_SUMMARY)

print(
    "\nExecution time:",
    round(time.time() - START_TIME, 2),
    "seconds"
)

print(
    "\nThis stage diagnoses prediction saturation and ranking ties. "
    "It does not change the recommendation algorithm."
)
