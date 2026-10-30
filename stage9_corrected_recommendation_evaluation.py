
# ================================================================
# STAGE 9: CORRECTED RECOMMENDATION RANKING AND FAIR EVALUATION
# Project: Serendipitous Novelty-Aware Movie Recommendation System
# Dataset: MovieLens 1M
# ================================================================

import os
import time
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


# ================================================================
# 1. SETTINGS AND PATHS
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
    "stage9_corrected_evaluation.csv"
)

OUTPUT_USER1 = os.path.join(
    PROJECT_PATH,
    "stage9_user1_recommendations.csv"
)

OUTPUT_ALL_USER1 = os.path.join(
    PROJECT_PATH,
    "stage9_user1_weight_comparison.csv"
)

SEED = 42
TOP_K = 10
NEIGHBOR_LIMIT = 50
EVALUATION_USER_LIMIT = 500

NOVELTY_WEIGHTS = [0.00, 0.01, 0.02, 0.05, 0.10, 0.15, 0.30]

print("=" * 72)
print("STAGE 9: CORRECTED RECOMMENDATION EVALUATION")
print("=" * 72)


# ================================================================
# 2. LOAD DATA
# ================================================================

print("\nSTEP 1: Loading datasets...")

ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)

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

movies = movies.drop_duplicates(
    subset=["movieId"],
    keep="last"
)

print("Ratings:", ratings.shape)
print("Movies:", movies.shape)
print("Unique users:", ratings["userId"].nunique())


# ================================================================
# 3. CREATE FIXED TRAIN-TEST SPLIT
# ================================================================

print("\nSTEP 2: Creating train-test split...")

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
print("Train-test overlap:", overlap)

if overlap != 0:
    raise ValueError("Train-test overlap detected.")


# ================================================================
# 4. CREATE TRAINING MATRICES
# ================================================================

print("\nSTEP 3: Creating training matrices...")

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

print("Training matrix shape:", rating_matrix.shape)


# ================================================================
# 5. CALCULATE USER SIMILARITY
# ================================================================

print("\nSTEP 4: Calculating user similarity...")

similarity_matrix = cosine_similarity(
    rating_matrix
).astype(np.float32)

np.fill_diagonal(similarity_matrix, 0)

print("Similarity calculation completed.")


# ================================================================
# 6. CALCULATE NOVELTY USING TRAINING DATA ONLY
# ================================================================

print("\nSTEP 5: Calculating movie novelty...")

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
# 7. PREDICT RATINGS WITH MEAN-CENTERED CF
# ================================================================

def predict_ratings(user_id):

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

        return np.full(
            len(movie_ids),
            user_means[user_index],
            dtype=np.float32
        )

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

    predictions = (
        user_means[user_index] + adjustments
    )

    return np.clip(
        predictions,
        1.0,
        5.0
    ).astype(np.float32)


# ================================================================
# 8. GENERATE RECOMMENDATIONS
# ================================================================

def recommend(user_id, predictions, novelty_weight):

    user_index = user_to_index[user_id]

    # A movie must be in the training matrix and unrated by this user.
    candidate_indices = np.flatnonzero(
        rated_mask[user_index] == 0
    )

    if len(candidate_indices) == 0:
        return pd.DataFrame()

    candidate_predictions = predictions[
        candidate_indices
    ]

    # Preference scores lie between 0 and 1.
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

    # Stable tie-breaking: movie ID determines order when scores tie.
    candidate_table = pd.DataFrame({
        "movie_index": candidate_indices,
        "movieId": movie_ids[candidate_indices],
        "predicted_rating": candidate_predictions,
        "preference_score": preference_scores,
        "novelty_score": candidate_novelty,
        "final_score": final_scores
    })

    candidate_table = candidate_table.sort_values(
        ["final_score", "movieId"],
        ascending=[False, True],
        kind="mergesort"
    )

    result = candidate_table.head(TOP_K).copy()

    result.insert(0, "userId", user_id)
    result["novelty_weight"] = novelty_weight

    result = result.merge(
        movies,
        on="movieId",
        how="left"
    )

    return result.reset_index(drop=True)


# ================================================================
# 9. DEFINE TEST RELEVANCE AND ELIGIBLE CANDIDATES
# ================================================================

print("\nSTEP 6: Preparing fair evaluation...")

# Ratings of 4 or 5 are treated as relevant.
test_relevant = test[
    test["rating"] >= 4
].copy()

# Only movies represented in the training matrix can be recommended.
test_relevant = test_relevant[
    test_relevant["movieId"].isin(movie_to_index)
    & test_relevant["userId"].isin(user_to_index)
].copy()

# A test movie is eligible for a user's recommendations only if
# that user did not rate that movie in the training set.
eligible_relevant = []

for row in test_relevant.itertuples(index=False):

    user_index = user_to_index[row.userId]
    movie_index = movie_to_index[row.movieId]

    if rated_mask[user_index, movie_index] == 0:

        eligible_relevant.append(
            (row.userId, row.movieId)
        )

eligible_relevant_df = pd.DataFrame(
    eligible_relevant,
    columns=["userId", "movieId"]
)

relevant_by_user = (
    eligible_relevant_df.groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

eligible_users = [
    user_id
    for user_id, relevant_movies in relevant_by_user.items()
    if len(relevant_movies) > 0
]

print("Eligible users:", len(eligible_users))
print(
    "Eligible relevant user-movie pairs:",
    len(eligible_relevant_df)
)

if not eligible_users:
    raise ValueError(
        "No eligible relevant test movies found."
    )

evaluation_rng = np.random.default_rng(SEED)

evaluation_users = evaluation_rng.choice(
    eligible_users,
    size=min(
        EVALUATION_USER_LIMIT,
        len(eligible_users)
    ),
    replace=False
).tolist()

print("Evaluation users:", len(evaluation_users))


# ================================================================
# 10. CALCULATE EVALUATION METRICS
# ================================================================

def calculate_ndcg(recommended_ids, relevant_ids):

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


def evaluate_user_recommendations(
    recommendations,
    relevant_ids
):

    recommended_ids = (
        recommendations["movieId"].tolist()
    )

    hits = sum(
        movie_id in relevant_ids
        for movie_id in recommended_ids
    )

    precision = hits / TOP_K

    recall = (
        hits / len(relevant_ids)
        if relevant_ids
        else 0.0
    )

    ndcg = calculate_ndcg(
        recommended_ids,
        relevant_ids
    )

    average_novelty = (
        recommendations["novelty_score"].mean()
        if not recommendations.empty
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "ndcg": ndcg,
        "novelty": average_novelty
    }


# ================================================================
# 11. EVALUATE ALL NOVELTY WEIGHTS
# ================================================================

print("\nSTEP 7: Evaluating novelty weights...")

metric_rows = []
all_user1_rows = []

# Cache predictions so they are calculated only once per user.
prediction_cache = {}

users_to_predict = set(evaluation_users)

# User 1 is added separately for a recommendation preview.
if 1 in user_to_index:
    users_to_predict.add(1)

for count, user_id in enumerate(
    users_to_predict,
    start=1
):

    prediction_cache[user_id] = predict_ratings(user_id)

    if count % 100 == 0:

        print(
            "Predictions calculated:",
            count
        )

for weight in NOVELTY_WEIGHTS:

    precisions = []
    recalls = []
    ndcgs = []
    novelty_values = []

    catalog = set()

    for user_id in evaluation_users:

        recommendations = recommend(
            user_id,
            prediction_cache[user_id],
            weight
        )

        relevant_ids = relevant_by_user.get(
            user_id,
            set()
        )

        metrics = evaluate_user_recommendations(
            recommendations,
            relevant_ids
        )

        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        ndcgs.append(metrics["ndcg"])
        novelty_values.append(metrics["novelty"])

        catalog.update(
            recommendations["movieId"].tolist()
        )

    metric_rows.append({
        "novelty_weight": weight,
        "preference_weight": 1.0 - weight,
        "precision_at_10": np.mean(precisions),
        "recall_at_10": np.mean(recalls),
        "ndcg_at_10": np.mean(ndcgs),
        "average_novelty": np.mean(novelty_values),
        "catalog_coverage": len(catalog) / len(movie_ids),
        "distinct_movies_recommended": len(catalog),
        "evaluation_users": len(evaluation_users)
    })

    # Save User 1's recommendations for every tested weight.
    if 1 in prediction_cache:

        user1_recommendations = recommend(
            1,
            prediction_cache[1],
            weight
        )

        if not user1_recommendations.empty:

            all_user1_rows.append(
                user1_recommendations
            )

    print(
        "Completed novelty weight:",
        f"{weight:.0%}"
    )


# ================================================================
# 12. SAVE OUTPUT FILES
# ================================================================

print("\nSTEP 8: Saving results...")

metrics_df = pd.DataFrame(metric_rows)

metrics_df.to_csv(
    OUTPUT_METRICS,
    index=False
)

if all_user1_rows:

    all_user1_df = pd.concat(
        all_user1_rows,
        ignore_index=True
    )

    all_user1_df.to_csv(
        OUTPUT_ALL_USER1,
        index=False
    )

    # Choose a conservative 0% novelty baseline for the separate
    # preview file; the full comparison remains available above.
    user1_baseline = all_user1_df[
        all_user1_df["novelty_weight"] == 0.0
    ].copy()

    user1_baseline.to_csv(
        OUTPUT_USER1,
        index=False
    )

else:

    all_user1_df = pd.DataFrame()

    pd.DataFrame(
        columns=[
            "userId",
            "movieId",
            "predicted_rating",
            "preference_score",
            "novelty_score",
            "final_score",
            "novelty_weight",
            "title",
            "genres"
        ]
    ).to_csv(
        OUTPUT_ALL_USER1,
        index=False
    )

    pd.DataFrame().to_csv(
        OUTPUT_USER1,
        index=False
    )


# ================================================================
# 13. DISPLAY EVALUATION RESULTS
# ================================================================

print("\n" + "=" * 72)
print("STAGE 9 RESULTS")
print("=" * 72)

print(
    metrics_df.to_string(
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
# 14. DISPLAY USER 1 RECOMMENDATIONS
# ================================================================

print("\n" + "=" * 72)
print("USER 1: BASELINE RECOMMENDATIONS")
print("=" * 72)

if not all_user1_df.empty:

    user1_baseline = all_user1_df[
        all_user1_df["novelty_weight"] == 0.0
    ]

    if not user1_baseline.empty:

        print(
            user1_baseline[
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

    print(
        "User 1 does not have sufficient training data "
        "for a recommendation preview."
    )


# ================================================================
# 15. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 72)
print("STAGE 9 COMPLETED SUCCESSFULLY")
print("=" * 72)

print("\nSaved files:")

print("1.", OUTPUT_METRICS)
print("2.", OUTPUT_USER1)
print("3.", OUTPUT_ALL_USER1)

print(
    "\nExecution time:",
    round(time.time() - START_TIME, 2),
    "seconds"
)

print(
    "\nImportant: These are offline evaluation results, "
    "not evidence of real-world user satisfaction."
)
