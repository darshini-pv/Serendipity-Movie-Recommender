
# ================================================================
# STAGE 7: NOVELTY WEIGHT TUNING AND IMPROVED COLLABORATIVE FILTERING
# Project: Serendipitous Novelty-Aware Movie Recommendation System
# Dataset: MovieLens 1M
# ================================================================

import os
import time
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


# ================================================================
# 1. PROJECT PATHS AND SETTINGS
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
    PROJECT_PATH, "stage7_novelty_weight_results.csv"
)

OUTPUT_RECOMMENDATIONS = os.path.join(
    PROJECT_PATH, "stage7_user1_recommendations.csv"
)

OUTPUT_ALL_RECOMMENDATIONS = os.path.join(
    PROJECT_PATH, "stage7_user1_weight_comparison.csv"
)

RANDOM_SEED = 42
EVALUATION_USER_LIMIT = 500
TOP_K = 10
NEIGHBOR_LIMIT = 50

NOVELTY_WEIGHTS = [0.00, 0.05, 0.10, 0.15, 0.30]

print("=" * 72)
print("STAGE 7: NOVELTY WEIGHT TUNING")
print("=" * 72)


# ================================================================
# 2. LOAD AND VALIDATE DATA
# ================================================================

print("\nSTEP 1: Loading datasets...")

ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)

required_rating_columns = {"userId", "movieId", "rating"}
required_movie_columns = {"movieId", "title", "genres"}

if not required_rating_columns.issubset(ratings.columns):
    raise ValueError(
        "ratings.csv must contain userId, movieId and rating."
    )

if not required_movie_columns.issubset(movies.columns):
    raise ValueError(
        "movies.csv must contain movieId, title and genres."
    )

ratings = ratings[["userId", "movieId", "rating"]].copy()
movies = movies[["movieId", "title", "genres"]].copy()

ratings = ratings.dropna(
    subset=["userId", "movieId", "rating"]
)

ratings["userId"] = ratings["userId"].astype(int)
ratings["movieId"] = ratings["movieId"].astype(int)
ratings["rating"] = pd.to_numeric(
    ratings["rating"], errors="coerce"
)

ratings = ratings.dropna(subset=["rating"])

# Remove duplicate user-movie pairs if any exist.
ratings = ratings.drop_duplicates(
    subset=["userId", "movieId"],
    keep="last"
)

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)
print("Unique users:", ratings["userId"].nunique())
print("Unique rated movies:", ratings["movieId"].nunique())


# ================================================================
# 3. CREATE A REPRODUCIBLE TRAIN-TEST SPLIT
# ================================================================

print("\nSTEP 2: Creating train-test split...")

rng = np.random.default_rng(RANDOM_SEED)

permutation = rng.permutation(len(ratings))
split_point = int(len(ratings) * 0.80)

train = ratings.iloc[permutation[:split_point]].copy()
test = ratings.iloc[permutation[split_point:]].copy()

train_pairs = pd.MultiIndex.from_frame(
    train[["userId", "movieId"]]
)
test_pairs = pd.MultiIndex.from_frame(
    test[["userId", "movieId"]]
)

overlap_count = len(train_pairs.intersection(test_pairs))

print("Training ratings:", len(train))
print("Testing ratings:", len(test))
print("Train-test pair overlap:", overlap_count)

if overlap_count != 0:
    raise ValueError("Data leakage detected between train and test.")

# Only training data will be used to calculate recommendations.
# Test ratings are reserved for evaluation.


# ================================================================
# 4. CREATE THE USER-MOVIE RATING MATRIX
# ================================================================

print("\nSTEP 3: Creating user-movie matrix...")

rating_matrix_df = train.pivot(
    index="userId",
    columns="movieId",
    values="rating"
).sort_index().sort_index(axis=1)

user_ids = rating_matrix_df.index.to_numpy()
movie_ids = rating_matrix_df.columns.to_numpy()

rating_matrix = rating_matrix_df.fillna(0).to_numpy(
    dtype=np.float32
)

rated_mask = (rating_matrix > 0).astype(np.float32)

# Mean rating for each user, calculated from training ratings only.
rating_counts = rated_mask.sum(axis=1)

user_means = np.divide(
    rating_matrix.sum(axis=1),
    rating_counts,
    out=np.zeros(len(user_ids), dtype=np.float32),
    where=rating_counts > 0
)

# Center only observed ratings. Missing ratings remain zero.
centered_matrix = np.where(
    rated_mask > 0,
    rating_matrix - user_means[:, None],
    0
).astype(np.float32)

user_to_index = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}

movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(movie_ids)
}

print("Rating matrix shape:", rating_matrix.shape)
print("Users in training:", len(user_ids))
print("Movies in training:", len(movie_ids))


# ================================================================
# 5. CALCULATE USER SIMILARITY
# ================================================================

print("\nSTEP 4: Calculating cosine similarity...")

similarity_matrix = cosine_similarity(rating_matrix).astype(
    np.float32
)

# A user should not be their own neighbor.
np.fill_diagonal(similarity_matrix, 0)

print("Similarity matrix shape:", similarity_matrix.shape)
print("User similarity calculation completed.")


# ================================================================
# 6. CALCULATE MOVIE NOVELTY FROM TRAINING DATA
# ================================================================

print("\nSTEP 5: Calculating movie novelty...")

# Popularity means the number of distinct training users
# who rated each movie.
movie_popularity = rated_mask.sum(axis=0)

number_of_training_users = len(user_ids)

popularity_probability = (
    movie_popularity / number_of_training_users
)

# A less popular movie receives a higher novelty score.
raw_novelty = -np.log2(
    np.maximum(
        popularity_probability,
        1 / number_of_training_users
    )
)

maximum_novelty = np.log2(number_of_training_users)

normalized_novelty = (
    raw_novelty / maximum_novelty
).astype(np.float32)

novelty_by_movie = {
    movie_id: float(normalized_novelty[index])
    for index, movie_id in enumerate(movie_ids)
}

print("Novelty scores calculated.")
print("Minimum novelty:", round(float(normalized_novelty.min()), 4))
print("Maximum novelty:", round(float(normalized_novelty.max()), 4))


# ================================================================
# 7. IMPROVED USER-BASED COLLABORATIVE FILTERING
# ================================================================

def predict_user_ratings(user_id):
    """
    Predict ratings for every movie in the training matrix.

    Formula:
        Prediction =
        Target user's mean rating
        + weighted deviations of similar users

    Only positive-similarity neighbors are used.
    The target user's own ratings are not used as recommendations.
    """

    if user_id not in user_to_index:
        return None

    user_index = user_to_index[user_id]

    similarities = similarity_matrix[user_index].copy()

    # Keep only positive similarities.
    similarities[similarities <= 0] = 0

    # Select the strongest positive neighbors.
    positive_indices = np.flatnonzero(similarities > 0)

    if len(positive_indices) == 0:
        return np.full(
            len(movie_ids),
            user_means[user_index],
            dtype=np.float32
        )

    if len(positive_indices) > NEIGHBOR_LIMIT:
        strongest = np.argpartition(
            similarities[positive_indices],
            -NEIGHBOR_LIMIT
        )[-NEIGHBOR_LIMIT:]

        neighbor_indices = positive_indices[strongest]
    else:
        neighbor_indices = positive_indices

    neighbor_weights = similarities[neighbor_indices]

    neighbor_deviations = centered_matrix[neighbor_indices]
    neighbor_rated = rated_mask[neighbor_indices]

    # Weighted sum of neighbors' rating deviations.
    numerator = neighbor_weights @ neighbor_deviations

    # Sum weights only for neighbors who rated each movie.
    denominator = neighbor_weights @ neighbor_rated

    adjustments = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > 0
    )

    predictions = user_means[user_index] + adjustments

    # Clip predictions to the MovieLens rating scale.
    predictions = np.clip(predictions, 1.0, 5.0)

    return predictions.astype(np.float32)


# ================================================================
# 8. GENERATE RECOMMENDATIONS FOR DIFFERENT NOVELTY WEIGHTS
# ================================================================

def recommend_for_user(user_id, predictions, novelty_weight):
    """
    Combine preference and novelty.

    Final score =
        (1 - novelty_weight) * normalized preference
        + novelty_weight * novelty
    """

    if user_id not in user_to_index:
        return pd.DataFrame()

    user_index = user_to_index[user_id]

    # Exclude movies already rated by this user in training.
    already_rated = rated_mask[user_index] > 0

    candidate_mask = ~already_rated

    candidate_indices = np.flatnonzero(candidate_mask)

    if len(candidate_indices) == 0:
        return pd.DataFrame()

    candidate_predictions = predictions[candidate_indices]

    # Normalize preference predictions from 1–5 into 0–1.
    preference_scores = (
        candidate_predictions - 1.0
    ) / 4.0

    candidate_novelty = normalized_novelty[candidate_indices]

    final_scores = (
        (1.0 - novelty_weight) * preference_scores
        + novelty_weight * candidate_novelty
    )

    # Select the top recommendations without sorting every movie.
    top_count = min(TOP_K, len(candidate_indices))

    top_positions = np.argpartition(
        final_scores,
        -top_count
    )[-top_count:]

    # Sort selected recommendations from highest to lowest score.
    top_positions = top_positions[
        np.argsort(final_scores[top_positions])[::-1]
    ]

    selected_indices = candidate_indices[top_positions]

    recommendations = pd.DataFrame({
        "userId": user_id,
        "movieId": movie_ids[selected_indices],
        "predicted_rating": predictions[selected_indices],
        "novelty_score": normalized_novelty[selected_indices],
        "preference_score": (
            predictions[selected_indices] - 1.0
        ) / 4.0,
        "novelty_weight": novelty_weight,
        "final_score": final_scores[top_positions]
    })

    recommendations = recommendations.merge(
        movies,
        on="movieId",
        how="left"
    )

    return recommendations.sort_values(
        "final_score",
        ascending=False
    ).head(TOP_K).reset_index(drop=True)


# ================================================================
# 9. EVALUATION METRICS
# ================================================================

def calculate_ndcg(recommended_ids, relevant_ids, k=TOP_K):
    """Calculate NDCG@K for binary relevance."""

    dcg = 0.0

    for rank, movie_id in enumerate(recommended_ids[:k], start=1):
        if movie_id in relevant_ids:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), k)

    if ideal_hits == 0:
        return 0.0

    ideal_dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / ideal_dcg


def evaluate_recommendations(recommendations, relevant_ids):
    """Calculate ranking relevance and discovery metrics."""

    recommended_ids = recommendations["movieId"].tolist()

    if len(recommended_ids) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "ndcg": 0.0,
            "novelty": 0.0
        }

    hits = sum(
        movie_id in relevant_ids
        for movie_id in recommended_ids
    )

    precision = hits / TOP_K

    recall = (
        hits / len(relevant_ids)
        if len(relevant_ids) > 0
        else 0.0
    )

    ndcg = calculate_ndcg(
        recommended_ids,
        relevant_ids,
        TOP_K
    )

    average_novelty = recommendations[
        "novelty_score"
    ].mean()

    return {
        "precision": precision,
        "recall": recall,
        "ndcg": ndcg,
        "novelty": average_novelty
    }


# ================================================================
# 10. SELECT THE SAME EVALUATION USERS FOR EVERY MODEL
# ================================================================

print("\nSTEP 6: Selecting evaluation users...")

test_relevant = test[test["rating"] >= 4].copy()

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

evaluation_rng = np.random.default_rng(RANDOM_SEED)

evaluation_count = min(
    EVALUATION_USER_LIMIT,
    len(eligible_users)
)

evaluation_users = evaluation_rng.choice(
    eligible_users,
    size=evaluation_count,
    replace=False
).tolist()

print("Eligible users:", len(eligible_users))
print("Users selected for evaluation:", len(evaluation_users))
print("Relevant test ratings use rating >= 4.")


# ================================================================
# 11. PREDICT ONCE PER USER, THEN TEST ALL NOVELTY WEIGHTS
# ================================================================

print("\nSTEP 7: Evaluating novelty weights...")
print("This step may take a little time.")

metric_rows = []
user1_recommendations_by_weight = []

# Cache predictions so they are not recalculated for each weight.
prediction_cache = {}

users_to_predict = set(evaluation_users)

# Also generate recommendations for User 1.
if 1 in user_to_index:
    users_to_predict.add(1)

for counter, user_id in enumerate(users_to_predict, start=1):
    prediction_cache[user_id] = predict_user_ratings(user_id)

    if counter % 100 == 0:
        print(
            "Predictions calculated for",
            counter,
            "users..."
        )

for novelty_weight in NOVELTY_WEIGHTS:

    print(
        "\nEvaluating novelty weight:",
        f"{novelty_weight:.0%}"
    )

    precisions = []
    recalls = []
    ndcgs = []
    novelties = []

    recommended_movie_ids = set()

    for user_id in evaluation_users:

        predictions = prediction_cache[user_id]

        recommendations = recommend_for_user(
            user_id,
            predictions,
            novelty_weight
        )

        relevant_ids = relevant_by_user.get(
            user_id, set()
        )

        metrics = evaluate_recommendations(
            recommendations,
            relevant_ids
        )

        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        ndcgs.append(metrics["ndcg"])
        novelties.append(metrics["novelty"])

        recommended_movie_ids.update(
            recommendations["movieId"].tolist()
        )

    catalog_coverage = (
        len(recommended_movie_ids) / len(movie_ids)
        if len(movie_ids) > 0
        else 0.0
    )

    metric_rows.append({
        "novelty_weight": novelty_weight,
        "preference_weight": 1.0 - novelty_weight,
        "precision_at_10": np.mean(precisions),
        "recall_at_10": np.mean(recalls),
        "ndcg_at_10": np.mean(ndcgs),
        "average_novelty": np.mean(novelties),
        "catalog_coverage": catalog_coverage,
        "distinct_movies_recommended": len(
            recommended_movie_ids
        ),
        "evaluation_users": len(evaluation_users)
    })

    # Save User 1's recommendations for each weight.
    if 1 in prediction_cache:
        user1_recommendations = recommend_for_user(
            1,
            prediction_cache[1],
            novelty_weight
        )

        if not user1_recommendations.empty:
            user1_recommendations_by_weight.append(
                user1_recommendations
            )


# ================================================================
# 12. SAVE RESULTS
# ================================================================

print("\nSTEP 8: Saving results...")

metrics_df = pd.DataFrame(metric_rows)

metrics_df.to_csv(
    OUTPUT_METRICS,
    index=False
)

if user1_recommendations_by_weight:

    all_user1_recommendations = pd.concat(
        user1_recommendations_by_weight,
        ignore_index=True
    )

    all_user1_recommendations.to_csv(
        OUTPUT_ALL_RECOMMENDATIONS,
        index=False
    )

    # Save recommendations from the best novelty weight
    # after the evaluation results are compared below.
else:
    all_user1_recommendations = pd.DataFrame()


# ================================================================
# 13. DISPLAY COMPARISON TABLE
# ================================================================

print("\n" + "=" * 72)
print("STAGE 7 RESULTS: NOVELTY WEIGHT COMPARISON")
print("=" * 72)

display_columns = [
    "novelty_weight",
    "precision_at_10",
    "recall_at_10",
    "ndcg_at_10",
    "average_novelty",
    "catalog_coverage"
]

print(
    metrics_df[display_columns].to_string(
        index=False,
        formatters={
            "novelty_weight": "{:.0%}".format,
            "precision_at_10": "{:.4f}".format,
            "recall_at_10": "{:.4f}".format,
            "ndcg_at_10": "{:.4f}".format,
            "average_novelty": "{:.4f}".format,
            "catalog_coverage": "{:.2%}".format
        }
    )
)


# ================================================================
# 14. IDENTIFY A BALANCED WEIGHT
# ================================================================

# Choose the weight with the highest NDCG among weights whose
# precision is at least 80% of the best precision observed.
# If no weight meets that condition, use the highest-NDCG weight.
best_precision = metrics_df["precision_at_10"].max()

acceptable_rows = metrics_df[
    metrics_df["precision_at_10"] >= 0.80 * best_precision
]

if not acceptable_rows.empty:
    best_row = acceptable_rows.sort_values(
        ["ndcg_at_10", "average_novelty"],
        ascending=False
    ).iloc[0]
else:
    best_row = metrics_df.sort_values(
        ["ndcg_at_10", "average_novelty"],
        ascending=False
    ).iloc[0]

best_weight = float(best_row["novelty_weight"])

print("\n" + "=" * 72)
print("RECOMMENDED BALANCED NOVELTY WEIGHT")
print("=" * 72)

print("Selected novelty weight:", f"{best_weight:.0%}")
print(
    "Preference weight:",
    f"{1.0 - best_weight:.0%}"
)
print(
    "Precision@10:",
    f"{best_row['precision_at_10']:.4f}"
)
print(
    "NDCG@10:",
    f"{best_row['ndcg_at_10']:.4f}"
)
print(
    "Average novelty:",
    f"{best_row['average_novelty']:.4f}"
)
print(
    "Catalog coverage:",
    f"{best_row['catalog_coverage']:.2%}"
)


# ================================================================
# 15. SAVE USER 1'S RECOMMENDATIONS AT THE SELECTED WEIGHT
# ================================================================

if 1 in prediction_cache:

    selected_user1 = recommend_for_user(
        1,
        prediction_cache[1],
        best_weight
    )

    selected_user1.to_csv(
        OUTPUT_RECOMMENDATIONS,
        index=False
    )

    print("\nUser 1's recommendations at selected weight:")

    print(
        selected_user1[
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


# ================================================================
# 16. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 72)
print("STAGE 7 COMPLETED SUCCESSFULLY")
print("=" * 72)

print("Saved files:")
print(OUTPUT_METRICS)
print(OUTPUT_ALL_RECOMMENDATIONS)
print(OUTPUT_RECOMMENDATIONS)

print(
    "\nTotal execution time:",
    round(time.time() - START_TIME, 2),
    "seconds"
)

print("\nNext: review the results before proceeding to Stage 8.")
