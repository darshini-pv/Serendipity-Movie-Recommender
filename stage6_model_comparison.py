
# ================================================================
# PROJECT: SERENDIPITY - NOVELTY-AWARE MOVIE RECOMMENDATION
# STAGE 6: BASELINE VS NOVELTY-AWARE MODEL COMPARISON
# Dataset: MovieLens 1M
# IDE: Spyder
# ================================================================

import os
import time
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


# ================================================================
# STEP 1: LOAD DATASETS
# ================================================================

print("=" * 75)
print("STAGE 6: BASELINE VS NOVELTY-AWARE RECOMMENDATIONS")
print("=" * 75)

start_time = time.time()

PROJECT_DIR = (
    r"C:\Users\manoj\OneDrive\Desktop"
    r"\Serendipitous Recommendation Engine"
)

DATA_DIR = os.path.join(PROJECT_DIR, "data")

ratings_path = os.path.join(DATA_DIR, "ratings.csv")
movies_path = os.path.join(DATA_DIR, "movies.csv")

print("\nLoading datasets...")

if not os.path.exists(ratings_path):
    raise FileNotFoundError(
        f"Ratings file not found: {ratings_path}"
    )

if not os.path.exists(movies_path):
    raise FileNotFoundError(
        f"Movies file not found: {movies_path}"
    )

ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)

ratings.columns = ratings.columns.str.strip()
movies.columns = movies.columns.str.strip()

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
].drop_duplicates(
    subset=["movieId"]
).copy()

for column in ["userId", "movieId", "rating"]:
    ratings[column] = pd.to_numeric(
        ratings[column],
        errors="coerce"
    )

ratings = ratings.dropna(
    subset=["userId", "movieId", "rating"]
)

ratings = ratings[
    ratings["rating"].between(1, 5)
].copy()

ratings["userId"] = ratings["userId"].astype(int)
ratings["movieId"] = ratings["movieId"].astype(int)

# Handle any duplicate user-movie ratings
ratings = ratings.groupby(
    ["userId", "movieId"],
    as_index=False
)["rating"].mean()

movie_titles = movies.set_index(
    "movieId"
)["title"].to_dict()

movie_genres = movies.set_index(
    "movieId"
)["genres"].to_dict()

print("\nRatings shape:", ratings.shape)
print("Movies shape:", movies.shape)
print("Unique users:", ratings["userId"].nunique())
print("Unique movies:", ratings["movieId"].nunique())


# ================================================================
# STEP 2: CREATE A REPRODUCIBLE TRAIN/TEST SPLIT
# ================================================================

print("\n" + "=" * 75)
print("STEP 2: TRAIN/TEST SPLIT")
print("=" * 75)

# Keep the same random seed and split approach as Stage 5.
split_rng = np.random.default_rng(42)

permutation = split_rng.permutation(
    len(ratings)
)

split_position = int(
    len(ratings) * 0.80
)

train_indices = permutation[:split_position]
test_indices = permutation[split_position:]

train = ratings.iloc[
    train_indices
].copy()

test = ratings.iloc[
    test_indices
].copy()

train_pairs = pd.MultiIndex.from_frame(
    train[["userId", "movieId"]]
)

test_pairs = pd.MultiIndex.from_frame(
    test[["userId", "movieId"]]
)

overlap = len(
    train_pairs.intersection(test_pairs)
)

print("Training ratings:", len(train))
print("Testing ratings:", len(test))
print("Overlapping user-movie pairs:", overlap)

if overlap != 0:
    raise RuntimeError(
        "Training/test leakage detected."
    )


# ================================================================
# STEP 3: BUILD THE USER-MOVIE MATRIX
# ================================================================

print("\n" + "=" * 75)
print("STEP 3: USER-MOVIE MATRIX")
print("=" * 75)

rating_matrix = train.pivot(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0).astype(np.float32)

user_ids = rating_matrix.index.to_numpy()
movie_ids = rating_matrix.columns.to_numpy()

user_to_position = {
    user_id: position
    for position, user_id in enumerate(user_ids)
}

rating_values = rating_matrix.to_numpy(
    dtype=np.float32
)

print("Training matrix shape:", rating_matrix.shape)


# ================================================================
# STEP 4: CALCULATE USER SIMILARITY
# ================================================================

print("\n" + "=" * 75)
print("STEP 4: USER SIMILARITY")
print("=" * 75)

print("Calculating cosine similarity...")

similarity_matrix = cosine_similarity(
    rating_values
).astype(np.float32)

print("User similarity calculation completed.")


# ================================================================
# STEP 5: CALCULATE MOVIE NOVELTY
# ================================================================

print("\n" + "=" * 75)
print("STEP 5: MOVIE NOVELTY")
print("=" * 75)

# Popularity is calculated only from training ratings.
movie_popularity = (
    train.groupby("movieId")["userId"]
    .nunique()
    .reindex(movie_ids)
    .fillna(0)
    .to_numpy(dtype=np.float32)
)

number_of_training_users = max(
    train["userId"].nunique(),
    1
)

popularity_probability = (
    movie_popularity / number_of_training_users
)

raw_novelty = -np.log2(
    np.maximum(
        popularity_probability,
        1.0 / number_of_training_users
    )
)

max_novelty = raw_novelty.max()

if max_novelty > 0:
    novelty_array = raw_novelty / max_novelty
else:
    novelty_array = np.zeros_like(raw_novelty)

print("Novelty scores calculated.")


# ================================================================
# STEP 6: GENERATE RECOMMENDATIONS
# ================================================================

def recommend_for_user(
    user_id,
    preference_weight,
    top_n=10,
    number_of_neighbors=20
):
    """
    preference_weight = 1.00:
        Preference-only baseline.

    preference_weight = 0.70:
        Novelty-aware model, with 30% novelty weight.

    Both models use identical training data, similarities,
    neighbors and candidate movies.
    """

    if user_id not in user_to_position:
        return pd.DataFrame()

    user_position = user_to_position[user_id]

    user_similarities = similarity_matrix[
        user_position
    ].copy()

    # Do not use the target user as their own neighbor.
    user_similarities[user_position] = 0

    positive_positions = np.flatnonzero(
        user_similarities > 0
    )

    if len(positive_positions) == 0:
        return pd.DataFrame()

    # Select the most similar positive-similarity neighbors.
    sorted_positions = positive_positions[
        np.argsort(
            user_similarities[positive_positions]
        )[::-1]
    ]

    neighbor_positions = sorted_positions[
        :number_of_neighbors
    ]

    neighbor_weights = user_similarities[
        neighbor_positions
    ]

    neighbor_ratings = rating_values[
        neighbor_positions
    ]

    # Weighted ratings for each candidate movie.
    weighted_rating_sum = (
        neighbor_weights @ neighbor_ratings
    )

    rated_mask = (
        neighbor_ratings > 0
    ).astype(np.float32)

    similarity_weight_sum = (
        neighbor_weights @ rated_mask
    )

    neighbor_count = rated_mask.sum(
        axis=0
    )

    predicted_ratings = np.divide(
        weighted_rating_sum,
        similarity_weight_sum,
        out=np.zeros_like(weighted_rating_sum),
        where=similarity_weight_sum > 0
    )

    predicted_ratings = np.clip(
        predicted_ratings,
        1,
        5
    )

    # Never recommend movies already rated in training.
    target_user_ratings = rating_values[
        user_position
    ]

    unseen_mask = (
        target_user_ratings == 0
    )

    eligible_mask = (
        unseen_mask
        & (similarity_weight_sum > 0)
    )

    candidate_positions = np.flatnonzero(
        eligible_mask
    )

    if len(candidate_positions) == 0:
        return pd.DataFrame()

    candidate_ratings = predicted_ratings[
        candidate_positions
    ]

    candidate_novelty = novelty_array[
        candidate_positions
    ]

    # Normalize predicted ratings from 1-5 to 0-1.
    normalized_preference = (
        candidate_ratings - 1
    ) / 4.0

    # Model A: preference_weight = 1.00
    # Model B: preference_weight = 0.70
    final_scores = (
        preference_weight * normalized_preference
        + (1 - preference_weight) * candidate_novelty
    )

    recommendations = pd.DataFrame({
        "movieId": movie_ids[candidate_positions],
        "predicted_rating": candidate_ratings,
        "novelty_score": candidate_novelty,
        "final_score": final_scores,
        "neighbour_count": neighbor_count[
            candidate_positions
        ].astype(int)
    })

    recommendations["title"] = (
        recommendations["movieId"].map(movie_titles)
    )

    recommendations["genres"] = (
        recommendations["movieId"].map(movie_genres)
    )

    recommendations = recommendations.sort_values(
        by=["final_score", "predicted_rating"],
        ascending=[False, False]
    )

    return recommendations.head(
        top_n
    ).reset_index(drop=True)


# ================================================================
# STEP 7: PREPARE THE EVALUATION USERS
# ================================================================

print("\n" + "=" * 75)
print("STEP 7: PREPARE EVALUATION USERS")
print("=" * 75)

K = 10

# Held-out ratings of 4 or 5 are considered relevant.
test_relevant = test[
    test["rating"] >= 4
]

relevant_by_user = (
    test_relevant.groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

all_evaluation_users = [
    user_id
    for user_id in relevant_by_user
    if user_id in user_to_position
]

MAX_EVALUATION_USERS = 500

evaluation_rng = np.random.default_rng(42)

if len(all_evaluation_users) > MAX_EVALUATION_USERS:
    evaluation_users = sorted(
        evaluation_rng.choice(
            all_evaluation_users,
            size=MAX_EVALUATION_USERS,
            replace=False
        ).tolist()
    )
else:
    evaluation_users = sorted(
        all_evaluation_users
    )

print(
    "Evaluation users selected:",
    len(evaluation_users),
    "out of",
    len(all_evaluation_users)
)


# ================================================================
# STEP 8: EVALUATION FUNCTION
# ================================================================

def evaluate_model(preference_weight, model_name):
    """
    Evaluate one model using the same selected users.

    Returns average ranking metrics and catalog coverage.
    """

    precision_values = []
    recall_values = []
    ndcg_values = []
    novelty_values = []

    recommended_catalog = set()
    evaluated_users = 0

    for index, user_id in enumerate(
        evaluation_users,
        start=1
    ):

        recs = recommend_for_user(
            user_id=user_id,
            preference_weight=preference_weight,
            top_n=K,
            number_of_neighbors=20
        )

        if recs.empty:
            continue

        recommended_ids = (
            recs["movieId"].tolist()
        )

        relevant_ids = relevant_by_user[
            user_id
        ]

        hits = [
            1 if movie_id in relevant_ids else 0
            for movie_id in recommended_ids
        ]

        number_hits = sum(hits)

        # Precision@10: relevant recommendations / 10.
        precision = number_hits / K

        # Recall@10: relevant recommendations /
        # all relevant held-out movies for this user.
        recall = (
            number_hits / len(relevant_ids)
            if len(relevant_ids) > 0
            else 0
        )

        # NDCG@10: reward relevant movies appearing
        # higher in the ranked recommendation list.
        discounts = np.log2(
            np.arange(2, len(hits) + 2)
        )

        dcg = sum(
            hit / discount
            for hit, discount in zip(
                hits,
                discounts
            )
        )

        ideal_hits = min(
            len(relevant_ids),
            K
        )

        ideal_discounts = np.log2(
            np.arange(2, ideal_hits + 2)
        )

        ideal_dcg = np.sum(
            1 / ideal_discounts
        )

        ndcg = (
            dcg / ideal_dcg
            if ideal_dcg > 0
            else 0
        )

        precision_values.append(precision)
        recall_values.append(recall)
        ndcg_values.append(ndcg)

        novelty_values.extend(
            recs["novelty_score"].tolist()
        )

        recommended_catalog.update(
            recommended_ids
        )

        evaluated_users += 1

        if index % 100 == 0:
            print(
                f"{model_name}: "
                f"{index}/{len(evaluation_users)} users processed"
            )

    catalog_size = len(movie_ids)

    coverage = (
        len(recommended_catalog) / catalog_size
        if catalog_size > 0
        else 0
    )

    results = {
        "Model": model_name,
        "Preference weight": preference_weight,
        "Novelty weight": 1 - preference_weight,
        "Users evaluated": evaluated_users,
        "Precision@10": (
            np.mean(precision_values)
            if precision_values else 0
        ),
        "Recall@10": (
            np.mean(recall_values)
            if recall_values else 0
        ),
        "NDCG@10": (
            np.mean(ndcg_values)
            if ndcg_values else 0
        ),
        "Average novelty": (
            np.mean(novelty_values)
            if novelty_values else 0
        ),
        "Catalog coverage": coverage,
        "Catalog coverage (%)": coverage * 100,
        "Distinct movies recommended": len(
            recommended_catalog
        )
    }

    return results


# ================================================================
# STEP 9: EVALUATE BOTH MODELS
# ================================================================

print("\n" + "=" * 75)
print("STEP 9: MODEL A - BASELINE COLLABORATIVE FILTERING")
print("=" * 75)

baseline_results = evaluate_model(
    preference_weight=1.00,
    model_name="Baseline"
)

print("\nBaseline evaluation completed.")

print("\n" + "=" * 75)
print("STEP 10: MODEL B - NOVELTY-AWARE RECOMMENDATIONS")
print("=" * 75)

novelty_results = evaluate_model(
    preference_weight=0.70,
    model_name="Novelty-aware"
)

print("\nNovelty-aware evaluation completed.")


# ================================================================
# STEP 11: COMPARE RESULTS
# ================================================================

print("\n" + "=" * 75)
print("STEP 11: BASELINE VS NOVELTY-AWARE COMPARISON")
print("=" * 75)

comparison = pd.DataFrame([
    baseline_results,
    novelty_results
])

metrics_to_display = [
    "Model",
    "Preference weight",
    "Novelty weight",
    "Users evaluated",
    "Precision@10",
    "Recall@10",
    "NDCG@10",
    "Average novelty",
    "Catalog coverage (%)",
    "Distinct movies recommended"
]

print(
    comparison[metrics_to_display].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ================================================================
# STEP 12: CALCULATE CHANGES FROM THE BASELINE
# ================================================================

print("\n" + "=" * 75)
print("STEP 12: CHANGE RELATIVE TO BASELINE")
print("=" * 75)

comparison_metrics = [
    "Precision@10",
    "Recall@10",
    "NDCG@10",
    "Average novelty",
    "Catalog coverage (%)"
]

for metric in comparison_metrics:

    baseline_value = baseline_results[metric]
    novelty_value = novelty_results[metric]

    absolute_change = (
        novelty_value - baseline_value
    )

    if abs(baseline_value) > 1e-12:
        percentage_change = (
            absolute_change / abs(baseline_value)
        ) * 100

        print(
            f"{metric}: "
            f"change = {absolute_change:.4f}, "
            f"relative change = {percentage_change:.2f}%"
        )

    else:
        print(
            f"{metric}: "
            f"change = {absolute_change:.4f}; "
            "relative change is undefined because "
            "the baseline is zero."
        )


# ================================================================
# STEP 13: SAVE RESULTS
# ================================================================

print("\n" + "=" * 75)
print("STEP 13: SAVE OUTPUT FILES")
print("=" * 75)

comparison_path = os.path.join(
    PROJECT_DIR,
    "stage6_model_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)

print(
    "Model comparison saved to:",
    comparison_path
)

# Save a sample of recommendations from both models.
example_user_id = 1

baseline_recommendations = recommend_for_user(
    user_id=example_user_id,
    preference_weight=1.00,
    top_n=10
)

novelty_recommendations = recommend_for_user(
    user_id=example_user_id,
    preference_weight=0.70,
    top_n=10
)

baseline_recommendations.insert(
    0,
    "Model",
    "Baseline"
)

novelty_recommendations.insert(
    0,
    "Model",
    "Novelty-aware"
)

example_comparison = pd.concat(
    [
        baseline_recommendations,
        novelty_recommendations
    ],
    ignore_index=True
)

recommendations_path = os.path.join(
    PROJECT_DIR,
    "stage6_user1_recommendation_comparison.csv"
)

example_comparison.to_csv(
    recommendations_path,
    index=False
)

print(
    "User 1 recommendation comparison saved to:",
    recommendations_path
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 75)
print("STAGE 6 COMPLETED")
print("=" * 75)

print(
    "Total execution time:",
    f"{time.time() - start_time:.2f} seconds"
)

print("Models compared: 2")
print("Evaluation users:", len(evaluation_users))
print("Recommendation cutoff: Top 10")
print("Relevant held-out ratings: Rating >= 4")
print("Train/test split: Random 80/20")
print("Random seed: 42")

print("\nImportant interpretation:")
print(
    "- Higher novelty does not automatically mean better recommendations."
)

print(
    "- Compare relevance metrics against the baseline."
)

print(
    "- This experiment does not directly measure human-perceived serendipity."
)

print(
    "- Evaluation is a sampled random rating-holdout experiment, "
    "not a chronological future-prediction test."
)

print("=" * 75)
