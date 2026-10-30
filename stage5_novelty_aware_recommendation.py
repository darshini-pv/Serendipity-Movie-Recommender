
# ================================================================
# PROJECT: SERENDIPITY - NOVELTY-AWARE MOVIE RECOMMENDATION SYSTEM
# STAGE 5: PERSONALIZED RECOMMENDATIONS AND MODEL EVALUATION
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

print("=" * 70)
print("STAGE 5: NOVELTY-AWARE MOVIE RECOMMENDATION")
print("=" * 70)

start_time = time.time()

DATA_DIR = (
    r"C:\Users\manoj\OneDrive\Desktop"
    r"\Serendipitous Recommendation Engine\data"
)

PROJECT_DIR = (
    r"C:\Users\manoj\OneDrive\Desktop"
    r"\Serendipitous Recommendation Engine"
)

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

# Remove accidental spaces from column names
ratings.columns = ratings.columns.str.strip()
movies.columns = movies.columns.str.strip()

# Validate required columns
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

# Keep only the required columns
ratings = ratings[
    ["userId", "movieId", "rating"]
].copy()

movies = movies[
    ["movieId", "title", "genres"]
].drop_duplicates(
    subset=["movieId"]
).copy()

# Convert numerical columns to appropriate types
ratings["userId"] = pd.to_numeric(
    ratings["userId"], errors="coerce"
)

ratings["movieId"] = pd.to_numeric(
    ratings["movieId"], errors="coerce"
)

ratings["rating"] = pd.to_numeric(
    ratings["rating"], errors="coerce"
)

# Remove invalid records
ratings = ratings.dropna(
    subset=["userId", "movieId", "rating"]
)

ratings = ratings[
    ratings["rating"].between(1, 5)
].copy()

ratings["userId"] = ratings["userId"].astype(int)
ratings["movieId"] = ratings["movieId"].astype(int)

# MovieLens normally has one rating per user-movie pair.
# If duplicates exist, retain their average.
ratings = ratings.groupby(
    ["userId", "movieId"],
    as_index=False
)["rating"].mean()

print("\nDATASET INFORMATION")
print("-" * 70)

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)
print("Unique users:", ratings["userId"].nunique())
print("Unique movies:", ratings["movieId"].nunique())


# Precompute movie information for fast lookups
movie_titles = movies.set_index(
    "movieId"
)["title"].to_dict()

movie_genres = movies.set_index(
    "movieId"
)["genres"].to_dict()


# ================================================================
# STEP 2: TRAIN/TEST SPLIT
# ================================================================

print("\n" + "=" * 70)
print("STEP 2: TRAIN/TEST SPLIT")
print("=" * 70)

# Randomly hold out 20% of ratings for evaluation.
# The test ratings are never used to build the recommender.

rng = np.random.default_rng(42)

permutation = rng.permutation(len(ratings))

split_position = int(len(ratings) * 0.80)

train_indices = permutation[:split_position]
test_indices = permutation[split_position:]

train = ratings.iloc[train_indices].copy()
test = ratings.iloc[test_indices].copy()

# Confirm that no user-movie pair occurs in both sets
train_pairs = pd.MultiIndex.from_frame(
    train[["userId", "movieId"]]
)

test_pairs = pd.MultiIndex.from_frame(
    test[["userId", "movieId"]]
)

overlap_count = len(
    train_pairs.intersection(test_pairs)
)

print("Training ratings:", len(train))
print("Testing ratings:", len(test))
print("Overlapping user-movie pairs:", overlap_count)

if overlap_count != 0:
    raise RuntimeError(
        "Data leakage detected: training and test pairs overlap."
    )


# ================================================================
# STEP 3: CREATE USER-MOVIE RATING MATRIX
# ================================================================

print("\n" + "=" * 70)
print("STEP 3: USER-MOVIE MATRIX")
print("=" * 70)

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

movie_to_position = {
    movie_id: position
    for position, movie_id in enumerate(movie_ids)
}

rating_values = rating_matrix.to_numpy(
    dtype=np.float32
)

# 1 means the user rated the movie; 0 means not rated
rated_mask = (rating_values > 0).astype(
    np.float32
)

print("Training matrix shape:", rating_matrix.shape)
print("Users in training matrix:", len(user_ids))
print("Movies in training matrix:", len(movie_ids))


# ================================================================
# STEP 4: CALCULATE USER SIMILARITY
# ================================================================

print("\n" + "=" * 70)
print("STEP 4: USER SIMILARITY")
print("=" * 70)

print("Calculating cosine similarity...")

similarity_matrix = cosine_similarity(
    rating_values
).astype(np.float32)

print("User similarity calculation completed.")


# ================================================================
# STEP 5: CALCULATE MOVIE NOVELTY
# ================================================================

print("\n" + "=" * 70)
print("STEP 5: MOVIE NOVELTY")
print("=" * 70)

# Popularity is calculated using training ratings only.
# This prevents test ratings from influencing novelty scores.

movie_popularity = (
    train.groupby("movieId")["userId"]
    .nunique()
    .reindex(movie_ids)
    .fillna(0)
    .to_numpy(dtype=np.float32)
)

total_training_users = max(
    train["userId"].nunique(), 1
)

popularity_probability = (
    movie_popularity / total_training_users
)

# Less popular movies receive higher novelty.
raw_novelty = -np.log2(
    np.maximum(
        popularity_probability,
        1.0 / total_training_users
    )
)

maximum_novelty = raw_novelty.max()

if maximum_novelty > 0:
    novelty_array = (
        raw_novelty / maximum_novelty
    )
else:
    novelty_array = np.zeros_like(raw_novelty)

print("Movie novelty calculation completed.")


# ================================================================
# STEP 6: DEFINE RECOMMENDATION FUNCTION
# ================================================================

def recommend_for_user(
    user_id,
    top_n=10,
    number_of_neighbors=20,
    preference_weight=0.70
):
    """
    Generate personalized novelty-aware recommendations.

    preference_weight:
        1.00 = preference-only collaborative filtering
        0.70 = 70% preference + 30% novelty
        0.00 = novelty-only ranking

    Returns the top-N movies the user has not rated
    in the training dataset.
    """

    if user_id not in user_to_position:
        return pd.DataFrame()

    if not 0 <= preference_weight <= 1:
        raise ValueError(
            "preference_weight must be between 0 and 1."
        )

    user_position = user_to_position[user_id]

    user_similarities = similarity_matrix[
        user_position
    ].copy()

    # Exclude the target user from their own neighbors
    user_similarities[user_position] = 0

    # Select only users with positive similarity
    positive_neighbor_positions = np.flatnonzero(
        user_similarities > 0
    )

    if len(positive_neighbor_positions) == 0:
        return pd.DataFrame()

    # Sort potential neighbors by similarity
    sorted_neighbors = positive_neighbor_positions[
        np.argsort(
            user_similarities[
                positive_neighbor_positions
            ]
        )[::-1]
    ]

    neighbor_positions = sorted_neighbors[
        :number_of_neighbors
    ]

    neighbor_weights = user_similarities[
        neighbor_positions
    ]

    if len(neighbor_positions) == 0:
        return pd.DataFrame()

    # Ratings given by selected neighbors
    neighbor_ratings = rating_values[
        neighbor_positions
    ]

    # Weighted rating numerator for each movie
    weighted_rating_sum = (
        neighbor_weights @ neighbor_ratings
    )

    # Sum of similarities for neighbors who rated each movie
    neighbor_rated = (
        neighbor_ratings > 0
    ).astype(np.float32)

    similarity_weight_sum = (
        neighbor_weights @ neighbor_rated
    )

    # Actual number of neighbors who rated each movie
    neighbor_count = neighbor_rated.sum(
        axis=0
    )

    # Avoid division by zero
    predicted_ratings = np.divide(
        weighted_rating_sum,
        similarity_weight_sum,
        out=np.zeros_like(weighted_rating_sum),
        where=similarity_weight_sum > 0
    )

    # Keep predicted ratings within the MovieLens scale
    predicted_ratings = np.clip(
        predicted_ratings,
        1,
        5
    )

    # Identify movies not rated by the target user
    target_user_ratings = rating_values[
        user_position
    ]

    unseen_mask = target_user_ratings == 0

    # Only recommend movies supported by at least one neighbor
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

    # Normalize predicted ratings from 1-5 to 0-1
    normalized_preference = (
        candidate_ratings - 1
    ) / 4.0

    # Combine preference and novelty
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

    # Add movie details using dictionary lookups,
    # avoiding a DataFrame merge on every call.
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

    recommendations = recommendations.head(
        top_n
    )

    return recommendations[
        [
            "movieId",
            "title",
            "genres",
            "predicted_rating",
            "novelty_score",
            "final_score",
            "neighbour_count"
        ]
    ].reset_index(drop=True)


# ================================================================
# STEP 7: GENERATE RECOMMENDATIONS FOR USER 1
# ================================================================

print("\n" + "=" * 70)
print("STEP 7: PERSONALIZED RECOMMENDATIONS")
print("=" * 70)

target_user_id = 1

recommendations = recommend_for_user(
    user_id=target_user_id,
    top_n=10,
    number_of_neighbors=20,
    preference_weight=0.70
)

print("\nTarget user ID:", target_user_id)

if recommendations.empty:
    print("No recommendations available.")
else:
    print(
        recommendations.to_string(index=False)
    )


# ================================================================
# STEP 8: EVALUATE RECOMMENDATIONS
# ================================================================

print("\n" + "=" * 70)
print("STEP 8: MODEL EVALUATION")
print("=" * 70)

K = 10

# A held-out movie rated 4 or 5 is treated as relevant.
test_relevant = test[
    test["rating"] >= 4
]

relevant_by_user = (
    test_relevant.groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

# Only evaluate users present in the training matrix.
all_evaluation_users = [
    user_id
    for user_id in relevant_by_user
    if user_id in user_to_position
]

# Evaluate a reproducible sample to keep execution manageable.
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
    evaluation_users = all_evaluation_users

print(
    "Users selected for evaluation:",
    len(evaluation_users),
    "out of",
    len(all_evaluation_users)
)

precision_values = []
recall_values = []
ndcg_values = []
novelty_values = []

recommended_catalog = set()

evaluation_start = time.time()

for count, user_id in enumerate(
    evaluation_users, start=1
):

    recs = recommend_for_user(
        user_id=user_id,
        top_n=K,
        number_of_neighbors=20,
        preference_weight=0.70
    )

    if recs.empty:
        continue

    recommended_ids = recs[
        "movieId"
    ].tolist()

    relevant_ids = relevant_by_user[
        user_id
    ]

    # A hit means the recommendation appears in
    # this user's held-out relevant movie set.
    hits = [
        1 if movie_id in relevant_ids else 0
        for movie_id in recommended_ids
    ]

    number_hits = sum(hits)

    # Precision@K uses K as the denominator
    precision = number_hits / K

    # Recall@K measures the fraction of relevant
    # held-out movies recovered by the recommender.
    recall = (
        number_hits / len(relevant_ids)
        if len(relevant_ids) > 0
        else 0
    )

    # NDCG@K: higher-ranked relevant movies contribute more.
    discounts = np.log2(
        np.arange(2, len(hits) + 2)
    )

    dcg = sum(
        hit / discount
        for hit, discount in zip(hits, discounts)
    )

    ideal_hits = min(
        len(relevant_ids), K
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

    # Print progress periodically
    if count % 100 == 0:
        elapsed = time.time() - evaluation_start

        print(
            f"Evaluated {count}/"
            f"{len(evaluation_users)} users "
            f"({elapsed:.1f} seconds elapsed)"
        )


# ================================================================
# STEP 9: DISPLAY AND SAVE EVALUATION RESULTS
# ================================================================

print("\n" + "=" * 70)
print("STEP 9: EVALUATION RESULTS")
print("=" * 70)

metrics = {}

if len(precision_values) == 0:

    print("No users could be evaluated.")

else:

    mean_precision = np.mean(
        precision_values
    )

    mean_recall = np.mean(
        recall_values
    )

    mean_ndcg = np.mean(
        ndcg_values
    )

    mean_novelty = (
        np.mean(novelty_values)
        if len(novelty_values) > 0
        else 0
    )

    catalog_size = len(movie_ids)

    coverage = (
        len(recommended_catalog) / catalog_size
        if catalog_size > 0
        else 0
    )

    print(
        "Users evaluated:",
        len(precision_values)
    )

    print(
        f"Precision@{K}: {mean_precision:.4f}"
    )

    print(
        f"Recall@{K}: {mean_recall:.4f}"
    )

    print(
        f"NDCG@{K}: {mean_ndcg:.4f}"
    )

    print(
        f"Average normalized novelty: {mean_novelty:.4f}"
    )

    print(
        f"Catalog coverage: {coverage:.4f}"
    )

    print(
        f"Catalog coverage percentage: {coverage * 100:.2f}%"
    )

    print(
        "Distinct recommended movies:",
        len(recommended_catalog)
    )

    metrics = {
        "evaluation_users": len(precision_values),
        "precision_at_10": mean_precision,
        "recall_at_10": mean_recall,
        "ndcg_at_10": mean_ndcg,
        "average_normalized_novelty": mean_novelty,
        "catalog_coverage": coverage,
        "catalog_coverage_percentage": coverage * 100,
        "distinct_recommended_movies": len(recommended_catalog)
    }


# ================================================================
# STEP 10: SAVE OUTPUT FILES
# ================================================================

print("\n" + "=" * 70)
print("STEP 10: SAVE OUTPUT FILES")
print("=" * 70)

recommendations_path = os.path.join(
    PROJECT_DIR,
    "stage5_recommendations_user1.csv"
)

metrics_path = os.path.join(
    PROJECT_DIR,
    "stage5_evaluation_metrics.csv"
)

if not recommendations.empty:

    recommendations.to_csv(
        recommendations_path,
        index=False
    )

    print(
        "Recommendations saved to:",
        recommendations_path
    )

if metrics:

    pd.DataFrame(
        [metrics]
    ).to_csv(
        metrics_path,
        index=False
    )

    print(
        "Evaluation metrics saved to:",
        metrics_path
    )


# ================================================================
# FINAL SUMMARY
# ================================================================

total_time = time.time() - start_time

print("\n" + "=" * 70)
print("STAGE 5 COMPLETED")
print("=" * 70)

print(
    f"Total execution time: {total_time:.2f} seconds"
)

print(
    "Evaluation type: Random 80/20 rating holdout"
)

print(
    "Evaluation sample limit:",
    MAX_EVALUATION_USERS,
    "users"
)

print(
    "Note: Results are experimental and are not "
    "evidence of real-world user satisfaction."
)

print("=" * 70)
