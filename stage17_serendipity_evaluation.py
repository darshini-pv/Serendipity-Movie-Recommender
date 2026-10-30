
# ================================================================
# STAGE 17: SERENDIPITY EVALUATION
# Project: Serendipitous Novelty-Aware Movie Recommendation System
# ================================================================

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score

print("=" * 75)
print("STAGE 17: SERENDIPITY EVALUATION")
print("=" * 75)

# ------------------------------------------------
# STEP 1: LOAD DATASETS
# ------------------------------------------------

print("\nSTEP 1: Loading datasets...")

PROJECT_DIR = (
    r"C:\Users\manoj\OneDrive\Desktop"
    r"\Serendipitous Recommendation Engine"
)

DATA_DIR = os.path.join(PROJECT_DIR, "data")

ratings = pd.read_csv(
    os.path.join(DATA_DIR, "ratings.csv")
)

movies = pd.read_csv(
    os.path.join(DATA_DIR, "movies.csv")
)

ratings = ratings[["userId", "movieId", "rating"]].copy()
movies = movies.drop_duplicates("movieId").copy()

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)

# ------------------------------------------------
# STEP 2: RECREATE THE EXACT STAGE 14/16 SPLIT
# ------------------------------------------------

print("\nSTEP 2: Recreating the exact data split...")

from sklearn.model_selection import train_test_split

train_df, holdout_df = train_test_split(
    ratings,
    test_size=0.20,
    random_state=42
)

validation_df, test_df = train_test_split(
    holdout_df,
    test_size=0.50,
    random_state=43
)

print("Training ratings:", len(train_df))
print("Validation ratings:", len(validation_df))
print("Final test ratings:", len(test_df))

train_pairs = set(
    zip(train_df["userId"], train_df["movieId"])
)
test_pairs = set(
    zip(test_df["userId"], test_df["movieId"])
)

print("Train-test pair overlap:", len(train_pairs & test_pairs))

# ------------------------------------------------
# STEP 3: BUILD THE TRAINING MATRIX
# ------------------------------------------------

print("\nSTEP 3: Building the training matrix...")

train_matrix = train_df.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

user_means = train_matrix.mean(axis=1)

centered_matrix = train_matrix.sub(
    user_means,
    axis=0
)

centered_filled = centered_matrix.fillna(0)

user_ids = train_matrix.index
movie_ids = train_matrix.columns

n_users = len(user_ids)

print("Training matrix:", train_matrix.shape)
print("Training users:", n_users)
print("Training movies:", len(movie_ids))
print("Training rating mean:", round(train_df["rating"].mean(), 4))

# ------------------------------------------------
# STEP 4: CALCULATE USER SIMILARITIES
# ------------------------------------------------

print("\nSTEP 4: Calculating user similarities...")

similarity_matrix = cosine_similarity(
    centered_filled.to_numpy()
)

print("Similarity matrix:", similarity_matrix.shape)

# ------------------------------------------------
# STEP 5: CALCULATE MOVIE NOVELTY
# ------------------------------------------------

print("\nSTEP 5: Calculating movie novelty...")

movie_rater_counts = train_df.groupby(
    "movieId"
)["userId"].nunique()

movie_novelty = (
    -np.log2(movie_rater_counts / n_users)
    / np.log2(n_users)
)

print("Mean movie novelty:", round(movie_novelty.mean(), 4))
print("Minimum movie novelty:", round(movie_novelty.min(), 4))
print("Maximum movie novelty:", round(movie_novelty.max(), 4))

# ------------------------------------------------
# STEP 6: PREPARE MOVIE GENRE VECTORS
# ------------------------------------------------

print("\nSTEP 6: Preparing movie genre information...")

movies["genres"] = movies["genres"].fillna("")

all_genres = sorted({
    genre
    for genre_string in movies["genres"]
    for genre in genre_string.split("|")
    if genre and genre != "(no genres listed)"
})

genre_matrix = pd.DataFrame(
    0.0,
    index=movies["movieId"],
    columns=all_genres
)

for row in movies[["movieId", "genres"]].itertuples(index=False):
    movie_id = row.movieId
    genre_string = row.genres

    for genre in genre_string.split("|"):
        if genre in genre_matrix.columns:
            genre_matrix.loc[movie_id, genre] = 1.0

# Keep only movies available in the training recommendation catalog.
genre_matrix = genre_matrix.reindex(
    movie_ids,
    fill_value=0.0
)

movie_genres = movies.set_index("movieId")["genres"].to_dict()
movie_titles = movies.set_index("movieId")["title"].to_dict()

print("Number of genre categories:", len(all_genres))
print("Genre matrix:", genre_matrix.shape)

# ------------------------------------------------
# STEP 7: PREPARE HELD-OUT RELEVANT MOVIES
# ------------------------------------------------

print("\nSTEP 7: Preparing test relevance...")

# A held-out rating of 4 or 5 indicates relevance.
# Only use users and movies present in the training catalog.

test_relevant_df = test_df[
    (test_df["rating"] >= 4)
    & (test_df["userId"].isin(user_ids))
    & (test_df["movieId"].isin(movie_ids))
].copy()

test_relevant = (
    test_relevant_df.groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

eligible_users = np.array(list(test_relevant.keys()))

rng = np.random.default_rng(2027)

if len(eligible_users) < 500:
    raise ValueError(
        "Fewer than 500 eligible users. Check the data split."
    )

selected_users = rng.choice(
    eligible_users,
    size=500,
    replace=False
)

print("Eligible test users:", len(eligible_users))
print("Selected test users:", len(selected_users))
print("Relevant rating threshold: 4 or higher")

# ------------------------------------------------
# STEP 8: DEFINE THE SELECTED MODEL
# ------------------------------------------------

print("\nSTEP 8: Configuring the selected model...")

REGULARIZATION = 10.0
NOVELTY_WEIGHT = 0.005
TOP_K_NEIGHBORS = 50
TOP_N = 10

print("Regularization:", REGULARIZATION)
print("Novelty weight:", NOVELTY_WEIGHT)
print("Neighbors:", TOP_K_NEIGHBORS)
print("Recommendations per user:", TOP_N)

# ------------------------------------------------
# STEP 9: DEFINE RECOMMENDATION FUNCTION
# ------------------------------------------------

def recommend_for_user(user_id):

    user_position = train_matrix.index.get_loc(user_id)

    # Find the most similar users.
    similarities = similarity_matrix[user_position].copy()
    similarities[user_position] = 0.0

    positive_positions = np.where(similarities > 0)[0]

    if len(positive_positions) == 0:
        return pd.DataFrame()

    sorted_positions = positive_positions[
        np.argsort(similarities[positive_positions])[::-1]
    ]

    neighbor_positions = sorted_positions[:TOP_K_NEIGHBORS]

    neighbor_ids = user_ids[neighbor_positions]
    neighbor_similarities = similarities[neighbor_positions]

    # Candidate movies are movies the user did not rate in training.
    user_ratings = train_matrix.loc[user_id]

    candidate_movies = user_ratings[
        user_ratings.isna()
    ].index

    if len(candidate_movies) == 0:
        return pd.DataFrame()

    neighbor_ratings = train_matrix.loc[
        neighbor_ids, candidate_movies
    ]

    neighbor_centered = centered_filled.loc[
        neighbor_ids, candidate_movies
    ]

    # Weighted rating deviations from neighboring users.
    numerator = neighbor_centered.T.dot(
        pd.Series(
            neighbor_similarities,
            index=neighbor_ids
        )
    )

    # Only neighbors who rated a candidate contribute to the denominator.
    denominator = neighbor_ratings.notna().T.dot(
        pd.Series(
            neighbor_similarities,
            index=neighbor_ids
        )
    )

    user_mean = user_means.loc[user_id]

    predicted_deviation = numerator / (
        denominator + REGULARIZATION
    )

    predicted_ratings = (
        user_mean + predicted_deviation
    ).clip(1, 5)

    candidate_novelty = movie_novelty.reindex(
        candidate_movies
    ).fillna(0.0)

    normalized_rating = (predicted_ratings - 1) / 4

    ranking_score = (
        (1 - NOVELTY_WEIGHT) * normalized_rating
        + NOVELTY_WEIGHT * candidate_novelty
    )

    recommendations = pd.DataFrame({
        "movieId": candidate_movies,
        "predicted_rating": predicted_ratings.to_numpy(),
        "novelty": candidate_novelty.to_numpy(),
        "ranking_score": ranking_score.to_numpy()
    })

    recommendations = recommendations.sort_values(
        ["ranking_score", "predicted_rating", "movieId"],
        ascending=[False, False, True]
    ).head(TOP_N).reset_index(drop=True)

    return recommendations

# ------------------------------------------------
# STEP 10: DEFINE GENRE-BASED USER PROFILE
# ------------------------------------------------

print("\nSTEP 9: Evaluating recommendations...")
print("Unexpectedness is measured using genre dissimilarity.")

def calculate_unexpectedness(user_id, recommended_movie_ids):

    # Build the profile from movies rated 4 or 5 in training.
    liked_rows = train_df[
        (train_df["userId"] == user_id)
        & (train_df["rating"] >= 4)
    ]

    liked_movie_ids = liked_rows["movieId"].unique()

    liked_movie_ids = [
        movie_id
        for movie_id in liked_movie_ids
        if movie_id in genre_matrix.index
    ]

    if not liked_movie_ids:
        # Without a liked-movie history, the proxy is undefined.
        return np.full(len(recommended_movie_ids), np.nan)

    profile = genre_matrix.loc[liked_movie_ids].mean(axis=0).to_numpy()

    candidate_vectors = genre_matrix.reindex(
        recommended_movie_ids,
        fill_value=0.0
    ).to_numpy()

    if np.linalg.norm(profile) == 0:
        return np.full(len(recommended_movie_ids), np.nan)

    genre_similarity = cosine_similarity(
        candidate_vectors,
        profile.reshape(1, -1)
    ).ravel()

    # Dissimilarity is used as a proxy for unexpectedness.
    unexpectedness = 1.0 - genre_similarity

    return np.clip(unexpectedness, 0.0, 1.0)

# ------------------------------------------------
# STEP 11: EVALUATE EACH USER
# ------------------------------------------------

user_results = []
recommendation_details = []

for counter, user_id in enumerate(selected_users, start=1):

    recommendations = recommend_for_user(user_id)

    if recommendations.empty:
        continue

    recommended_ids = recommendations["movieId"].tolist()

    relevant_movies = test_relevant.get(user_id, set())

    hits = np.array([
        1 if movie_id in relevant_movies else 0
        for movie_id in recommended_ids
    ])

    actual_top_n = len(recommended_ids)

    precision = hits.sum() / TOP_N
    recall = (
        hits.sum() / len(relevant_movies)
        if relevant_movies else 0.0
    )

    discounts = 1 / np.log2(
        np.arange(2, actual_top_n + 2)
    )

    dcg = float(np.sum(hits * discounts))

    ideal_hits = min(len(relevant_movies), TOP_N)

    if ideal_hits > 0:
        ideal_dcg = float(np.sum(
            1 / np.log2(np.arange(2, ideal_hits + 2))
        ))
        ndcg = dcg / ideal_dcg
    else:
        ndcg = 0.0

    # Novelty of the recommended list.
    average_novelty = recommendations["novelty"].mean()

    # User-profile-based unexpectedness.
    unexpectedness = calculate_unexpectedness(
        user_id,
        recommended_ids
    )

    average_unexpectedness = (
        float(np.nanmean(unexpectedness))
        if np.isfinite(unexpectedness).any()
        else np.nan
    )

    # Serendipity proxy: relevant hits weighted by unexpectedness.
    # This is a proxy, not a direct measure of user-perceived surprise.
    valid_unexpectedness = np.nan_to_num(
        unexpectedness,
        nan=0.0
    )

    serendipity_proxy = float(
        np.sum(hits * valid_unexpectedness) / TOP_N
    )

    # Intra-list diversity: average pairwise genre dissimilarity.
    recommended_genres = genre_matrix.reindex(
        recommended_ids,
        fill_value=0.0
    ).to_numpy()

    if len(recommended_ids) >= 2:
        pairwise_similarity = cosine_similarity(
            recommended_genres
        )

        upper_triangle = np.triu_indices(
            len(recommended_ids),
            k=1
        )

        pairwise_distances = (
            1 - pairwise_similarity[upper_triangle]
        )

        diversity = float(np.mean(pairwise_distances))
    else:
        diversity = np.nan

    user_results.append({
        "userId": user_id,
        "precision_at_10": precision,
        "recall_at_10": recall,
        "ndcg_at_10": ndcg,
        "average_novelty": average_novelty,
        "average_unexpectedness": average_unexpectedness,
        "serendipity_proxy_at_10": serendipity_proxy,
        "intra_list_diversity": diversity,
        "relevant_hits": int(hits.sum()),
        "relevant_test_movies": len(relevant_movies)
    })

    for rank, row in recommendations.iterrows():

        movie_id = int(row["movieId"])
        hit = int(movie_id in relevant_movies)

        recommendation_details.append({
            "userId": user_id,
            "rank": rank + 1,
            "movieId": movie_id,
            "title": movie_titles.get(movie_id, "Unknown"),
            "genres": movie_genres.get(movie_id, ""),
            "predicted_rating": row["predicted_rating"],
            "ranking_score": row["ranking_score"],
            "novelty": row["novelty"],
            "unexpectedness": unexpectedness[rank],
            "heldout_relevant": hit
        })

    if counter % 50 == 0:
        print(
            f"Evaluated {counter} of {len(selected_users)} users..."
        )

# ------------------------------------------------
# STEP 12: SUMMARIZE RESULTS
# ------------------------------------------------

print("\nSTEP 10: Summarizing results...")

user_metrics = pd.DataFrame(user_results)
recommendation_details = pd.DataFrame(
    recommendation_details
)

if user_metrics.empty:
    raise ValueError("No users were successfully evaluated.")

metric_columns = [
    "precision_at_10",
    "recall_at_10",
    "ndcg_at_10",
    "average_novelty",
    "average_unexpectedness",
    "serendipity_proxy_at_10",
    "intra_list_diversity"
]

summary_rows = []

for metric in metric_columns:
    values = user_metrics[metric].dropna()

    summary_rows.append({
        "metric": metric,
        "mean": values.mean(),
        "median": values.median(),
        "std": values.std(),
        "minimum": values.min(),
        "maximum": values.max(),
        "users_with_valid_value": len(values)
    })

summary_df = pd.DataFrame(summary_rows)

# Overall catalog coverage across all recommendations.
catalog_coverage = (
    recommendation_details["movieId"].nunique()
    / len(movie_ids)
)

users_with_hits = int(
    (user_metrics["relevant_hits"] > 0).sum()
)

print("\n" + "=" * 75)
print("STAGE 17: FINAL SERENDIPITY EVALUATION")
print("=" * 75)

print("\nMODEL CONFIGURATION")
print("Regularization:", REGULARIZATION)
print("Novelty weight:", NOVELTY_WEIGHT)
print("Users evaluated:", len(user_metrics))
print("Recommendations per user:", TOP_N)

print("\nAVERAGE USER-LEVEL METRICS")
print(
    user_metrics[metric_columns]
    .mean()
    .round(5)
    .to_string()
)

print("\nADDITIONAL RESULTS")
print("Users with at least one relevant hit:",
      f"{users_with_hits}/{len(user_metrics)}")
print("Distinct recommended movies:",
      recommendation_details["movieId"].nunique())
print("Catalog coverage:",
      round(catalog_coverage * 100, 2), "%")
print("Recommendation rows:",
      len(recommendation_details))

print("\nMETRIC DEFINITIONS")
print("Precision@10: Relevant recommendations divided by 10.")
print("Recall@10: Relevant recommendations divided by held-out relevant movies.")
print("NDCG@10: Ranking quality with higher ranks receiving more weight.")
print("Novelty: Popularity-based information measure from training data.")
print("Unexpectedness: Genre dissimilarity from the user's liked-movie profile.")
print("Serendipity proxy: Relevant hits weighted by genre-based unexpectedness.")
print("Diversity: Average pairwise genre dissimilarity within the top 10.")

# ------------------------------------------------
# STEP 13: SAVE OUTPUT FILES
# ------------------------------------------------

print("\nSTEP 11: Saving output files...")

summary_df.to_csv(
    os.path.join(PROJECT_DIR, "stage17_serendipity_summary.csv"),
    index=False
)

user_metrics.to_csv(
    os.path.join(PROJECT_DIR, "stage17_user_level_metrics.csv"),
    index=False
)

recommendation_details.to_csv(
    os.path.join(PROJECT_DIR, "stage17_recommendation_details.csv"),
    index=False
)

print("Saved:")
print("stage17_serendipity_summary.csv")
print("stage17_user_level_metrics.csv")
print("stage17_recommendation_details.csv")

print("\n" + "=" * 75)
print("STAGE 17 COMPLETED")
print("=" * 75)
