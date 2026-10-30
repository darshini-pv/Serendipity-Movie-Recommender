
# =============================================================================
# STAGE 22: HYBRID MOVIE RECOMMENDATION SYSTEM
# Collaborative Filtering + Content-Based Filtering
# Project: Serendipity - Novelty-Aware Personalized Movie Recommendation System
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# =============================================================================
# STEP 1: PROJECT PATHS AND SETTINGS
# =============================================================================

PROJECT_PATH = (
    r"C:\Users\manoj\OneDrive\Desktop"
    r"\Serendipitous Recommendation Engine"
)

DATA_PATH = os.path.join(PROJECT_PATH, "data")

RATINGS_FILE = os.path.join(DATA_PATH, "ratings.csv")
MOVIES_FILE = os.path.join(DATA_PATH, "movies.csv")

TOP_N = 10
NUMBER_OF_NEIGHBORS = 50
REGULARIZATION_LAMBDA = 10

VALIDATION_USER_SAMPLE = 300
TEST_USER_SAMPLE = 500

RANDOM_STATE_VALIDATION = 2026
RANDOM_STATE_TEST = 2027

# Collaborative filtering weights to evaluate.
# Content weight = 1 - CF weight.
HYBRID_WEIGHTS = [1.00, 0.75, 0.50, 0.25, 0.00]


# =============================================================================
# STEP 2: LOAD DATA
# =============================================================================

print("=" * 80)
print("STAGE 22: HYBRID RECOMMENDATION SYSTEM")
print("=" * 80)

print("\nSTEP 1: Loading datasets...")

ratings = pd.read_csv(RATINGS_FILE)
movies = pd.read_csv(MOVIES_FILE)

ratings.columns = ratings.columns.str.lower()
movies.columns = movies.columns.str.lower()

required_rating_columns = {"userid", "movieid", "rating"}
required_movie_columns = {"movieid", "title", "genres"}

if not required_rating_columns.issubset(ratings.columns):
    raise ValueError("Ratings file is missing required columns.")

if not required_movie_columns.issubset(movies.columns):
    raise ValueError("Movies file is missing required columns.")

ratings = ratings[["userid", "movieid", "rating"]].copy()
movies = movies[["movieid", "title", "genres"]].copy()

ratings["userid"] = ratings["userid"].astype(int)
ratings["movieid"] = ratings["movieid"].astype(int)
movies["movieid"] = movies["movieid"].astype(int)

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)
print("Unique users:", ratings["userid"].nunique())
print("Unique rated movies:", ratings["movieid"].nunique())


# =============================================================================
# STEP 3: CREATE TRAIN / VALIDATION / TEST SPLIT
# =============================================================================

print("\nSTEP 2: Creating train-validation-test split...")

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

train_pairs = set(
    zip(train_df["userid"], train_df["movieid"])
)

validation_pairs = set(
    zip(validation_df["userid"], validation_df["movieid"])
)

test_pairs = set(
    zip(test_df["userid"], test_df["movieid"])
)

print("Training ratings:", len(train_df))
print("Validation ratings:", len(validation_df))
print("Test ratings:", len(test_df))

print("Train-validation overlap:", len(train_pairs & validation_pairs))
print("Train-test overlap:", len(train_pairs & test_pairs))
print("Validation-test overlap:", len(validation_pairs & test_pairs))

if train_pairs & validation_pairs:
    raise ValueError("Train-validation leakage detected.")

if train_pairs & test_pairs:
    raise ValueError("Train-test leakage detected.")

if validation_pairs & test_pairs:
    raise ValueError("Validation-test leakage detected.")


# =============================================================================
# STEP 4: CREATE USER AND MOVIE INDEXES
# =============================================================================

print("\nSTEP 3: Creating user and movie indexes...")

user_ids = np.sort(train_df["userid"].unique())
movie_ids = np.sort(train_df["movieid"].unique())

user_to_index = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}

movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(movie_ids)
}

index_to_movie = {
    index: movie_id
    for movie_id, index in movie_to_index.items()
}

number_of_users = len(user_ids)
number_of_movies = len(movie_ids)

print("Training users:", number_of_users)
print("Training movies:", number_of_movies)


# =============================================================================
# STEP 5: CREATE USER-MOVIE RATING MATRIX
# =============================================================================

print("\nSTEP 4: Creating user-movie rating matrix...")

train_df["user_index"] = train_df["userid"].map(user_to_index)
train_df["movie_index"] = train_df["movieid"].map(movie_to_index)

rating_matrix = csr_matrix(
    (
        train_df["rating"].values,
        (
            train_df["user_index"].values,
            train_df["movie_index"].values
        )
    ),
    shape=(number_of_users, number_of_movies)
)

user_rating_counts = np.diff(rating_matrix.indptr)

user_means = np.asarray(
    rating_matrix.sum(axis=1)
).flatten() / np.maximum(user_rating_counts, 1)

global_mean = float(train_df["rating"].mean())

print("Rating matrix shape:", rating_matrix.shape)
print("Training global mean:", round(global_mean, 4))


# =============================================================================
# STEP 6: BUILD MEAN-CENTERED COLLABORATIVE FILTERING MATRIX
# =============================================================================

print("\nSTEP 5: Building mean-centered collaborative filtering model...")

centered_matrix = rating_matrix.copy().astype(float)

for user_index in range(number_of_users):
    start = centered_matrix.indptr[user_index]
    end = centered_matrix.indptr[user_index + 1]

    if start < end:
        centered_matrix.data[start:end] -= user_means[user_index]

# Binary matrix indicates whether a user rated a movie.
rating_presence_matrix = rating_matrix.copy()
rating_presence_matrix.data = np.ones_like(
    rating_presence_matrix.data
)

print("Mean-centered matrix created.")


# =============================================================================
# STEP 7: CALCULATE USER SIMILARITIES
# =============================================================================

print("\nSTEP 6: Calculating user similarities...")

similarity_matrix = cosine_similarity(
    centered_matrix,
    dense_output=True
)

np.fill_diagonal(similarity_matrix, 0)

print("Similarity matrix shape:", similarity_matrix.shape)


# =============================================================================
# STEP 8: STORE TOP POSITIVE NEIGHBORS
# =============================================================================

print("\nSTEP 7: Selecting top neighbors...")

neighbor_indices = []
neighbor_similarities = []

for user_index in range(number_of_users):

    similarities = similarity_matrix[user_index]

    positive_indices = np.where(similarities > 0)[0]

    if len(positive_indices) == 0:
        neighbor_indices.append(np.array([], dtype=int))
        neighbor_similarities.append(np.array([], dtype=float))
        continue

    positive_values = similarities[positive_indices]

    number_to_select = min(
        NUMBER_OF_NEIGHBORS,
        len(positive_indices)
    )

    selected_positions = np.argpartition(
        positive_values,
        -number_to_select
    )[-number_to_select:]

    selected_indices = positive_indices[selected_positions]
    selected_values = positive_values[selected_positions]

    order = np.argsort(selected_values)[::-1]

    neighbor_indices.append(selected_indices[order])
    neighbor_similarities.append(selected_values[order])

print("Top-neighbor lists created.")


# =============================================================================
# STEP 9: CREATE CONTENT-BASED GENRE FEATURES
# =============================================================================

print("\nSTEP 8: Creating content-based genre features...")

movies["genres"] = movies["genres"].fillna("")

genre_features = movies["genres"].str.get_dummies(sep="|")

genre_feature_columns = genre_features.columns.tolist()

genre_matrix_df = pd.concat(
    [
        movies[["movieid"]].reset_index(drop=True),
        genre_features.reset_index(drop=True)
    ],
    axis=1
)

genre_matrix_df = genre_matrix_df.set_index("movieid")

# Keep only movies present in the training rating matrix.
genre_matrix_df = genre_matrix_df.reindex(
    movie_ids,
    fill_value=0
)

genre_matrix = genre_matrix_df.to_numpy(dtype=float)

print("Genre feature matrix shape:", genre_matrix.shape)
print("Number of genres:", len(genre_feature_columns))


# =============================================================================
# STEP 10: BUILD USER PROFILES USING TRAINING RATINGS ONLY
# =============================================================================

print("\nSTEP 9: Building user genre profiles...")

liked_train_df = train_df[train_df["rating"] >= 4].copy()

user_profiles = np.zeros(
    (number_of_users, len(genre_feature_columns)),
    dtype=float
)

for user_id, group in liked_train_df.groupby("userid"):

    user_index = user_to_index[user_id]

    liked_movie_indices = group["movie_index"].values

    if len(liked_movie_indices) > 0:

        user_profiles[user_index] = genre_matrix[
            liked_movie_indices
        ].mean(axis=0)

profile_norms = np.linalg.norm(
    user_profiles,
    axis=1
)

print(
    "Users with content profiles:",
    int(np.sum(profile_norms > 0))
)


# =============================================================================
# STEP 11: CREATE TRAINING POPULARITY SCORES
# =============================================================================

print("\nSTEP 10: Creating popularity fallback...")

movie_rating_counts = np.asarray(
    rating_presence_matrix.sum(axis=0)
).flatten()

max_movie_count = max(
    movie_rating_counts.max(),
    1
)

popularity_scores = movie_rating_counts / max_movie_count

print("Popularity fallback created.")


# =============================================================================
# STEP 12: PREPARE USER HISTORY AND RELEVANT ITEMS
# =============================================================================

print("\nSTEP 11: Preparing evaluation data...")

train_seen_movies = {}

for user_id, group in train_df.groupby("userid"):
    train_seen_movies[user_id] = set(
        group["movieid"].astype(int)
    )


def build_relevant_items(dataframe):

    relevant_items = {}

    relevant_df = dataframe[dataframe["rating"] >= 4]

    for user_id, group in relevant_df.groupby("userid"):

        user_id = int(user_id)

        if user_id not in user_to_index:
            continue

        relevant_movie_ids = set(
            group["movieid"].astype(int)
        )

        relevant_movie_ids = (
            relevant_movie_ids
            & set(movie_ids)
        )

        relevant_movie_ids -= train_seen_movies.get(
            user_id,
            set()
        )

        if len(relevant_movie_ids) > 0:
            relevant_items[user_id] = relevant_movie_ids

    return relevant_items


validation_relevant = build_relevant_items(validation_df)
test_relevant = build_relevant_items(test_df)

print(
    "Validation users with relevant items:",
    len(validation_relevant)
)

print(
    "Test users with relevant items:",
    len(test_relevant)
)


# =============================================================================
# STEP 13: SELECT EVALUATION USERS
# =============================================================================

print("\nSTEP 12: Selecting validation and test users...")

validation_user_candidates = np.array(
    sorted(validation_relevant.keys())
)

test_user_candidates = np.array(
    sorted(test_relevant.keys())
)

validation_rng = np.random.default_rng(
    RANDOM_STATE_VALIDATION
)

test_rng = np.random.default_rng(
    RANDOM_STATE_TEST
)

validation_sample_size = min(
    VALIDATION_USER_SAMPLE,
    len(validation_user_candidates)
)

test_sample_size = min(
    TEST_USER_SAMPLE,
    len(test_user_candidates)
)

validation_users = validation_rng.choice(
    validation_user_candidates,
    size=validation_sample_size,
    replace=False
)

test_users = test_rng.choice(
    test_user_candidates,
    size=test_sample_size,
    replace=False
)

print("Validation sample:", len(validation_users))
print("Test sample:", len(test_users))


# =============================================================================
# STEP 14: DEFINE HYBRID RECOMMENDATION FUNCTION
# =============================================================================

print("\nSTEP 13: Defining recommendation function...")


def recommend_for_user(user_id, collaborative_weight):

    content_weight = 1.0 - collaborative_weight

    user_index = user_to_index[user_id]

    # Candidate movies are training-catalog movies not rated by this user.
    seen_movies = train_seen_movies.get(user_id, set())

    candidate_movie_indices = np.array(
        [
            index
            for index, movie_id in index_to_movie.items()
            if movie_id not in seen_movies
        ],
        dtype=int
    )

    if len(candidate_movie_indices) == 0:
        return []

    # ---------------------------------------------------------
    # COLLABORATIVE FILTERING SCORE
    # ---------------------------------------------------------

    neighbor_ids = neighbor_indices[user_index]
    neighbor_weights = neighbor_similarities[user_index]

    if len(neighbor_ids) > 0:

        neighbor_centered_ratings = centered_matrix[
            neighbor_ids
        ]

        neighbor_presence = rating_presence_matrix[
            neighbor_ids
        ]

        weighted_deviations = (
            neighbor_weights @ neighbor_centered_ratings
        )

        weighted_counts = (
            neighbor_weights @ neighbor_presence
        )

        weighted_deviations = np.asarray(
            weighted_deviations
        ).flatten()

        weighted_counts = np.asarray(
            weighted_counts
        ).flatten()

        cf_predictions = (
            user_means[user_index]
            + weighted_deviations
            / (
                weighted_counts
                + REGULARIZATION_LAMBDA
            )
        )

    else:

        cf_predictions = np.full(
            number_of_movies,
            user_means[user_index]
        )

    cf_predictions = np.clip(
        cf_predictions,
        1,
        5
    )

    # Normalize predicted ratings to the range 0 to 1.
    cf_scores = (cf_predictions - 1) / 4

    # ---------------------------------------------------------
    # CONTENT-BASED SCORE
    # ---------------------------------------------------------

    user_profile = user_profiles[user_index]

    profile_norm = np.linalg.norm(user_profile)

    if profile_norm > 0:

        candidate_genres = genre_matrix[
            candidate_movie_indices
        ]

        candidate_norms = np.linalg.norm(
            candidate_genres,
            axis=1
        )

        content_scores = np.zeros(
            len(candidate_movie_indices),
            dtype=float
        )

        valid_norms = candidate_norms > 0

        content_scores[valid_norms] = (
            candidate_genres[valid_norms]
            @ user_profile
        ) / (
            candidate_norms[valid_norms]
            * profile_norm
        )

    else:

        # Fallback for users without liked training movies.
        content_scores = popularity_scores[
            candidate_movie_indices
        ]

    # ---------------------------------------------------------
    # HYBRID SCORE
    # ---------------------------------------------------------

    hybrid_scores = (
        collaborative_weight
        * cf_scores[candidate_movie_indices]
        +
        content_weight
        * content_scores
    )

    # Stable tie-breaking by movie index.
    order = np.lexsort(
        (
            candidate_movie_indices,
            -hybrid_scores
        )
    )

    top_positions = order[:TOP_N]

    top_movie_indices = candidate_movie_indices[
        top_positions
    ]

    top_scores = hybrid_scores[
        top_positions
    ]

    recommendations = []

    for movie_index, score in zip(
        top_movie_indices,
        top_scores
    ):

        movie_id = index_to_movie[movie_index]

        recommendations.append(
            {
                "movieid": int(movie_id),
                "score": float(score)
            }
        )

    return recommendations


# =============================================================================
# STEP 15: EVALUATION METRICS
# =============================================================================

print("\nSTEP 14: Defining evaluation metrics...")


def calculate_metrics(
    evaluation_users,
    relevant_items,
    collaborative_weight
):

    precision_values = []
    recall_values = []
    ndcg_values = []
    novelty_values = []

    all_recommended_movies = set()

    total_recommendations = 0
    users_with_hit = 0

    for user_id in evaluation_users:

        recommendations = recommend_for_user(
            int(user_id),
            collaborative_weight
        )

        recommended_movies = [
            item["movieid"]
            for item in recommendations
        ]

        relevant = relevant_items.get(
            int(user_id),
            set()
        )

        hits = [
            1 if movie_id in relevant else 0
            for movie_id in recommended_movies
        ]

        hit_count = sum(hits)

        if hit_count > 0:
            users_with_hit += 1

        precision = hit_count / TOP_N

        recall = (
            hit_count / len(relevant)
            if len(relevant) > 0
            else 0
        )

        dcg = sum(
            hit / np.log2(rank + 2)
            for rank, hit in enumerate(hits)
        )

        ideal_hits = min(
            len(relevant),
            TOP_N
        )

        idcg = sum(
            1 / np.log2(rank + 2)
            for rank in range(ideal_hits)
        )

        ndcg = (
            dcg / idcg
            if idcg > 0
            else 0
        )

        # Novelty based on training unique-rater counts.
        novelty_scores = []

        for movie_id in recommended_movies:

            movie_index = movie_to_index[movie_id]

            count = movie_rating_counts[movie_index]

            probability = count / number_of_users

            if probability > 0:

                novelty = (
                    -np.log2(probability)
                    / np.log2(number_of_users)
                )

            else:

                novelty = 1.0

            novelty_scores.append(novelty)

        average_novelty = (
            np.mean(novelty_scores)
            if len(novelty_scores) > 0
            else 0
        )

        precision_values.append(precision)
        recall_values.append(recall)
        ndcg_values.append(ndcg)
        novelty_values.append(average_novelty)

        all_recommended_movies.update(
            recommended_movies
        )

        total_recommendations += len(
            recommended_movies
        )

    number_of_evaluated_users = len(
        evaluation_users
    )

    coverage = (
        len(all_recommended_movies)
        / number_of_movies
        if number_of_movies > 0
        else 0
    )

    results = {
        "CF_Weight": collaborative_weight,
        "Content_Weight": 1.0 - collaborative_weight,
        "Users_Evaluated": number_of_evaluated_users,
        "Precision@10": np.mean(precision_values),
        "Recall@10": np.mean(recall_values),
        "NDCG@10": np.mean(ndcg_values),
        "Novelty": np.mean(novelty_values),
        "Coverage": coverage,
        "Distinct_Movies": len(all_recommended_movies),
        "Total_Recommendations": total_recommendations,
        "Users_With_Hit": users_with_hit
    }

    return results


# =============================================================================
# STEP 16: VALIDATION WEIGHT TUNING
# =============================================================================

print("\nSTEP 15: Tuning hybrid weights on validation data...")
print("=" * 80)

validation_results = []

for cf_weight in HYBRID_WEIGHTS:

    content_weight = 1.0 - cf_weight

    print(
        "\nEvaluating CF weight:",
        cf_weight,
        "| Content weight:",
        content_weight
    )

    result = calculate_metrics(
        validation_users,
        validation_relevant,
        cf_weight
    )

    validation_results.append(result)

    print(
        "Precision@10:",
        round(result["Precision@10"], 5)
    )

    print(
        "Recall@10:",
        round(result["Recall@10"], 5)
    )

    print(
        "NDCG@10:",
        round(result["NDCG@10"], 5)
    )

    print(
        "Novelty:",
        round(result["Novelty"], 5)
    )

    print(
        "Coverage:",
        round(result["Coverage"], 5)
    )


validation_results_df = pd.DataFrame(
    validation_results
)

validation_results_df = validation_results_df.sort_values(
    by=["NDCG@10", "Precision@10"],
    ascending=False
)

best_cf_weight = float(
    validation_results_df.iloc[0]["CF_Weight"]
)

best_content_weight = 1.0 - best_cf_weight

print("\n" + "=" * 80)
print("BEST VALIDATION CONFIGURATION")
print("=" * 80)

print("Best CF weight:", best_cf_weight)
print("Best content weight:", best_content_weight)

print(
    "Best validation NDCG@10:",
    round(
        validation_results_df.iloc[0]["NDCG@10"],
        5
    )
)


# =============================================================================
# STEP 17: FINAL TEST EVALUATION
# =============================================================================

print("\nSTEP 16: Evaluating selected model on untouched test data...")
print("=" * 80)

final_test_result = calculate_metrics(
    test_users,
    test_relevant,
    best_cf_weight
)

print("\nFINAL TEST RESULTS")

for metric, value in final_test_result.items():

    if isinstance(value, (float, np.floating)):
        print(metric, ":", round(float(value), 5))

    else:
        print(metric, ":", value)


# =============================================================================
# STEP 18: COMPARE PURE CF AND SELECTED HYBRID MODEL
# =============================================================================

print("\nSTEP 17: Comparing pure CF and selected hybrid model...")

baseline_test_result = calculate_metrics(
    test_users,
    test_relevant,
    1.0
)

comparison_results = pd.DataFrame(
    [
        {
            "Model": "Pure Collaborative Filtering",
            **baseline_test_result
        },
        {
            "Model": "Selected Hybrid Model",
            **final_test_result
        }
    ]
)

print("\nMODEL COMPARISON")

print(
    comparison_results[
        [
            "Model",
            "Precision@10",
            "Recall@10",
            "NDCG@10",
            "Novelty",
            "Coverage",
            "Distinct_Movies"
        ]
    ].to_string(index=False)
)


# =============================================================================
# STEP 19: GENERATE SAMPLE RECOMMENDATIONS
# =============================================================================

print("\nSTEP 18: Generating sample hybrid recommendations...")

sample_user_id = int(test_users[0])

sample_recommendations = recommend_for_user(
    sample_user_id,
    best_cf_weight
)

recommendation_rows = []

for rank, recommendation in enumerate(
    sample_recommendations,
    start=1
):

    movie_id = recommendation["movieid"]

    movie_info = movies[
        movies["movieid"] == movie_id
    ]

    if len(movie_info) > 0:

        title = movie_info.iloc[0]["title"]
        genres = movie_info.iloc[0]["genres"]

    else:

        title = "Unknown"
        genres = "Unknown"

    recommendation_rows.append(
        {
            "userId": sample_user_id,
            "rank": rank,
            "movieId": movie_id,
            "title": title,
            "genres": genres,
            "hybrid_score": recommendation["score"]
        }
    )

sample_recommendations_df = pd.DataFrame(
    recommendation_rows
)

print("\nSample user:", sample_user_id)

print(
    sample_recommendations_df.to_string(
        index=False
    )
)


# =============================================================================
# STEP 20: SAVE RESULTS
# =============================================================================

print("\nSTEP 19: Saving Stage 22 results...")

validation_results_df.to_csv(
    os.path.join(
        PROJECT_PATH,
        "stage22_validation_weight_results.csv"
    ),
    index=False
)

comparison_results.to_csv(
    os.path.join(
        PROJECT_PATH,
        "stage22_final_test_model_comparison.csv"
    ),
    index=False
)

sample_recommendations_df.to_csv(
    os.path.join(
        PROJECT_PATH,
        "stage22_sample_hybrid_recommendations.csv"
    ),
    index=False
)

final_summary = pd.DataFrame(
    [final_test_result]
)

final_summary.to_csv(
    os.path.join(
        PROJECT_PATH,
        "stage22_final_test_summary.csv"
    ),
    index=False
)


# =============================================================================
# STEP 21: FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("STAGE 22 COMPLETED")
print("=" * 80)

print("Selected CF weight:", best_cf_weight)
print("Selected content weight:", best_content_weight)

print(
    "Final Test Precision@10:",
    round(final_test_result["Precision@10"], 5)
)

print(
    "Final Test Recall@10:",
    round(final_test_result["Recall@10"], 5)
)

print(
    "Final Test NDCG@10:",
    round(final_test_result["NDCG@10"], 5)
)

print(
    "Final Test Novelty:",
    round(final_test_result["Novelty"], 5)
)

print(
    "Final Test Coverage:",
    round(final_test_result["Coverage"], 5)
)

print("\nSaved files:")
print("1. stage22_validation_weight_results.csv")
print("2. stage22_final_test_model_comparison.csv")
print("3. stage22_sample_hybrid_recommendations.csv")
print("4. stage22_final_test_summary.csv")


 