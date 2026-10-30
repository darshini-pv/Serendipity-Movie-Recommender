
# ============================================================
# STAGE 13: FINAL MODEL SELECTION AND TUNING
# ============================================================
# Project: Serendipitous Novelty-Aware Movie Recommendation
# Dataset: MovieLens 1M
#
# Models:
# 1. User-mean baseline
# 2. Mean-centered user-based collaborative filtering
# 3. Regularized mean-centered user-based CF
#
# Evaluation:
# MAE, RMSE, Precision@10, Recall@10, NDCG@10,
# Average Novelty, Catalog Coverage
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error


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

SEED = 42
TOP_N = 10
MAX_NEIGHBORS = 50
N_EVALUATION_USERS = 500
N_TEST_RATINGS = 20000

# Shrinkage strength for the regularized model.
REGULARIZATION_VALUES = [2, 5, 10]

# Lower weights help avoid overwhelming relevance with novelty.
NOVELTY_WEIGHTS = [0.00, 0.005, 0.01, 0.02, 0.05]


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 75)
print("STAGE 13: FINAL MODEL SELECTION AND TUNING")
print("=" * 75)

print("\nSTEP 1: Loading datasets...")

ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)

ratings = ratings[["userId", "movieId", "rating"]].copy()
movies = movies.drop_duplicates(subset="movieId").copy()

print("Ratings:", ratings.shape)
print("Movies:", movies.shape)


# ============================================================
# 3. TRAIN-TEST SPLIT
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

overlap = (
    set(zip(train_df["userId"], train_df["movieId"]))
    &
    set(zip(test_df["userId"], test_df["movieId"]))
)

print("Train-test pair overlap:", len(overlap))

if overlap:
    raise RuntimeError("Data leakage: train-test pairs overlap.")


# ============================================================
# 4. CREATE TRAINING MATRIX
# ============================================================

print("\nSTEP 3: Building user-movie matrix...")

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

movie_ids = rating_matrix.columns.to_numpy()
user_ids = rating_matrix.index.to_numpy()

print("Matrix shape:", rating_matrix.shape)
print("Training users:", len(user_ids))
print("Training movies:", len(movie_ids))
print("Global mean:", round(global_mean, 4))


# ============================================================
# 5. CALCULATE USER SIMILARITY
# ============================================================

print("\nSTEP 4: Calculating user similarity...")

similarity_array = cosine_similarity(centered_filled)

np.fill_diagonal(similarity_array, 0)

similarity_df = pd.DataFrame(
    similarity_array,
    index=rating_matrix.index,
    columns=rating_matrix.index
)

print("Similarity calculation completed.")


# ============================================================
# 6. CALCULATE CORRECTED NOVELTY
# ============================================================

print("\nSTEP 5: Calculating movie novelty...")

unique_raters = (
    train_df.groupby("movieId")["userId"]
    .nunique()
    .reindex(movie_ids)
    .fillna(0)
)

n_users = len(user_ids)

popularity = unique_raters / n_users

safe_popularity = popularity.clip(
    lower=1 / n_users,
    upper=1.0
)

novelty = (
    -np.log2(safe_popularity)
    / np.log2(n_users)
).clip(0, 1)

movie_novelty = pd.Series(
    novelty.to_numpy(),
    index=movie_ids
)

print("Mean novelty:", round(movie_novelty.mean(), 4))


# ============================================================
# 7. HELPER: SELECT TOP NEIGHBORS
# ============================================================

def get_neighbors(user_id):

    similarities = similarity_df.loc[user_id].drop(user_id)

    similarities = similarities[similarities > 0]

    similarities = similarities.sort_values(
        ascending=False,
        kind="mergesort"
    ).head(MAX_NEIGHBORS)

    return similarities


# ============================================================
# 8. HELPER: PREDICT ALL UNSEEN MOVIES
# ============================================================

def predict_all_movies(user_id, regularization=0):

    user_ratings = rating_matrix.loc[user_id]
    user_mean = user_means.loc[user_id]

    candidate_mask = user_ratings.isna()

    similarities = get_neighbors(user_id)

    if similarities.empty:

        predictions = pd.Series(
            user_mean,
            index=rating_matrix.columns,
            dtype=float
        )

    else:

        neighbor_ids = similarities.index
        weights = similarities.to_numpy()

        neighbor_ratings = rating_matrix.loc[neighbor_ids]
        neighbor_centered = centered_matrix.loc[neighbor_ids]

        observed_mask = neighbor_ratings.notna().astype(float)

        numerator = (
            neighbor_centered.fillna(0)
            .mul(weights, axis=0)
            .sum(axis=0)
        )

        denominator = (
            observed_mask
            .mul(weights, axis=0)
            .sum(axis=0)
        )

        if regularization > 0:

            # Regularized deviation:
            # numerator / (denominator + lambda)
            deviations = numerator.div(
                denominator + regularization
            )

        else:

            deviations = numerator.div(
                denominator.replace(0, np.nan)
            )

        predictions = user_mean + deviations

        predictions = predictions.fillna(user_mean)

    predictions = predictions.clip(1, 5)

    return predictions.loc[candidate_mask]


# ============================================================
# 9. HELPER: PREDICT ONE HELD-OUT RATING
# ============================================================

def predict_single_rating(user_id, movie_id, regularization=0):

    user_mean = user_means.loc[user_id]

    similarities = get_neighbors(user_id)

    if similarities.empty:
        return float(user_mean)

    # Retain only neighbors who rated this movie.
    neighbor_ratings = rating_matrix.loc[
        similarities.index, movie_id
    ]

    observed = neighbor_ratings.notna()

    neighbor_ratings = neighbor_ratings[observed]
    selected_similarities = similarities.loc[
        neighbor_ratings.index
    ]

    if len(neighbor_ratings) == 0:
        return float(user_mean)

    deviations = (
        neighbor_ratings
        - user_means.loc[neighbor_ratings.index]
    )

    numerator = (
        selected_similarities * deviations
    ).sum()

    denominator = selected_similarities.sum()

    if regularization > 0:
        predicted_deviation = (
            numerator / (denominator + regularization)
        )
    elif denominator > 0:
        predicted_deviation = numerator / denominator
    else:
        return float(user_mean)

    prediction = user_mean + predicted_deviation

    return float(np.clip(prediction, 1, 5))


# ============================================================
# 10. EVALUATE RATING PREDICTION ACCURACY
# ============================================================

print("\nSTEP 6: Preparing held-out rating sample...")

eligible_test = test_df[
    test_df["userId"].isin(rating_matrix.index)
    & test_df["movieId"].isin(rating_matrix.columns)
].copy()

test_sample = eligible_test.sample(
    n=min(N_TEST_RATINGS, len(eligible_test)),
    random_state=SEED
).reset_index(drop=True)

print("Eligible held-out ratings:", len(eligible_test))
print("Sampled held-out ratings:", len(test_sample))


print("\nSTEP 7: Evaluating prediction accuracy...")
print("This step may take some time.")

actual = test_sample["rating"].to_numpy()

user_mean_predictions = []
cf_predictions = []
regularized_predictions = {
    value: [] for value in REGULARIZATION_VALUES
}

for position, row in test_sample.iterrows():

    user_id = row["userId"]
    movie_id = row["movieId"]

    user_mean_predictions.append(
        user_means.loc[user_id]
    )

    cf_predictions.append(
        predict_single_rating(
            user_id, movie_id, regularization=0
        )
    )

    for value in REGULARIZATION_VALUES:

        regularized_predictions[value].append(
            predict_single_rating(
                user_id,
                movie_id,
                regularization=value
            )
        )

    if (position + 1) % 5000 == 0:
        print(
            "Evaluated",
            position + 1,
            "of",
            len(test_sample),
            "ratings"
        )


prediction_results = []


def add_prediction_metrics(model_name, predictions):

    predictions = np.asarray(predictions)

    prediction_results.append({
        "model": model_name,
        "MAE": mean_absolute_error(actual, predictions),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predictions)
        ),
        "mean_prediction": np.mean(predictions),
        "prediction_std": np.std(predictions),
        "fraction_at_5": np.mean(
            np.isclose(predictions, 5.0)
        )
    })


add_prediction_metrics(
    "User-mean baseline",
    user_mean_predictions
)

add_prediction_metrics(
    "Mean-centered user-based CF",
    cf_predictions
)

for value in REGULARIZATION_VALUES:

    add_prediction_metrics(
        f"Regularized CF lambda={value}",
        regularized_predictions[value]
    )


prediction_metrics_df = pd.DataFrame(prediction_results)

print("\nRATING PREDICTION RESULTS")
print(
    prediction_metrics_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.5f}"
    )
)

prediction_metrics_path = os.path.join(
    PROJECT_DIR,
    "stage13_prediction_accuracy_comparison.csv"
)

prediction_metrics_df.to_csv(
    prediction_metrics_path,
    index=False
)


# ============================================================
# 11. PREPARE RELEVANT HELD-OUT MOVIES
# ============================================================

print("\nSTEP 8: Preparing recommendation evaluation...")

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

rng = np.random.default_rng(SEED)

evaluation_users = rng.choice(
    eligible_users,
    size=min(N_EVALUATION_USERS, len(eligible_users)),
    replace=False
).tolist()

print("Eligible users:", len(eligible_users))
print("Evaluation users:", len(evaluation_users))


# ============================================================
# 12. CACHE CF AND REGULARIZED PREDICTIONS
# ============================================================

print("\nSTEP 9: Caching recommendation predictions...")

prediction_cache = {}

# Compare ordinary CF with regularization strengths.
# Regularized CF with lambda=5 is one candidate, not a
# preselected winner.

regularization_candidates = [0, 2, 5, 10]

for position, user_id in enumerate(evaluation_users, start=1):

    prediction_cache[user_id] = {}

    for value in regularization_candidates:

        prediction_cache[user_id][value] = (
            predict_all_movies(
                user_id,
                regularization=value
            )
        )

    if position % 100 == 0:
        print(
            "Cached",
            position,
            "of",
            len(evaluation_users),
            "users"
        )


# ============================================================
# 13. RECOMMENDATION METRICS
# ============================================================

def make_recommendations(
    user_id,
    regularization,
    novelty_weight
):

    predictions = prediction_cache[user_id][regularization]

    candidates = pd.DataFrame({
        "movieId": predictions.index.to_numpy(),
        "predicted_rating": predictions.to_numpy()
    })

    candidates["novelty"] = (
        candidates["movieId"]
        .map(movie_novelty)
        .fillna(0.0)
    )

    preference_score = (
        (candidates["predicted_rating"] - 1) / 4
    )

    candidates["score"] = (
        (1 - novelty_weight) * preference_score
        + novelty_weight * candidates["novelty"]
    )

    candidates = candidates.sort_values(
        ["score", "movieId"],
        ascending=[False, True],
        kind="mergesort"
    )

    return candidates.head(TOP_N).reset_index(drop=True)


def calculate_metrics(recommendations, relevant_items):

    recommended_ids = recommendations["movieId"].tolist()

    hits = [
        int(movie_id in relevant_items)
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

    ndcg = dcg / idcg if idcg else 0.0

    average_novelty = (
        recommendations["novelty"].mean()
        if not recommendations.empty else 0.0
    )

    return precision, recall, ndcg, average_novelty


# ============================================================
# 14. TUNE REGULARIZATION AND NOVELTY WEIGHT
# ============================================================

print("\nSTEP 10: Tuning regularization and novelty...")

ranking_results = []

for regularization in regularization_candidates:

    for weight in NOVELTY_WEIGHTS:

        precision_values = []
        recall_values = []
        ndcg_values = []
        novelty_values = []

        distinct_movies = set()

        for user_id in evaluation_users:

            recommendations = make_recommendations(
                user_id,
                regularization,
                weight
            )

            metrics = calculate_metrics(
                recommendations,
                relevant_by_user[user_id]
            )

            precision_values.append(metrics[0])
            recall_values.append(metrics[1])
            ndcg_values.append(metrics[2])
            novelty_values.append(metrics[3])

            distinct_movies.update(
                recommendations["movieId"].tolist()
            )

        ranking_results.append({
            "regularization": regularization,
            "novelty_weight": weight,
            "evaluation_users": len(evaluation_users),
            "precision_at_10": np.mean(precision_values),
            "recall_at_10": np.mean(recall_values),
            "ndcg_at_10": np.mean(ndcg_values),
            "average_novelty": np.mean(novelty_values),
            "catalog_coverage": (
                len(distinct_movies) / len(movie_ids)
            ),
            "distinct_recommended_movies": len(distinct_movies)
        })

        print(
            "Lambda:", regularization,
            "| Novelty:", f"{weight:.1%}",
            "| P@10:", round(np.mean(precision_values), 5),
            "| R@10:", round(np.mean(recall_values), 5),
            "| NDCG:", round(np.mean(ndcg_values), 5),
            "| Novelty score:", round(np.mean(novelty_values), 4),
            "| Coverage:", round(
                100 * len(distinct_movies) / len(movie_ids), 2
            ), "%"
        )


ranking_metrics_df = pd.DataFrame(ranking_results)

ranking_metrics_path = os.path.join(
    PROJECT_DIR,
    "stage13_ranking_tuning_results.csv"
)

ranking_metrics_df.to_csv(
    ranking_metrics_path,
    index=False
)


# ============================================================
# 15. SELECT A CANDIDATE BASED ON NDCG
# ============================================================

print("\nSTEP 11: Selecting a candidate configuration...")

# Choose the configuration with the highest NDCG.
# If tied, prefer higher precision, then lower novelty weight.
# This is a provisional selection on the tuning sample.

selection_table = ranking_metrics_df.sort_values(
    [
        "ndcg_at_10",
        "precision_at_10",
        "novelty_weight"
    ],
    ascending=[False, False, True],
    kind="mergesort"
)

best_configuration = selection_table.iloc[0]

print("\nPROVISIONAL BEST CONFIGURATION")
print("Regularization:", best_configuration["regularization"])
print("Novelty weight:", best_configuration["novelty_weight"])
print("Precision@10:", best_configuration["precision_at_10"])
print("Recall@10:", best_configuration["recall_at_10"])
print("NDCG@10:", best_configuration["ndcg_at_10"])
print("Average novelty:", best_configuration["average_novelty"])
print("Catalog coverage:", best_configuration["catalog_coverage"])


# ============================================================
# 16. SAVE FINAL SUMMARY
# ============================================================

summary_path = os.path.join(
    PROJECT_DIR,
    "stage13_model_selection_summary.csv"
)

summary_df = pd.DataFrame([{
    "selected_regularization": best_configuration["regularization"],
    "selected_novelty_weight": best_configuration["novelty_weight"],
    "precision_at_10": best_configuration["precision_at_10"],
    "recall_at_10": best_configuration["recall_at_10"],
    "ndcg_at_10": best_configuration["ndcg_at_10"],
    "average_novelty": best_configuration["average_novelty"],
    "catalog_coverage": best_configuration["catalog_coverage"],
    "selection_basis": "Highest NDCG@10 on the tuning evaluation sample"
}])

summary_df.to_csv(summary_path, index=False)

print("\nFILES SAVED")
print(prediction_metrics_path)
print(ranking_metrics_path)
print(summary_path)

print("\n" + "=" * 75)
print("STAGE 13 COMPLETED")
print("=" * 75)
