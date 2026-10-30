
# ============================================================
# STAGE 14: INDEPENDENT FINAL MODEL EVALUATION
# ============================================================
# Project: Serendipitous Novelty-Aware Movie Recommendation
# Dataset: MovieLens 1M
#
# 80% Training / 10% Validation / 10% Final Test
#
# Validation is used for model selection.
# The final test set is evaluated only after selection.
#
# Models:
# 1. User-mean baseline
# 2. Mean-centered user-based CF
# 3. Regularized mean-centered user-based CF
#
# Ranking:
# Collaborative filtering + optional novelty reranking
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


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

TOP_N = 10
MAX_NEIGHBORS = 50

N_VALIDATION_USERS = 500
N_TEST_USERS = 500
N_TEST_RATINGS = 10000

REGULARIZATION_VALUES = [0, 2, 5, 10]
NOVELTY_WEIGHTS = [0.00, 0.005, 0.01, 0.02, 0.05]

SEED = 42


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 75)
print("STAGE 14: INDEPENDENT FINAL MODEL EVALUATION")
print("=" * 75)

print("\nSTEP 1: Loading datasets...")

ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)

ratings = ratings[["userId", "movieId", "rating"]].copy()
movies = movies.drop_duplicates(subset="movieId").copy()

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)


# ============================================================
# 3. CREATE THREE SEPARATE DATA SPLITS
# ============================================================

print("\nSTEP 2: Creating training, validation and test sets...")

# First: 80% training, 20% temporary holdout.
train_df, holdout_df = train_test_split(
    ratings,
    test_size=0.20,
    random_state=42
)

# Divide the remaining 20% equally:
# 10% validation and 10% final test.
validation_df, test_df = train_test_split(
    holdout_df,
    test_size=0.50,
    random_state=43
)

train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Training ratings:", len(train_df))
print("Validation ratings:", len(validation_df))
print("Final test ratings:", len(test_df))

train_pairs = set(
    zip(train_df["userId"], train_df["movieId"])
)

validation_pairs = set(
    zip(validation_df["userId"], validation_df["movieId"])
)

test_pairs = set(
    zip(test_df["userId"], test_df["movieId"])
)

print(
    "Train-validation pair overlap:",
    len(train_pairs & validation_pairs)
)

print(
    "Train-test pair overlap:",
    len(train_pairs & test_pairs)
)

print(
    "Validation-test pair overlap:",
    len(validation_pairs & test_pairs)
)

if (
    (train_pairs & validation_pairs)
    or (train_pairs & test_pairs)
    or (validation_pairs & test_pairs)
):
    raise RuntimeError("Rating-pair leakage detected.")


# ============================================================
# 4. BUILD TRAINING MATRIX
# ============================================================

print("\nSTEP 3: Building training-only user-movie matrix...")

rating_matrix = train_df.pivot_table(
    index="userId",
    columns="movieId",
    values="rating",
    aggfunc="mean"
)

user_means = rating_matrix.mean(axis=1)

global_mean = train_df["rating"].mean()

centered_matrix = rating_matrix.sub(
    user_means,
    axis=0
)

centered_filled = centered_matrix.fillna(0)

user_ids = rating_matrix.index.to_numpy()
movie_ids = rating_matrix.columns.to_numpy()

print("Training matrix:", rating_matrix.shape)
print("Training users:", len(user_ids))
print("Training catalog movies:", len(movie_ids))
print("Training global mean:", round(global_mean, 4))


# ============================================================
# 5. CALCULATE SIMILARITIES USING TRAINING DATA ONLY
# ============================================================

print("\nSTEP 4: Calculating user similarities...")

similarity_values = cosine_similarity(centered_filled)

np.fill_diagonal(similarity_values, 0)

similarity_df = pd.DataFrame(
    similarity_values,
    index=rating_matrix.index,
    columns=rating_matrix.index
)

print("Similarity matrix completed.")


# ============================================================
# 6. CALCULATE TRAINING-ONLY MOVIE NOVELTY
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

novelty_values = (
    -np.log2(safe_popularity)
    / np.log2(n_users)
).clip(0, 1)

movie_novelty = pd.Series(
    novelty_values.to_numpy(),
    index=movie_ids
)

print("Mean movie novelty:", round(movie_novelty.mean(), 4))


# ============================================================
# 7. SELECT SIMILAR USERS
# ============================================================

def get_neighbors(user_id):

    similarities = similarity_df.loc[user_id].drop(user_id)

    similarities = similarities[similarities > 0]

    return similarities.sort_values(
        ascending=False,
        kind="mergesort"
    ).head(MAX_NEIGHBORS)


# ============================================================
# 8. PREDICT RATINGS FOR UNSEEN MOVIES
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
# 9. PREDICT A SINGLE HELD-OUT RATING
# ============================================================

def predict_single_rating(user_id, movie_id, regularization=0):

    user_mean = user_means.loc[user_id]

    similarities = get_neighbors(user_id)

    if similarities.empty:
        return float(user_mean)

    neighbor_ratings = rating_matrix.loc[
        similarities.index,
        movie_id
    ]

    observed = neighbor_ratings.notna()

    neighbor_ratings = neighbor_ratings[observed]

    selected_similarities = similarities.loc[
        neighbor_ratings.index
    ]

    if neighbor_ratings.empty:
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
# 10. PREPARE RELEVANT HELD-OUT MOVIES
# ============================================================

def prepare_relevant_movies(heldout_df):

    relevant_by_user = {}

    relevant_df = heldout_df[
        (heldout_df["rating"] >= 4)
        & heldout_df["userId"].isin(rating_matrix.index)
        & heldout_df["movieId"].isin(rating_matrix.columns)
    ]

    for user_id, group in relevant_df.groupby("userId"):

        eligible_movies = set(group["movieId"].tolist())

        if eligible_movies:
            relevant_by_user[user_id] = eligible_movies

    return relevant_by_user


# ============================================================
# 11. SELECT USERS FOR VALIDATION AND FINAL TEST
# ============================================================

print("\nSTEP 6: Preparing independent evaluation users...")

validation_relevant = prepare_relevant_movies(validation_df)

test_relevant = prepare_relevant_movies(test_df)

validation_eligible_users = np.array(
    list(validation_relevant.keys())
)

test_eligible_users = np.array(
    list(test_relevant.keys())
)

rng_validation = np.random.default_rng(2026)
rng_test = np.random.default_rng(2027)

validation_users = rng_validation.choice(
    validation_eligible_users,
    size=min(N_VALIDATION_USERS, len(validation_eligible_users)),
    replace=False
).tolist()

test_users = rng_test.choice(
    test_eligible_users,
    size=min(N_TEST_USERS, len(test_eligible_users)),
    replace=False
).tolist()

print("Validation users selected:", len(validation_users))
print("Final test users selected:", len(test_users))


# ============================================================
# 12. CACHE PREDICTIONS FOR VALIDATION USERS
# ============================================================

print("\nSTEP 7: Caching validation predictions...")

validation_cache = {}

for position, user_id in enumerate(validation_users, start=1):

    validation_cache[user_id] = {}

    for regularization in REGULARIZATION_VALUES:

        validation_cache[user_id][regularization] = (
            predict_all_movies(
                user_id,
                regularization=regularization
            )
        )

    if position % 100 == 0:
        print(
            "Validation prediction cache:",
            position,
            "of",
            len(validation_users)
        )


# ============================================================
# 13. GENERATE RECOMMENDATIONS
# ============================================================

def make_recommendations(
    user_id,
    regularization,
    novelty_weight,
    prediction_cache
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

    candidates["final_score"] = (
        (1 - novelty_weight) * preference_score
        + novelty_weight * candidates["novelty"]
    )

    candidates = candidates.sort_values(
        ["final_score", "movieId"],
        ascending=[False, True],
        kind="mergesort"
    )

    return candidates.head(TOP_N).reset_index(drop=True)


# ============================================================
# 14. CALCULATE TOP-10 METRICS
# ============================================================

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

    ndcg = dcg / idcg if idcg > 0 else 0.0

    average_novelty = (
        recommendations["novelty"].mean()
        if not recommendations.empty else 0.0
    )

    return precision, recall, ndcg, average_novelty


# ============================================================
# 15. TUNE CONFIGURATIONS ON VALIDATION DATA ONLY
# ============================================================

print("\nSTEP 8: Tuning on validation data...")

validation_results = []

for regularization in REGULARIZATION_VALUES:

    for weight in NOVELTY_WEIGHTS:

        precision_values = []
        recall_values = []
        ndcg_values = []
        novelty_values = []

        distinct_movies = set()

        for user_id in validation_users:

            recommendations = make_recommendations(
                user_id,
                regularization,
                weight,
                validation_cache
            )

            metrics = calculate_metrics(
                recommendations,
                validation_relevant[user_id]
            )

            precision_values.append(metrics[0])
            recall_values.append(metrics[1])
            ndcg_values.append(metrics[2])
            novelty_values.append(metrics[3])

            distinct_movies.update(
                recommendations["movieId"].tolist()
            )

        validation_results.append({
            "regularization": regularization,
            "novelty_weight": weight,
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
            "| NDCG@10:", round(np.mean(ndcg_values), 5),
            "| Novelty:", round(np.mean(novelty_values), 4)
        )


validation_results_df = pd.DataFrame(validation_results)

validation_path = os.path.join(
    PROJECT_DIR,
    "stage14_validation_tuning_results.csv"
)

validation_results_df.to_csv(
    validation_path,
    index=False
)


# ============================================================
# 16. SELECT MODEL USING VALIDATION RESULTS
# ============================================================

print("\nSTEP 9: Selecting configuration using validation data...")

# Primary criterion: NDCG@10.
# Tie-breakers: Precision@10, then lower novelty weight.
# The final test data has not been used for this selection.

selection_table = validation_results_df.sort_values(
    [
        "ndcg_at_10",
        "precision_at_10",
        "novelty_weight"
    ],
    ascending=[False, False, True],
    kind="mergesort"
)

best = selection_table.iloc[0]

best_regularization = int(best["regularization"])
best_novelty_weight = float(best["novelty_weight"])

print("\nSELECTED CONFIGURATION")
print("Regularization:", best_regularization)
print("Novelty weight:", best_novelty_weight)
print("Validation Precision@10:", best["precision_at_10"])
print("Validation Recall@10:", best["recall_at_10"])
print("Validation NDCG@10:", best["ndcg_at_10"])
print("Validation novelty:", best["average_novelty"])
print("Validation coverage:", best["catalog_coverage"])


# ============================================================
# 17. CACHE FINAL TEST PREDICTIONS
# ============================================================

print("\nSTEP 10: Preparing final test evaluation...")

# Only the selected model and the no-novelty CF baseline
# are evaluated on the final test recommendation sample.

test_cache = {}

models_to_cache = sorted(
    set([0, best_regularization])
)

for position, user_id in enumerate(test_users, start=1):

    test_cache[user_id] = {}

    for regularization in models_to_cache:

        test_cache[user_id][regularization] = (
            predict_all_movies(
                user_id,
                regularization=regularization
            )
        )

    if position % 100 == 0:
        print(
            "Final test prediction cache:",
            position,
            "of",
            len(test_users)
        )


# ============================================================
# 18. EVALUATE BASELINE AND SELECTED MODEL ON TEST
# ============================================================

print("\nSTEP 11: Evaluating on untouched final test data...")

test_configurations = [
    {
        "name": "Mean-centered CF baseline",
        "regularization": 0,
        "novelty_weight": 0.0
    },
    {
        "name": "Validation-selected model",
        "regularization": best_regularization,
        "novelty_weight": best_novelty_weight
    }
]

final_test_results = []

for config in test_configurations:

    precision_values = []
    recall_values = []
    ndcg_values = []
    novelty_values = []

    distinct_movies = set()

    for user_id in test_users:

        recommendations = make_recommendations(
            user_id,
            config["regularization"],
            config["novelty_weight"],
            test_cache
        )

        metrics = calculate_metrics(
            recommendations,
            test_relevant[user_id]
        )

        precision_values.append(metrics[0])
        recall_values.append(metrics[1])
        ndcg_values.append(metrics[2])
        novelty_values.append(metrics[3])

        distinct_movies.update(
            recommendations["movieId"].tolist()
        )

    final_test_results.append({
        "model": config["name"],
        "regularization": config["regularization"],
        "novelty_weight": config["novelty_weight"],
        "test_users": len(test_users),
        "precision_at_10": np.mean(precision_values),
        "recall_at_10": np.mean(recall_values),
        "ndcg_at_10": np.mean(ndcg_values),
        "average_novelty": np.mean(novelty_values),
        "catalog_coverage": (
            len(distinct_movies) / len(movie_ids)
        ),
        "distinct_recommended_movies": len(distinct_movies)
    })


final_test_df = pd.DataFrame(final_test_results)

print("\nFINAL TEST RECOMMENDATION RESULTS")
print(
    final_test_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.5f}"
    )
)

final_test_path = os.path.join(
    PROJECT_DIR,
    "stage14_final_test_recommendation_results.csv"
)

final_test_df.to_csv(
    final_test_path,
    index=False
)


# ============================================================
# 19. FINAL TEST RATING-PREDICTION ACCURACY
# ============================================================

print("\nSTEP 12: Evaluating rating-prediction accuracy...")

eligible_test_ratings = test_df[
    test_df["userId"].isin(rating_matrix.index)
    & test_df["movieId"].isin(rating_matrix.columns)
].copy()

test_rating_sample = eligible_test_ratings.sample(
    n=min(N_TEST_RATINGS, len(eligible_test_ratings)),
    random_state=SEED
).reset_index(drop=True)

actual_ratings = test_rating_sample["rating"].to_numpy()

user_mean_predictions = []
cf_predictions = []
selected_model_predictions = []

for position, row in test_rating_sample.iterrows():

    user_id = row["userId"]
    movie_id = row["movieId"]

    user_mean_predictions.append(
        user_means.loc[user_id]
    )

    cf_predictions.append(
        predict_single_rating(
            user_id,
            movie_id,
            regularization=0
        )
    )

    selected_model_predictions.append(
        predict_single_rating(
            user_id,
            movie_id,
            regularization=best_regularization
        )
    )

    if (position + 1) % 2500 == 0:
        print(
            "Evaluated final test ratings:",
            position + 1,
            "of",
            len(test_rating_sample)
        )


def prediction_summary(name, predictions):

    predictions = np.asarray(predictions)

    return {
        "model": name,
        "sampled_test_ratings": len(actual_ratings),
        "MAE": mean_absolute_error(
            actual_ratings, predictions
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                actual_ratings, predictions
            )
        )
    }


rating_test_results = [
    prediction_summary(
        "User-mean baseline",
        user_mean_predictions
    ),
    prediction_summary(
        "Mean-centered CF",
        cf_predictions
    ),
    prediction_summary(
        "Validation-selected regularized CF",
        selected_model_predictions
    )
]

rating_test_df = pd.DataFrame(rating_test_results)

print("\nFINAL TEST RATING-PREDICTION RESULTS")
print(
    rating_test_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.5f}"
    )
)

rating_test_path = os.path.join(
    PROJECT_DIR,
    "stage14_final_test_rating_accuracy.csv"
)

rating_test_df.to_csv(
    rating_test_path,
    index=False
)


# ============================================================
# 20. SAVE MODEL SELECTION SUMMARY
# ============================================================

summary_df = pd.DataFrame([{
    "selected_regularization": best_regularization,
    "selected_novelty_weight": best_novelty_weight,
    "validation_ndcg_at_10": best["ndcg_at_10"],
    "validation_precision_at_10": best["precision_at_10"],
    "final_test_ndcg_at_10": final_test_df.iloc[1]["ndcg_at_10"],
    "final_test_precision_at_10": (
        final_test_df.iloc[1]["precision_at_10"]
    ),
    "final_test_recall_at_10": (
        final_test_df.iloc[1]["recall_at_10"]
    ),
    "final_test_average_novelty": (
        final_test_df.iloc[1]["average_novelty"]
    ),
    "final_test_catalog_coverage": (
        final_test_df.iloc[1]["catalog_coverage"]
    ),
    "selection_basis": (
        "Validation NDCG@10; final test evaluated after selection"
    )
}])

summary_path = os.path.join(
    PROJECT_DIR,
    "stage14_final_model_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 21. COMPLETION
# ============================================================

print("\nFILES SAVED:")

print(validation_path)
print(final_test_path)
print(rating_test_path)
print(summary_path)

print("\n" + "=" * 75)
print("STAGE 14 COMPLETED")
print("=" * 75)
