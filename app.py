import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# SERENDIPITY - Streamlit application
# Uses the Stage 16 split and selected model:
# mean-centered user-based CF, 50 neighbors, lambda=10,
# novelty weight=0.005, Top-10 recommendations.
# ============================================================

st.set_page_config(
    page_title="Serendipity | Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
MOVIES_PATH = os.path.join(DATA_DIR, "movies.csv")

TOP_N = 10
MAX_NEIGHBORS = 50
SPLIT_SEED = 42
HOLDOUT_SPLIT_SEED = 43
REGULARIZATION = 10.0
NOVELTY_WEIGHT = 0.005


@st.cache_resource(show_spinner="Preparing the recommendation model (first run may take a minute)...")
def load_model():
    if not os.path.exists(RATINGS_PATH):
        raise FileNotFoundError(
            f"Cannot find {RATINGS_PATH}. Put ratings.csv inside the project's data folder."
        )
    if not os.path.exists(MOVIES_PATH):
        raise FileNotFoundError(
            f"Cannot find {MOVIES_PATH}. Put movies.csv inside the project's data folder."
        )

    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)

    ratings.columns = ratings.columns.str.strip()
    movies.columns = movies.columns.str.strip()

    required_ratings = {"userId", "movieId", "rating"}
    required_movies = {"movieId", "title", "genres"}

    if not required_ratings.issubset(ratings.columns):
        raise ValueError("ratings.csv must contain userId, movieId and rating columns.")
    if not required_movies.issubset(movies.columns):
        raise ValueError("movies.csv must contain movieId, title and genres columns.")

    ratings = ratings[["userId", "movieId", "rating"]].copy()
    movies = movies[["movieId", "title", "genres"]].drop_duplicates("movieId").copy()

    # Reproduce the Stage 16 train/validation/test split exactly.
    train_df, holdout_df = train_test_split(
        ratings, test_size=0.20, random_state=SPLIT_SEED
    )
    validation_df, test_df = train_test_split(
        holdout_df, test_size=0.50, random_state=HOLDOUT_SPLIT_SEED
    )

    # The application fits recommendation data on training ratings only.
    rating_matrix = train_df.pivot_table(
        index="userId", columns="movieId", values="rating", aggfunc="mean"
    )
    user_means = rating_matrix.mean(axis=1)
    centered_matrix = rating_matrix.sub(user_means, axis=0)
    centered_filled = centered_matrix.fillna(0.0)

    user_ids = rating_matrix.index.to_numpy()
    movie_ids = rating_matrix.columns.to_numpy()

    rating_values = rating_matrix.to_numpy(dtype=np.float32)
    centered_values = centered_filled.to_numpy(dtype=np.float32)
    rated_mask = np.isfinite(rating_values)

    user_to_index = {uid: i for i, uid in enumerate(user_ids)}
    movie_to_index = {mid: i for i, mid in enumerate(movie_ids)}

    # Novelty is calculated from training data only, as in Stage 16.
    unique_raters = (
        train_df.groupby("movieId")["userId"]
        .nunique()
        .reindex(movie_ids)
        .fillna(0)
    )
    n_users = len(user_ids)
    popularity = unique_raters.to_numpy(dtype=float) / n_users
    safe_popularity = np.clip(popularity, 1 / n_users, 1.0)
    novelty_values = np.clip(
        -np.log2(safe_popularity) / np.log2(n_users), 0, 1
    )

    movie_info = movies.set_index("movieId")
    movie_info = movie_info.reindex(movie_ids)

    return {
        "train_df": train_df,
        "rating_matrix": rating_matrix,
        "user_means": user_means,
        "centered_values": centered_values,
        "rating_values": rating_values,
        "rated_mask": rated_mask,
        "user_ids": user_ids,
        "movie_ids": movie_ids,
        "user_to_index": user_to_index,
        "movie_to_index": movie_to_index,
        "novelty_values": novelty_values,
        "movie_info": movie_info,
        "n_users": n_users,
        "n_movies": len(movie_ids),
    }


def recommend_for_user(model, user_id):
    user_idx = model["user_to_index"][user_id]
    user_mean = float(model["user_means"].loc[user_id])

    # Calculate this user's similarity against all training users.
    # This is mathematically equivalent to taking one row from the
    # full Stage 16 cosine-similarity matrix, but uses less memory.
    similarities = cosine_similarity(
        model["centered_values"][user_idx:user_idx + 1],
        model["centered_values"]
    ).ravel()
    similarities[user_idx] = 0.0

    positive_indices = np.flatnonzero(similarities > 0)
    if len(positive_indices):
        order = np.argsort(-similarities[positive_indices], kind="mergesort")
        neighbor_indices = positive_indices[order[:MAX_NEIGHBORS]]
        neighbor_weights = similarities[neighbor_indices]
    else:
        neighbor_indices = np.array([], dtype=int)
        neighbor_weights = np.array([], dtype=float)

    user_ratings = model["rating_values"][user_idx]
    candidate_mask = ~np.isfinite(user_ratings)
    candidate_indices = np.flatnonzero(candidate_mask)

    if len(candidate_indices) == 0:
        return pd.DataFrame()

    if len(neighbor_indices):
        neighbor_ratings = model["rating_values"][neighbor_indices]
        neighbor_centered = np.where(
            np.isfinite(neighbor_ratings),
            neighbor_ratings - np.nan_to_num(
                model["rating_values"][neighbor_indices].mean(axis=1),
                nan=0.0
            )[:, None],
            0.0
        )

        # Match Stage 16's per-user-centered matrix exactly.
        neighbor_centered = model["centered_values"][neighbor_indices]
        neighbor_presence = np.isfinite(neighbor_ratings).astype(np.float32)

        numerator = neighbor_weights @ neighbor_centered
        denominator = neighbor_weights @ neighbor_presence
        deviations = numerator / np.where(
            denominator + REGULARIZATION > 0,
            denominator + REGULARIZATION,
            np.nan
        )
        predictions = user_mean + deviations
        predictions = np.where(np.isfinite(predictions), predictions, user_mean)
        support_counts = neighbor_presence.sum(axis=0)
    else:
        predictions = np.full(len(model["movie_ids"]), user_mean, dtype=float)
        support_counts = np.zeros(len(model["movie_ids"]), dtype=int)

    predictions = np.clip(predictions, 1, 5)

    candidate_predictions = predictions[candidate_indices]
    candidate_novelty = model["novelty_values"][candidate_indices]

    ranking_scores = (
        (1 - NOVELTY_WEIGHT) * ((candidate_predictions - 1) / 4)
        + NOVELTY_WEIGHT * candidate_novelty
    )

    recs = pd.DataFrame({
        "movieId": model["movie_ids"][candidate_indices],
        "Predicted rating": candidate_predictions,
        "Novelty": candidate_novelty,
        "Recommendation score": ranking_scores,
        "Similar users who rated it": support_counts[candidate_indices].astype(int)
    })

    recs = recs.sort_values(
        ["Recommendation score", "movieId"],
        ascending=[False, True],
        kind="mergesort"
    ).head(TOP_N).reset_index(drop=True)

    info = model["movie_info"]
    recs["Movie"] = recs["movieId"].map(info["title"]).fillna("Unknown title")
    recs["Genres"] = recs["movieId"].map(info["genres"]).fillna("Unknown")
    recs["Why recommended"] = recs.apply(
        lambda r: (
            f"Predicted rating {r['Predicted rating']:.2f}/5; "
            f"{int(r['Similar users who rated it'])} of the selected nearest neighbours rated it; "
            f"novelty score {r['Novelty']:.3f}."
        ),
        axis=1
    )
    return recs[
        [
            "Movie", "Genres", "Predicted rating", "Novelty",
            "Recommendation score", "Similar users who rated it",
            "Why recommended", "movieId"
        ]
    ]


st.title("🎬 Serendipity")
st.subheader("Novelty-Aware Personalized Movie Recommendation System")
st.write(
    "Discover movies beyond your usual preferences using "
    "mean-centered user-based collaborative filtering."
)

try:
    model = load_model()
except Exception as exc:
    st.error(str(exc))
    st.info(
        "Expected layout: app.py beside a data folder containing ratings.csv and movies.csv."
    )
    st.stop()

with st.sidebar:
    st.header("Recommendation settings")
    selected_user = st.selectbox(
        "Select an existing MovieLens user",
        options=[int(x) for x in model["user_ids"]],
        index=0
    )
    st.caption("This demo supports users present in the MovieLens training split.")
    st.divider()
    st.write("**Selected model**")
    st.write("Mean-centered user-based CF")
    st.write(f"Nearest neighbours: {MAX_NEIGHBORS}")
    st.write(f"Regularization λ: {REGULARIZATION:g}")
    st.write(f"Novelty weight: {NOVELTY_WEIGHT}")
    st.write(f"Recommendation list: Top {TOP_N}")

col1, col2, col3 = st.columns(3)
col1.metric("Training users", f"{model['n_users']:,}")
col2.metric("Training movies", f"{model['n_movies']:,}")
col3.metric("Selected user", str(selected_user))

st.markdown("---")
st.header(f"Your Top {TOP_N} recommendations")

with st.spinner("Generating personalized recommendations..."):
    recommendations = recommend_for_user(model, selected_user)

if recommendations.empty:
    st.warning("No unseen catalogue movies are available for this user.")
else:
    for rank, row in recommendations.iterrows():
        with st.container(border=True):
            left, right = st.columns([3, 1])
            with left:
                st.markdown(f"**{rank + 1}. {row['Movie']}**")
                st.caption(row["Genres"])
                st.write(row["Why recommended"])
            with right:
                st.metric("Predicted rating", f"{row['Predicted rating']:.2f}/5")
                st.metric("Ranking score", f"{row['Recommendation score']:.4f}")

    csv_data = recommendations.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download recommendations as CSV",
        data=csv_data,
        file_name=f"serendipity_user_{selected_user}_recommendations.csv",
        mime="text/csv"
    )

with st.expander("How does Serendipity work?"):
    st.markdown(
        """
        1. Ratings are split using the same Stage 16 random seeds.
        2. User rating averages are subtracted to account for different rating habits.
        3. Cosine similarity identifies up to 50 similar users.
        4. Neighbour ratings estimate scores for movies the selected user has not rated.
        5. A small novelty adjustment is applied to the ranking score.

        **Ranking score:**  
        `(1 - 0.005) × normalized predicted rating + 0.005 × novelty`

        Novelty is computed from movie popularity in the training data only.
        """
    )

st.caption(
    "Research prototype using MovieLens 1M. Offline evaluation does not establish "
    "real-world user satisfaction. New-user cold-start handling is not implemented."
)
