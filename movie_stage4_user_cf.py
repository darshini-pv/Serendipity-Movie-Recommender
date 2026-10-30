
# ============================================================
# STAGE 4: USER-BASED COLLABORATIVE FILTERING
# Project: Serendipity - A Novelty-Aware Movie Recommender
# ============================================================

import pandas as pd
from tkinter import Tk, filedialog
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# 1. SELECT DATASETS
# ------------------------------------------------------------

root = Tk()
root.withdraw()
root.attributes("-topmost", True)

print("Select ratings.csv")
ratings_path = filedialog.askopenfilename(
    title="Select ratings.csv",
    filetypes=[("CSV files", "*.csv")]
)

print("Select movies.csv")
movies_path = filedialog.askopenfilename(
    title="Select movies.csv",
    filetypes=[("CSV files", "*.csv")]
)

root.destroy()

if not ratings_path or not movies_path:
    raise ValueError("Please select both CSV files.")

ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)

# ------------------------------------------------------------
# 2. CREATE USER-MOVIE RATING MATRIX
# ------------------------------------------------------------

print("\nCreating user-movie rating matrix...")

rating_matrix = ratings.pivot(
    index="userId",
    columns="movieId",
    values="rating"
)

print("Matrix shape:", rating_matrix.shape)

# ------------------------------------------------------------
# 3. FILL MISSING RATINGS WITH ZERO FOR SIMILARITY
# ------------------------------------------------------------
# Zero means "not rated" here, not a real movie rating.

rating_matrix_filled = rating_matrix.fillna(0)

# ------------------------------------------------------------
# 4. CALCULATE USER SIMILARITY
# ------------------------------------------------------------

print("\nCalculating user similarity...")

similarity_matrix = cosine_similarity(rating_matrix_filled)

user_similarity = pd.DataFrame(
    similarity_matrix,
    index=rating_matrix.index,
    columns=rating_matrix.index
)

print("User similarity calculation completed.")

# ------------------------------------------------------------
# 5. SELECT A TARGET USER
# ------------------------------------------------------------

user_id = int(input(
    "\nEnter a user ID between 1 and 6040: "
))

if user_id not in rating_matrix.index:
    raise ValueError("This user ID does not exist in the rating matrix.")

# ------------------------------------------------------------
# 6. FIND SIMILAR USERS
# ------------------------------------------------------------

similar_users = user_similarity[user_id].drop(user_id)

similar_users = similar_users.sort_values(ascending=False)

# Select the 20 most similar users
top_similar_users = similar_users.head(20)

print("\nTop 10 similar users:")
print(top_similar_users.head(10).round(3))

# ------------------------------------------------------------
# 7. FIND MOVIES THE TARGET USER HAS NOT RATED
# ------------------------------------------------------------

target_user_ratings = rating_matrix.loc[user_id]

unrated_movies = target_user_ratings[
    target_user_ratings.isna()
].index

# ------------------------------------------------------------
# 8. CALCULATE PERSONALIZED SCORES
# ------------------------------------------------------------
# Predict each candidate movie's score using ratings
# from the most similar users, weighted by similarity.

candidate_ratings = ratings[
    ratings["userId"].isin(top_similar_users.index)
    & ratings["movieId"].isin(unrated_movies)
].copy()

candidate_ratings["similarity"] = candidate_ratings["userId"].map(
    top_similar_users.to_dict()
)

candidate_ratings["weighted_rating"] = (
    candidate_ratings["rating"] * candidate_ratings["similarity"]
)

movie_scores = candidate_ratings.groupby("movieId").agg(
    weighted_sum=("weighted_rating", "sum"),
    similarity_sum=("similarity", "sum"),
    neighbour_count=("userId", "nunique")
).reset_index()

# Avoid division by zero
movie_scores = movie_scores[
    movie_scores["similarity_sum"] > 0
].copy()

movie_scores["predicted_rating"] = (
    movie_scores["weighted_sum"] / movie_scores["similarity_sum"]
)

# ------------------------------------------------------------
# 9. SORT AND DISPLAY TOP 10 RECOMMENDATIONS
# ------------------------------------------------------------

recommendations = movie_scores.merge(
    movies[["movieId", "title", "genres"]],
    on="movieId",
    how="left"
)

recommendations = recommendations.sort_values(
    by=["predicted_rating", "neighbour_count"],
    ascending=[False, False]
)

print("\n" + "=" * 70)
print("PERSONALIZED MOVIE RECOMMENDATIONS")
print("Target user ID:", user_id)
print("=" * 70)

if recommendations.empty:
    print("No recommendations found for this user.")
else:
    print(
        recommendations[
            [
                "title",
                "genres",
                "predicted_rating",
                "neighbour_count"
            ]
        ].head(10).round(3).to_string(index=False)
    )

print("\nStage 4 completed.")
