
# ============================================================
# STAGE 3: POPULARITY-BASED MOVIE RECOMMENDER
# Project: Serendipity - A Novelty-Aware Personalized
#          Movie Recommendation System
# ============================================================

import pandas as pd
from tkinter import Tk, filedialog

# ------------------------------------------------------------
# 1. SELECT THE DATASETS
# ------------------------------------------------------------

root = Tk()
root.withdraw()
root.attributes("-topmost", True)

print("Select your ratings.csv file")
ratings_path = filedialog.askopenfilename(
    title="Select ratings.csv",
    filetypes=[("CSV files", "*.csv")]
)

print("Select your movies.csv file")
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
# 2. CHECK THE DATA
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STAGE 3: POPULARITY-BASED RECOMMENDER")
print("=" * 60)

print("\nRatings dataset shape:", ratings.shape)
print("Movies dataset shape:", movies.shape)

# ------------------------------------------------------------
# 3. SPLIT RATINGS INTO TRAINING AND TESTING DATA
# ------------------------------------------------------------
# Keep 20% of each user's ratings for testing.
# Use training data to calculate movie popularity.

train_parts = []
test_parts = []

for user_id, group in ratings.groupby("userId"):
    group = group.sample(frac=1, random_state=42)

    # Keep at least one rating in training.
    if len(group) >= 2:
        test_count = max(1, int(len(group) * 0.2))
        test_count = min(test_count, len(group) - 1)

        test_parts.append(group.iloc[:test_count])
        train_parts.append(group.iloc[test_count:])
    else:
        # Users with only one rating stay in training.
        train_parts.append(group)

train_ratings = pd.concat(train_parts).copy()
test_ratings = (
    pd.concat(test_parts).copy()
    if test_parts
    else pd.DataFrame(columns=ratings.columns)
)

print("\nTraining ratings:", len(train_ratings))
print("Testing ratings:", len(test_ratings))
print("Training users:", train_ratings["userId"].nunique())
print("Testing users:", test_ratings["userId"].nunique())

# ------------------------------------------------------------
# 4. CALCULATE MOVIE STATISTICS USING TRAINING DATA ONLY
# ------------------------------------------------------------

movie_stats = train_ratings.groupby("movieId").agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "count")
).reset_index()

# Overall average rating in the training dataset
C = train_ratings["rating"].mean()

# Minimum rating-count threshold
m = 100

print("\nOverall training rating average:", round(C, 3))
print("Minimum ratings required for ranking:", m)

# ------------------------------------------------------------
# 5. CALCULATE WEIGHTED RATINGS
# ------------------------------------------------------------
# Weighted rating balances a movie's average rating
# with the overall average rating.
#
# WR = (v / (v + m)) * R + (m / (v + m)) * C
#
# R = movie's average rating
# v = number of ratings for that movie
# C = overall average rating
# m = minimum rating-count threshold

movie_stats["weighted_score"] = (
    (movie_stats["rating_count"] /
     (movie_stats["rating_count"] + m))
    * movie_stats["average_rating"]
    +
    (m / (movie_stats["rating_count"] + m)) * C
)

# ------------------------------------------------------------
# 6. FILTER MOVIES WITH ENOUGH RATINGS
# ------------------------------------------------------------

popular_movies = movie_stats[
    movie_stats["rating_count"] >= m
].copy()

# Sort movies from highest to lowest weighted score
popular_movies = popular_movies.sort_values(
    by=["weighted_score", "rating_count"],
    ascending=[False, False]
)

# ------------------------------------------------------------
# 7. ADD MOVIE TITLES AND GENRES
# ------------------------------------------------------------

recommendations = popular_movies.merge(
    movies[["movieId", "title", "genres"]],
    on="movieId",
    how="left"
)

recommendations = recommendations[
    [
        "movieId",
        "title",
        "genres",
        "average_rating",
        "rating_count",
        "weighted_score"
    ]
]

# ------------------------------------------------------------
# 8. DISPLAY THE TOP 10 MOVIES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TOP 10 POPULARITY-BASED MOVIE RECOMMENDATIONS")
print("=" * 60)

if recommendations.empty:
    print("No movies met the minimum rating-count threshold.")
else:
    print(
        recommendations.head(10).round(3).to_string(index=False)
    )

# ------------------------------------------------------------
# 9. SAVE THE RECOMMENDATIONS
# ------------------------------------------------------------

output_path = filedialog.asksaveasfilename(
    title="Save popularity recommendations",
    defaultextension=".csv",
    initialfile="popularity_recommendations.csv",
    filetypes=[("CSV files", "*.csv")]
)

if output_path:
    recommendations.to_csv(output_path, index=False)
    print("\nRecommendations saved to:")
    print(output_path)
else:
    print("\nSaving skipped.")

print("\nStage 3 completed successfully.")


