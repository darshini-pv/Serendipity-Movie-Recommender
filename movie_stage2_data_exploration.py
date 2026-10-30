
# ==========================================================
# PROJECT: SERENDIPITY-AWARE MOVIE RECOMMENDATION SYSTEM
# STAGE 2: DATA EXPLORATION AND QUALITY CHECKS
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# ----------------------------------------------------------
# 1. LOAD THE DATASETS
# ----------------------------------------------------------

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

print("Select users.csv")
users_path = filedialog.askopenfilename(
    title="Select users.csv",
    filetypes=[("CSV files", "*.csv")]
)

root.destroy()

if not ratings_path or not movies_path or not users_path:
    raise ValueError("Please select all three CSV files.")

ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)
users = pd.read_csv(users_path)

# ----------------------------------------------------------
# 2. BASIC DATASET INFORMATION
# ----------------------------------------------------------

print("\n" + "=" * 55)
print("DATASET INFORMATION")
print("=" * 55)

print("\nRatings shape:", ratings.shape)
print("Movies shape:", movies.shape)
print("Users shape:", users.shape)

print("\nRatings data types:")
print(ratings.dtypes)

# ----------------------------------------------------------
# 3. CHECK MISSING VALUES
# ----------------------------------------------------------

print("\n" + "=" * 55)
print("MISSING VALUE CHECK")
print("=" * 55)

print("\nRatings missing values:")
print(ratings.isnull().sum())

print("\nMovies missing values:")
print(movies.isnull().sum())

print("\nUsers missing values:")
print(users.isnull().sum())

# ----------------------------------------------------------
# 4. CHECK DUPLICATE ROWS
# ----------------------------------------------------------

print("\n" + "=" * 55)
print("DUPLICATE ROW CHECK")
print("=" * 55)

print("Duplicate rating rows:", ratings.duplicated().sum())
print("Duplicate movie rows:", movies.duplicated().sum())
print("Duplicate user rows:", users.duplicated().sum())

# ----------------------------------------------------------
# 5. COUNT USERS AND MOVIES
# ----------------------------------------------------------

print("\n" + "=" * 55)
print("UNIQUE USER AND MOVIE COUNTS")
print("=" * 55)

print("Unique users in ratings:", ratings["userId"].nunique())
print("Unique movies in ratings:", ratings["movieId"].nunique())
print("Unique movies in movie metadata:", movies["movieId"].nunique())

# Check whether ratings reference missing movie metadata
missing_movie_ids = set(ratings["movieId"]) - set(movies["movieId"])

print("Rated movie IDs missing from metadata:", len(missing_movie_ids))

# ----------------------------------------------------------
# 6. RATING DISTRIBUTION
# ----------------------------------------------------------

print("\n" + "=" * 55)
print("RATING DISTRIBUTION")
print("=" * 55)

print(ratings["rating"].value_counts().sort_index())

ratings["rating"].value_counts().sort_index().plot(
    kind="bar",
    xlabel="Rating",
    ylabel="Number of ratings",
    title="MovieLens Rating Distribution"
)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 7. MOST-RATED MOVIES
# ----------------------------------------------------------

print("\n" + "=" * 55)
print("TOP 10 MOST-RATED MOVIES")
print("=" * 55)

movie_counts = ratings.groupby("movieId").size()

top_movies = (
    movie_counts.rename("rating_count")
    .reset_index()
    .merge(movies, on="movieId", how="left")
    .sort_values("rating_count", ascending=False)
)

print(top_movies[["title", "rating_count"]].head(10).to_string(index=False))

# ----------------------------------------------------------
# 8. GENRE ANALYSIS
# ----------------------------------------------------------

print("\n" + "=" * 55)
print("MOST COMMON GENRES")
print("=" * 55)

genre_counts = (
    movies["genres"]
    .dropna()
    .str.split("|")
    .explode()
    .value_counts()
)

print(genre_counts.head(10))

genre_counts.head(10).sort_values().plot(
    kind="barh",
    xlabel="Number of movies",
    title="Top 10 Movie Genres"
)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 9. SUMMARY
# ----------------------------------------------------------

print("\n" + "=" * 55)
print("STAGE 2 COMPLETED")
print("=" * 55)
print("Data exploration and quality checks completed.")
