
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# STAGE 21: CONTENT-BASED MOVIE RECOMMENDATION
# ============================================================

PROJECT_DIR = (
    r"C:\Users\manoj\OneDrive\Desktop"
    r"\Serendipitous Recommendation Engine"
)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
MOVIES_PATH = os.path.join(DATA_DIR, "movies.csv")

print("=" * 70)
print("STAGE 21: CONTENT-BASED MOVIE RECOMMENDATION")
print("=" * 70)

# STEP 1: Load datasets
print("\nSTEP 1: Loading datasets...")

ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)

# STEP 2: Detect column names
print("\nSTEP 2: Detecting column names...")

# Normalize column names to lowercase for consistent access.
movies.columns = movies.columns.str.strip().str.lower()
ratings.columns = ratings.columns.str.strip().str.lower()

required_movie_columns = {"movieid", "title", "genres"}
required_rating_columns = {"userid", "movieid", "rating"}

if not required_movie_columns.issubset(movies.columns):
    raise ValueError(
        "Expected movie columns are missing. Found: "
        + str(movies.columns.tolist())
    )

if not required_rating_columns.issubset(ratings.columns):
    raise ValueError(
        "Expected rating columns are missing. Found: "
        + str(ratings.columns.tolist())
    )

print("Movie columns:", movies.columns.tolist())
print("Rating columns:", ratings.columns.tolist())

# STEP 3: Create genre features
print("\nSTEP 3: Creating genre feature matrix...")

movies["genres"] = movies["genres"].fillna("").astype(str)

genre_matrix = movies["genres"].str.get_dummies(sep="|")

print("Genre feature matrix shape:", genre_matrix.shape)
print("Genres:", genre_matrix.columns.tolist())

# STEP 4: Calculate movie-to-movie similarity
print("\nSTEP 4: Calculating movie similarity...")

movie_similarity = cosine_similarity(genre_matrix)

movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=movies["movieid"],
    columns=movies["movieid"]
)

print("Similarity matrix shape:", movie_similarity_df.shape)

# STEP 5: Build user genre profiles
print("\nSTEP 5: Creating user genre profiles...")

liked_ratings = ratings[ratings["rating"] >= 4].copy()

movie_genres = genre_matrix.copy()
movie_genres.index = movies["movieid"]

user_profiles = {}

for user_id, user_data in liked_ratings.groupby("userid"):
    liked_movie_ids = user_data["movieid"].values

    available_ids = [
        movie_id for movie_id in liked_movie_ids
        if movie_id in movie_genres.index
    ]

    if not available_ids:
        continue

    user_profiles[user_id] = movie_genres.loc[
        available_ids
    ].mean(axis=0)

print("Users with genre profiles:", len(user_profiles))

# STEP 6: Generate recommendations
print("\nSTEP 6: Generating recommendations...")

sample_user_id = 1

if sample_user_id not in user_profiles:
    raise ValueError(
        f"User {sample_user_id} has no liked movies for a profile."
    )

user_profile = user_profiles[sample_user_id]

content_scores = cosine_similarity(
    user_profile.to_numpy().reshape(1, -1),
    genre_matrix.to_numpy()
)[0]

# Work with a fresh copy of the movie catalog.
recommendation_catalog = movies.copy()
recommendation_catalog["content_score"] = content_scores

rated_movie_ids = set(
    ratings.loc[
        ratings["userid"] == sample_user_id,
        "movieid"
    ]
)

recommendations = recommendation_catalog.loc[
    ~recommendation_catalog["movieid"].isin(rated_movie_ids)
].copy()

recommendations = recommendations.sort_values(
    by="content_score",
    ascending=False
)

top_10 = recommendations.head(10).copy()

print(f"\nTop 10 recommendations for User {sample_user_id}:")

# IMPORTANT: use lowercase column names.
print(
    top_10[
        ["movieid", "title", "genres", "content_score"]
    ].to_string(index=False)
)

# STEP 7: Save outputs
print("\nSTEP 7: Saving outputs...")

top_10.to_csv(
    os.path.join(
        PROJECT_DIR,
        "stage21_user1_content_recommendations.csv"
    ),
    index=False
)

genre_matrix_with_id = genre_matrix.copy()
genre_matrix_with_id.insert(
    0, "movieid", movies["movieid"].to_numpy()
)

genre_matrix_with_id.to_csv(
    os.path.join(
        PROJECT_DIR,
        "stage21_movie_genre_features.csv"
    ),
    index=False
)

movie_similarity_df.to_csv(
    os.path.join(
        PROJECT_DIR,
        "stage21_movie_similarity_matrix.csv"
    )
)

print("\n" + "=" * 70)
print("STAGE 21 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("Movies:", len(movies))
print("Genres:", genre_matrix.shape[1])
print("User profiles:", len(user_profiles))
print("Recommendations:", len(top_10))
print("\nSaved files:")
print("- stage21_user1_content_recommendations.csv")
print("- stage21_movie_genre_features.csv")
print("- stage21_movie_similarity_matrix.csv")
