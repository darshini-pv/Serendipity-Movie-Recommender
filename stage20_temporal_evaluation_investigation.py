
import os
import pandas as pd
import numpy as np

# ============================================================
# STAGE 20: TEMPORAL EVALUATION INVESTIGATION
# ============================================================

PROJECT_DIR = (
    r"C:\Users\manoj\OneDrive\Desktop"
    r"\Serendipitous Recommendation Engine"
)

DATA_DIR = os.path.join(PROJECT_DIR, "data")

RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")

print("=" * 75)
print("STAGE 20: TEMPORAL EVALUATION INVESTIGATION")
print("=" * 75)

# ------------------------------------------------------------
# STEP 1: Load ratings
# ------------------------------------------------------------

print("\nSTEP 1: Loading ratings dataset...")

ratings = pd.read_csv(RATINGS_PATH)

print("Dataset shape:", ratings.shape)
print("\nColumns:")
print(ratings.columns.tolist())

# ------------------------------------------------------------
# STEP 2: Check timestamp information
# ------------------------------------------------------------

print("\nSTEP 2: Investigating timestamps...")

if "timestamp" not in ratings.columns:

    raise ValueError(
        "Timestamp column not found. "
        "Check the structure of ratings.csv."
    )

ratings["timestamp"] = pd.to_numeric(
    ratings["timestamp"],
    errors="coerce"
)

print("\nMissing timestamps:", ratings["timestamp"].isna().sum())

print("Earliest timestamp:")
print(pd.to_datetime(
    ratings["timestamp"].min(),
    unit="s"
))

print("\nLatest timestamp:")
print(pd.to_datetime(
    ratings["timestamp"].max(),
    unit="s"
))

# ------------------------------------------------------------
# STEP 3: Check rating activity over time
# ------------------------------------------------------------

print("\nSTEP 3: Ratings per year...")

ratings["rating_date"] = pd.to_datetime(
    ratings["timestamp"],
    unit="s"
)

ratings["year"] = ratings["rating_date"].dt.year

yearly_counts = (
    ratings.groupby("year")
    .size()
    .reset_index(name="rating_count")
)

print(yearly_counts)

# ------------------------------------------------------------
# STEP 4: Investigate user rating histories
# ------------------------------------------------------------

print("\nSTEP 4: User rating history investigation...")

user_history = (
    ratings.groupby("userId")
    .agg(
        total_ratings=("rating", "count"),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max")
    )
    .reset_index()
)

user_history["history_duration_days"] = (
    user_history["last_timestamp"]
    - user_history["first_timestamp"]
) / 86400

print("\nNumber of users:", len(user_history))

print("\nUser history duration statistics:")
print(
    user_history["history_duration_days"]
    .describe()
)

# ------------------------------------------------------------
# STEP 5: Investigate a chronological global split
# ------------------------------------------------------------

print("\nSTEP 5: Investigating chronological split...")

ratings_sorted = ratings.sort_values(
    "timestamp"
).reset_index(drop=True)

split_position = int(len(ratings_sorted) * 0.80)

train_df = ratings_sorted.iloc[:split_position].copy()

test_df = ratings_sorted.iloc[split_position:].copy()

print("\nTraining records:", len(train_df))
print("Testing records:", len(test_df))

print("\nTraining time range:")

print(
    pd.to_datetime(
        train_df["timestamp"].min(),
        unit="s"
    )
)

print(
    pd.to_datetime(
        train_df["timestamp"].max(),
        unit="s"
    )
)

print("\nTesting time range:")

print(
    pd.to_datetime(
        test_df["timestamp"].min(),
        unit="s"
    )
)

print(
    pd.to_datetime(
        test_df["timestamp"].max(),
        unit="s"
    )
)

# ------------------------------------------------------------
# STEP 6: Check users and movies in the test set
# ------------------------------------------------------------

print("\nSTEP 6: User and movie availability...")

train_users = set(train_df["userId"].unique())

test_users = set(test_df["userId"].unique())

train_movies = set(train_df["movieId"].unique())

test_movies = set(test_df["movieId"].unique())

known_test_users = train_users.intersection(test_users)

known_test_movies = train_movies.intersection(test_movies)

print("Unique test users:", len(test_users))

print("Test users also present in training:",
      len(known_test_users))

print("Unique test movies:", len(test_movies))

print("Test movies also present in training:",
      len(known_test_movies))

print("\nUnseen test users:",
      len(test_users - train_users))

print("Unseen test movies:",
      len(test_movies - train_movies))

# ------------------------------------------------------------
# STEP 7: Save investigation results
# ------------------------------------------------------------

print("\nSTEP 7: Saving investigation outputs...")

yearly_counts.to_csv(
    os.path.join(
        PROJECT_DIR,
        "stage20_ratings_per_year.csv"
    ),
    index=False
)

user_history.to_csv(
    os.path.join(
        PROJECT_DIR,
        "stage20_user_history_analysis.csv"
    ),
    index=False
)

split_summary = pd.DataFrame([{
    "training_records": len(train_df),
    "testing_records": len(test_df),
    "training_users": len(train_users),
    "testing_users": len(test_users),
    "known_test_users": len(known_test_users),
    "unseen_test_users": len(test_users - train_users),
    "training_movies": len(train_movies),
    "testing_movies": len(test_movies),
    "known_test_movies": len(known_test_movies),
    "unseen_test_movies": len(test_movies - train_movies),
    "training_latest_timestamp": train_df["timestamp"].max(),
    "testing_earliest_timestamp": test_df["timestamp"].min()
}])

split_summary.to_csv(
    os.path.join(
        PROJECT_DIR,
        "stage20_temporal_split_summary.csv"
    ),
    index=False
)

print("\nFiles saved:")
print("stage20_ratings_per_year.csv")
print("stage20_user_history_analysis.csv")
print("stage20_temporal_split_summary.csv")

print("\n" + "=" * 75)
print("STAGE 20 INVESTIGATION COMPLETED")
print("=" * 75)
