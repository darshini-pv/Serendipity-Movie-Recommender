
# ==========================================================
# PROJECT: SERENDIPITY-AWARE MOVIE RECOMMENDATION SYSTEM
# STAGE 1: LOAD AND INSPECT THE DATASET
# ==========================================================

import pandas as pd
from tkinter import Tk, filedialog

# ----------------------------------------------------------
# 1. SELECT THE THREE CSV FILES
# ----------------------------------------------------------

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

print("Select your users.csv file")

users_path = filedialog.askopenfilename(
    title="Select users.csv",
    filetypes=[("CSV files", "*.csv")]
)

root.destroy()

# ----------------------------------------------------------
# 2. CHECK WHETHER FILES WERE SELECTED
# ----------------------------------------------------------

if not ratings_path or not movies_path or not users_path:
    raise ValueError(
        "A file was not selected. Run the script again "
        "and select all three CSV files."
    )

# ----------------------------------------------------------
# 3. LOAD THE DATASETS
# ----------------------------------------------------------

ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)
users = pd.read_csv(users_path)

print("\nAll three datasets loaded successfully!")

# ----------------------------------------------------------
# 4. DISPLAY DATASET DIMENSIONS
# ----------------------------------------------------------

print("\nRATINGS DATASET")
print("Rows and columns:", ratings.shape)

print("\nMOVIES DATASET")
print("Rows and columns:", movies.shape)

print("\nUSERS DATASET")
print("Rows and columns:", users.shape)

# ----------------------------------------------------------
# 5. DISPLAY COLUMN NAMES
# ----------------------------------------------------------

print("\nRatings columns:")
print(ratings.columns.tolist())

print("\nMovies columns:")
print(movies.columns.tolist())

print("\nUsers columns:")
print(users.columns.tolist())

# ----------------------------------------------------------
# 6. DISPLAY FIRST FIVE ROWS
# ----------------------------------------------------------

print("\nFirst five ratings:")
print(ratings.head())

print("\nFirst five movies:")
print(movies.head())

print("\nFirst five users:")
print(users.head())

print("\nSTAGE 1 COMPLETED SUCCESSFULLY!")
