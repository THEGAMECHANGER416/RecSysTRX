import os

# Always resolve path relative to the root of the project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # src/config.py location
PROJECT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))  # move up to project root

RATINGS_FILE = os.path.join(PROJECT_ROOT, "data", "ml-32m", "ratings.csv")
MOVIES_FILE = os.path.join(PROJECT_ROOT, "data", "tmdb_dataset", "TMDB_movie_dataset_v11.csv")

# For FAISS or vector settings
VECTOR_DIM = 100
FAISS_INDEX_TYPE = "IndexFlatIP"  # or "IndexFlatL2"

# Output
TOP_K = 10  # Number of recommendations to retrieve
