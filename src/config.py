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
TOP_K = 10  # [10] Number of recommendations to retrieve

# ALS Settings
ALS_FACTORS = 64 # [64]
ALS_ITERATIONS = 2 # [15]
ALS_REGULARIZATION = 0.1 # [0.1]

# HNSW Index Settings
HNSW_INDEX_SPACE = 'cosine'
HNSW_INDEX_EF_BUILD = 200  # [200] Controls depth of search during build (higher = better quality)
HNSW_INDEX_M = 32  # [32] Number of bi-directional links per node (higher = better recall, more memory)
HNSW_BATCH_SIZE = 10000 # [10000]
HNSW_INDEX_BUILD_THREADS = 4 # [4]
HNSW_INDEX_EF_QUERY = 150  # [150] Controls depth of search during queries (higher = better recall)

# Output paths
MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "user_user")
HNSW_INDEX_PATH = os.path.join(MODEL_OUTPUT_PATH, "hnsw_index.bin")
USER_MOVIE_MAPPINGS_PATH = os.path.join(MODEL_OUTPUT_PATH, "user_movie_mappings.npz")
ALS_MODEL_PATH = os.path.join(MODEL_OUTPUT_PATH, "als_model.pkl")
USER_VECTORS_PATH = os.path.join(MODEL_OUTPUT_PATH, "user_vectors.npz")
RATINGS_DATASET_PATH = os.path.join(MODEL_OUTPUT_PATH, "ratings_dataset.csv")
USER_RATINGS_DICT_PATH = os.path.join(MODEL_OUTPUT_PATH, "user_ratings_dict.npz")