from .base import BaseRecommender
from src.data_loader import CSVDataLoader
from src.data_preprocessing import RatingsDataPreprocessor
from src.config import RATINGS_FILE, ALS_REGULARIZATION, ALS_ITERATIONS, ALS_FACTORS
import faiss
import numpy as np
from implicit.als import AlternatingLeastSquares
import polars as pl
from scipy.sparse import csr_matrix

class UserUserRecommender(BaseRecommender):
    def __init__(self, ratings_path: str = RATINGS_FILE):
        """
        Initialize the recommender system.
        
        :param ratings_path: Path to the ratings data file (default is RATINGS_FILE from config)
        """
        self.ratings_path = ratings_path
        self.ratings_df = CSVDataLoader(self.ratings_path).load_data(["userId","movieId","rating"])
        self.ratings_df = self._preprocess_data(self.ratings_df)

        self.user_similarity_index = None
        self.user_vectors = None

        self.user_id_to_index = None
        self.index_to_user_id = None

        self.movie_id_to_index = None
        self.index_to_movie_id = None

        self.als_model = None
    
    def _preprocess_data(self, ratings_df):
        preprocessor = RatingsDataPreprocessor(ratings_df)
        preprocessor.clean()
        preprocessor.handle_missing_values()
        preprocessor.transform()
        return preprocessor.df
        
    def build_model(self):
        """
        Build the user-user similarity model using FAISS.
        
        This should create the FAISS index for fast user similarity lookup.
        """
        print("[Build Model] Fitting ALS for User Embeddings")
        self.user_vectors = self._prepare_user_vectors(self.ratings_df)
        print(f"[Build Model] Successfully built User Embedding of shape {self.user_vectors.shape}")


        # TODO: Step 2 - Use FAISS to build the similarity index (e.g., using cosine or Euclidean distance)
        self.user_similarity_index = self._build_faiss_index(self.user_vectors)
        print("User-user similarity model built.")

    def _prepare_user_vectors(self, ratings_df):
        # Convert to efficient types
        ratings_df = ratings_df.with_columns([
            pl.col("userId").cast(pl.UInt32),
            pl.col("movieId").cast(pl.UInt32),
            pl.col("rating").cast(pl.Float32)
        ])

        # Create user and movie ID mappings
        unique_users = ratings_df["userId"].unique().sort().to_list()
        unique_movies = ratings_df["movieId"].unique().sort().to_list()

        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.index_to_user_id = {v: k for k, v in self.user_id_to_index.items()}

        self.movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
        self.index_to_movie_id = {v: k for k, v in self.movie_id_to_index.items()}

        # Map original IDs to contiguous indices
        user_indices = ratings_df.select(
            pl.col("userId").replace(self.user_id_to_index)
        ).to_numpy().flatten()
        
        movie_indices = ratings_df.select(
            pl.col("movieId").replace(self.movie_id_to_index)
        ).to_numpy().flatten()

        # Build sparse matrix with correct dimensions
        sparse_matrix = csr_matrix(
            (ratings_df["rating"].to_numpy(), (user_indices, movie_indices)),
            shape=(len(unique_users), ratings_df["movieId"].max() + 1),
            dtype=np.float32
        )

        # Train ALS
        self.als_model = AlternatingLeastSquares(factors=ALS_FACTORS, iterations=ALS_ITERATIONS, regularization=ALS_ITERATIONS)
        self.als_model.fit(sparse_matrix)
        
        return self.als_model.user_factors

    def _build_faiss_index(self, user_vectors):
        """
        Build a FAISS index for user-user similarity based on feature vectors.
        
        :param user_vectors: NumPy array containing user feature vectors
        :return: FAISS index
        """
        # TODO: Implement this method to build a FAISS index
        print("Building FAISS index...")
        # Stub: Using FAISS's L2 distance index as an example
        dim = user_vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(user_vectors.astype(np.float32))  # Add user vectors to FAISS index
        return index

    def inference(self, user_id: int, k: int = 5):
        """
        Perform inference to recommend similar users to the given user.

        :param user_id: The user for whom to find similar users
        :param k: The number of similar users to return (default 5)
        :return: A list of recommended user IDs (similar users)
        """
        print(f"Running inference for user {user_id}...")
        
        # TODO: Step 1 - Get the feature vector for the given user
        user_vector = self._get_user_vector(user_id)
        
        # TODO: Step 2 - Use the FAISS index to find similar users
        similar_users = self._get_similar_users(user_vector, k)
        
        # TODO: Step 3 - Return similar user IDs (or actual recommendations)
        return similar_users

    def _get_user_vector(self, user_id: int):
        """
        Get the feature vector for a given user.
        
        :param user_id: The user ID
        :return: A feature vector representing the user
        """
        # TODO: Implement this method to extract the feature vector for the given user
        print(f"Getting vector for user {user_id}...")
        return np.random.rand(1, 50)  # Placeholder: Return random vector for now

    def _get_similar_users(self, user_vector, k):
        """
        Get the most similar users to the given user.

        :param user_vector: The feature vector for the user
        :param k: The number of similar users to return
        :return: List of similar user IDs
        """
        # TODO: Implement this method to query the FAISS index and return k similar users
        print(f"Finding {k} similar users...")
        distances, indices = self.user_similarity_index.search(user_vector.astype(np.float32), k)
        return indices[0]  # Return the indices of similar users (placeholder)
