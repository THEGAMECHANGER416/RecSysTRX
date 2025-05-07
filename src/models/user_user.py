from .base import BaseRecommender
from src.data_loader import CSVDataLoader
from src.data_preprocessing import RatingsDataPreprocessor
from src.config import *
from src.utils import Utils
import numpy as np
from implicit.als import AlternatingLeastSquares
import polars as pl
from scipy.sparse import csr_matrix
import hnswlib
import pickle

class UserUserRecommender(BaseRecommender):
    def __init__(self):
        """
        Initialize the recommender system.
        
        :param ratings_path: Path to the ratings data file (default is RATINGS_FILE from config)
        """
        self.ratings_df = None
        self.user_ratings_dict = None
        self.movie_details_dict = None

        self.user_similarity_index = None
        self.user_vectors = None

        self.user_id_to_index = None
        self.index_to_user_id = None

        self.movie_id_to_index = None
        self.index_to_movie_id = None

        self.als_model = None
    
    def __preprocess_data(self):
        print(f"[Preprocess Data] Preprocessing data...")
        preprocessor = RatingsDataPreprocessor(self.ratings_df)
        print(f"[Preprocess Data] Successfully loaded data with shape {self.ratings_df.shape}")
        preprocessor.clean()
        print(f"[Preprocess Data] Successfully cleaned data with shape {self.ratings_df.shape}")
        preprocessor.handle_missing_values()
        print(f"[Preprocess Data] Successfully handled missing values with shape {self.ratings_df.shape}")
        preprocessor.transform()
        print(f"[Preprocess Data] Successfully transformed data with shape {self.ratings_df.shape}")
        self.ratings_df = preprocessor.df
        print(f"[Preprocess Data] Successfully preprocessed data with shape {self.ratings_df.shape}")
        
    def __prepare_user_vectors(self, ratings_df):
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
            shape=(len(unique_users), len(unique_movies)),
            dtype=np.float32
        )

        # Train ALS
        self.als_model = AlternatingLeastSquares(factors=ALS_FACTORS, iterations=ALS_ITERATIONS, regularization=ALS_ITERATIONS)
        self.als_model.fit(sparse_matrix)
        
        return self.als_model.user_factors

    def __build_hnsw_index(self, user_vectors):
        """
        Builds an HNSW index for approximate nearest neighbor search of user vectors.
        
        :param user_vectors: NumPy array of shape (num_users, embedding_dim)
        """
        # Normalize vectors for cosine similarity
        user_vectors = np.ascontiguousarray(user_vectors)
        norms = np.linalg.norm(user_vectors, axis=1, keepdims=True)
        user_vectors_normalized = user_vectors / np.where(norms == 0, 1e-10, norms)

        # Initialize index
        dim = user_vectors_normalized.shape[1]
        self.user_similarity_index = hnswlib.Index(space=HNSW_INDEX_SPACE, dim=dim)
        
        # For 1M+ users: M=32, ef_construction=200 provides good quality/speed balance
        self.user_similarity_index.init_index(
            max_elements=len(user_vectors_normalized),
            ef_construction=HNSW_INDEX_EF_BUILD,
            M=HNSW_INDEX_M
        )
        
        # Add items in batches for large datasets
        batch_size = HNSW_BATCH_SIZE
        for i in range(0, len(user_vectors_normalized), batch_size):
            end_idx = min(i + batch_size, len(user_vectors_normalized))
            batch = user_vectors_normalized[i:end_idx]
            self.user_similarity_index.add_items(batch, num_threads=HNSW_INDEX_BUILD_THREADS)
        
        # Set query-time parameters
        self.user_similarity_index.set_ef(HNSW_INDEX_EF_QUERY)  

    def __save_model(self):
        """
        Save the model to disk. This includes the ratings_df, HNSW index, user and movie ID mappings, and the ALS model.
        """
        # Delete the directory if it already exists
        if os.path.exists(MODEL_OUTPUT_PATH):
            import shutil
            shutil.rmtree(MODEL_OUTPUT_PATH)

        # Ensure the output directory exists
        os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

        # Save self.user_ratings_dict to disk
        np.savez_compressed(
            USER_RATINGS_DICT_PATH,
            user_ratings_dict=self.user_ratings_dict,
            movie_details_dict=self.movie_details_dict
        )

        # Save the HNSW index to disk
        self.user_similarity_index.save_index(HNSW_INDEX_PATH)

        # Save user and movie ID mappings
        np.savez_compressed(
            USER_MOVIE_MAPPINGS_PATH,
            user_id_to_index=self.user_id_to_index,
            index_to_user_id=self.index_to_user_id,
            movie_id_to_index=self.movie_id_to_index,
            index_to_movie_id=self.index_to_movie_id
        )

        # Save embedding vectors
        np.savez_compressed(
            USER_VECTORS_PATH,
            user_vectors=self.user_vectors
        )

        # Save the ALS model
        with open(ALS_MODEL_PATH, "wb") as f:
            pickle.dump(self.als_model, f)

    def __recommend_movies(self, watched_movies, labels, distances):
        """
        Recommend movies based on the watched movies and similar users.
        
        :param watched_movies: List of movie IDs that the user has watched.
        :param labels: Indices of similar users.
        :param distances: Distances to the similar users.
        :return: List of recommended movie IDs.
        """
        watched_movies_set = set(watched_movies)

        # Sort similar users by distance
        sorted_indices = np.argsort(distances[0])
        sorted_labels = labels[0][sorted_indices]
        sorted_distances = distances[0][sorted_indices]

        recommended_movies = {}
        for idx, user_index in enumerate(sorted_labels):
            user_id = self.index_to_user_id[user_index]
            user_ratings = self.user_ratings_dict[user_id]
            # Get the movie IDs and ratings for the user
            for movie_id, rating in user_ratings:
                # Check if the movie is already watched
                if movie_id not in watched_movies_set:
                    if movie_id not in recommended_movies:
                        recommended_movies[movie_id] = 0
                    # Weight by proximity and the rating
                    proximity_weight = (1 / (1 + sorted_distances[idx]))
                    recommended_movies[movie_id] += rating * np.power(proximity_weight, PROXIMITY_INFLUENCE_FACTOR)

        # Sort recommendations by score
        recommended_movies = sorted(recommended_movies.items(), key=lambda x: x[1], reverse=True)

        # Make tuple of movie ID and score
        recommended_movies = [(movie_id, score) for movie_id, score in recommended_movies[:TOP_K]]
        return recommended_movies

    def load_model(self):
        """
        Load the model from disk. This includes the ratings_df, HNSW index, user and movie ID mappings, and the ALS model.
        """
        # Check if the model output path exists
        if not os.path.exists(MODEL_OUTPUT_PATH):
            raise FileNotFoundError(f"Model output path {MODEL_OUTPUT_PATH} does not exist. Please build the model first.")
        
        # Load the user ratings dictionary from disk
        user_ratings_dict = np.load(USER_RATINGS_DICT_PATH, allow_pickle=True)
        self.user_ratings_dict = user_ratings_dict["user_ratings_dict"].item()
        self.movie_details_dict = user_ratings_dict["movie_details_dict"].item()

        # Load the HNSW index from disk
        self.user_similarity_index = hnswlib.Index(space=HNSW_INDEX_SPACE, dim=VECTOR_DIM)
        self.user_similarity_index.load_index(HNSW_INDEX_PATH)

        # Load user and movie ID mappings
        mappings = np.load(USER_MOVIE_MAPPINGS_PATH, allow_pickle=True)
        self.user_id_to_index = mappings["user_id_to_index"].item()
        self.index_to_user_id = mappings["index_to_user_id"].item()
        self.movie_id_to_index = mappings["movie_id_to_index"].item()
        self.index_to_movie_id = mappings["index_to_movie_id"].item()

        # Load embedding vectors
        user_vectors = np.load(USER_VECTORS_PATH, allow_pickle=True)["user_vectors"]
        self.user_vectors = user_vectors

        # Load the ALS model
        with open(ALS_MODEL_PATH, "rb") as f:
            self.als_model = pickle.load(f)

        print(f"[Load Model] Successfully loaded model from {MODEL_OUTPUT_PATH}")

    def build_model(self):
        """
        Build the user-user similarity model using FAISS.
        
        This should create the FAISS index for fast user similarity lookup.
        """
        # Load and preprocess data
        print(f"[Build Model] Loading data from {RATINGS_FILE}")
        data_loader = CSVDataLoader(RATINGS_FILE)
        self.ratings_df = data_loader.load_data()
        print(f"[Build Model] Successfully loaded data with shape {self.ratings_df.shape}")

        # Preprocess data
        print(f"[Build Model] Preprocessing data...")
        self.__preprocess_data()
        print(f"[Build Model] Successfully preprocessed data with shape {self.ratings_df.shape}")

        # Store a dictionary to fast fetch movies and ratings by userID
        print(f"[Build Model] Building user ratings dictionary")
        self.user_ratings_dict = Utils.build_user_ratings_dict(self.ratings_df)
        print(f"[Build Model] Successfully built user ratings dictionary with {len(self.user_ratings_dict)} users")

        # Store a dictionary to fast fetch movie details by movieID
        print(f"[Build Model] Building movie details dictionary")
        self.movie_details_dict = Utils.build_movie_details_dict()
        print(f"[Build Model] Successfully built movie details dictionary with {len(self.movie_details_dict)} movies")

        # Prepare User Movie Sparse Matrix
        print("[Build Model] Fitting ALS for User Embeddings")
        self.user_vectors = self.__prepare_user_vectors(self.ratings_df)
        print(f"[Build Model] Successfully built User Embedding of shape {self.user_vectors.shape}")

        # Build HNSW Index
        print(f"[Build Model] Building HNSW Index")
        self.__build_hnsw_index(self.user_vectors)
        print(f"[Build Model] HNSW Index built")

        # TODO: Save Model In Disk
        print(f"[Build Model] Saving model in disk")
        self.__save_model()
        print(f"[Build Model] Successfully saved model at {MODEL_OUTPUT_PATH}")

    def __inference_user_id(self, user_id: int):
        """
        Perform inference to get recommendations for a given user ID.
        
        :param user_id: The user ID for which to get recommendations.
        :return: List of recommended movie IDs.
        """
        if self.user_similarity_index is None:
            raise ValueError("Model not built. Please call build_model() first.")

        if user_id not in self.user_id_to_index:
            raise ValueError(f"User ID {user_id} not found in the model.")

        # Get the index of the user
        user_index = self.user_id_to_index[user_id]

        # Get the top K similar users
        labels, distances = self.user_similarity_index.knn_query(self.user_vectors[user_index], k=TOP_K)
        print(f"[Inference] Similar user labels: {labels}")
        
        # Get the watched movies of the user
        watched_movies = self.user_ratings_dict[user_id]
        watched_movies = [movie_id for movie_id, _ in watched_movies]
        print(f"[Inference] Watched movies: {watched_movies}")

        # Recommend movies based on the watched movies and similar users
        recommended_movies = self.__recommend_movies(watched_movies, labels, distances)
        print(f"[Inference] Recommended movies: {recommended_movies}")
        return recommended_movies

    def __inference_watched_movies(self, watched_movies: list):
        """
        Perform inference to get recommendations for a given list of watched movies.

        :param watched_movies: List of movie IDs that the user has watched.
        :return: List of recommended movie IDs with scores.
        """
        if self.user_similarity_index is None or self.als_model is None:
            raise ValueError("Model not built or loaded. Please call build_model() or load_model() first.")

        # Convert watched movies to indices and filter out movies not in the model
        watched_movie_indices = [
            self.movie_id_to_index[movie_id] for movie_id in watched_movies
            if movie_id in self.movie_id_to_index
        ]

        if not watched_movie_indices:
            print("Warning: None of the watched movies are in the model. Cannot generate recommendations.")
            return [] # Return empty list if no watched movies are in the model

        # --- Corrected Approach: Average Item Factors ---
        # Get the item factors for the watched movies
        item_factors = self.als_model.item_factors[watched_movie_indices]

        # Compute the average item factor vector to represent the user
        user_vector_embedding = np.mean(item_factors, axis=0).reshape(1, -1)

        # Normalize the user vector embedding
        user_vector_embedding = np.ascontiguousarray(user_vector_embedding)
        norms = np.linalg.norm(user_vector_embedding, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors
        user_vector_embedding /= (norms + 1e-10)

        print(f"[Inference Watched Movies] User vector embedding shape: {user_vector_embedding.shape}")

        # Get the top K similar users using the HNSW index
        # HNSW knn_query returns labels (indices) and distances
        labels, distances = self.user_similarity_index.knn_query(user_vector_embedding, k=TOP_K)
        print(f"[Inference Watched Movies] Similar user labels: {labels}")

        # Recommend movies based on the watched movies and similar users
        recommended_movies = self.__recommend_movies(watched_movies, labels, distances)
        print(f"[Inference Watched Movies] Recommended movies: {recommended_movies}")
        return recommended_movies

    def inference(self, type: str, user_id: int = None, watched_movies: list = None):
        """
        Perform inference to get recommendations for a given user ID or watched movies.
        
        :param type: Type of inference ('user_id' or 'watched_movies').
        :param user_id: The user ID for which to get recommendations (if type is 'user_id').
        :param watched_movies: List of movie IDs that the user has watched (if type is 'watched_movies').
        :return: List of recommended movie IDs.
        """
        if type == "user_id":
            return self.__inference_user_id(user_id)
        elif type == "watched_movies":
            return self.__inference_watched_movies(watched_movies)
        else:
            raise ValueError("Invalid inference type. Use 'user_id' or 'watched_movies'.")