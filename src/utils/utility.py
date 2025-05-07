import polars as pl
import numpy as np
import os
from src.config import *
from src.data_loader import CSVDataLoader

class Utils:
    @staticmethod
    def build_user_ratings_dict(ratings_df):
        """
        Build a dictionary mapping user IDs to their ratings.

        Args:
            ratings_df (pl.DataFrame): DataFrame containing user ratings.
        
        Returns:
            dict: Dictionary mapping user IDs to their ratings.
        """
        user_ratings_dict = {}
        grouped = ratings_df.group_by("userId").agg([
            pl.col("movieId"),
            pl.col("rating")
        ])
        for row in grouped.iter_rows():
            user_id = row[0]
            movie_ids = row[1]
            ratings = row[2]
            user_ratings_dict[user_id] = list(zip(movie_ids, ratings))
        return user_ratings_dict
    
    @staticmethod
    def build_movie_details_dict():
        """
        Build a dictionary mapping movie IDs to their details.

        Returns:
            dict: Dictionary mapping movie IDs to their details.
        """
        links_df = CSVDataLoader(LINKS_FILE).load_lazy_data()
        tmdb_movies_df = CSVDataLoader(MOVIES_FILE).load_lazy_data()
        
        # Join on TMDB ID
        merged_df = links_df.join(tmdb_movies_df, left_on="tmdbId", right_on="id", how="inner")
        
        # Create a dictionary mapping movie IDs to their details (All columns except 'id')
        movie_details_dict = {}
        for row in merged_df.iter_rows():
            movie_id = row[0]
            details = {
                str(col): row[i] for i, col in enumerate(merged_df.columns) if col != "id"
            }
            movie_details_dict[movie_id] = details
        return movie_details_dict
    
    @staticmethod
    def get_movie_details(movie_id, movie_details_dict):
        """
        Get movie details for a given movie ID.

        Args:
            movie_id (int): Movie ID.
            movie_details_dict (dict): Dictionary mapping movie IDs to their details.
        
        Returns:
            dict: Movie details for the given movie ID.
        """
        return movie_details_dict.get(movie_id, None)