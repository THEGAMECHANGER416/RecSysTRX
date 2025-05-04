from .base import DataPreprocessor
import polars as pl

class RatingsDataPreprocessor(DataPreprocessor):
    def clean(self):
        """
        Perform cleaning operations like removing duplicates and unnecessary columns.
        """
        initial_count = self.df.shape[0]
        self.df = self.df.unique(subset=["userId", "movieId"])
        final_count = self.df.shape[0]
        print(f"[Clean] Removed {initial_count - final_count} duplicate rows.")

    def transform(self):
        """
        Perform transformations like data normalization.
        """
        min_rating = self.df["rating"].min()
        max_rating = self.df["rating"].max()
        if max_rating == min_rating:
            print("[Transform] Skipping normalization (all ratings are the same).")
            return
        self.df = self.df.with_columns(
            ((pl.col("rating") - min_rating) / (max_rating - min_rating)).alias("rating")
        )
        print(f"[Transform] Normalized ratings to range [0, 1].")

    def handle_missing_values(self):
        """
        Handle missing and NaN values by replacing them with the mean rating.
        """
        null_mask = self.df["rating"].is_null()
        nan_mask = self.df["rating"].is_nan()

        null_count = null_mask.sum()
        nan_count = nan_mask.sum()
        total_missing = null_count + nan_count

        if total_missing == 0:
            print("[Missing] No missing or NaN ratings found.")
            return

        mean_rating = self.df["rating"].mean()

        self.df = self.df.with_columns(
            pl.when(pl.col("rating").is_nan() | pl.col("rating").is_null())
            .then(mean_rating)
            .otherwise(pl.col("rating"))
            .alias("rating")
        )

        print(f"[Missing] Filled {total_missing} missing/NaN ratings with mean value: {mean_rating:.2f}")
