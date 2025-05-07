from .base import DataPreprocessor
import polars as pl

class RatingsDataPreprocessor(DataPreprocessor):
    def clean(self):
        """
        Perform cleaning operations like removing duplicates and unnecessary columns.
        """
        try:
            initial_count = self.df.shape[0]
            self.df = self.df.unique(subset=["userId", "movieId"])
            final_count = self.df.shape[0]
            print(f"[Clean] Removed {initial_count - final_count} duplicate rows.")
        except Exception as e:
            print(f"[Clean] Error while removing duplicates: {e}")

    def transform(self):
        """
        Perform transformations like data normalization at the user level.
        Normalize ratings within each user.
        """
        print("[Transform] Normalizing ratings at userId level")

        # Normalize ratings within each userId group
        self.df = self.df.with_columns(
            pl.when(pl.col("rating").max().over("userId") == pl.col("rating").min().over("userId"))
            .then(0)
            .otherwise(
                (pl.col("rating") - pl.col("rating").min().over("userId")) /
                (pl.col("rating").max().over("userId") - pl.col("rating").min().over("userId"))
            )
            .alias("rating")
        )

        print("[Transform] Normalized ratings at the user level.")



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
