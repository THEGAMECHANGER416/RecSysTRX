import polars as pl

class DataPreprocessor:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def clean(self):
        """
        This method should be implemented in subclasses for cleaning the data.
        """
        raise NotImplementedError("Subclasses should implement the clean method")

    def transform(self):
        """
        This method should be implemented in subclasses for data transformation.
        """
        raise NotImplementedError("Subclasses should implement the transform method")

    def handle_missing_values(self):
        """
        Handle missing values by dropping rows with null values or any other strategy.
        """
        self.df = self.df.drop_nulls(subset=["rating"])
        return self.df
