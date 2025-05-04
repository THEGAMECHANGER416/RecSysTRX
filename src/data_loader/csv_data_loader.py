import polars as pl
from pathlib import Path
from .base_loader import BaseLoader

class CSVDataLoader(BaseLoader):
    def __init__(self, data_path):
        self.data_path = Path(data_path)

    def load_data(self, columns=None) -> pl.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"[CSVDataLoader] File not found: {self.data_path}")

        print(f"[CSVDataLoader] Loading data from: {self.data_path}")
        df = pl.read_csv(self.data_path, columns=columns)
        print(f"[CSVDataLoader] Loaded shape: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
