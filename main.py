from src.models import UserUserRecommender
from src.config import RATINGS_FILE

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    model = UserUserRecommender(RATINGS_FILE)
    model.build_model()
