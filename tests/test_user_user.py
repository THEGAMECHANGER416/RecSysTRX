from src.models import UserUserRecommender
from src.config import RATINGS_FILE

def run_user_user_pipeline():
    model = UserUserRecommender(RATINGS_FILE)
    model.build_model()

if __name__ == "__main__":
    run_user_user_pipeline()
