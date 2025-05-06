from src.models import UserUserRecommender

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    model = UserUserRecommender()
    # model.build_model()
    model.load_model()
    
    # model.inference(type="user_id",user_id=99999)
    # [(858, np.float32(30.46161)), (1193, np.float32(27.594316)), (1674, np.float32(26.192596)), (296, np.float32(24.806093)), (50, np.float32(24.127657)), (1252, np.float32(23.366873)), (1244, np.float32(22.65704)), (1089, np.float32(21.99973)), (1240, np.float32(21.992231)), (1230, np.float32(21.987974))]
    
    watched_movies = [5349, 8636, 52722, 95510, 110553, 195159, 201773]
    model.inference(type="watched_movies", watched_movies=watched_movies)
    # [(1193, np.float32(36.097908)), (50, np.float32(35.36721)), (1198, np.float32(34.63244)), (1230, np.float32(34.627655)), (913, np.float32(34.62595)), (296, np.float32(33.88787)), (903, np.float32(32.424686)), (858, np.float32(32.397694)), (750, np.float32(30.953785)), (919, np.float32(30.926365))]
    
