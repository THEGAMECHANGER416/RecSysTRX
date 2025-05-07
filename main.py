from src.models import UserUserRecommender
from src.utils import Utils

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    model = UserUserRecommender()
    # model.build_model()
    model.load_model()
    
    # model.inference(type="user_id",user_id=5130)
    # [(2571, np.float32(2.7919884)), (73017, np.float32(2.2052264)), (69844, np.float32(2.1642516)), (110102, np.float32(2.0178792)), (68319, np.float32(2.0009112)), (79132, np.float32(1.9409686)), (6539, np.float32(1.8630883)), (59784, np.float32(1.8367908)), (53125, np.float32(1.7862867)), (1580, np.float32(1.7704163))]
    
    # watched_movies = [3114, 4993, 49272, 118696, 120799, 111362, 1240, 33794, 125916, 91658, 81834, 72998, 8961, 116977, 4306, 68954, 59315, 119155, 4262, 1221, 135887, 5952, 296, 2716, 89745, 130634, 112852, 76093, 1, 122892, 44191, 858, 364, 7153, 78499, 91529, 589, 87232, 58559]
    # model.inference(type="watched_movies", watched_movies=watched_movies)
    # [(2571, np.float32(2.7919884)), (73017, np.float32(2.2052264)), (69844, np.float32(2.1642516)), (110102, np.float32(2.0178792)), (68319, np.float32(2.0009112)), (79132, np.float32(1.9409686)), (6539, np.float32(1.8630883)), (59784, np.float32(1.8367908)), (53125, np.float32(1.7862867)), (1580, np.float32(1.7704163))]
