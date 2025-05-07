import streamlit as st
from src.models import UserUserRecommender
from src.utils import Utils
from dotenv import load_dotenv
import ast

load_dotenv()

@st.cache_resource
def load_model():
    model = UserUserRecommender()
    model.load_model()
    return model

model = load_model()

st.title("🎬 Movie Recommender")

input_type = st.radio("Choose input type:", ("user_id", "watched_movies"))

if input_type == "user_id":
    user_id = st.number_input("Enter User ID:", min_value=1, step=1)
    if st.button("Recommend"):
        recommendations = model.inference(type="user_id", user_id=user_id)

        st.subheader("🎯 Recommendations")
        for movie_id, score in recommendations:
            details = Utils.get_movie_details(movie_id, model.movie_details_dict)
            st.write({
                "movie_id": movie_id,
                "title": details["title"],
                "genres": details["genres"],
                "score": float(score)
            })

else:
    # Create mapping from movie titles to IDs
    movie_name_to_id = {v["title"]: k for k, v in model.movie_details_dict.items()}
    movie_titles = sorted(movie_name_to_id.keys())

    selected_titles = st.multiselect("Select Watched Movies by Title", movie_titles)

    watched_list = [movie_name_to_id[title] for title in selected_titles]
    watched_list = [int(movie_id) for movie_id in watched_list]
    
    if st.button("Recommend"):
        recommendations = model.inference(type="watched_movies", watched_movies=watched_list)

        st.subheader("📽️ Watched Movies")
        for movie_id in watched_list:
            st.write(Utils.get_movie_details(movie_id, model.movie_details_dict))

        st.subheader("🎯 Recommendations")
        for movie_id, score in recommendations:
            details = Utils.get_movie_details(movie_id, model.movie_details_dict)
            st.write({
                "movie_id": movie_id,
                details["title"]: details,
                "score": float(score)
            })
