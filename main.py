import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval
from surprise import Reader, Dataset, SVD
import ast

@st.cache_data
def load_data():
    # Load datasets
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = pd.read_csv("tmdb_5000_movies.csv")

    # Rename 'movie_id' to 'id' in credits
    credits.rename(columns={'movie_id': 'id'}, inplace=True)

    # Convert 'id' in credits to numeric, coerce errors to NaN
    credits['id'] = pd.to_numeric(credits['id'], errors='coerce')

    # Filter out rows with invalid 'id's
    credits = credits[credits['id'].apply(lambda x: isinstance(x, (int, float)) and not np.isnan(x))]

    # Convert 'id' in credits to integer
    credits['id'] = credits['id'].astype(int)

    # Convert 'id' in movies to integer
    movies['id'] = movies['id'].astype(int)

    # Merge datasets on 'id' and specify suffixes
    movies = movies.merge(credits, on='id', suffixes=('', '_credit'))

    # Drop the 'title' from credits if it exists
    if 'title_credit' in movies.columns:
        movies.drop(columns=['title_credit'], inplace=True)

    # Remove duplicate titles
    movies.drop_duplicates(subset='title', inplace=True)

    return movies

# Rest of your code...

# Demographic Filtering: Weighted Rating
def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)

@st.cache_data
def load_ratings_data():
    ratings = pd.read_csv("ratings_small.csv")
    return ratings

# Content-Based Filtering: TF-IDF and Cosine Similarity
def get_recommendations(title, cosine_sim, movies):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Collaborative Filtering: SVD
def collaborative_filtering(user_id, movie_id):
    prediction = svd.predict(user_id, movie_id).est
    return prediction

# Main Function
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Movie Recommendation System",
        page_icon="🎬",
        initial_sidebar_state="collapsed"
    )

    # Inject custom CSS for Netflix-style styling
    st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');

        /* Define color variables */
        :root {
            --bg-color: #14171A;
            --text-color: #FFFFFF;
            --accent-color: #E50914; /* Netflix red */
            --border-color: #5A5A5A;
            --header-bg: #1C1F24;
            --light-gray: #666666;
        }

        /* Set body styles */
        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Poppins', sans-serif;
            margin: 0;
            padding: 20px;
        }

        /* Style headers */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-color);
            font-weight: bold;
            margin-top: 20px;
        }

        /* Style text */
        p, li {
            color: var(--text-color);
            line-height: 1.6;
        }

        /* Style buttons */
        .stButton button {
            background-color: var(--accent-color);
            color: var(--text-color);
            font-weight: bold;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
        }

        /* Remove hover effect for buttons */
        .stButton button:hover {
            background-color: var(--accent-color);
        }

        /* Style select boxes */
        .stSelectbox {
            background-color: #000000; /* Black background */
            color: var(--text-color);
            border: 1px solid var(--accent-color); /* Red border */
            border-radius: 4px;
            padding: 5px;
            font-family: inherit;
        }

        /* Style input fields */
        .stTextInput input {
            background-color: #000000; /* Black background */
            color: var(--text-color); /* White text */
            border: 1px solid var(--accent-color); /* Red border */
            border-radius: 4px;
            padding: 5px;
        }

        /* Style tables */
        table {
            width: 100%;
            border-collapse: collapse;
            color: var(--text-color);
        }

        th, td {
            border: 1px solid var(--border-color);
            padding: 8px;
            text-align: left;
        }

        th {
            background-color: var(--header-bg);
        }

        /* Style code blocks */
        pre {
            background-color: #282C34;
            color: var(--text-color);
            border-radius: 4px;
            padding: 10px;
            overflow-x: auto;
        }

        /* Add spacing between sections */
        .section {
            margin-bottom: 40px;
        }

        /* Center text */
        .centered {
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)
    st.image("nflogo.jpeg", width=250)
    st.title("Recommendation System")
    st.write("-By Bhoomika Ramchandani")
    
    

    # Load data
    movies = load_data()
    ratings = load_ratings_data()

    # Dropdown to select recommendation technique
    technique = st.selectbox(
        "Select Recommendation Technique",
        ["Demographic Filtering", "Content-Based Filtering", "Predict Movie Rating"],
        key="technique_selectbox",
        label_visibility="collapsed"
    )

    if technique == "Demographic Filtering":
        st.write("Top Rated Movies :")
        C = movies['vote_average'].mean()
        m = movies['vote_count'].quantile(0.9)
        q_movies = movies.copy().loc[movies['vote_count'] >= m]
        q_movies['score'] = q_movies.apply(weighted_rating, axis=1, args=(m, C))
        q_movies = q_movies.sort_values('score', ascending=False)
        st.write(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10))

    elif technique == "Content-Based Filtering":
        st.write("Select a movie title :")
        movie_title = st.selectbox(
            "🔍 Select a movie:",
            movies['title'].values,
            key="movie_selectbox",
            label_visibility="collapsed"
        )
        if st.button("Get Recommendations"):
            # TF-IDF Vectorization
            tfidf = TfidfVectorizer(stop_words='english')
            movies['overview'] = movies['overview'].fillna('')
            tfidf_matrix = tfidf.fit_transform(movies['overview'])
            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

            # Get recommendations
            global indices
            indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
            recommendations = get_recommendations(movie_title, cosine_sim, movies)
            st.write(recommendations)
              
    elif technique == "Predict Movie Rating":
        st.write("Predict Rating for a Movie :")
        user_id = st.number_input("Enter User ID:", min_value=1, max_value=671, value=1)
        movie_id = st.number_input("Enter Movie ID:", min_value=1, max_value=100000, value=302)
        if st.button("Predict Rating"):
            # Load ratings data
            ratings = pd.read_csv("ratings_small.csv")
            reader = Reader()
            data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

            # Train SVD model
            global svd
            svd = SVD()
            trainset = data.build_full_trainset()
            svd.fit(trainset)

            # Predict rating
            prediction = collaborative_filtering(user_id, movie_id)
            st.write(f"🎬 Predicted Rating for Movie ID {movie_id}: {prediction:.2f}")

if __name__ == "__main__":
    main()