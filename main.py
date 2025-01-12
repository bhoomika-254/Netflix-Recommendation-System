import gradio as gr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval
from scikit-surprise import Reader, Dataset, SVD
import ast

# Cache implementation remains the same
_cached_movies = None
_cached_ratings = None

# Data loading functions remain the same
def load_data():
    global _cached_movies
    if _cached_movies is not None:
        return _cached_movies
        
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits.rename(columns={'movie_id': 'id'}, inplace=True)
    credits['id'] = pd.to_numeric(credits['id'], errors='coerce')
    credits = credits[credits['id'].apply(lambda x: isinstance(x, (int, float)) and not np.isnan(x))]
    credits['id'] = credits['id'].astype(int)
    movies['id'] = movies['id'].astype(int)
    movies = movies.merge(credits, on='id', suffixes=('', '_credit'))
    if 'title_credit' in movies.columns:
        movies.drop(columns=['title_credit'], inplace=True)
    movies.drop_duplicates(subset='title', inplace=True)
    _cached_movies = movies
    return movies

def load_ratings_data():
    global _cached_ratings
    if _cached_ratings is not None:
        return _cached_ratings
    ratings = pd.read_csv("ratings_small.csv")
    _cached_ratings = ratings
    return ratings

# Core logic functions remain the same
def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)

def get_top_rated_movies():
    movies = load_data()
    C = movies['vote_average'].mean()
    m = movies['vote_count'].quantile(0.9)
    q_movies = movies.copy().loc[movies['vote_count'] >= m]
    q_movies['score'] = q_movies.apply(weighted_rating, axis=1, args=(m, C))
    q_movies = q_movies.sort_values('score', ascending=False)
    result = q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)
    formatted_result = ""
    for idx, row in result.iterrows():
        formatted_result += f"Title: {row['title']}\n"
        formatted_result += f"Vote Count: {row['vote_count']}\n"
        formatted_result += f"Vote Average: {row['vote_average']:.1f}\n"
        formatted_result += f"Score: {row['score']:.2f}\n"
        formatted_result += "-" * 50 + "\n"
    return formatted_result

def get_recommendations(title):
    movies = load_data()
    tfidf = TfidfVectorizer(stop_words='english')
    movies['overview'] = movies['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movies['overview'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = movies['title'].iloc[movie_indices]
    return "\n".join([f"{i+1}. {title}" for i, title in enumerate(recommendations)])

def predict_rating(user_id, movie_id):
    ratings = load_ratings_data()
    reader = Reader()
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    svd = SVD()
    trainset = data.build_full_trainset()
    svd.fit(trainset)
    prediction = svd.predict(user_id, movie_id).est
    return f"🎬 Predicted Rating: {prediction:.2f} / 5.0"

# Custom CSS for Netflix theme
custom_css = """
#component-0 {
    max-width: 800px !important;
    margin: auto !important;
    padding: 20px !important;
    background-color: #141414 !important;
}

.logo-container {
    text-align: center;
    margin: 0 !important;
}

.logo-image {
    max-width: 200px;
    margin: 0 !important;
}

.gradio-container {
    background-color: #141414 !important;
}

.tabs.svelte-710i53 {
    background-color: #141414 !important;
    border-bottom: 2px solid #DC1A22 !important;
    margin-bottom: 25px !important;
}

.tab-nav {
    background-color: #141414 !important;
    border: none !important;
    margin-bottom: 20px !important;
}

button.selected {
    background-color: #DC1A22 !important;
    color: white !important;
}

button {
    background-color: #DC1A22 !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 8px 16px !important;
    font-size: 14px !important;
    cursor: pointer !important;
    transition: background-color 0.3s !important;
    margin: 10px 5px !important;
}

button:hover {
    background-color: #B2070E !important;
}

.input-box, .output-box, select, textarea {
    background-color: #242424 !important;
    border: 1px solid #DC1A22 !important;
    color: white !important;
    border-radius: 4px !important;
    margin-top: 10px !important;
    margin-bottom: 15px !important;
}

label {
    color: #FFFFFF !important;
    margin-bottom: 8px !important;
}

.markdown {
    color: #FFFFFF !important;
    margin-bottom: 20px !important;
}

.tabs > div:first-child {
    border-bottom: 2px solid #DC1A22 !important;
    margin-bottom: 20px !important;
}

.tab-selected {
    color: #DC1A22 !important;
    border-bottom: 2px solid #DC1A22 !important;
}

/* Add space between elements */
.block {
    margin-bottom: 20px !important;
}

.row {
    margin-bottom: 15px !important;
}
"""

def create_interface():
    movies = load_data()
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
        # Load and display logo using gr.Image
        gr.Image("nflogo.jpeg", show_label=False, container=False, scale=1, min_width=100, show_download_button=False, interactive=False, show_fullscreen_button=False)
        
        gr.Markdown("# Movie Recommendation System")
        gr.Markdown("### - By Bhoomika Ramchandani")
        
        with gr.Tabs() as tabs:
            with gr.Tab("Top Rated"):
                with gr.Column(scale=1):
                    demo_button = gr.Button("Show Top Rated Movies", scale=0.4)
                    gr.Markdown("     ")  # Adding space
                    demo_output = gr.Textbox(label="Results", lines=15)
                demo_button.click(get_top_rated_movies, inputs=[], outputs=demo_output)
            
            with gr.Tab("Find Similar"):
                with gr.Column(scale=1):
                    with gr.Row():
                        movie_dropdown = gr.Dropdown(
                            choices=movies['title'].tolist(),
                            label="Select a movie",
                            container=False,
                            scale=0.7
                        )
                        content_button = gr.Button("Get Recommendations", scale=0.3)
                    gr.Markdown("     ")  # Adding space
                    content_output = gr.Textbox(label="Recommended Movies", lines=10)
                content_button.click(get_recommendations, inputs=[movie_dropdown], outputs=content_output)
            
            with gr.Tab("Predict Rating"):
                with gr.Column(scale=1):
                    with gr.Row():
                        user_id = gr.Number(
                            minimum=1,
                            maximum=671,
                            value=1,
                            label="User ID",
                            scale=0.3
                        )
                        movie_id = gr.Number(
                            minimum=1,
                            maximum=100000,
                            value=302,
                            label="Movie ID",
                            scale=0.3
                        )
                        predict_button = gr.Button("Predict", scale=0.2)
                    gr.Markdown("     ")  # Adding space
                    predict_output = gr.Textbox(label="Predicted Rating")
                predict_button.click(predict_rating, inputs=[user_id, movie_id], outputs=predict_output)
    
    return demo

if __name__ == "__main__":
<<<<<<< HEAD
    demo = create_interface()
    demo.launch()
=======
    main()
>>>>>>> 8c7aa701a1c283a01a075ed12422cab98f921adb
