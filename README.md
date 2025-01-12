# Netflix Recommendation System 🎥🍿
This project is designed to provide personalized movie recommendations using three different techniques: Demographic Filtering, Content-Based Filtering, and Collaborative Filtering. The system is built using Python, Streamlit, and machine learning libraries like Scikit-learn and Surprise.

![image](https://github.com/user-attachments/assets/7f6cb09c-d883-4cf6-bab7-d4a29dc5890f)

# Features:
- Netflix-Themed UI: A sleek and modern interface inspired by Netflix.
- Three Recommendation Techniques:
  1. Demographic Filtering: Top-rated movies based on weighted ratings.
  2. Content-Based Filtering: Movies similar to a selected movie.
  3. Collaborative Filtering: Predicts a user's rating for a specific movie.
- Interactive Interface: Built with Streamlit for easy interaction.
- Detailed Output: Displays recommendations in a table with title, genres, type, and IMDb rating.

# Technologies Used :
- Python: The core programming language used for the project.
- Streamlit: For building the interactive web application.
- Pandas: For data manipulation and analysis.
- NumPy: For numerical computations.
- Scikit-learn: For TF-IDF vectorization and cosine similarity.
- Surprise: For collaborative filtering using the SVD algorithm.
- CSS: Custom styling for the Netflix-inspired UI.

# Project Overview :
The Netflix-Style Movie Recommendation System is a web-based application that leverages machine learning techniques to provide personalized movie recommendations. The system is built using Streamlit, a popular framework for creating interactive web applications, and incorporates three distinct recommendation methods:
- Demographic Filtering:
  This method recommends movies based on their overall popularity and ratings. It uses a weighted rating formula that balances the average rating and the number of votes to ensure that highly-rated movies with a significant number of votes are prioritized.
- Content-Based Filtering:
  This method recommends movies similar to a selected movie based on their overview (description). It uses TF-IDF (Term Frequency-Inverse Document Frequency) to vectorize the movie descriptions and cosine similarity to find movies with similar content.
- Collaborative Filtering:
  This method predicts a user's rating for a specific movie using the Singular Value Decomposition (SVD) algorithm. It analyzes user-movie interactions from a ratings dataset to make personalized predictions.

