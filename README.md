# Movie-Recommendation-System
This is a movie recommendation system that provides movie suggestions based on user input.
The system uses cosine similarity to find movies that are similar to the user's favorite movie.
![image about the final project](<Movie Recommendation System.png>)

## Prerequisites
Before running the code, make sure you have the following dependencies installed:
- Pandas
- scikit-learn (for feature extraction and cosine similarity)
- difflib (for finding close matches of movie names)

## Overview of the Code
1-Load the Movies dataset and display the first and last five rows.

2-Select the most relevant features for movie recommendation, which include 'genres', 'keywords', 'tagline', 'cast', and 'director'.

3-Check for missing values in the dataset and perform data cleaning for the selected features by filling missing values with empty strings.

4-Combine the selected features into a single text-based feature, 'combined_features'.

5-Convert the text-based 'combined_features' into meaningful numeric columns using TfidfVectorizer.

6-Calculate the similarity scores between movies using cosine similarity.

7-Create a list of movie names.

8-The recommendation system allows users to enter their favorite movie name, finds close matches, and recommends the top 20 similar movies to the user's input.

## Flask App Structure
- app.py: Contains Flask routes for rendering the web interface and handling predictions.
- templates/: Directory with HTML templates for the web pages.
- static/: Directory for static files as CSS.

## Contribution
Contributions to this project are welcome. You can enhance the recommendation system by implementing more advanced algorithms, improving the recommendation accuracy, or adding user interface features. 
Feel free to make any contributions and submit pull requests.

