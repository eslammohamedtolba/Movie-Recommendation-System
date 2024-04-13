# Load dependencies
from flask import Flask, render_template, request
import pandas as pd
import pickle as pkl
# To convert the user input into meaningful input suitable for the dataset
import difflib

# Create application
app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/")
# Load the dataset
Movies_dataset = pd.read_csv('./preparingsimilarities/movies.csv')
# Create a list for all the movies names
movies_names = Movies_dataset['title'].tolist()
# Get the similarity scores using pickle
similarity = pkl.load(open('./preparingsimilarities/similarity.pkl','rb'))




# Create Home api
@app.route('/', methods = ['GET','POST'])
def index():
    return render_template('index.html', list_movies = [])

# Create prediction api
@app.route('/predict',methods = ['POST'])
def predict():
    list_movies = [] 
    # Get the movie name from the user and required number of similar movies
    user_movie_name = request.form.get('input')
    num_recommended_movies = int(request.form.get('num_movies'))
    # Find the close matches between the user input and the movies names
    close_matches_names = difflib.get_close_matches(user_movie_name, movies_names)
    if close_matches_names:
        close_matches_names = close_matches_names[0]
        # Find the index of the closest movie of the user input
        closest_movie = Movies_dataset[Movies_dataset.title == close_matches_names]['index']
        index_closest_movie = closest_movie.values[0]
        # Get a list of similar movies
        similarity_list = list(enumerate(similarity[index_closest_movie]))
        # Sort the list of similarity based on the similarity values and reverse it from biggest to smallest
        similarity_list_sorted = sorted(similarity_list, key=lambda x:x[1],reverse=True)
        # Make the Recommendation program for the user input
        for NumMovie in range(len(similarity_list_sorted)):
            if NumMovie >= num_recommended_movies:
                break
            list_movies.append(f"({NumMovie+1}) {Movies_dataset.iloc[similarity_list_sorted[NumMovie][0],:].title}")
    return render_template('index.html', list_movies = list_movies)


if __name__ == "__main__":
    app.run(debug=True)


