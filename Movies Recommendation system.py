# Import required dependencies
import pandas as pd
# To convert the user input into meaningful input suitable for the dataset
import difflib
# Convert textual columns in selected features into meaningful numeric columns
from sklearn.feature_extraction.text import TfidfVectorizer
# Find the similarity between each move and all other movies with shape of samples_size * samples_size
from sklearn.metrics.pairwise import cosine_similarity


# Load the dataset
Movies_dataset = pd.read_csv('/content/movies.csv')
# Show the first five rows in the dataset
Movies_dataset.head()
# Show the last five rows in the dataset
Movies_dataset.tail()
# Show the dataset shape
Movies_dataset.shape



# Select the most relevant features for the recommendation
selected_features = ['genres','keywords','tagline','cast','director']
selected_features

# Check about the none(missing) values in the dataset to decide if will make a data cleaning or not
Movies_dataset.isnull().sum()
# Make a data cleaning for 'genres','keywords','tagline','cast' and 'director' choosed columns 
for feature in selected_features:
    Movies_dataset[feature].fillna('',inplace=True)

# Choose the selected features for the recommendation and make a new dataset
combined_features = Movies_dataset['genres']+' '+Movies_dataset['keywords']+' '+Movies_dataset['tagline']+' '+Movies_dataset['cast']+' '+Movies_dataset['director']
combined_features

# Convert textual columns in selected features into meaningful numeric columns
Vectorizer = TfidfVectorizer(min_df=1,stop_words='english', lowercase=True)
vectorized_features = Vectorizer.fit_transform(combined_features)
print(vectorized_features)
# Find the similarity scores using cosine similarity
similarity = cosine_similarity(vectorized_features)
similarity
# Create a list for all the movies names
movies_names = Movies_dataset['title'].tolist()
print(movies_names)




# Making a recommendation system by taking the user movie name and find the first 20 similar movies for the user input

# Get the movie name from the user
user_movie_name = input("Enter your favourite movie name: ")

# Find the close matches between the user input and the movies names
close_matches_names = difflib.get_close_matches(user_movie_name,movies_names)
if close_matches_names:
    close_matches_names = close_matches_names[0]
print(close_matches_names)

# Find the index of the closest movie of the user input
index_closest_movie = Movies_dataset[Movies_dataset.title==close_matches_names]['index'].values[0]
print(index_closest_movie)

# Get a list of similar movies
similarity_list = list(enumerate(similarity[index_closest_movie]))
similarity_list

# Sort the list of similarity based on the similarity values and reverse it from biggest to smallest
similarity_list_sorted = sorted(similarity_list,key=lambda x:x[1],reverse=True)
similarity_list_sorted

# Make the Recommendation program for the user input
num_recommended_movies=20
for NumMovie in range(len(similarity_list_sorted)):
    if NumMovie==num_recommended_movies:
        break
    print(f"{NumMovie+1}: ",Movies_dataset.iloc[similarity_list_sorted[NumMovie][0],:].title)
