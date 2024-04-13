# Import required dependencies
import pandas as pd
import pickle as pkl
# Convert textual columns in selected features into meaningful numeric columns
from sklearn.feature_extraction.text import TfidfVectorizer
# Find the similarity between each move and all other movies with shape of samples_size * samples_size
from sklearn.metrics.pairwise import cosine_similarity


# Load the dataset
Movies_dataset = pd.read_csv('movies.csv')
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
print(combined_features)
print(combined_features.shape)

# Convert textual columns in selected features into meaningful numeric columns
Vectorizer = TfidfVectorizer(min_df=1,stop_words='english', lowercase=True)
vectorized_features = Vectorizer.fit_transform(combined_features)
print(vectorized_features)
print(vectorized_features.shape)
# Find the similarity scores using cosine similarity
similarity = cosine_similarity(vectorized_features)
print(similarity)

# Save similarities of the movies
pkl.dump(similarity,open('similarity.pkl','wb'))



