import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the datasets
movies = pd.read_csv('movies.csv')  # MovieID, Title
ratings = pd.read_csv('ratings.csv')  # UserID, MovieID, Rating

# Show the first few rows of each dataframe
print(movies.head())
print()
print(ratings.head())
print()

# Merge the datasets on MovieID
movie_data = pd.merge(ratings, movies, on='movieId')
print(movie_data.head())

# Create the user-item matrix
user_movie_ratings = movie_data.pivot_table(index='userId', columns='title', values='rating')

# Show the matrix
print(user_movie_ratings.head())

# Define the Reader object to interpret the data
reader = Reader(rating_scale=(0.5, 5.0))

# Load the dataset using Surprise
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Use Singular Value Decomposition (SVD) for collaborative filtering
model = SVD()

# Train the model
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model using RMSE (Root Mean Squared Error)
rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse}')

