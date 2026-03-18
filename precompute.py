import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

print("Loading data...")
df1 = pd.read_csv('Movie Recommendation System copy/tmdb_5000_credits.csv')
df2 = pd.read_csv('Movie Recommendation System copy/tmdb_5000_movies.csv')

# Merge on movie id
df1 = df1.rename(columns={'movie_id': 'id'})
df2 = df2.merge(df1, how='inner', on='id')
df2.drop(columns=['title_x'], inplace=True)
df2.rename(columns={'title_y': 'title'}, inplace=True)
print(f"Merged dataset: {df2.shape[0]} movies")

# Parse JSON string columns
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)

# Extract director from crew
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return ''

# Get top 3 names from list feature
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        return names[:3] if len(names) > 3 else names
    return []

df2['director'] = df2['crew'].apply(get_director)
for feature in ['cast', 'keywords', 'genres']:
    df2[feature] = df2[feature].apply(get_list)

# Save display fields BEFORE cleaning (clean_data makes everything lowercase/no-spaces for matching)
df2['genres_display'] = df2['genres'].apply(lambda x: ', '.join(x) if x else '')
df2['cast_display'] = df2['cast'].apply(lambda x: ', '.join(x) if x else '')
df2['director_display'] = df2['director'].copy()

# Clean: lowercase, remove spaces (for soup matching)
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(' ', '')) for i in x]
    return str.lower(x.replace(' ', '')) if isinstance(x, str) else ''

for feature in ['cast', 'keywords', 'director', 'genres']:
    df2[feature] = df2[feature].apply(clean_data)

# Build metadata soup (note: director is a string, not a list)
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

df2['soup'] = df2.apply(create_soup, axis=1)

# Reset index and build title -> index mapping
df2 = df2.reset_index(drop=True)
indices = pd.Series(df2.index, index=df2['title'])

print("Computing cosine similarity matrix (may take ~30 seconds)...")
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
print(f"Matrix shape: {cosine_sim2.shape}")

# Save similarity matrix compressed (float32 saves ~half the space)
np.savez_compressed('cosine_sim2.npz', matrix=cosine_sim2.astype(np.float32))
size_mb = os.path.getsize('cosine_sim2.npz') / 1024 / 1024
print(f"Saved cosine_sim2.npz ({size_mb:.1f} MB)")

# Save movie metadata
movie_data = df2[['title', 'id', 'overview', 'release_date', 'vote_average', 'vote_count',
                   'genres_display', 'cast_display', 'director_display']].copy()
movie_data['year'] = pd.to_datetime(movie_data['release_date'], errors='coerce').dt.year.fillna(0).astype(int)

with open('movie_data.pkl', 'wb') as f:
    pickle.dump(movie_data, f)
print(f"Saved movie_data.pkl ({os.path.getsize('movie_data.pkl') / 1024:.0f} KB)")

with open('indices.pkl', 'wb') as f:
    pickle.dump(indices, f)
print(f"Saved indices.pkl ({os.path.getsize('indices.pkl') / 1024:.0f} KB)")

print(f"\n✅ Done! {len(movie_data)} movies processed.")
