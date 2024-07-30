import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
movies_data=pd.read_csv("minorpro.csv")
movies_data.shape
selected_features=['Genre','Director','actor_1_name','actor_2_name','MovieName']
print(selected_features)
combined_features = movies_data['Genre']+'   '+movies_data['Director']+'  '+movies_data['actor_1_name']+'   '+movies_data['MovieName']+'  '+movies_data['actor_2_name']+'  '+movies_data['actor_3_name']

vectorizer=TfidfVectorizer()
feature_vectors=vectorizer.fit_transform(combined_features)


similarity=cosine_similarity(feature_vectors)
movie_name=input("Enter your Favourite movie name:")

list_of_all_titles=movies_data['MovieName'].tolist()
print(list_of_all_titles)

find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)
print(find_close_match)
close_match =find_close_match[0]
print(close_match)

index_of_movie=movies_data[movies_data.MovieName == close_match]['Index'].values[0]
print(index_of_movie)



similarity_score=list(enumerate(similarity[index_of_movie]))
sorted_similar_movies=sorted(similarity_score, key=lambda x:x[1], reverse=True)

print("Movie suggested for you : \n")
i=1
for movie in sorted_similar_movies:
    index=movie[0]
    title_from_index=movies_data[movies_data.index==index]['MovieName'].values[0]+' - '
    if(i<20):
        print(i,'.',title_from_index,'(similarity score:',movie[1],')')
        i+=1