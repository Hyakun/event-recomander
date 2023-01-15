# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Dependencies
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#Data Collection and Pre-Processing

#load data to pandas
event_data = pd.read_csv('C:/Users/xdany/Desktop/events.csv')

#print the first 5 rows of the data frame
event_data.head()

#number of rows and columns in df
event_data.shape

# cleaning data
event_data['keywords']=event_data['keywords'].str.replace(r'\b(\w{1,2})\b', '').str[:-1].str.replace('#','').str.replace(',','')
#print(event_data['Tags'][1])


#selecting the relevant features for recomandation
selected_features = ['keywords']

# replacing the null values with null string
for feature in selected_features:
    event_data[feature] = event_data[feature].fillna('')
    


# combining all the 2 selected feaures
combined_features = event_data['keywords'] 

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)


print(feature_vectors)

#cosine Simirality
#getting similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)

print(similarity.shape)


eventId = input('Enter event Id: ') #Adauga Id dupa care cauti aici!!!!


index_of_the_event = event_data.index[event_data['_id'] == eventId][0]
print(index_of_the_event)

print(event_data.iloc[index_of_the_event])

#getting a list of similar events
similarity_score = list(enumerate(similarity[index_of_the_event]))

print(similarity_score)
print(len(similarity_score))


# sorting the events based on their similarity score
sorted_similar_events = sorted(similarity_score, key = lambda x:x[1], reverse = True)
print(sorted_similar_events)



list_of_similar_events = []

i = 1

for event in sorted_similar_events:
    index = event[0]
    id_from_index = event_data.iloc[index]['_id']
    if (i<31):
        list_of_similar_events.append(id_from_index)
        i+=1
    if(i == 31):
        break


print(list_of_similar_events) # lista finala cu Id-uri















