#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Downloading necessary NLTK resources
nltk.download('wordnet')


# In[2]:


"""Load dataset, remove nulls, and keep unique entries based on 'names' and 'overview'."""
def load_data(filepath):
    df = pd.read_csv(filepath).dropna(subset=['names', 'overview'])
    
    # Droping duplicate rows based on 'names' and 'overview'
    df = df.drop_duplicates(subset=['names', 'overview'])
    
    # randomly selecting 500 unique rows
    df = df.sample(n=500,random_state=42)
    
    return df


# In[3]:


"""Preprocess text by removing special characters and applying lemmatization."""
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Removing special characters
    text = re.sub(r'\s+', ' ', text)  # Removing extra spaces
    text = text.lower().strip()  # Converting all words to lowercase
    
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)


# In[4]:


"""Convert text descriptions into TF-IDF vectors using bigrams & trigrams for better similarity."""
def build_tfidf_matrix(descriptions):
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        ngram_range=(1, 3), 
        max_features=50000, 
        strip_accents='unicode', #Different accents are converted to unicode eg., å -> a,a˙ -> a
        max_df=0.85,  # Ignore words that appear in more than 85% of the documents
        min_df=2,     # Ignore words that appear in less than 2 documents
    )
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    return vectorizer, tfidf_matrix


# In[5]:


"""Recommend top-N similar items based on user input using cosine similarity."""
def recommend_items(user_input, vectorizer, tfidf_matrix, df, top_n=5):
    user_tfidf = vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    top_indices = cosine_similarities.argsort()[::-1][:top_n]
    recommendations = df.iloc[top_indices][['names', 'overview']]
    scores = cosine_similarities[top_indices]
    
    return recommendations.reset_index(drop=True), scores  # Reset index for clean output


# In[6]:


"""Plot similarity scores of top recommendations."""
def plot_recommendations(recommendations, scores):
    plt.figure(figsize=(8, 4))
    sns.barplot(x=scores, y=recommendations['names'], palette='coolwarm')
    plt.xlabel('Similarity Score')
    plt.ylabel('Recommended Movies')
    plt.title('Top Recommended Movies')
    plt.show()


# In[8]:


"""Main execution function."""
def main():
    dataset_path = 'imdb_movies.csv'  
    
    #load dataset
    df = load_data(dataset_path)
    
    #Preprocess text
    df['overview'] = df['overview'].apply(preprocess_text)
    
    #Build tf-idf vectorizer
    vectorizer, tfidf_matrix = build_tfidf_matrix(df['overview'])
    
    #Collecting user input
    user_input = input("\nEnter your preference (e.g., 'I love thrilling action movies set in space, with a comedic twist.'):\n")
    
    #Getting recommendations and similarity scores
    recommendations, scores = recommend_items(user_input, vectorizer, tfidf_matrix, df, top_n=5)
    
    if recommendations.empty:
        print("\nNo recommendations found. Try a different description.\n")
        return
    
    print("\nTop Recommendations:")
    for idx, row in recommendations.iterrows():
        print(f"\n{idx+1}. {row['names']}")
        print(f"   {row['overview']}\n")
    
    plot_recommendations(recommendations, scores)




# In[ ]:
if __name__ == "__main__":
    main()



