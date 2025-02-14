import pickle
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

# Load saved TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as file:
    tfidf_vectorizer = pickle.load(file)

# Load saved TF-IDF encoded matrix
tfidf_encoding = load_npz("tfidf_encoding.npz")

# Load saved cosine similarity matrix (only top-k values are stored)
book_cosine_sim = load_npz("final_book_cosine_sim.npz")

# Load the book names (assuming they were stored separately)
book_names = pd.Series(pd.read_csv("book_list.csv")["Name"])  # Replace with correct file

def recommend_books(book_name, top_n=5):
    """Recommend books similar to the given book using precomputed similarity scores."""
    
    # Get the index of the input book
    if book_name.lower() not in book_names.str.lower().values:
        print("Book not found in dataset!")
        return []
    
    input_idx = book_names.str.lower()[book_names.str.lower() == book_name.lower()].index[0]
    
    # Get similarity scores for this book
    sim_scores = book_cosine_sim[input_idx].toarray().flatten()
    
    # Get top-n similar books (excluding itself)
    top_indices = np.argsort(sim_scores)[::-1][1:top_n+1]
    
    return book_names.iloc[top_indices].tolist()

# Test the recommendation system
book_name = "The Catcher in the Rye"  # Change this to any book name in your dataset
recommended_books = recommend_books(book_name, top_n=5)

print(f"Recommended books similar to '{book_name}':")
print(recommended_books)
