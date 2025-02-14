from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

app = Flask(__name__)

# Load saved cosine similarity matrix
book_cosine_sim = load_npz("final_book_cosine_sim.npz")

# Load book names
book_names = pd.Series(pd.read_csv("book_list.csv")["Name"])  # Ensure correct path

def recommend_books(book_name, top_n=5):
    """Recommend books similar to the given book."""
    
    # Check if book exists
    if book_name.lower() not in book_names.str.lower().values:
        return {"error": "Book not found!"}
    
    input_idx = book_names.str.lower()[book_names.str.lower() == book_name.lower()].index[0]
    
    # Get similarity scores
    sim_scores = book_cosine_sim[input_idx].toarray().flatten()
    
    # Get top-n similar books (excluding itself)
    top_indices = np.argsort(sim_scores)[::-1][1:top_n+1]
    
    return {"recommended_books": book_names.iloc[top_indices].tolist()}

@app.route('/')
def home():
    return "Book Recommendation API is running!"

@app.route('/recommend', methods=['GET'])
def recommend():
    book_name = request.args.get('book')
    
    if not book_name:
        return jsonify({"error": "Please provide a book name"}), 400

    recommendations = recommend_books(book_name)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
