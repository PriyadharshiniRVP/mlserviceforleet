from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast
import os

app = Flask(__name__)

# --- STEP 1: Load and preprocess dataset ---
csv_path = r"leetcode_problems.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at {csv_path}")

df = pd.read_csv(csv_path)

# Fill missing values
df['topics'] = df['topics'].fillna("[]")
df['difficulty'] = df['difficulty'].fillna("Unknown")
df['title'] = df['title'].fillna("Untitled")

# ✅ Auto-generate slug from title for LeetCode URL
df['slug'] = df['title'].apply(
    lambda x: x.strip().lower().replace(' ', '-').replace("'", "").replace('"', '')
)

# Convert topics string to list and then to space-separated string for vectorizer
df['topics_list'] = df['topics'].apply(
    lambda x: ' '.join(ast.literal_eval(x)) if isinstance(x, str) else ''
)

# Use CountVectorizer to turn topics into bag-of-words matrix
vectorizer = CountVectorizer()
topic_matrix = vectorizer.fit_transform(df['topics_list'])

# --- STEP 2: Recommendation logic ---
def recommend_problems(solved_ids, top_n=5):
    valid_ids = df['frontendQuestionId'].astype(str).values
    solved_ids = [str(i).strip() for i in solved_ids if str(i).strip() in valid_ids]
    if not solved_ids:
        return []

    unsolved_mask = ~df['frontendQuestionId'].astype(str).isin(solved_ids)
    unsolved_df = df[unsolved_mask].copy()

    solved_mask = df['frontendQuestionId'].astype(str).isin(solved_ids)
    if solved_mask.sum() == 0:
        return []

    solved_vectors = topic_matrix[solved_mask]
    user_vector = np.asarray(solved_vectors.mean(axis=0))

    unsolved_vectors = topic_matrix[unsolved_mask]
    similarity_scores = cosine_similarity(user_vector, unsolved_vectors).flatten()

    unsolved_df.loc[:, 'similarity'] = similarity_scores

    recommendations = unsolved_df.sort_values(by='similarity', ascending=False).head(top_n)

    output = []
    for _, row in recommendations.iterrows():
        topics_list = ast.literal_eval(row['topics']) if isinstance(row['topics'], str) else []
        output.append({
            "id": row['frontendQuestionId'],
            "title": row['title'],
            "difficulty": row['difficulty'],
            "topics": topics_list,
            "similarity": round(row['similarity'], 4),
            "url": row['slug']  # ✅ Frontend will format full LeetCode URL
        })
    return output

# --- STEP 3: Flask API endpoints ---
@app.route('/')
def home():
    return jsonify({"status": "API is running. Use /recommend?solved=1,2,3"})

@app.route('/recommend', methods=['GET'])
def recommend_api():
    solved_param = request.args.get('solved', '')
    if not solved_param:
        return jsonify({"error": "Please provide solved problem IDs like ?solved=1,2,3"}), 400

    solved_ids = solved_param.split(',')
    recommendations = recommend_problems(solved_ids, top_n=5)
    return jsonify({"recommendations": recommendations})

# --- Run the app ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
