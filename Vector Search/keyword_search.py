from sklearn.feature_extraction.text import TfidfVectorizer
import cosine_similarity

def generate_embeddings_tfidf(data_set):
    embeddings = []
    for data in data_set:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(data)
        embeddings.append(X)
    return embeddings

def calculate_tfidf_similarity(query, text_data):
 
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_data)
    query_tfidf_vector = vectorizer.transform([query])

    tfidf_scores = cosine_similarity(query_tfidf_vector, tfidf_matrix).flatten()
    
    return tfidf_scores
