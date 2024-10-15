import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import keyword_search

def generate_embeddings(model_id, data):
    model = SentenceTransformer(model_id)
    semantic_embeddings = model.encode(data['text'].values)
    keyword_embeddings = generate_embeddings_tfidf
    data["embeddings"] = semantic_embeddings
    return data