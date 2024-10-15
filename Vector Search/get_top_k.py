import pandas as pd
import numpy as np
import cosine_similarity
import euclidean_distance
scores = {}
def get_top_k(query_vector, keyword_embedddings, semantic_embeddings, top_k, distance_metric):

    for  index, key_embed, semant_embed in enumerate(zip(keyword_embedddings, semantic_embeddings)):
        if distance_metric == 'euclidean':

            key_score = euclidean_distance(query_vector, key_embed)
            semant_score = euclidean_distance(query_vector, semant_embed)
            score = (key_score + semant_score)/2
            scores[index] = score

        elif distance_metric == 'cosine':
            key_score = cosine_similarity(query_vector, key_embed)  
            semant_score = cosine_similarity(query_vector, semant_embed)
            score = (key_score + semant_score)/2
            scores[index] = score


    top_k = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k])
    return top_k


