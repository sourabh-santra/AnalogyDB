import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer
import time

class Index:

    def __init__(self, model_id, data, index_path, index_size, index_ef_construction, index_m):
        self.model_id = model_id
        self.data = data
        self.index_path = index_path
        self.index_size = index_size
        self.index_ef_construction = index_ef_construction
        self.index_m = index_m

    def create_index(self, embeddings):
        index = hnswlib.Index(space='cosine', dim=embeddings.shape[1])
        index.init_index(max_elements=self.index_size, ef_construction=self.index_ef_construction, M=self.index_m)
        index.add_items(embeddings)
        start_time = time.time()
        index.save_index(self.index_path)
        end_time = time.time()
        print(f"Time taken to create index: {end_time - start_time}")
        return index

    def query_index(self, query_vector, index_ef_search):
        index = hnswlib.Index(space='cosine', dim=query_vector.shape[0])
        index.load_index(self.index_path, max_elements=self.index_size)
        index.set_ef(index_ef_search)
        start_time = time.time()
        labels, distances = index.knn_query(query_vector, k=5)
        end_time = time.time()
        print(f"Time taken to search: {end_time - start_time}")
        return labels, distances

    def update_index(self, new_embeddings):
        index = hnswlib.Index(space='cosine', dim=new_embeddings.shape[1])
        index.load_index(self.index_path)
        index.add_items(new_embeddings)
        index.save_index(self.index_path)
        return index
    
    def delete_the_index(self):
        index = hnswlib.Index(space='cosine', dim=self.data.shape[1])
        index.load_index(self.index_path)
        index.clear_index()
        index.save_index(self.index_path)
        return index