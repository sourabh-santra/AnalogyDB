
import get_top_k
from sentence_transformers import SentenceTransformer
def search(query_vector, vectors, top_k, distance_metric,model_id):
    model = SentenceTransformer(model_id)
    top_k = get_top_k.get_top_k(query_vector, vectors, top_k, distance_metric)
    top_k = model.decode(top_k)
    return top_k

