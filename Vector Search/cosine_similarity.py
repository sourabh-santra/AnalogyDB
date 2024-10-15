import numpy as np
from numpy.linalg import norm

# define two lists or array
A = np.array([2,1,2,3,2,9])
B = np.array([3,4,2,4,5,5])
 
def cosine_similarity(A, B):
    score = np.dot(A, B) / (norm(A) * norm(B))
    print(score)
    return score

cosine_similarity(A, B)