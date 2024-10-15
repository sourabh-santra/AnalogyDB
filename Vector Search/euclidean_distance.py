import numpy as np
 
# initializing points in
# numpy arrays
point1 = np.array((1, 2, 3))
point2 = np.array((1, 1, 1))
 
def euclidean_distance(point1, point2):
    dist = np.linalg.norm(point1 - point2)
    return dist

euclidean_distance(point1, point2)
 
