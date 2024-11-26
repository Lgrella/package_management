# import numpy as np
# from scipy.spatial import distance_matrix
# from scipy.sparse.csgraph import shortest_path
# from scipy.linalg import eigh
# from pyspark.sql import SparkSession
# from pyspark import RDD

# def SparkIsomap(X, k, n_components):
    
#     # X.persist()
    
#     #TODO: Making the adjacency graph !!
#     pairwise_distances = distance_matrix(X, X)
    
#     n_samples = X.shape[0]
#     knnGraph = np.full(pairwise_distances.shape, np.inf)
#     for i in range(n_samples):
#         nearestNeighbors = np.argsort(pairwise_distances[i])[:k + 1]
#         knnGraph[i, nearestNeighbors] = pairwise_distances[i, nearestNeighbors]
        
#     #NOTE: Need to make sure that the distances are symmetric
#     knnGraph = np.minimum(knnGraph, knnGraph.T)
    
#     #TODO: Shortest Paths
#     geodesicDistances = shortest_path(csgraph=knnGraph, directed=False)
    
#     #TODO: Metrics MDS
#     n = geodesicDistances.shape[0]
#     H = np.eye(n) - np.ones((n, n)) / n
#     K = -0.5 * H @ (geodesicDistances ** 2) @ H

#     #TODO: Components and Eigenvectors
#     eigenvalues, eigenvectors = eigh(K, subset_by_index=[n - n_components, n - 1])
    
#     #TODO: Sort so we can pick like the few components we want
#     idx = np.argsort(eigenvalues)[::-1]
#     eigenvalues = eigenvalues[idx]
#     eigenvectors = eigenvectors[:, idx]
    
#     # Compute the final embedding
#     embedding = eigenvectors[:, :n_components] * np.sqrt(eigenvalues[:n_components])
#     return embedding

from pyspark import SparkContext
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.linalg import eigh

def SparkIsomap(X, k, n_components):
    """
    ISOMAP implementation compatible with Apache Spark.

    Parameters:
        X (RDD): Input data as an RDD of (index, vector).
        k (int): Number of nearest neighbors.
        n_components (int): Number of dimensions for the reduced embedding.

    Returns:
        RDD: Lower-dimensional embedding as an RDD of vectors.
    """
    
    # Step 1: Compute pairwise distances
    pairwise_distances = X.cartesian(X).map(
        lambda pair: ((pair[0][0], pair[1][0]), np.linalg.norm(np.array(pair[0][1]) - np.array(pair[1][1])))
    )
    
    # Step 2: Build the k-nearest neighbors graph
    # Build the k-nearest neighbors graph
    knn_graph = pairwise_distances.map(
        lambda pair: (pair[0][0], (pair[0][1], pair[1]))
    ).groupByKey().mapValues(
        lambda distances: list(distances)  # Ensure distances is an iterable
    ).mapValues(
        lambda distances: sorted(distances, key=lambda x: x[1])[:k]  # Sort and pick top k
    ).flatMap(
        lambda x: [(x[0], neighbor[0], neighbor[1]) for neighbor in x[1]]
    ).distinct()

    
    # Convert KNN graph to adjacency matrix format (local computation for now)
    adjacency_matrix = knn_graph.collect()
    n = X.count()
    graph = np.full((n, n), np.inf)
    for (i, j, dist) in adjacency_matrix:
        graph[i, j] = dist
    graph = np.minimum(graph, graph.T)  # Ensure symmetry
    
    # Step 3: Compute geodesic distances
    geodesic_distances = shortest_path(csgraph=graph, directed=False)
    
    # Step 4: Apply classical MDS
    H = np.eye(n) - np.ones((n, n)) / n
    K = -0.5 * H @ (geodesic_distances ** 2) @ H
    eigenvalues, eigenvectors = eigh(K)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components
    embedding = eigenvectors[:, :n_components] * np.sqrt(eigenvalues[:n_components])
    return embedding
    # Convert embedding to RDD
    # embedding_rdd = X.context.parallelize(enumerate(embedding)).map(lambda x: (x[0], x[1].tolist()))
    # return embedding_rdd

# Example Usage
if __name__ == "__main__":
    sc = SparkContext.getOrCreate()
    
    # Example data
    data = np.random.rand(100, 3)  # 100 points in 3D space
    rdd = sc.parallelize(enumerate(data))
    
    # Run ISOMAP
    reduced_data = SparkIsomap(rdd, k=10, n_components=2)
    print(reduced_data)
