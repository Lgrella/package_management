from pyspark import SparkContext
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.linalg import eigh

def SparkIsomap(X, k, n_components):
    
    X.persist()

    pairwise_distances = X.cartesian(X).map(
        lambda pair: ((pair[0][0], pair[1][0]), np.linalg.norm(np.array(pair[0][1]) - np.array(pair[1][1])))
    )
    
    # Build the k-nearest neighbors graph
    knn_graph = pairwise_distances.map(
        lambda pair: (pair[0][0], (pair[0][1], pair[1]))
    ).groupByKey().mapValues(
        lambda distances: list(distances)
    ).mapValues(
        lambda distances: sorted(distances, key=lambda x: x[1])[:k]
    ).flatMap(
        lambda x: [(x[0], neighbor[0], neighbor[1]) for neighbor in x[1]]
    ).distinct()

    
    # Convert KNN graph to adjacency matrix format (local computation for now)
    adjacency_matrix = knn_graph.collect()
    
    n = X.count()
    
    graph = np.full((n, n), np.inf)
    
    for (i, j, dist) in adjacency_matrix:
        graph[i, j] = dist
    graph = np.minimum(graph, graph.T)
    
    # geodesic distances
    geodesic_distances = shortest_path(csgraph=graph, directed=False)
    
    # classical MDS
    H = np.eye(n) - np.ones((n, n)) / n
    K = -0.5 * H @ (geodesic_distances ** 2) @ H
    eigenvalues, eigenvectors = eigh(K)
    
    # Sort eigenvalues and eigenvectors 
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components
    embedding = eigenvectors[:, :n_components] * np.sqrt(eigenvalues[:n_components])
    return embedding
