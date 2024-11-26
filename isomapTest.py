import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from isomap.main import SparkIsomap
from sklearn.datasets import make_swiss_roll
from pyspark.sql import SparkSession
import numpy as np
from sklearn.manifold import MDS, Isomap
from pyspark import SparkContext

df = pd.read_csv('data/diabetes.csv') 
#print(df.head)
y = df["Outcome"]
X = df.drop("Outcome", axis=1)

# Run the test
if __name__ == "__main__":
    # test_spark_pca_against_sklearn()
    
    #NOTE: Swiss Roll Data for Mapping Purposes
    # load swiss roll dataset
    n_samples = 10000
    noise = 0.05

    np.random.seed(2024) # use random seed

    X, y = make_swiss_roll(n_samples, noise=noise)

    # plot the data colored by class labels
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:,2], c= y, cmap='RdPu')
    ax.set_title("Initial Data in 3D")
    ax.set_xlabel('x');  ax.set_ylabel('y'); ax.set_zlabel('z'); 
    ax.view_init(20, 70)
    
    
    # #NOTE: Plotting the sklearn version of dimension reduction mapping with IsoMap 
    # # Creating IsoMap Representation 
    # embedding = Isomap(n_neighbors=10)
    # X_new = embedding.fit_transform(X)
    
    # # plot the data colored by class labels
    # fig = plt.figure(figsize=(12, 4))
    # ax = fig.add_subplot(131)
    # ax.scatter(X_new[:, 0], X_new[:, 1], c= y, cmap='RdPu')
    # ax.set_title("data in 2D with Isomap (10 neighbors)")
    # ax.set_xlabel('x') 
    # ax.set_ylabel('y')
    # plt.show()
    
    #NOTE: Plotting the Sparkit version of dimension reduction mapping with IsoMap 
    # Creating IsoMap Representation 
    sc = SparkContext.getOrCreate()
    X_new = sc.parallelize(enumerate(X)) 
    embedding = SparkIsomap(X_new, k=10, n_components=10)

    # plot the data colored by class labels
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(131)
    ax.scatter((embedding[:, 0] * -1), embedding[:, 1], c= y, cmap='RdPu')
    ax.set_title("data in 2D with SparkIt (10 neighbors)")
    ax.set_xlabel('x') 
    ax.set_ylabel('y')
    plt.show()
    
    
    
    
