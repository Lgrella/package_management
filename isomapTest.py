import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from isomap.main import SparkIsomap
from sklearn.datasets import make_swiss_roll
from pyspark.sql import SparkSession
import numpy as np
from sklearn.manifold import MDS, Isomap

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
    
    
    #NOTE: Plotting the sklearn dimension reduction mapping
    # Creating Isompa Representation 
    embedding = Isomap(n_neighbors=10)
    X_new = embedding.fit_transform(X)
    
    # plot the data colored by class labels
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(131)
    ax.scatter(X_new[:, 0], X_new[:, 1], c= y, cmap='RdPu')
    ax.set_title("data in 2D with Isomap (10 neighbors)")
    ax.set_xlabel('x') 
    ax.set_ylabel('y')
    plt.show()
    
    
    # #NOTE: Initialize Spark session
    # spark = SparkSession.builder.appName("SparkPCATest").getOrCreate()
    # sc = spark.sparkContext
    # rdd = sc.parallelize(X.tolist()) 

    # #NOTE: SparkIt PCA
    # spark_pca = SparkPCA(n_components=2)
    # spark_pca.fit(rdd)
    # spark_transformed_rdd = spark_pca.transform(rdd)
    # spark_transformed_data = np.array(spark_transformed_rdd.collect()) 
    
    # # print("SparkPCA Components:")
    # # print(spark_transformed_data.components)
    
    # fig = plt.figure(figsize=(12, 4))
    # ax = fig.add_subplot(131)
    # ax.scatter(spark_transformed_data[:, 0], spark_transformed_data[:, 1], c= y, cmap='RdPu')
    # ax.set_title("SparkIt Results in 2D (PCA 10000 Samples)")
    # ax.set_xlabel('x') 
    # ax.set_ylabel('y')
    # plt.show()
    
    
