import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dimension_reduction.main import SparkPCA
from sklearn.decomposition import PCA
from pyspark.sql import SparkSession
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll
import time

from sklearn.decomposition import PCA
from sklearn.datasets import make_swiss_roll
from pyspark.sql import SparkSession
import numpy as np

df = pd.read_csv('data/diabetes.csv') 
#print(df.head)
y = df["Outcome"]
X = df.drop("Outcome", axis=1)

# Run the test
if __name__ == "__main__":
    # test_spark_pca_against_sklearn()
    
    #NOTE: Swiss Roll Data for Mapping Purposes
    start = time.time()
    # load swiss roll dataset
    n_samples = 10000 # please change accordingly
    noise = 0.05

    np.random.seed(2024) # use random seed

    X, y = make_swiss_roll(n_samples, noise=noise)
    end = time.time()
    print("Load and process data:", end - start, "seconds")


    # plot the data colored by class labels
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:,2], c= y, cmap='RdPu')
    ax.set_title("Initial Data in 3D")
    ax.set_xlabel('x');  ax.set_ylabel('y'); ax.set_zlabel('z'); 
    ax.view_init(20, 70)
    
    
    #NOTE: Plotting the sklearn dimension reduction mapping
    pca_results = PCA(n_components=2)
    pca_results = pca_results.fit_transform(X)
    
    # print("Sklearn PCA Components:")
    # print(pca_results.components_)

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(131)
    ax.scatter(pca_results[:, 0], pca_results[:, 1], c= y, cmap='RdPu')
    ax.set_title("SKLearn Results in 2D (PCA 10000 Samples)")
    ax.set_xlabel('x') 
    ax.set_ylabel('y')
    plt.show()
    
    
    #NOTE: Initialize Spark session
    spark = SparkSession.builder.appName("SparkPCATest").getOrCreate()
    sc = spark.sparkContext
    rdd = sc.parallelize(X.tolist()) 

    #NOTE: SparkIt PCA
    spark_pca = SparkPCA(n_components=2)
    spark_pca.fit(rdd)
    spark_transformed_rdd = spark_pca.transform(rdd)
    spark_transformed_data = np.array(spark_transformed_rdd.collect()) 
    
    # print("SparkPCA Components:")
    # print(spark_transformed_data.components)
    
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(131)
    ax.scatter(spark_transformed_data[:, 0], spark_transformed_data[:, 1], c= y, cmap='RdPu')
    ax.set_title("SparkIt Results in 2D (PCA 10000 Samples)")
    ax.set_xlabel('x') 
    ax.set_ylabel('y')
    plt.show()
    
    
