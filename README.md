# CS 512: Distributed Systems - Scikit-Learn on Spark

## Project Overview

This project is a part of the CS 512: Distributed Systems course. The goal is to build a machine learning library with a distributed computing approach using Apache Spark. This will allow for scalable machine learning operations on large datasets.

## Features

- **Distributed Data Processing**: Leverage Spark's distributed computing capabilities to handle large datasets efficiently.
- **Machine Learning Algorithms**: Implement core machine learning algorithms such as linear regression, logistic regression, k-means clustering, and decision trees.
- **User-Friendly API**: Maintain a user-friendly API similar to Scikit-Learn for ease of use.

## Installation

To install the project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/sparkit.git
    ```

2. Navigate to the project directory:

    ```bash
    cd sparkit
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Here is a simple example of how to use the library:

```python
# Create a Spark session
spark = SparkSessionBuilder().get_session()

# Load your data
data = spark.read.csv('/workspaces/sparkit/data/diabetes.csv', header=True, inferSchema=True)

data_rdd = preprocess_data(data)

training_rdd, test_rdd = data_rdd.randomSplit([0.8, 0.2])

# Initialize and train LinearRegression model
model = LinearRegression(w_shape=(len(training_rdd.first().data),), batch_size = 32, b_shape=(1,), lr=0.01)
model.train(training_rdd, num_epochs=10)

```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
