from linear_regression import LinearRegression
from utils import SparkSessionBuilder

# Initialize Spark session
spark = SparkSessionBuilder().get_session()

# Load your data
data = spark.read.csv('data.csv', header=True, inferSchema=True)

# Initialize and train the model
model = LinearRegression()
model.train(data)

# Make predictions
predictions = model.predict(data)
### For now this is a numpy array
print(predictions)
