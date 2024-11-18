from pyspark.sql import functions as F
from core.point import LabeledPoint

def preprocess_data(data, feature_columns=None, label_column=None):
    """
    Preprocess the input DataFrame: scales features and converts to RDD of LabeledPoint.

    Parameters:
        data (DataFrame): The input DataFrame with features and labels.
        feature_columns (list): List of feature column names. If None, all except the last column are used.
        label_column (str): Name of the label column. If None, the last column is used.

    Returns:
        RDD[LabeledPoint]: An RDD of LabeledPoint objects with scaled features and labels.
    """
    # Infer feature and label columns if not provided
    if feature_columns is None:
        feature_columns = data.columns[:-1]
    if label_column is None:
        label_column = data.columns[-1]

    # Calculate mean and standard deviation for each feature
    feature_stats = data.select(
        *[F.mean(c).alias(f"{c}_mean") for c in feature_columns],
        *[F.stddev(c).alias(f"{c}_std") for c in feature_columns]
    ).collect()[0]

    # Normalize features manually
    for col in feature_columns:
        mean = feature_stats[f"{col}_mean"]
        std = feature_stats[f"{col}_std"]
        data = data.withColumn(f"{col}_scaled", (F.col(col) - mean) / std)

    # Combine scaled features into vectors manually
    scaled_feature_columns = [f"{col}_scaled" for col in feature_columns]
    data = data.withColumn("features", F.array(*scaled_feature_columns))

    # Convert to RDD of LabeledPoint
    data_rdd = data.rdd.map(lambda row: LabeledPoint(row.features, row[label_column]))

    return data_rdd
