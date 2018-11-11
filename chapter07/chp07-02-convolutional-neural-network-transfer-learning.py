#!/usr/bin/python

""" chp07-02-convolutional-neural-network-transfer-learning.py: Classify images using Transfer Learning from a pre-trained convolutional neural network """

# (1) Import the required PySpark and Spark Deep Learning libraries
from sparkdl import DeepImageFeaturizer
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.ml.image import ImageSchema
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

__author__ = "Jillur Quddus"
__credits__ = ["Jillur Quddus"]
__version__ = "1.0.0"
_maintainer__ = "Jillur Quddus"
__email__ = "jillur.quddus@keisan.io"
__status__ = "Development"

# (2) Create a Spark Session using the Spark Context instantiated from spark-submit
spark = SparkSession.builder.appName("Convolutional Neural Networks - Transfer Learning - Image Recognition").getOrCreate()

# (3) Load the Plane and Bird images into Spark DataFrames and define a literal label column
path_to_img_directory = '/data/workspaces/jillur.quddus/jupyter/notebooks/Machine-Learning-with-Apache-Spark-QuickStart-Guide/chapter07/data/image-recognition-data'
birds_df = ImageSchema.readImages(path_to_img_directory + "/birds").withColumn("label", lit(0))
planes_df = ImageSchema.readImages(path_to_img_directory + "/planes").withColumn("label", lit(1))

# (4) Create Training and Test DataFrames respectively
planes_train_df, planes_test_df = planes_df.randomSplit([0.75, 0.25], seed=12345)
birds_train_df, birds_test_df = birds_df.randomSplit([0.75, 0.25], seed=12345)
train_df = planes_train_df.unionAll(birds_train_df)
test_df = planes_test_df.unionAll(birds_test_df)

# (5) Transform the Images into Numeric Feature Vectors using Transfer Learning and the pre-trained InceptionV3 Convolutional Neural Network
featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")

# (6) Train a Logistic Regression Model to classify our images
logistic_regression = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")

# (7) Execute the Featurizer and Logistic Regression estimator within a Pipeline to generate the Trained Model
pipeline = Pipeline(stages=[featurizer, logistic_regression])
model = pipeline.fit(train_df)

# (8) Apply the Trained Image Classification Model to the Test DataFrame to make predictions
test_predictions_df = model.transform(test_df)
test_predictions_df.select("image.origin", "prediction").show(truncate=False)

# (9) Compute the accuracy of our Trained Image Classification Model
accuracy_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Accuracy on Test Dataset = %g" % accuracy_evaluator.evaluate(test_predictions_df.select("label", "prediction")))
