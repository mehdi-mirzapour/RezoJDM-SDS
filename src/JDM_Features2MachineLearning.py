
import sys
import os
import pyspark as ps
import warnings
import re
import datetime
from pyspark import SparkContext 
from pyspark.sql import SQLContext

from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.types import *
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.functions import udf


from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, LinearSVC, MultilayerPerceptronClassifier, NaiveBayes, DecisionTreeClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.sql.functions import when
from pyspark.ml.feature import VectorAssembler



CSV_PATH_Test = "gs://lrec_dataset/Dataset/testFeatures.json/"
CSV_PATH_Train = "gs://lrec_dataset/Dataset/trainFeatures.json/"

CSV_File_Test = "part-00000-8e99b14a-75b2-4dd0-b953-c94fc02638f5-c000.json"
CSV_File_Train = "part-00000-477754da-c5bf-406c-be15-9cbe50b7b4c8-c000.json"

EXPORT_FILE= "gs://lrec_dataset/Dataset/model_outcome.csv"

relations=['r_agent', 'r_carac', 'r_causatif', 'r_conseq', 'r_has_part', 'r_holo','r_instr', 'r_isa', 'r_lieu', 'r_patient']

try:
      sc = ps.SparkContext()
      sc.setLogLevel("ERROR")
      sqlContext = ps.sql.SQLContext(sc)
      print('Created a SparkContext')
except ValueError:
      warnings.warn('SparkContext already exists')


schema = StructType([
                      StructField("source_name",StringType(),True),
                      StructField("destination_name",StringType(),True),
                      StructField("r_agent",StringType(),True),
                      StructField("r_carac",StringType(),True),
                      StructField("r_causatif",StringType(),True),
                      StructField("r_conseq",StringType(),True),
                      StructField("r_has_part",StringType(),True),
                      StructField("r_holo",StringType(),True),
                      StructField("r_instr",StringType(),True),
                      StructField("r_isa",StringType(),True),
                      StructField("r_lieu",StringType(),True),
                      StructField("r_patient",StringType(),True),
                      StructField("sourceFeatures",VectorUDT(),True),
                      StructField("destinationFeatures",VectorUDT(),True)
                    ])


testData = (sqlContext
               .read
               .schema(schema)
               .format("json")
               .load(CSV_PATH_Test+CSV_File_Test))

trainData = (sqlContext
               .read
               .schema(schema)
               .format("json")
               .load(CSV_PATH_Train+CSV_File_Train))



testData = testData.withColumn("r_isa", \
              when(testData["r_isa"] == "2", "1").otherwise(testData["r_isa"]))

trainData = trainData.withColumn("r_isa", \
              when(trainData["r_isa"] == "2", "1").otherwise(trainData["r_isa"]))





testData = VectorAssembler(
    inputCols=["sourceFeatures", "destinationFeatures"],
    outputCol="allFeatures").transform(testData)

trainData = VectorAssembler(
    inputCols=["sourceFeatures", "destinationFeatures"],
    outputCol="allFeatures").transform(trainData)


for rel in relations:

    testData = testData.withColumn(rel+"_label", testData[rel].cast('int'))
    trainData = trainData.withColumn(rel+"_label", trainData[rel].cast('int'))

# maxDepth=25,numTrees=35

    rf_RF = RandomForestClassifier(labelCol=rel+"_label",featuresCol="allFeatures",maxDepth=30,numTrees=50)
    model_RF = rf_RF.fit(trainData)
    testData = model_RF.transform(testData)
    testData=testData.withColumnRenamed("prediction",rel+"_RF_pred")
    testData = testData.drop(*["rawPrediction","probability"])       
    testData.show(5)

# maxDepth=25

    rf_DT=DecisionTreeClassifier(labelCol=rel+"_label",featuresCol="allFeatures",maxDepth=30)
    model_DT = rf_DT.fit(trainData)
    testData = model_DT.transform(testData)
    testData=testData.withColumnRenamed("prediction",rel+"_DT_pred")
    testData = testData.drop(*["rawPrediction","probability"])       
    testData.show(5)

# maxIter=250


    rf_SVC=LinearSVC(labelCol=rel+"_label",featuresCol="allFeatures", maxIter=300,aggregationDepth=5)
    model_SVC = rf_SVC.fit(trainData)
    testData = model_SVC.transform(testData)
    testData=testData.withColumnRenamed("prediction",rel+"_SVC_pred")
    testData = testData.drop(*["rawPrediction","probability"])       
    testData.show(5)



relations_nested=[ [x+"_label",x+"_RF_pred",x+"_DT_pred",x+"_SVC_pred"] for x in relations]

relations_labels = []
for sublist in relations_nested:
    for item in sublist:
        relations_labels.append(item)


testData.select(relations_labels).coalesce(1).write.option("header", "true").csv(EXPORT_FILE)


sc.stop()


