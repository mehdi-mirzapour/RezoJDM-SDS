
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



CSV_PATH_Test = "gs://lrec_dataset/testDataBig.json/"
CSV_PATH_Train = "gs://lrec_dataset/trainDataBig.json/"

CSV_File_Test = "part-00000-9da2ec43-155a-4c78-907a-a67c2e79b6b1-c000.json"
CSV_File_Train = "part-00000-12c761d2-6d82-4c04-88f4-672ea5a1cc0f-c000.json"

EXPORT_FILE= "model_all.csv"

# relations=["r_agent","r_agentif_role","r_carac","r_causatif"]

# relations=["r_isa","r_lieu","r_make","r_patient"]

relations=["r_agent","r_agentif_role","r_carac","r_causatif", "r_conseq", "r_has_part","r_holo","r_instr","r_isa","r_lieu","r_make","r_patient","r_processus>agent","r_processus>patient","r_product_of","r_telic_role","r_time"]

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
                      StructField("r_agentif_role",StringType(),True),
                      StructField("r_carac",StringType(),True),
                      StructField("r_causatif",StringType(),True),
                      StructField("r_conseq",StringType(),True),
                      StructField("r_has_part",StringType(),True),
                      StructField("r_holo",StringType(),True),
                      StructField("r_instr",StringType(),True),
                      StructField("r_isa",StringType(),True),
                      StructField("r_lieu",StringType(),True),
                      StructField("r_make",StringType(),True),
                      StructField("r_patient",StringType(),True),
                      StructField("r_processus>agent",StringType(),True),
                      StructField("r_processus>patient",StringType(),True),
                      StructField("r_product_of",StringType(),True),
                      StructField("r_telic_role",StringType(),True),
                      StructField("r_time",StringType(),True),
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


for rel in relations:

    totalTest=testData.count()
    totalTrain=trainData.count()

    print(testData.groupBy(rel).count().collect())
    print(trainData.groupBy(rel).count().collect())
    print((testData.groupBy(rel).count().collect()[1][1])/totalTest)
    print((trainData.groupBy(rel).count().collect()[1][1])/totalTrain)
    # print(rel+ " distribution in testdata is "+  testData.groupBy(rel).count().collect()[1][1]/totalTest+"in traindata is "+  trainData.groupBy(rel).count().collect()[1][1]/totalTrain)




sc.stop()


