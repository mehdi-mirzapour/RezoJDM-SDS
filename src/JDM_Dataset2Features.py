
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
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer, NGram, CountVectorizer, IDF, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import PipelineModel

try:
      sc = ps.SparkContext()
      sc.setLogLevel("ERROR")
      sqlContext = ps.sql.SQLContext(sc)
      print('Created a SparkContext')
except ValueError:
      warnings.warn('SparkContext already exists')

# reload(sys)
# sys.setdefaultencoding('cp1252')

CSV_PATH = "gs://lrec_dataset/Dataset/"

testDataBig = (sqlContext.read.format("csv")
            .option("header", "true")
            .load(CSV_PATH+"test.csv"))

trainDataBig= (sqlContext.read.format("csv")
            .option("header", "true")
            .load(CSV_PATH+"train.csv"))


language_model = sc.textFile('gs://lrec_dataset/cc.fr.300.vec')
language_model=(language_model.map(lambda x: x.split("\n"))
                              .filter(lambda x: x!=['2000000 300'])
                              .map(lambda x: (x[0].split(" ")[0],
                                              Vectors.dense([float(i) for i in x[0].split(" ")[1:]]) )))


from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors


df = sqlContext.createDataFrame(language_model, ["lan_name","features"])

pca = PCA(k=20, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df)

language_model_dataframe = model.transform(df).select(["lan_name","pcaFeatures"])
language_model_dataframe.registerTempTable('language_model_table')


# First Round

x=testDataBig.alias("x")
y=language_model_dataframe.alias("y")

z = (x.join(y, x.source_name == y.lan_name)
                   .drop("lan_name")
                   .withColumnRenamed("pcaFeatures","sourceFeatures"))

w = (z.join(y, z.destination_name == y.lan_name)
                   .drop("lan_name")
                   .withColumnRenamed("pcaFeatures","destinationFeatures"))

w.show()
w.coalesce(1).write.format('json').save(CSV_PATH+"testFeatures.json")

print(w.schema)


# Second Round

x=trainDataBig.alias("x")
y=language_model_dataframe.alias("y")

z = (x.join(y, x.source_name == y.lan_name)
                   .drop("lan_name")
                   .withColumnRenamed("pcaFeatures","sourceFeatures"))

w = (z.join(y, z.destination_name == y.lan_name)
                   .drop("lan_name")
                   .withColumnRenamed("pcaFeatures","destinationFeatures"))

w.show()
w.coalesce(1).write.format('json').save(CSV_PATH+"trainFeatures.json")

print(w.schema)

sc.stop()


