
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


reload(sys)
sys.setdefaultencoding('cp1252')

jdm_dump = sc.textFile('gs://lrec_dataset/09032019-LEXICALNET-JEUXDEMOTS-FR-NOHTML_UTF8.txt')

language_model = sc.textFile('gs://lrec_dataset/cc.fr.300.vec')


#  Generating NodeTypes Dataframe from JDM Dump
print("Generating NodeTypes Dataframe from JDM Dump")
print(datetime.datetime.now())

from pyspark.sql import Row
node_types = [
                 (0,'n_generic'),(1,'n_term'),(2,'n_acception'),(3,'n_definition'),
                 (4,'n_pos'),(5,'n_concept'),(6,'n_flpot'),(7,'n_hub'),
                 (8,'n_chunk'),(9,'n_question'),(10,'n_relation'),(18,'n_data'),
                 (36,'n_data_pot'),(666,'AKI'),(777,'wikipedia')
             ]
rdd = sc.parallelize(node_types)
node_types_dataframe = rdd.map(lambda x: Row(id=int(x[0]), name=x[1]))
node_types_dataframe = sqlContext.createDataFrame(node_types_dataframe)
node_types_dataframe.registerTempTable('node_types_table')
node_types_dataframe.show()


#  Generating RelationTypes Dataframe from JDM Dump
print("Generating RelationTypes Dataframe from JDM Dump")
print(datetime.datetime.now())


def head_eliminator(rel_list):
    return([clean_item(rel_list[0][5:]), 
            clean_item(rel_list[1][5:]), 
            clean_item(rel_list[2][11:]), 
            clean_item(rel_list[3][5:])])


def clean_item(rel_list_item):
    len_item=len(rel_list_item)
    if rel_list_item[len_item-1]=="\"":
        rel_list_item=rel_list_item[0:len_item-1]
    if rel_list_item[0]=="\"":
        rel_list_item=rel_list_item[1:]
    return(rel_list_item.strip())


relation_types=(jdm_dump.map(lambda x: x.split("|")) 
               .filter(lambda x: x[0][:5]=="rtid=")
               .map(lambda x: head_eliminator(x))
       )

from pyspark.sql import Row

relation_types_dataframe = relation_types.map(lambda x: Row(id=int(x[0]), name=x[1],extended_name=x[2], info=x[3]))
relation_types_dataframe = sqlContext.createDataFrame(relation_types_dataframe)
relation_types_dataframe.registerTempTable('relation_types_table')
relation_types_dataframe.show()


#  Generating Nodes Dataframe from JDM Dump
print("Generating Nodes Dataframe from JDM Dump")
print(datetime.datetime.now())


import re
from pyspark.sql import Row

nodes_dataframe=(jdm_dump.map(lambda x: x.split("\n"))
               .filter(lambda x: "eid=" in x[0])
               .filter(lambda x: x[0][0:2]!="//")
               .map(lambda x: Row(id=int(re.findall("eid=(\d+)\|n=", x[0])[0]), 
                                          name=re.findall("n=\"(.+)\"\|t=", x[0])[0],
                                          type=int(re.findall("t=(-?\d+)\|w=", x[0])[0]), 
                                          weight=float(re.findall("w=(-?\d+)", x[0])[0])))
      )

nodes_dataframe = sqlContext.createDataFrame(nodes_dataframe)
nodes_dataframe.registerTempTable('nodes_table')

# Important to discriminate big vs small size datasets
# nodes_dataframe=nodes_dataframe[nodes_dataframe["weight"]>50]

nodes_dataframe.show()



# Generating Relations Dataframe from JDM Dump
print("Generating Relations Dataframe from JDM Dump")
print(datetime.datetime.now())


def head_eliminator(rel_list):
    return([clean_item(rel_list[0][4:]), 
            clean_item(rel_list[1][3:]), 
            clean_item(rel_list[2][3:]), 
            clean_item(rel_list[3][2:]),
            clean_item(rel_list[4][2:])
           ])


def clean_item(rel_list_item):
    len_item=len(rel_list_item)
    if rel_list_item[len_item-1]=="\"":
        rel_list_item=rel_list_item[0:len_item-1]
    if rel_list_item[0]=="\"":
        rel_list_item=rel_list_item[1:]
    return(rel_list_item.strip())


relations=(jdm_dump.map(lambda x: x.split("|")) 
               .filter(lambda x: x[0][:4]=="rid=")
               .map(lambda x: head_eliminator(x))
       )

from pyspark.sql import Row


relations_dataframe = relations.map(lambda x: Row(id=int(x[0]), source=int(x[1]),destination=int(x[2]),type=int(x[3]) ,weight=float(x[4])))
relations_dataframe = sqlContext.createDataFrame(relations_dataframe)
relations_dataframe.registerTempTable('relation_table')
relations_dataframe=relations_dataframe[relations_dataframe["weight"]>24]


relations_dataframe.show()


#  Language Model Words Filtering
print("Language Model Words Filtering")
print(datetime.datetime.now())


language_model=(language_model.map(lambda x: x.split("\n"))
                              .filter(lambda x: x!=['2000000 300'])
                              .map(lambda x: [x[0].split(" ")[0]]))

from pyspark.sql.types import *

mySchema = StructType([StructField("lan_name", StringType(), True)])
language_model_dataframe = sqlContext.createDataFrame(language_model,schema=mySchema)
language_model_dataframe.registerTempTable('language_model_table')

x=nodes_dataframe.alias("x")
y=language_model_dataframe.alias("y")

nodes_dataframe = x.join(y, x.name == y.lan_name).drop("lan_name")


#  Cleaning-up
print("Cleaning-up")
print(datetime.datetime.now())



Node_Types_Filters = [
      1,  # n_term
      2,  # n_form
      4,  # n_pos
      6,  # n_flpot
     18,  # r_data
     36,  # r_data_pot
    666,  # n_AKI
    777  # n_wikipedia
]

Semantic_Relation_Types=[
        1, # r_raff_sem
        6, # r_isa
        9, # r_has_part
        10, # r_holo
        13, # r_agent
        14, # r_patient
        15, # r_lieu
        16, # r_instr
        17, # r_carac
        37, # r_relic_role
        38, # r_agentif_role
        41, # r_conseq
        42, # r_causatif
        49, # r_time
        53, # r_make
        54, # r_product_of
        70, # r_processus>agent
        76, # r_processus>patient
        80, # r_processus>instr
        119, # r_but
]


valid_node_collection=node_types_dataframe[["id"]].collect()
valid_node_list=[valid_node_collection[i].id for i in range(len(valid_node_collection))]

nodes_dataframe=nodes_dataframe[nodes_dataframe['type'].isin(valid_node_list)]
nodes_dataframe=nodes_dataframe[nodes_dataframe['type'].isin(Node_Types_Filters)]



relations_dataframe=relations_dataframe[relations_dataframe['type'].isin(Semantic_Relation_Types)]


relations_dataframe.show()


# Filtering
print("Filtering")
print(datetime.datetime.now())



a=relations_dataframe.alias("a")
b=nodes_dataframe.alias("b")
c=relation_types_dataframe.alias("c")

a=(a.join(b,a.source==b.id)
                       .select("a.id","a.source","b.name","a.destination","a.type","a.weight")
                       .withColumnRenamed("name","source_name"))

a=(a.join(b,a.destination==b.id)
                       .select("a.id","a.source","source_name","a.destination","b.name","a.type","a.weight")
                       .withColumnRenamed("name","destination_name"))

a=(a.join(c,a.type==c.id)
                       .select("a.id","a.source","source_name","a.destination","destination_name","a.type","c.name","a.weight")
                       .withColumnRenamed("name","relation_type_name"))

d=a.groupBy("source_name","destination_name").pivot("relation_type_name").count().na.fill(0)

e=a.join(d,(a.source_name==d.source_name) & (a.destination_name==d.destination_name)).drop(d.source_name).drop(d.destination_name)

e=e.drop("id").drop("source").drop("destination").drop("type").drop("relation_type_name").drop("weight")

e.show()


#  Train-Test Dataset Splitting
print("Train-Test Dataset Splitting")
print(datetime.datetime.now())



from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

train, test = e.randomSplit([0.8, 0.2], seed=1)


# Saving Files into Hard Disk
print("Saving Trainset into Hard Disk")
print(datetime.datetime.now())


(train.repartition(1)
                 .write.format("com.databricks.spark.csv")
                 .option("header", "true")
                 .save("gs://lrec_dataset/train")
                 )

print("Saving Testset into Hard Disk")
print(datetime.datetime.now())

(test.repartition(1)
                 .write.format("com.databricks.spark.csv")
                 .option("header", "true")
                 .save("gs://lrec_dataset/test")
                 )

# print("Saving XLSX Files into Hard Disk")
# print(datetime.datetime.now())

# test.toPandas().to_excel('gs://lrec_dataset/test.xlsx', sheet_name = 'Sheet1', index = True)
# train.toPandas().to_excel('gs://lrec_dataset/train.xlsx', sheet_name = 'Sheet1', index = True)

# print("Task Completion")
# print(datetime.datetime.now())

sc.stop()


