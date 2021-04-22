# Databricks notebook source
sensorData = spark.table("maxfisher.sensor")

# generate fake sensor data
## add in anomalous data into the "stream"

# using the original training dataset + new data coming in, calcualte dataset drift

# save results to delta lake

# create visual / query in SQLA

# COMMAND ----------

# MAGIC %sql select * from maxfisher.sensor

# COMMAND ----------

# MAGIC %md
# MAGIC https://en.wikipedia.org/wiki/Hellinger_distance
# MAGIC 
# MAGIC 
# MAGIC https://stackoverflow.com/questions/45741850/python-hellinger-formula-explanation
# MAGIC 
# MAGIC 
# MAGIC https://medium.com/@evgeni.dubov/classifying-imbalanced-data-using-hellinger-distance-f6a4330d6f9a

# COMMAND ----------

import random
import time

sample_jsons = [ {
    "columns": [
        "Sensor1",
        "Sensor2",
        "Sensor3",
        "Sensor4"
    ],
    "data": [
        [random.randint(50,80)+random.random(), 
         random.randint(11000,21000)+random.random(),
         random.randint(50,140)+random.random(),	
         random.randint(50,80)+random.random()]
    ]
} for i in range(0, 10000)]


anomalous_jsons = [ {
    "columns": [
        "Sensor1",
        "Sensor2",
        "Sensor3",
        "Sensor4"
    ],
    "data": [
        [random.randint(90,110)+random.random(), 
         random.randint(21500,23000)+random.random(),
         random.randint(145,165)+random.random(),	
         random.randint(87,107)+random.random()]
    ]
} for i in range(0, 100)]

# COMMAND ----------

# - weekly rolling distribution
# - distribution drifted

