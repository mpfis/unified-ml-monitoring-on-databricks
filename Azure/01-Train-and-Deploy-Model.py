# Databricks notebook source
# MAGIC %md
# MAGIC ### Deploy Model as a Web Service in AML
# MAGIC <img src="https://mcg1stanstor00.blob.core.windows.net/images/demos/Ignite/deploywebservice.jpg" alt="Model Deployment" width="800">
# MAGIC </br></br>
# MAGIC The MLFlow model will conainerized and deployed as a web service with AML and Azure Container Instances

# COMMAND ----------

# MAGIC %pip install azureml-mlflow

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
import mlflow
import mlflow.spark
import mlflow.sklearn
import mlflow.azureml
from mlflow.exceptions import RestException
import azureml
import azureml.core

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Upload and Read Sensor Dataset  
# MAGIC   
# MAGIC For the training dataset, you will need to upload some data to the Databricks File System (DBFS). Go to File > Upload Data and click "Browse" in the middle box to bring up your file explorer for your local computer. Navigate to the place where you downloaded the artifacts for this workshop and go into the `/Datasets` folder and choose `sensordata.csv`. Once you see a green checkmark, then you just need to press **Next** and then **Done** on the next screen.

# COMMAND ----------

# MAGIC %md
# MAGIC Here we will be creating a database to store some of the tables that we will create during this workshop. The first table will be a Delta Lake table that will hold our uploaded sensor data.

# COMMAND ----------

DB_NAME = "UMLWorkshop"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
username = spark.sql("SELECT current_user()").collect()[0][0]
sensorData = spark.read.csv(f"dbfs:/FileStore/shared_uploads/{username}/sensordata.csv", header=True, inferSchema=True)
sensorData.write.saveAsTable(f"{DB_NAME}.sensor", format="delta", mode="overwrite")
dataDf = spark.table("sensor").where(col('Device') == 'Device001')

# COMMAND ----------

# MAGIC %md
# MAGIC With our sensor data table saved, we can create an MLFlow Experiment to house the metrics that we log during our training runs.  
# MAGIC   
# MAGIC First, within our workspace, we need to create a location for the experiment to be created. Go to a location and create a folder to hold the experiment. Copy the path to the newly-created folder into the `PATH_TO_MLFLOW_EXPERIMENT` variable.  
# MAGIC   
# MAGIC Next, we first establish a `MlflowClient()` which gives us a connection to the MLFlow Tracking Server and allows us to issue commands via the MLFlow API to do things like create experiments, which we do using `client.create_experiment()`.

# COMMAND ----------

from mlflow.tracking import MlflowClient
PATH_TO_MLFLOW_EXPERIMENT = "ENTER PATH HERE"
client = MlflowClient()
try:
  experiment_id = client.create_experiment(PATH_TO_MLFLOW_EXPERIMENT)
except RestException as r:
  print("Experiment Already Exists, loading experiment...")
except Exception as e:
  print(e)
mlflow.set_experiment(PATH_TO_MLFLOW_EXPERIMENT)

# COMMAND ----------

# MAGIC %md
# MAGIC In this next cell, we take the sensor data and create test and training datasets by splitting up the table using a randomized distribution of data.

# COMMAND ----------

import pandas as pd
import random

#Setup Test/Train datasets
data = dataDf.toPandas()
x = data.drop(["Device", "Time", "Sensor5"], axis=1)
y = data[["Sensor5"]]
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.20, random_state=30)

# COMMAND ----------

# MAGIC %md
# MAGIC With our training and test dataset ready, we are able to run a few training runs for our model.  
# MAGIC   
# MAGIC We are going to run three runs of the `RandomForestClassifier` from sklearn using the following parameters:  
# MAGIC ```python
# MAGIC numEstimators = [10, 15, 20]
# MAGIC maxDepths = [15, 20, 25]
# MAGIC ```  
# MAGIC   
# MAGIC We will also be leveraging the sklearn autologger to log the parameters for each run, as well as key metrics and the trained model itself. The autologging capabilities of MLFlow make it even easier to get started with training your models and tracking your model performance over various training runs. 

# COMMAND ----------

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def fetch_logged_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts

numEstimators = [10, 15, 20]
maxDepths = [15, 20, 25]

mlflow.sklearn.autolog()

for (numEstimator, maxDepth) in [(numTree, maxDepth) for numEstimator in numEstimators for maxDepth in maxDepths]:
  with mlflow.start_run(run_name="Sensor Regression") as run:
    # Fit, train, and score the model
    model = RandomForestRegressor(max_depth = maxDepth, n_estimators = numEstimator)
    model.fit(train_x, train_y)
    preds = model.predict(test_x)

# COMMAND ----------

params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC **Optional: For Those with AzureML**  
# MAGIC   
# MAGIC The next few steps will walk you through the deployment of the model to Azure Machine Leaning Service. If you **do not** have an AzureML Service stood up, that is fine. You will be able to complete the lab without one.

# COMMAND ----------

from azureml.core import Workspace
from azureml.core.webservice import AciWebservice, AksWebservice
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC In this cell, we grab the latest run ID, which is needed to know what model from what run to deploy to AzureML.

# COMMAND ----------

expId = mlflow.get_experiment_by_name(PATH_TO_MLFLOW_EXPERIMENT).experiment_id  
last_run_id = spark.read.format("mlflow-experiment").load(expId).orderBy(col("end_time").desc()).select("run_id").limit(1).collect()[0][0]
model_uri = "runs:/"+last_run_id+"/model"

# COMMAND ----------

# MAGIC %md
# MAGIC Here we establish a connection to the AzureML workspace so that we can deploy the model that is logged in MLFlow.

# COMMAND ----------

workspace_name = "ENTER WORKSPACE NAME"
subscription_id = "ENTER SUBSCRIPTION ID" ## add as secret?
resource_group = "RESOURCE GROUP NAME"

ws = Workspace.get(name=workspace_name,
               subscription_id=subscription_id,
               resource_group=resource_group)

# COMMAND ----------

# MAGIC %md
# MAGIC In this cell, we establish a `Deployment Configuration` in which we enable logging to Application Insights to ensure that we can capture and query model response and trace information for monitoring purposes.

# COMMAND ----------

service_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1, enable_app_insights=True, collect_model_data=True)

# COMMAND ----------

# MAGIC %md
# MAGIC In this last cell, the model from MLFlow and the Deployment Configuration are packaged together, sent to Azure ML, and deployed as an endpoint hosted on an Azure Container Instance.

# COMMAND ----------

azure_service, azure_model = mlflow.azureml.deploy(model_uri=model_uri,
                                                   service_name=endpointName,
                                                   workspace=ws,
                                                   deployment_config=service_config,
                                                   synchronous=True,
                                                   tags={"mlflowExperiment":PATH_TO_MLFLOW_EXPERIMENT})
