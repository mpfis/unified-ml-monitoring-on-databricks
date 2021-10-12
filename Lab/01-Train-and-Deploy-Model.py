# Databricks notebook source
# MAGIC %md
# MAGIC # Train and Deploy Model using MLFlow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 0: Setup Environment

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
import zipfile
import mlflow
import mlflow.spark
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from mlflow.exceptions import RestException

# COMMAND ----------

# MAGIC %md
# MAGIC **NOTE**: If you get a missing library error for mlflow for the above cell. Create a cell **above** the previous cell with all the library imports and add this:  
# MAGIC `%pip install mlflow` and then try again. If you are still seeing an issue, put in the Lab Live Chat your specific issue.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Upload and Read Sensor Dataset  
# MAGIC 
# MAGIC The dataset that we will use to train the model is provided in the GitHub repository for this workshop. We can simply clone this repository to the cluster itself and read the data file. This makes it easy for us here in the lab environment, but typically our datasets are not this small. This step would usually be a read operation from Azure Data Lake Storage, AWS S3, or Google Cloud Storage (GCS).

# COMMAND ----------

# MAGIC %run ../Lab/00-Setup

# COMMAND ----------

# MAGIC %sh
# MAGIC git clone https://github.com/mpfis/unified-ml-monitoring-on-databricks.git

# COMMAND ----------

with zipfile.ZipFile("/databricks/driver/unified-ml-monitoring-on-databricks/Datasets/sensordata.csv.zip","r") as zip_ref:
    zip_ref.extractall("/databricks/driver/unified-ml-monitoring-on-databricks/Datasets/data/")

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
username = spark.sql("SELECT current_user()").collect()[0][0]
dbutils.fs.cp("file:/databricks/driver/unified-ml-monitoring-on-databricks/Datasets/data/sensordata.csv", f"dbfs:/FileStore/shared_uploads/{username}/sensordata.csv")
sensorData = spark.read.csv(f"dbfs:/FileStore/shared_uploads/{username}/sensordata.csv", header=True, inferSchema=True)
sensorData.write.saveAsTable(f"{DB_NAME}.sensor", format="delta", mode="overwrite")
dataDf = spark.table(f"{DB_NAME}.sensor").where(col('Device') == 'Device001')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Train the Model with MLFlow

# COMMAND ----------

# MAGIC %md
# MAGIC With our sensor data table saved, we can create an MLFlow Experiment to house the metrics that we log during our training runs.  
# MAGIC   
# MAGIC First, within our workspace, we need to create a location for the experiment to be created. Go to a location and create a folder to hold the experiment. Copy the path to the newly-created folder into the `PATH_TO_MLFLOW_EXPERIMENT` variable.  
# MAGIC   
# MAGIC Next, we first establish a `MlflowClient()` which gives us a connection to the MLFlow Tracking Server and allows us to issue commands via the MLFlow API to do things like create experiments, which we do using `client.create_experiment()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create or Load Experiment

# COMMAND ----------

from mlflow.tracking import MlflowClient
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
# MAGIC ### Prepare Data for Training

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
# MAGIC ### Train Model  
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

numEstimators = [10, 15, 20]
maxDepths = [15, 20, 25]

mlflow.sklearn.autolog()

for (numEstimator, maxDepth) in [(numEstimator, maxDepth) for numEstimator in numEstimators for maxDepth in maxDepths]:
  with mlflow.start_run(run_name="Sensor Regression") as run:
    # Fit, train, and score the model
    model = RandomForestRegressor(max_depth = maxDepth, n_estimators = numEstimator)
    model.fit(train_x, train_y)
    preds = model.predict(test_x)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Register Model with MLFlow Model Registry

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the Model in the MLFlow Model Registry

# COMMAND ----------

randID = str(random.randint(1000,5000))
modelName = f"sensor-prediction-model-{randID}"
mlflow.register_model(
    f"runs:/{run.info.run_id}/model",
    modelName
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transition the Model to "Staging"

# COMMAND ----------

client = MlflowClient()
client.transition_model_version_stage(
    name=modelName,
    version=1,
    stage="Staging"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test and Move Model to Production using MLFlow Registry

# COMMAND ----------

# MAGIC %md
# MAGIC In this part of the lab, we will demonstrate a potential flow for testing the model and then transitioning the model to the "Production" state.  
# MAGIC   
# MAGIC We will use this example request to test the model:  
# MAGIC ```json
# MAGIC {"columns": ["Sensor1", "Sensor2", "Sensor3", "Sensor4"], "data": [[70.4843795946883, 18347.021600937504, 51.38152543807783, 71.97184461064535]]}
# MAGIC ```
# MAGIC   
# MAGIC On your own, follow along in the MLFlow Registry by going to the "Models" tab on the left-hand bar of the Databricks Workspace.  
# MAGIC   
# MAGIC After we are done in the registry, we will come back to this notebook.  
# MAGIC   
# MAGIC **NOTE**: All of this can be automated by using the MLFlow APIs.

# COMMAND ----------

# MAGIC %md
# MAGIC ## OPTIONAL: Deploy to Azure Machine Learning Service  
# MAGIC The next few steps will walk you through the deployment of the model to Azure Machine Leaning Service. If you **do not** have an AzureML Service stood up, that is fine. You will be able to complete the lab without one.

# COMMAND ----------

import azureml
import azureml.core
import mlflow.azureml
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
subscription_id = "ENTER SUBSCRIPTION ID"
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
