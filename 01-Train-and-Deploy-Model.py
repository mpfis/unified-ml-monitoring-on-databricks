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
import azureml
import azureml.core

# COMMAND ----------

loanDF = spark.read.csv("/databricks-datasets/lending-club-loan-stats/LoanStats_2018Q2.csv", header=True, inferSchema=True)

# COMMAND ----------

cleaned_loanDF = loanDF.withColumn("int_rate_cleaned", split(loanDF.int_rate, "%").getItem(0).cast("float")) \
                                          .drop("int_rate").withColumnRenamed("int_rate_cleaned", "int_rate") \
                                          .withColumn("term_cleaned", split(loanDF.term, " ")[0].cast("integer")).drop("term").withColumnRenamed("term_cleaned", "term") \
                                          .drop("idCol")

# COMMAND ----------

from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

PATH_TO_MLFLOW_EXPERIMENT = "/Shared/UnifiedMLMonitoringWorkshop/LoanRiskClassifierModel"
mlflow.set_experiment(PATH_TO_MLFLOW_EXPERIMENT)

# COMMAND ----------

maxDepth = [10, 15, 20]
maxBins = [25, 30, 10]

for params in param_groups:
  with mlflow.start_run() as run:
    run_id = run.info.run_uuid

    ## configure stages for Pipeline 
    categoricalFeatures = ["grade","sub_grade", "application_type", "purpose"]
    numericFeatures = ["term", "int_rate", "annual_inc", "tax_liens"]
    stages = []
    for categoricalCol in categoricalFeatures:
        stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
        stages += [stringIndexer, encoder]
    labelIndexer = StringIndexer(inputCol="loan_status", outputCol="label")
    stages += [labelIndexer]
    assemblerInputs = [col + "classVec" for col in categoricalFeatures] + numericFeatures
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]
    ## Log Information to Tracking Server
    mlflow.log_param("features", categoricalFeatures+numericFeatures)

    ## Fit Pipeline to Dataset
    from pyspark.ml import Pipeline
    pipeline = Pipeline(stages = stages)
    pipelineModel = pipeline.fit(cleaned_loanDF)
    df = pipelineModel.transform(cleaned_loanDF)
    selectedCols = ['label', 'features'] + categoricalFeatures + numericFeatures
    cleanedAndLabeledDF = df.select(*selectedCols)
    train, test = cleanedAndLabeledDF.randomSplit([0.7, 0.3], seed = 1234)
    ## Log Information to Tracking Server
    mlflow.log_param("seed", 1234)

    # configure and fit model
    rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', maxDepth=, maxBins=)
    rfModel = rf.fit(train)
    predictions = rfModel.transform(test)
    predCols = ['rawPrediction', 'probability', 'prediction']
    mlflow.log_param("maxDepth", maxDepth)
    mlflow.log_param("maxBins", maxDepth)
    mlflow.spark.log_model(rfModel, f"loan_risk_model_{run_id}")

    # measure performance of model
    evaluator = BinaryClassificationEvaluator()
    print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
    mlflow.log_metric("Area Under ROC", float(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

# COMMAND ----------

from azureml.core import Workspace
from azureml.core.webservice import AciWebservice, AksWebservice
from datetime import datetime

# COMMAND ----------

expId = mlflow.get_experiment_by_name(PATH_TO_MLFLOW_EXPERIMENT).experiment_id  
last_run_id = spark.read.format("mlflow-experiment").load(expId).orderBy(col("end_time").desc()).select("run_id").limit(1).collect()[0][0]
model_uri = "runs:/"+last_run_id+"/model"

# COMMAND ----------

workspace_name = ""
subscription_id = ""
resource_group = ""

ws = Workspace.get(name=workspace_name,
               subscription_id=subscription_id,
               resource_group=resource_group)

# COMMAND ----------

service_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1, enable_app_insights=True, collect_model_data=True)

# COMMAND ----------

azure_service, azure_model = mlflow.azureml.deploy(model_uri=model_uri,
                                                   service_name=endpointName,
                                                   workspace=ws,
                                                   deployment_config=service_config,
                                                   synchronous=True,
                                                   tags={"mlflowExperiment":PATH_TO_MLFLOW_EXPERIMENT})

# COMMAND ----------

# Add Scoring URI for the Experiment to the internal table tracking them
now = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
spark.createDataFrame([[PATH_TO_MLFLOW_EXPERIMENT,endpointName,model_uri, now]], 
                      ["experimentName", "endpointName", "scoringURI", "_ts"]) \
              .write.saveAsTable(f"{DB_NAME}.experimentScoringURIs", format="delta", mode="append")
