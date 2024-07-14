# Databricks notebook source
import pandas as pd
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, length, regexp_extract, to_date, weekofyear, year, lag, to_timestamp, trim, unix_timestamp, current_date, count, datediff, greatest, avg
from pyspark.sql.types import IntegerType, FloatType, TimestampType
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

# COMMAND ----------

spark = SparkSession.builder.appName("BigData").getOrCreate()

# COMMAND ----------

dbutils.widgets.text("CONTAINER_NAME", "<CONTAINER_NAME>", "BLOB STORAGE CONTAINER NAME")
dbutils.widgets.text("STORAGE_ACCOUNT_NAME", "<STORAGE_ACCOUNT_NAME>", "BLOB STORAGE ACCOUNT NAME")

# COMMAND ----------

container_name = dbutils.widgets.get("CONTAINER_NAME")
storage_account_name = dbutils.widgets.get("STORAGE_ACCOUNT_NAME")
spark.conf.set(f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net", dbutils.secrets.get(scope="bigdata", key="blobkey"))


data_folder = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/"
dbutils.fs.ls(f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/")

# COMMAND ----------

def read_latest_part_csv(data_folder):
    spark = SparkSession.builder.getOrCreate()

    files = dbutils.fs.ls(f"{data_folder}/")
    files_filtered = [file for file in files if file.name.startswith("part")]

    if files:
        latest_file = max(files_filtered, key=lambda x: x.modificationTime).path
        print(latest_file)
    
        df_specific_file = spark.read.csv(latest_file, header=True, multiLine=True)

        return df_specific_file
    else:
        print(f"No se encontró ningún archivo que comience con 'part' en la carpeta: {data_folder}")
        return None

# COMMAND ----------

df_bt_users_transactions = read_latest_part_csv(f"{data_folder}/bt_users_transactions")
df_lk_onboarding = read_latest_part_csv(f"{data_folder}/lk_onboarding")
df_lk_users = read_latest_part_csv(f"{data_folder}/lk_users")

# COMMAND ----------

# MAGIC %md
# MAGIC # Crear Dataframe final para mostrar los casos de onboarding

# COMMAND ----------

df_final = df_lk_users.join(df_lk_onboarding, on="user_id", how="inner")

df_final = df_final.select(
    "user_id",
    "rubro",
    "birth_dt",
    "first_login_dt",
    "habito",
    "habito_dt",
    "activacion",
    "activacion_dt",
    "setup",
    "setup_dt",
    "return",
    "return_dt",
    "transaction_count"
)

df_final.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC # Nueva Columna Para agrupar según edades

# COMMAND ----------

current_year = year(current_date())

df_final = df_final.withColumn(
    "age_group",
    when((current_year - year(col("birth_dt"))) < 18, "Under 18")
    .when((current_year - year(col("birth_dt"))) < 60, "Adult")
    .otherwise("Elderly")
)

df_final.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC # Nueva columna para indicar si es seller o no

# COMMAND ----------

df_final = df_final.withColumn(
    "is_seller",
    when(col("rubro") != 0, "Seller").otherwise("Non-Seller")
)

df_final.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC # Nueva columna para indicar si dropeo la app

# COMMAND ----------

df_final = df_final.withColumn(
    "drop",
    when(
        (col('first_login_dt').isNotNull()) &
        (col('activacion') == 0) &
        (col('habito') == 0) &
        (col('setup') == 0),
        1
    ).otherwise(0)
)
df_final.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC # Nueva columna dias desde el primer inicio de sesión

# COMMAND ----------

df_final = df_final.withColumn("days_since_first_login", datediff(current_date(), col("first_login_dt")))

df_final.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC # Nueva columna dias desde la ultima transacción

# COMMAND ----------

df_final = df_final.withColumn("days_since_last_transaction", datediff(current_date(), greatest(col("habito_dt"), col("activacion_dt"), col("setup_dt"), col("return_dt"), col("first_login_dt"))))

df_final.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC # Nueva columna sin eventos

# COMMAND ----------

df_final = df_final.withColumn('no_events', when((col('activacion') == 0) & (col('setup') == 0) & (col('habito') == 0), 1).otherwise(0))

df_final.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC # Nueva columna dias hasta la primera transaccion

# COMMAND ----------

df_final = df_final.withColumn("days_until_first_transaction", 
                               when(col("activacion_dt").isNull(), -1)
                               .otherwise(datediff(col("activacion_dt"), col("first_login_dt"))))

df_final.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC # Nueva columna nivel de actividad con KMeans

# COMMAND ----------

df_final = df_final.withColumn("transaction_count", col("transaction_count").cast("int"))

features = ['transaction_count', 'days_until_first_transaction', 'days_since_first_login', 'days_since_last_transaction']
assembler = VectorAssembler(inputCols=features, outputCol="features")

df_assembled = assembler.transform(df_final)

kmeans = KMeans(k=3, seed=1)
model = kmeans.fit(df_assembled.select("features"))
predictions = model.transform(df_assembled)

# COMMAND ----------

centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

predictions.select('user_id', 'transaction_count', 'days_until_first_transaction', 'days_since_first_login', 'days_since_last_transaction', 'prediction').show(10)

pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")
pca_model = pca.fit(predictions)
pca_result = pca_model.transform(predictions).select("pcaFeatures", "prediction")

pandas_df = pca_result.toPandas()
pandas_df['pca1'] = pandas_df['pcaFeatures'].apply(lambda x: x[0])
pandas_df['pca2'] = pandas_df['pcaFeatures'].apply(lambda x: x[1])

plt.figure(figsize=(10, 6))
plt.scatter(pandas_df['pca1'], pandas_df['pca2'], c=pandas_df['prediction'], cmap='viridis', marker='o')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Clustering Results (PCA reduced)')
plt.colorbar(label='Cluster')
plt.show()

# COMMAND ----------

pandas_predictions = predictions.select("transaction_count", "days_until_first_transaction", "days_since_first_login", "days_since_last_transaction", "prediction").toPandas()

cluster_0 = pandas_predictions[pandas_predictions['prediction'] == 0]
cluster_1 = pandas_predictions[pandas_predictions['prediction'] == 1]
cluster_2 = pandas_predictions[pandas_predictions['prediction'] == 2]

sns.pairplot(pandas_predictions, hue="prediction", vars=["transaction_count", "days_until_first_transaction", "days_since_first_login", "days_since_last_transaction"])
plt.show()

# COMMAND ----------

df_final = df_final.join(predictions.select('user_id', 'prediction'), on='user_id', how='inner')
df_final = df_final.withColumnRenamed('prediction', 'activity_level')

df_final.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC # Conexion con Azure SQL Database

# COMMAND ----------

jdbc_url = dbutils.secrets.get(scope="bigdata", key="sqljdbcurl")

# COMMAND ----------

# MAGIC %md
# MAGIC # Guardar en la Base De Datos SQL Azure

# COMMAND ----------

df_final.write.jdbc(url=jdbc_url, table="analysis", mode="overwrite")
