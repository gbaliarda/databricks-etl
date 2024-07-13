# Databricks notebook source
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, length, regexp_extract, to_date, weekofyear, year, lag, to_timestamp, trim, unix_timestamp, current_date, count, lit
from pyspark.sql.types import IntegerType, FloatType, TimestampType
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates
from azure.storage.blob import BlobServiceClient

# COMMAND ----------

spark = SparkSession.builder.appName("BigData").getOrCreate()

# COMMAND ----------

dbutils.widgets.text("CONTAINER_NAME", "<CONTAINER_NAME>", "BLOB STORAGE CONTAINER NAME")
dbutils.widgets.text("STORAGE_ACCOUNT_NAME", "<STORAGE_ACCOUNT_NAME>", "BLOB STORAGE ACCOUNT NAME")
dbutils.widgets.text("TARGET_CONTAINER_NAME", "<TARGET_CONTAINER_NAME>", "BLOB STORAGE TARGET CONTAINER NAME")

# COMMAND ----------

# Configuration for Azure Blob Storage 
container_name = dbutils.widgets.get("CONTAINER_NAME")
storage_account_name = dbutils.widgets.get("STORAGE_ACCOUNT_NAME")
spark.conf.set(f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net", dbutils.secrets.get(scope="bigdata", key="blobkey"))


data_folder = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/"
dbutils.fs.ls(f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/")

# COMMAND ----------

df_bt_users_transactions = spark.read.csv(f"{data_folder}/bt_users_transactions.csv", header=True)
df_lk_onboarding = spark.read.csv(f"{data_folder}/lk_onboarding.csv", header=True)
df_lk_users = spark.read.csv(f"{data_folder}/lk_users.csv", header=True, multiLine=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # Eliminar duplicados

# COMMAND ----------

print("Count before removing duplicates:")
print(f"lk_users: {df_lk_users.count()}")
print(f"bt_users_transactions: {df_bt_users_transactions.count()}")
print(f"lk_onboarding: {df_lk_onboarding.count()}")

df_lk_users = df_lk_users.dropDuplicates()
df_bt_users_transactions = df_bt_users_transactions.dropDuplicates()
df_lk_onboarding = df_lk_onboarding.dropDuplicates(["user_id"])

print("Count after removing duplicates:")
print(f"lk_users: {df_lk_users.count()}")
print(f"bt_users_transactions: {df_bt_users_transactions.count()}")
print(f"lk_onboarding: {df_lk_onboarding.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Renombrar campo id

# COMMAND ----------

if '_c0' in df_lk_users.columns:
    df_lk_users = df_lk_users.withColumnRenamed("_c0", "serial_user")
if '_c0' in df_bt_users_transactions.columns:
    df_bt_users_transactions = df_bt_users_transactions.withColumnRenamed("_c0", "serial_transaction")
if '_c0' in df_lk_onboarding.columns:
    df_lk_onboarding = df_lk_onboarding.withColumnRenamed("_c0", "serial_onboarding")

print("Renamed columns:")
print("Users DataFrame:")
print(df_lk_users.columns)
print("Transactions DataFrame:")
print(df_bt_users_transactions.columns)
print("Onboarding DataFrame:")
print(df_lk_onboarding.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC # Eliminar columnas innecesarias

# COMMAND ----------

df_lk_users = df_lk_users.select("serial_user", "user_id", "rubro", "birth_dt")
df_lk_onboarding = df_lk_onboarding.select("serial_onboarding", "first_login_dt", "user_id", 
                                         "habito", "habito_dt", "activacion", "activacion_dt", 
                                         "setup", "setup_dt", "return", "return_dt")
print("Users DataFrame:")
print(df_lk_users.columns)
print("Onboarding DataFrame:")
print(df_lk_onboarding.columns)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Reemplazar habito vacio por 0 

# COMMAND ----------

print("habito values in df_lk_onboarding:")
df_lk_onboarding.select('habito').distinct().show()

df_lk_onboarding = df_lk_onboarding.withColumn("habito", when(col("habito").isNull(), "0").otherwise(col("habito")))
df_lk_onboarding = df_lk_onboarding.withColumn("habito", col("habito").cast(IntegerType()))
df_lk_onboarding.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Columnas temporales sin flag activo

# COMMAND ----------

df_habito = df_lk_onboarding.filter((col('habito_dt').isNotNull()) & (col('habito') == 0))
df_activacion = df_lk_onboarding.filter((col('activacion_dt').isNotNull()) & (col('activacion') == 0))
df_setup = df_lk_onboarding.filter((col('setup_dt').isNotNull()) & (col('setup') == 0))
df_return = df_lk_onboarding.filter((col('return_dt').isNotNull()) & (col('return') == 0))

print("Rows with habito_dt not null and habito 0:")
df_habito.show()

print("Rows with activacion_dt not null and activacion 0:")
df_activacion.show()

print("Rows with setup_dt not null and setup 0:")
df_setup.show()

print("Rows with return_dt not null and return 0:")
df_return.show()

print(f"Count before inconsistencies: {df_lk_onboarding.count()}")
df_lk_onboarding = df_lk_onboarding.subtract(df_habito.union(df_activacion).union(df_setup).union(df_return))

print(f"Count after inconsistencies: {df_lk_onboarding.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Eliminar usuarios sin first login datetime

# COMMAND ----------

df_lk_onboarding = df_lk_onboarding.filter(col('first_login_dt').isNotNull())

print(f"Filas después de eliminar first_login_dt nulos: {df_lk_onboarding.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Eliminar inconsistencias (1 en habito 0 en activación)

# COMMAND ----------

inconsistent_rows = df_lk_onboarding.filter((col('activacion') == 0) & (col('habito') == 1))

num_inconsistent_rows = inconsistent_rows.count()
print(f"Rows with inconsistency: {num_inconsistent_rows}")

inconsistent_rows.show()

df_lk_onboarding = df_lk_onboarding.join(inconsistent_rows, on='serial_onboarding', how='left_anti')

print(f"Rows without inconsistency: {df_lk_onboarding.count()}")


# COMMAND ----------

# MAGIC %md
# MAGIC # Fecha de Hábito Anterior a Activación

# COMMAND ----------

inconsistent_rows = df_lk_onboarding.filter(col('habito_dt') < col('activacion_dt'))

num_inconsistent_rows = inconsistent_rows.count()
print(f"Rows with habit_dt before activacion_dt: {num_inconsistent_rows}")

inconsistent_rows.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Fecha de activación antes que setup

# COMMAND ----------

inconsistent_rows = df_lk_onboarding.filter(col('activacion_dt') < col('setup_dt'))

num_inconsistent_rows = inconsistent_rows.count()
print(f"Rows with activacion_dt before setup_dt: {num_inconsistent_rows}")

inconsistent_rows.show()

df_lk_onboarding = df_lk_onboarding.join(inconsistent_rows, on='serial_onboarding', how='left_anti')

print(f"Rows without this inconsistency: {df_lk_onboarding.count()}")

df_lk_onboarding.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 0 Setup 1 Activación

# COMMAND ----------

inconsistent_rows = df_lk_onboarding.filter((col('activacion') == 1) & (col('setup') == 0))

num_inconsistent_rows = inconsistent_rows.count()
print(f"Rows with inconsistencies: {num_inconsistent_rows}")

df_lk_onboarding = df_lk_onboarding.join(inconsistent_rows, on='serial_onboarding', how='left_anti')

print(f"Rows without this inconsistency:: {df_lk_onboarding.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Eliminar fecha de Return antes que first login

# COMMAND ----------

inconsistent_return_rows = df_lk_onboarding.filter(col('return_dt') < col('first_login_dt'))

num_inconsistent_return_rows = inconsistent_return_rows.count()
print(f"Rows with return_dt before first_login_dt: {num_inconsistent_return_rows}")

df_lk_onboarding = df_lk_onboarding.join(inconsistent_return_rows, on='serial_onboarding', how='left_anti')

print(f"Rows without this inconsistency: {df_lk_onboarding.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Inconsistencias entre cantidad de transacciones y flags

# COMMAND ----------

df_lk_onboarding.show()

# COMMAND ----------

df_bt_users_aggregated = df_bt_users_transactions.groupBy("user_id").agg(count("*").alias("transaction_count"))

df_bt_users_aggregated.show()

df_lk_onboarding.show()

df_lk_onboarding_with_transactions = df_lk_onboarding.join(df_bt_users_aggregated, on="user_id", how="left")

df_lk_onboarding_with_transactions.show()

df_lk_onboarding_with_transactions = df_lk_onboarding_with_transactions.withColumn("transaction_count", when(col("transaction_count").isNull(), "0").otherwise(col("transaction_count")))

df_lk_onboarding_with_transactions.show()

# COMMAND ----------

df_filtered = df_lk_onboarding_with_transactions.filter((col("activacion") == 1) & (col("transaction_count") == 0))

print(f"Amount of rows with activacion but no transaction: {df_filtered.count()}")

df_lk_onboarding = df_lk_onboarding_with_transactions.withColumn(
    "activacion",
    when((col("activacion") == 1) & (col("transaction_count") == 0), 0).otherwise(col("activacion"))
).withColumn(
    "activacion_dt",
    when(~((col("activacion_dt").isNotNull()) & (col("transaction_count") == 0)), col("activacion_dt"))
)

print(f"DataFrame df_lk_onboarding updated:")
df_lk_onboarding.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Guardar dataframes procesados

# COMMAND ----------

target_container_name = dbutils.widgets.get("TARGET_CONTAINER_NAME")
blob_service_client = BlobServiceClient(
    account_url=f"https://{storage_account_name}.blob.core.windows.net",
    credential=dbutils.secrets.get(scope="bigdata", key="blobkey")
)

try:
    container_client = blob_service_client.create_container(target_container_name)
    print(f"Contenedor '{target_container_name}' creado con éxito.")
except Exception as e:
    print(f"El contenedor '{target_container_name}' ya existe o hubo un error: {e}")

# COMMAND ----------

target_data_folder = f"wasbs://{target_container_name}@{storage_account_name}.blob.core.windows.net/"

df_bt_users_transactions.write.csv(f"{target_data_folder}/bt_users_transactions", header=True, mode="overwrite")
df_lk_onboarding.write.csv(f"{target_data_folder}/lk_onboarding", header=True, mode="overwrite")
df_lk_users.write.csv(f"{target_data_folder}/lk_users", header=True, mode="overwrite")


# COMMAND ----------

# MAGIC %md
# MAGIC
