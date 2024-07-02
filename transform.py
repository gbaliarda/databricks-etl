# Databricks notebook source
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, length, regexp_extract, to_date, weekofyear, year, lag, to_timestamp, trim, unix_timestamp, current_date, count
from pyspark.sql.types import IntegerType, FloatType, TimestampType
from pyspark.sql.window import Window
import numpy as np
import matplotlib.dates as mdates

# COMMAND ----------

spark = SparkSession.builder.appName("BigData").getOrCreate()

# COMMAND ----------

dbutils.widgets.text("STORAGE_ACCOUNT_KEY", "<STORAGE_ACCOUNT_KEY>", "STORAGE ACCOUNT KEY")
dbutils.widgets.text("CONTAINER_NAME", "<CONTAINER_NAME>", "BLOB STORAGE CONTAINER NAME")
dbutils.widgets.text("STORAGE_ACCOUNT_NAME", "<STORAGE_ACCOUNT_NAME>", "BLOB STORAGE ACCOUNT NAME")

# COMMAND ----------

storage_account_key = dbutils.widgets.get("STORAGE_ACCOUNT_KEY")
container_name = dbutils.widgets.get("CONTAINER_NAME")
storage_account_name = dbutils.widgets.get("STORAGE_ACCOUNT_NAME")
spark.conf.set(f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net", storage_account_key)


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

df_bt_users_aggregated = df_bt_users_transactions.groupBy("user_id").agg(count("*").alias("transaction_count"))
df_lk_onboarding_with_transactions = df_lk_onboarding.join(df_bt_users_aggregated, on="user_id", how="left")

df_lk_onboarding_with_transactions = df_lk_onboarding_with_transactions.withColumn("transaction_count", when(col("transaction_count").isNull(), "0").otherwise(col("transaction_count")))

# COMMAND ----------

df_final = df_lk_users.join(df_lk_onboarding_with_transactions, on="user_id", how="inner")

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
    when((current_year - year(col("birth_dt"))) < 18, "Adolescent")
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
# MAGIC # Conexion con Azure SQL Database

# COMMAND ----------

dbutils.widgets.text("JDBC_URL", "<JDBC_URL>", "JDBC URL")

# COMMAND ----------

jdbc_url = dbutils.widgets.get("JDBC_URL")

# COMMAND ----------

# MAGIC %md
# MAGIC # Guardar en la Base De Datos SQL Azure

# COMMAND ----------

df_final.write.jdbc(url=jdbc_url, table="analysis", mode="overwrite")
