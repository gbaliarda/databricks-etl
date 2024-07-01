# Databricks notebook source
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, length, regexp_extract, to_date, weekofyear, year, lag, to_timestamp, trim, unix_timestamp
from pyspark.sql.types import IntegerType, FloatType, TimestampType
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates

# COMMAND ----------

spark = SparkSession.builder.appName("BigData").getOrCreate()

# COMMAND ----------

dbutils.widgets.text("data_folder", "/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data", "Data Folder")
data_folder = dbutils.widgets.get("data_folder")

# COMMAND ----------

df_bt_users_transactions = spark.read.csv(f"file:{data_folder}/bt_users_transactions.csv", header=True)
df_lk_onboarding = spark.read.csv(f"file:{data_folder}/lk_onboarding.csv", header=True)
df_lk_users = spark.read.csv(f"file:{data_folder}/lk_users.csv", header=True, multiLine=True)

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
# MAGIC ## Usuarios que no son seller tendran rubro 0 en vez de None

# COMMAND ----------

print("Rubro values in df_lk_users:")
df_lk_users.select('rubro').distinct().show()

df_lk_users = df_lk_users.withColumn("rubro", when(col("rubro").isNull(), "0").otherwise(col("rubro")))
df_lk_users = df_lk_users.withColumn("rubro", col("rubro").cast(IntegerType()))

df_lk_onboarding = df_lk_onboarding.withColumn("habito", col("habito").cast(IntegerType()))

df_lk_users.show(10)  

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
# MAGIC # Cantidad de DROP (Usuarios que no hicieron más nada despues del primer login)

# COMMAND ----------

drops = df_lk_onboarding.filter(
    (col('first_login_dt').isNotNull()) &
    (col('activacion') == 0) &
    (col('habito') == 0) &
    (col('setup') == 0)
)

num_drop_users = drops.count()
print(f"Number of drop users: {num_drop_users}")

drops.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocesamiento de los dataframes en conjunto

# COMMAND ----------

# MAGIC %md
# MAGIC
