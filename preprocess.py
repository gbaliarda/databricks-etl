# Databricks notebook source
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

# Contar filas antes de eliminar duplicados
print("Count before removing duplicates:")
print(f"lk_users: {df_lk_users.count()}")
print(f"bt_users_transactions: {df_bt_users_transactions.count()}")
print(f"lk_onboarding: {df_lk_onboarding.count()}")

# Eliminar duplicados
df_users = df_lk_users.dropDuplicates()
df_transactions = df_bt_users_transactions.dropDuplicates()
df_onboarding = df_lk_onboarding.dropDuplicates()

# Contar filas después de eliminar duplicados
print("Count after removing duplicates:")
print(f"lk_users: {df_users.count()}")
print(f"bt_users_transactions: {df_transactions.count()}")
print(f"lk_onboarding: {df_onboarding.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Renombrar campo id

# COMMAND ----------

# Rename _c0 columns
if '_c0' in df_users.columns:
    df_users = df_users.withColumnRenamed("_c0", "serial_user")
if '_c0' in df_transactions.columns:
    df_transactions = df_transactions.withColumnRenamed("_c0", "serial_transaction")
if '_c0' in df_onboarding.columns:
    df_onboarding = df_onboarding.withColumnRenamed("_c0", "serial_onboarding")

# Show the renamed columns for verification
print("Renamed columns:")
print("Users DataFrame:")
print(df_users.columns)
print("Transactions DataFrame:")
print(df_transactions.columns)
print("Onboarding DataFrame:")
print(df_onboarding.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC # Eliminar columnas innecesarias

# COMMAND ----------

df_users = df_users.select("serial_user", "user_id", "rubro", "birth_dt")
df_onboarding = df_onboarding.select("serial_onboarding", "first_login_dt", "user_id", 
                                         "habito", "habito_dt", "activacion", "activacion_dt", 
                                         "setup", "setup_dt", "return", "return_dt")
print("Users DataFrame:")
print(df_users.columns)
print("Transactions DataFrame:")
print(df_transactions.columns)
print("Onboarding DataFrame:")
print(df_onboarding.columns)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Usuarios que no son seller tendran rubro 0 en vez de None

# COMMAND ----------

# Obtener tipos de rubros presentes en df_users
rubros = df_users.select('rubro').distinct().collect()

# Imprimir los tipos de rubros
print("Tipos de rubros presentes en df_users:")
for row in rubros:
    print(row['rubro'])

# Reemplazar None en rubro con '0' para usuarios que no son seller
df_users = df_users.withColumn("rubro", when(col("rubro").isNull(), "0").otherwise(col("rubro")))
df_users = df_users.withColumn("rubro", col("rubro").cast(IntegerType()))

df_onboarding = df_onboarding.withColumn("habito", col("habito").cast(IntegerType()))

df_users.show(10)  

# COMMAND ----------

df_onboarding.show(10)
print("-----------------------")
df_transactions.show(10)
print("-----------------------")
df_users.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC # Eliminar inconsistencias (1 en habito 0 en activación)

# COMMAND ----------

# Identificar filas con inconsistencias
inconsistent_rows = df_onboarding.filter((col('activacion') == 0) & (col('habito') == 1))

# Contar el número de filas inconsistentes
num_inconsistent_rows = inconsistent_rows.count()
print(f"Número de filas con inconsistencias: {num_inconsistent_rows}")

# Visualizar algunas filas inconsistentes para revisión
inconsistent_rows.show()

# Eliminar filas inconsistentes de df_users utilizando un anti-join
df_onboarding = df_onboarding.join(inconsistent_rows, on='serial_onboarding', how='left_anti')

#Mostarar el número de filas sin esta inconsistencia
print(f"Número de filas sin esta incosistencia: {df_onboarding.count()}")


# COMMAND ----------

# MAGIC %md
# MAGIC # Fecha de Hábito Anterior a Activación

# COMMAND ----------

# Filtrar filas donde habito_dt es anterior a activacion_dt
inconsistent_rows = df_onboarding.filter(col('habito_dt') < col('activacion_dt'))

# Contar el número de filas con esta inconsistencia
num_inconsistent_rows = inconsistent_rows.count()
print(f"Número de filas con fecha de hábito anterior a activación: {num_inconsistent_rows}")

# Mostrar algunas filas inconsistentes para revisión
inconsistent_rows.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Fecha de activación antes que setup

# COMMAND ----------

# Filtrar filas donde activacion_dt es anterior a setup_dt
inconsistent_rows = df_onboarding.filter(col('activacion_dt') < col('setup_dt'))

# Contar el número de filas con esta inconsistencia
num_inconsistent_rows = inconsistent_rows.count()
print(f"Número de filas con activación antes que setup: {num_inconsistent_rows}")

# Mostrar algunas filas inconsistentes para revisión
inconsistent_rows.show()

# Eliminar filas inconsistentes de df_users utilizando un anti-join
df_onboarding = df_onboarding.join(inconsistent_rows, on='serial_onboarding', how='left_anti')

#Mostarar el número de filas sin esta inconsistencia
print(f"Número de filas sin esta incosistencia: {df_onboarding.count()}")

# Verificar los cambios después de eliminar filas inconsistentes
df_onboarding.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 0 Setup 1 Activación

# COMMAND ----------

# Identificar filas con inconsistencias
inconsistent_rows = df_onboarding.filter((col('activacion') == 1) & (col('setup') == 0))

# Contar el número de filas inconsistentes
num_inconsistent_rows = inconsistent_rows.count()
print(f"Número de filas con inconsistencias: {num_inconsistent_rows}")

# Visualizar algunas filas inconsistentes para revisión
inconsistent_rows.show()

# Eliminar filas inconsistentes de df_users utilizando un anti-join
df_onboarding = df_onboarding.join(inconsistent_rows, on='serial_onboarding', how='left_anti')

#Mostarar el número de filas sin esta inconsistencia
print(f"Número de filas sin esta incosistencia: {df_onboarding.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Eliminar fecha de Return antes que first login

# COMMAND ----------

# Filtrar filas donde return_dt es anterior a first_login_dt
inconsistent_return_rows = df_onboarding.filter(col('return_dt') < col('first_login_dt'))

# Contar el número de filas con esta inconsistencia
num_inconsistent_return_rows = inconsistent_return_rows.count()
print(f"Número de filas con return antes que first login: {num_inconsistent_return_rows}")

# Mostrar algunas filas inconsistentes para revisión
inconsistent_return_rows.show()

# Eliminar filas inconsistentes de df_users utilizando un anti-join
df_onboarding = df_onboarding.join(inconsistent_return_rows, on='serial_onboarding', how='left_anti')

# Verificar los cambios después de eliminar filas inconsistentes
print(f"Numero de filas despues de eliminar esta inconsistencia: {df_onboarding.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Cantidad de DROP (Usuarios que no hicieron más nada despues del primer login)

# COMMAND ----------

# Filtrar usuarios con 'first_login_dt' pero sin eventos de 'activacion', 'habito' ni 'setup'
drops = df_onboarding.filter(
    (col('first_login_dt').isNotNull()) &
    (col('activacion') == 0) &
    (col('habito') == 0) &
    (col('setup') == 0)
)

# Contar el número de usuarios que se consideran "DROP"
num_drop_users = drops.count()
print(f"Número de usuarios DROP: {num_drop_users}")

# Mostrar algunas filas de usuarios DROP para revisión
drops.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
