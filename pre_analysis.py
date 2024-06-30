# Databricks notebook source
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

spark = SparkSession.builder.appName("BigData").getOrCreate()

# COMMAND ----------

df_bt_users_transactions = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/bt_users_transactions.csv", header=True)
df_lk_onboarding = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/lk_onboarding.csv", header=True)
df_lk_users = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/lk_users.csv", header=True)

# COMMAND ----------

df_bt_users_transactions.show()

# COMMAND ----------

df_lk_onboarding.show() 

# COMMAND ----------

df_lk_users.show()

# Note that lk_users CSV contains lots of \n that makes the default read be incorrect.
csv_options = {
    "header": True,
    "multiline": True, 
    "escape": "\"",
}

df_lk_users = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/lk_users.csv", header=True, multiLine=True)
df_lk_users.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Analizar Completitud (datos no nulos)

# COMMAND ----------

def calculate_completeness_percentages(df, columns):
    completeness_percentages = {}
    total_rows = df.count()
    
    # For each column, check how many not nulls elements there are by checking the nulls
    for col_name in columns:
        null_count = df.filter(col(col_name).isNull()).count()
        completeness_percentage = 100 - (null_count / total_rows) * 100
        completeness_percentages[col_name] = completeness_percentage
        
    return completeness_percentages

# COMMAND ----------

def plot_completeness_percentages(completeness_percentages):
    df_completeness_percentages = pd.DataFrame(list(completeness_percentages.items()), columns=['Column', 'Completeness Percentage'])

    df_completeness_percentages = df_completeness_percentages.sort_values(by='Completeness Percentage', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    ax = sns.barplot(x='Completeness Percentage', y='Column', data=df_completeness_percentages, palette="viridis")

    for p in ax.patches:
        percentage = '{:.1f}%'.format(p.get_width())
        x = p.get_x() + p.get_width() + 0.5
        y = p.get_y() + p.get_height() / 2
        ax.annotate(percentage, (x, y), ha='center', va='center', fontsize=10, color='black')


    ax.set_title('Completeness Percentage per Column')
    ax.set_xlabel('Completeness Percentage')
    ax.set_ylabel('Column')

    plt.show()

# COMMAND ----------

df_bt_users_transactions_columns = ["_c0", "user_id", "transaction_dt", "type", "segment"]
completeness_percentages_bt_users_transactions = calculate_completeness_percentages(df_bt_users_transactions, df_bt_users_transactions_columns)
plot_completeness_percentages(completeness_percentages_bt_users_transactions)

df_lk_onboarding_columns = ["_c0", "Unnamed: 0", "first_login_dt", "week_year", "user_id", "habito", "habito_dt", "activacion", "activacion_dt", "setup", "setup_dt", "return", "return_dt"]
completeness_percentages_lk_onboarding = calculate_completeness_percentages(df_lk_onboarding, df_lk_onboarding_columns)
plot_completeness_percentages(completeness_percentages_lk_onboarding)

df_lk_users_columns = ["_c0", "user_id", "name", "email", "address", "birth_dt", "phone", "type", "rubro"]
completeness_percentages_lk_users = calculate_completeness_percentages(df_lk_users, df_lk_users_columns)
plot_completeness_percentages(completeness_percentages_lk_users)

# COMMAND ----------

# MAGIC %md
# MAGIC # Analizar unicidad (datos repetidos)

# COMMAND ----------

def calculate_uniqueness_percentage(df, columns):
    total_rows = df.count()
    distinct_counts = []
    
    for col_name in columns:
        distinct_count = df.select(col_name).distinct().count()
        distinct_counts.append(distinct_count)
    
    uniqueness_percentages = [(count / total_rows) * 100 for count in distinct_counts]
    
    return uniqueness_percentages

# COMMAND ----------

df_bt_users_transactions_columns = ["_c0", "user_id"]
df_lk_onboarding_columns = ["_c0", "user_id"]
df_lk_users_columns = ["_c0", "user_id"]

percentages_bt_users_transactions = calculate_uniqueness_percentage(df_bt_users_transactions, df_bt_users_transactions_columns)
percentages_lk_onboarding = calculate_uniqueness_percentage(df_lk_onboarding, df_lk_onboarding_columns)
percentages_lk_users = calculate_uniqueness_percentage(df_lk_users, df_lk_users_columns)

# COMMAND ----------

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

bars1 = ax1.bar(df_bt_users_transactions_columns, percentages_bt_users_transactions, color='blue')
ax1.set_title('df_bt_users_transactions')
ax1.set_xlabel('Columns')
ax1.set_ylabel('Uniqueness Percentage (%)')

for bar in bars1:
    yval = bar.get_height()
    ax1.annotate(f'{yval:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, yval), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom')

bars2 = ax2.bar(df_lk_onboarding_columns, percentages_lk_onboarding, color='green')
ax2.set_title('df_lk_onboarding')
ax2.set_xlabel('Columns')

for bar in bars2:
    yval = bar.get_height()
    ax2.annotate(f'{yval:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, yval), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom')

bars3 = ax3.bar(df_lk_users_columns, percentages_lk_users, color='orange')
ax3.set_title('df_lk_users')
ax3.set_xlabel('Columns')

for bar in bars3:
    yval = bar.get_height()
    ax3.annotate(f'{yval:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, yval), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Integridad de los datos

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Relación entre DFs

# COMMAND ----------

total_unique_user_ids = (
    df_bt_users_transactions.select("user_id")
    .union(df_lk_onboarding.select("user_id"))
    .union(df_lk_users.select("user_id"))
    .select("user_id")
    .distinct()
    .count()
)

unique_user_ids_bt = df_bt_users_transactions.select("user_id").distinct().count()
unique_user_ids_onboarding = df_lk_onboarding.select("user_id").distinct().count()
unique_user_ids_lk_users = df_lk_users.select("user_id").distinct().count()

# COMMAND ----------

dataframes = ['df_bt_users_transactions', 'df_lk_onboarding', 'df_lk_users']
user_counts = [unique_user_ids_bt, unique_user_ids_onboarding, unique_user_ids_lk_users]

percentages = [count / total_unique_user_ids * 100 for count in user_counts]

plt.figure(figsize=(10, 6))

bars = plt.bar(dataframes, user_counts, color=['blue', 'green', 'orange'])

plt.xlabel('DataFrames')
plt.ylabel('Amount of unique users')
plt.title('Amount of unique users by DataFrame')

plt.axhline(y=total_unique_user_ids, color='red', linestyle='--', linewidth=2, label='Total Unique Users')

plt.text(0.5, total_unique_user_ids + 0.5, f'Total Unique Users: {total_unique_user_ids}', ha='right', va='bottom', color='red')

for i, bar in enumerate(bars):
    percentage = percentages[i]
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10, color='black')


plt.legend()

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Analisis estadístico descriptivo

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Analisis temporales

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Analisis de correlacion

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Analisis de Categorizacion

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Analisis de Anomalías

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
