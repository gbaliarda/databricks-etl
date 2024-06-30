# Databricks notebook source
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, regexp_extract, to_date, weekofyear, year, lag, to_timestamp, trim, unix_timestamp
from pyspark.sql.types import IntegerType, FloatType, TimestampType
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates

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

df_bt_users_transactions = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/bt_users_transactions.csv", header=True)
df_lk_onboarding = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/lk_onboarding.csv", header=True)
df_lk_users = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/lk_users.csv", header=True, multiLine=True)

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

df_bt_users_transactions = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/bt_users_transactions.csv", header=True)
df_lk_onboarding = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/lk_onboarding.csv", header=True)
df_lk_users = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/lk_users.csv", header=True, multiLine=True)

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

df_bt_users_transactions = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/bt_users_transactions.csv", header=True)
df_lk_onboarding = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/lk_onboarding.csv", header=True)
df_lk_users = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/lk_users.csv", header=True, multiLine=True)

# COMMAND ----------

# Checks if _c0 is SERIAL.
def check_c0(df):
    df = df.withColumn("_c0_int", col("_c0").cast(IntegerType()))
    window = Window.orderBy("_c0_int")
    df = df.withColumn("prev_c0", lag("_c0_int", 1).over(window))
    df = df.withColumn("diff", col("_c0_int") - col("prev_c0"))
    total = df.filter(col("_c0").isNotNull()).count()
    valid = df.filter((col("_c0").isNotNull()) & ((col("diff") == 1) | (col("prev_c0").isNull()))).count()
    return (valid / total) * 100

# COMMAND ----------

# Check if user starts with MLB
def check_user_id(df):
    total = df.filter(col("user_id").isNotNull()).count()
    valid = df.filter(col("user_id").rlike("^MLB.*$")).count()
    return (valid / total) * 100

# COMMAND ----------

# Check if rubro is a float
def check_rubro(df):
    total = df.filter(col("rubro").isNotNull()).count()
    valid = df.filter(col("rubro").cast(FloatType()).isNotNull()).count()
    return (valid / total) * 100

# COMMAND ----------

# Check if date is well written
def check_date_column(df, column):
    df = df.withColumn("parsed_date", to_timestamp(col(column)))
    
    total = df.filter(col(column).isNotNull()).count()
    valid = df.filter(col("parsed_date").isNotNull()).count()
    
    return (valid / total) * 100 if total > 0 else 100.0

# COMMAND ----------

# Check if type is between 1 and 9
def check_type(df):
    total = df.filter(col("type").isNotNull()).count()
    valid = df.filter(col("type").between(1, 9)).count()
    return (valid / total) * 100

# COMMAND ----------

# Check if segment is 1 or 2
def check_segment(df):
    total = df.filter(col("segment").isNotNull()).count()
    valid = df.filter(col("segment").isin([1, 2])).count()
    return (valid / total) * 100

# COMMAND ----------

# Check if week year is correct
def check_week_year(df):
    df = df.withColumn("first_login_dt", to_timestamp(col("first_login_dt")))
    df = df.withColumn("week_year_actual", weekofyear(col("first_login_dt")))
    total = df.filter(col("week_year").isNotNull() & col("week_year_actual").isNotNull()).count()
    valid = df.filter(col("week_year") == col("week_year_actual")).count()
    return (valid / total) * 100

# COMMAND ----------

# Check if column is 0 or 1
def check_binary_column(df, column):
    df = df.withColumn(column, col(column).cast("integer"))
    
    total = df.filter(col(column).isNotNull()).count()
    valid = df.filter(col(column).isin([0, 1])).count()
    
    return (valid / total) * 100 if total > 0 else 100.0

# COMMAND ----------

def generate_validation_plots(df, df_name):
    functions = validation_functions.get(df_name, {})
    
    percentages = []
    for column, validation_func in functions.items():
        integrity_percentage = validation_func(df)
        percentages.append((column, integrity_percentage))
    
    percentages.sort(key=lambda x: x[1], reverse=True)
    
    columns = [col for col, _ in percentages]
    percentages = [perc for _, perc in percentages]
    
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")

    ax = sns.barplot(x=percentages, y=columns, palette="viridis")

    for p in ax.patches:
        percentage = '{:.1f}%'.format(p.get_width())
        x = p.get_x() + p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.annotate(percentage, (x, y), ha='center', va='center', fontsize=10, color='black')

    ax.set_title(f'Integrity of data by column in {df_name}')
    ax.set_xlabel('Integrity (%)')
    ax.set_ylabel('Column')

    plt.show()

# COMMAND ----------

validation_functions = {
    "df_bt_users_transactions": {
        "_c0": check_c0,
        "user_id": check_user_id,
        "transaction_dt": lambda df: check_date_column(df, "transaction_dt"),
        "type": check_type,
        "segment": check_segment
    },
    "df_lk_onboarding": {
        "_c0": check_c0,
        "user_id": check_user_id,
        "first_login_dt": lambda df: check_date_column(df, "first_login_dt"),
        "habito_dt": lambda df: check_date_column(df, "habito_dt"),
        "activacion_dt": lambda df: check_date_column(df, "activacion_dt"),
        "setup_dt": lambda df: check_date_column(df, "setup_dt"),
        "return_dt": lambda df: check_date_column(df, "return_dt"),
        "week_year": check_week_year,
        "habito": lambda df: check_binary_column(df, "habito"),
        "activacion": lambda df: check_binary_column(df, "activacion"),
        "setup": lambda df: check_binary_column(df, "setup"),
        "return": lambda df: check_binary_column(df, "return")
    },
    "df_lk_users": {
        "_c0": check_c0,
        "user_id": check_user_id,
        "rubro": check_rubro,
        "birth_dt": lambda df: check_date_column(df, "birth_dt")
    }
}

# COMMAND ----------

generate_validation_plots(df_bt_users_transactions, "df_bt_users_transactions")
generate_validation_plots(df_lk_onboarding, "df_lk_onboarding")
generate_validation_plots(df_lk_users, "df_lk_users")

# COMMAND ----------

# MAGIC %md
# MAGIC # Relación entre DFs

# COMMAND ----------

df_bt_users_transactions = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/bt_users_transactions.csv", header=True)
df_lk_onboarding = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/lk_onboarding.csv", header=True)
df_lk_users = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/lk_users.csv", header=True, multiLine=True)

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
# MAGIC # Analisis de frecuencia

# COMMAND ----------

df_bt_users_transactions = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/bt_users_transactions.csv", header=True)
df_lk_onboarding = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/lk_onboarding.csv", header=True)
df_lk_users = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/lk_users.csv", header=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### lk_users

# COMMAND ----------

rubro_data = df_lk_users.filter(df_lk_users["rubro"].isNotNull()).select("rubro").toPandas()

plt.figure(figsize=(10, 6))
sns.histplot(rubro_data["rubro"], bins=20, kde=True)
plt.title("'Rubro' Distribution")
plt.xlabel("Rubro")
plt.ylabel("Count")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### bt_users_transactions

# COMMAND ----------

type_data = df_bt_users_transactions.filter(df_bt_users_transactions["type"].isNotNull()).select("type").toPandas()

plt.figure(figsize=(8, 6))
sns.histplot(type_data["type"], bins=9, kde=True)
plt.title("'Type' Distribution")
plt.xlabel("Type")
plt.ylabel("Count")
plt.show()

# COMMAND ----------

segment_counts = df_bt_users_transactions.groupBy("segment").count().toPandas()

plt.figure(figsize=(8, 6))
sns.barplot(x="segment", y="count", data=segment_counts)
plt.title("'Segment' Distribution")
plt.xlabel("Segment")
plt.ylabel("Count")
plt.show()

# COMMAND ----------

onboarding_data = df_lk_onboarding.select("habito", "activacion", "setup", "return").toPandas()

habit_data = onboarding_data["habito"].value_counts().sort_index()
activation_data = onboarding_data["activacion"].value_counts().sort_index()
setup_data = onboarding_data["setup"].value_counts().sort_index()
return_data = onboarding_data["return"].value_counts().sort_index()

total_habit = habit_data.sum()
percent_habit_0 = (habit_data[0] / total_habit) * 100
percent_habit_1 = (habit_data[1] / total_habit) * 100

total_activation = activation_data.sum()
percent_activation_0 = (activation_data[0] / total_activation) * 100
percent_activation_1 = (activation_data[1] / total_activation) * 100

total_setup = setup_data.sum()
percent_setup_0 = (setup_data[0] / total_setup) * 100
percent_setup_1 = (setup_data[1] / total_setup) * 100

total_return = return_data.sum()
percent_return_0 = (return_data[0] / total_return) * 100
percent_return_1 = (return_data[1] / total_return) * 100

labels = ["habito", "activacion", "setup", "return"]
x = range(len(labels))

plt.figure(figsize=(10, 6))

bar_width = 0.35

bar_positions = [i for i in x]

plt.bar(bar_positions[0], habit_data[0], width=bar_width, label='0', color='lightcyan')
plt.bar(bar_positions[0], habit_data[1], width=bar_width, label='1', color='cyan', bottom=habit_data[0])

plt.bar(bar_positions[1], activation_data[0], width=bar_width, label='0', color='lightgreen')
plt.bar(bar_positions[1], activation_data[1], width=bar_width, label='1', color='green', bottom=activation_data[0])

plt.bar(bar_positions[2], setup_data[0], width=bar_width, label='0', color='lightcoral')
plt.bar(bar_positions[2], setup_data[1], width=bar_width, label='1', color='red', bottom=setup_data[0])

plt.bar(bar_positions[3], return_data[0], width=bar_width, label='0', color='lightyellow')
plt.bar(bar_positions[3], return_data[1], width=bar_width, label='1', color='yellow', bottom=return_data[0])


plt.text(bar_positions[0], habit_data[0] + habit_data[1] / 2, f"{percent_habit_1:.1f}%", ha='center', va='center', color='black')
plt.text(bar_positions[0], habit_data[0] / 2, f"{percent_habit_0:.1f}%", ha='center', va='center', color='black')

plt.text(bar_positions[1], activation_data[0] + activation_data[1] / 2, f"{percent_activation_1:.1f}%", ha='center', va='center', color='black')
plt.text(bar_positions[1], activation_data[0] / 2, f"{percent_activation_0:.1f}%", ha='center', va='center', color='black')

plt.text(bar_positions[2], setup_data[0] + setup_data[1] / 2, f"{percent_setup_1:.1f}%", ha='center', va='center', color='black')
plt.text(bar_positions[2], setup_data[0] / 2, f"{percent_setup_0:.1f}%", ha='center', va='center', color='black')

plt.text(bar_positions[3], return_data[0] + return_data[1] / 2, f"{percent_return_1:.1f}%", ha='center', va='center', color='black')
plt.text(bar_positions[3], return_data[0] / 2, f"{percent_return_0:.1f}%", ha='center', va='center', color='black')

plt.xlabel("Column")
plt.ylabel("Count")
plt.xticks(x, labels)
plt.legend(title="Value")

plt.tight_layout()
plt.show()

# COMMAND ----------

week_year_data = df_lk_onboarding.filter(df_lk_onboarding["week_year"].isNotNull()).select("week_year").toPandas()

plt.figure(figsize=(12, 6))
sns.histplot(week_year_data["week_year"], bins=52, kde=False)
plt.title("'week_year' Distribution")
plt.xlabel("Week Year")
plt.ylabel("Count")
plt.show()

# COMMAND ----------

date_columns = ["first_login_dt", "habito_dt", "activacion_dt", "setup_dt"]

date_counts = {col_name: df_lk_onboarding.select(to_timestamp(col(col_name)).alias(col_name))
               .filter(col(col_name).isNotNull())
               .groupBy(col_name)
               .count()
               .toPandas()
               for col_name in date_columns}

fig, axs = plt.subplots(len(date_columns), figsize=(12, 10), sharex=True)

for i, col_name in enumerate(date_columns):
    data = date_counts[col_name].sort_values(by=col_name)
    
    axs[i].bar(data[col_name], data['count'], width=0.8, edgecolor='black')
    axs[i].set_title(col_name)
    axs[i].set_ylabel("Count")
    axs[i].tick_params(axis='x', rotation=45)

    axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.xlabel("Dates")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Analisis temporales

# COMMAND ----------

df_bt_users_transactions = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/bt_users_transactions.csv", header=True)
df_lk_onboarding = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/lk_onboarding.csv", header=True)
df_lk_users = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/lk_users.csv", header=True, multiLine=True)

# COMMAND ----------

df = df_bt_users_transactions.alias('df_bt') \
    .join(df_lk_onboarding.alias('df_lk_onboarding'), col('df_bt.user_id') == col('df_lk_onboarding.user_id'), 'inner') \
    .join(df_lk_users.alias('df_lk_users'), col('df_bt.user_id') == col('df_lk_users.user_id'), 'inner') \
    .select('df_bt.*', 'df_lk_onboarding.*', 'df_lk_users.*')

df.show()

# COMMAND ----------




# COMMAND ----------

# MAGIC %md
# MAGIC # Analisis de correlacion

# COMMAND ----------

df_bt_users_transactions = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/bt_users_transactions.csv", header=True)
df_lk_onboarding = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/lk_onboarding.csv", header=True)
df_lk_users = spark.read.csv("file:/Workspace/Repos/gbaliarda@itba.edu.ar/databricks-etl/data/lk_users.csv", header=True, multiLine=True)

# COMMAND ----------

df_bt_users_transactions = df_bt_users_transactions.withColumn("transaction_dt_cast", unix_timestamp(to_date("transaction_dt")))
df_bt_users_transactions = df_bt_users_transactions.withColumn("bt_users_transactions_type_cast", df_bt_users_transactions["type"].cast(IntegerType()))
df_bt_users_transactions = df_bt_users_transactions.withColumn("segment_cast", df_bt_users_transactions["segment"].cast(FloatType()))

df_bt_columns_to_correlate = ["transaction_dt_cast", "bt_users_transactions_type_cast", "segment_cast"]

df_bt_correlation_matrix = df_bt_users_transactions.select(df_bt_columns_to_correlate).toPandas().corr(numeric_only=False)

plt.figure(figsize=(10, 8))
sns.heatmap(df_bt_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Matrix - df_bt_users_transactions')
plt.show()

# COMMAND ----------

df_lk_onboarding = df_lk_onboarding.withColumn("first_login_dt_cast", unix_timestamp(to_date("first_login_dt")).cast("double"))
df_lk_onboarding = df_lk_onboarding.withColumn("habito_dt_cast", unix_timestamp(to_date("habito_dt")).cast("double"))
df_lk_onboarding = df_lk_onboarding.withColumn("activacion_dt_cast", unix_timestamp(to_date("activacion_dt")).cast("double"))
df_lk_onboarding = df_lk_onboarding.withColumn("return_dt_cast", unix_timestamp(to_date("return_dt")).cast("double"))
df_lk_onboarding = df_lk_onboarding.withColumn("Unnamed_0_cast", df_lk_onboarding["Unnamed: 0"].cast(IntegerType()))
df_lk_onboarding = df_lk_onboarding.withColumn("week_year_cast", df_lk_onboarding["week_year"].cast(IntegerType()))
df_lk_onboarding = df_lk_onboarding.withColumn("habito_cast", df_lk_onboarding["habito"].cast(IntegerType()))
df_lk_onboarding = df_lk_onboarding.withColumn("activacion_cast", df_lk_onboarding["activacion"].cast(IntegerType()))
df_lk_onboarding = df_lk_onboarding.withColumn("setup_cast", df_lk_onboarding["setup"].cast(IntegerType()))
df_lk_onboarding = df_lk_onboarding.withColumn("return_cast", df_lk_onboarding["return"].cast(IntegerType()))

df_lk_onboarding_columns_to_correlate = [
    "first_login_dt_cast", "habito_dt_cast", "activacion_dt_cast", "return_dt_cast",
    "Unnamed_0_cast", "week_year_cast", "habito_cast", "activacion_cast", "setup_cast", "return_cast"
]

df_lk_onboarding_correlation_matrix = df_lk_onboarding.select(df_lk_onboarding_columns_to_correlate).toPandas().corr()

plt.figure(figsize=(10, 8))
sns.heatmap(df_lk_onboarding_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Matrix - df_lk_onboarding')
plt.show()

# COMMAND ----------

df_lk_users = df_lk_users.withColumn("birth_dt_cast", unix_timestamp(to_date("birth_dt")).cast("double"))
df_lk_users = df_lk_users.withColumn("lk_users_type_cast", df_lk_users["type"].cast(IntegerType()))
df_lk_users = df_lk_users.withColumn("rubro_cast", df_lk_users["rubro"].cast(FloatType()))

df_lk_users_columns_to_correlate = ["birth_dt_cast", "lk_users_type_cast", "rubro_cast"]

df_lk_users_correlation_matrix = df_lk_users.select(df_lk_users_columns_to_correlate).toPandas().corr()

plt.figure(figsize=(10, 8))
sns.heatmap(df_lk_users_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Matrix - df_lk_users')
plt.show()

# COMMAND ----------

df = df_bt_users_transactions.alias('df_bt') \
    .join(df_lk_onboarding.alias('df_lk_onboarding'), col('df_bt.user_id') == col('df_lk_onboarding.user_id'), 'inner') \
    .join(df_lk_users.alias('df_lk_users'), col('df_bt.user_id') == col('df_lk_users.user_id'), 'inner') \
    .select('df_bt.*', 'df_lk_onboarding.*', 'df_lk_users.*')

# COMMAND ----------

cast_columns = [col for col in df.columns if col.endswith("_cast")]

df_correlation_matrix = df.select(df_bt_columns_to_correlate + df_lk_onboarding_columns_to_correlate + df_lk_users_columns_to_correlate).toPandas().corr()

# Visualizar la matriz de correlación como heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Matrix')
plt.show()

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
