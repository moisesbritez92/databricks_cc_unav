# Databricks notebook source
# MAGIC %md
# MAGIC # 🧪 Notebook de Prueba - Conexión a Databricks
# MAGIC 
# MAGIC Este notebook verifica que la conexión con Databricks funciona correctamente.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Verificar Spark Context

# COMMAND ----------

# Verificar que Spark está disponible
print(f"Spark Version: {spark.version}")
print(f"Spark App Name: {spark.sparkContext.appName}")
print(f"Master: {spark.sparkContext.master}")
print("\n✅ Spark está funcionando correctamente!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Crear un DataFrame de Prueba

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# Crear datos de ejemplo sobre calidad del aire
data = [
    ("Madrid", "2024-01-01", 35.5, 20.3, 45.2, "Moderado"),
    ("Barcelona", "2024-01-01", 28.7, 18.5, 38.9, "Bueno"),
    ("Valencia", "2024-01-01", 42.3, 25.6, 52.1, "Dañino"),
    ("Sevilla", "2024-01-01", 31.2, 19.8, 41.5, "Moderado"),
    ("Madrid", "2024-01-02", 38.9, 22.1, 48.7, "Moderado"),
]

schema = StructType([
    StructField("ciudad", StringType(), True),
    StructField("fecha", StringType(), True),
    StructField("pm25", DoubleType(), True),
    StructField("pm10", DoubleType(), True),
    StructField("aqi", DoubleType(), True),
    StructField("categoria", StringType(), True)
])

df = spark.createDataFrame(data, schema)

print("✅ DataFrame creado exitosamente!")
df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Operaciones Básicas con Spark

# COMMAND ----------

# Contar registros
print(f"Total de registros: {df.count()}")

# Estadísticas descriptivas
print("\n📊 Estadísticas de PM2.5:")
df.select("pm25").describe().show()

# Agrupar por ciudad
print("\n🏙️ Promedio de AQI por ciudad:")
df.groupBy("ciudad").agg(
    F.avg("aqi").alias("aqi_promedio"),
    F.max("aqi").alias("aqi_maximo")
).orderBy(F.desc("aqi_promedio")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verificar MLlib

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Preparar features para un modelo simple
assembler = VectorAssembler(
    inputCols=["pm25", "pm10"],
    outputCol="features"
)

df_features = assembler.transform(df)

# Crear un modelo simple
lr = LinearRegression(featuresCol="features", labelCol="aqi")

print("✅ MLlib está disponible y funcionando!")
print(f"Modelo creado: {type(lr).__name__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verificar MLflow

# COMMAND ----------

import mlflow

print(f"MLflow Version: {mlflow.__version__}")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print("\n✅ MLflow está disponible!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Resumen de Verificación
# MAGIC 
# MAGIC Si llegaste hasta aquí sin errores, tu entorno está listo para:
# MAGIC - ✅ Trabajar con Spark DataFrames
# MAGIC - ✅ Realizar transformaciones de datos
# MAGIC - ✅ Entrenar modelos con MLlib
# MAGIC - ✅ Usar MLflow para tracking de experimentos
# MAGIC 
# MAGIC **🎉 ¡Todo listo para comenzar el proyecto de calidad del aire!**
