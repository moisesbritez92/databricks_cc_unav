# Databricks notebook source
# MAGIC %md
# MAGIC # 🧹 Data Cleaning & Exploratory Data Analysis
# MAGIC 
# MAGIC ## Dataset: Beijing PM2.5 (UCI Machine Learning Repository)
# MAGIC 
# MAGIC Este notebook continúa después de cargar el dataset de Beijing PM2.5.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Cargar el dataset descargado

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *

# Si ya tienes el DataFrame 'df' de la celda anterior, salta esta celda
# Si no, recarga desde la URL:

import requests
import pandas as pd
from io import StringIO

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"

response = requests.get(url)
df_pandas = pd.read_csv(StringIO(response.text))
df = spark.createDataFrame(df_pandas)

print(f"✅ Dataset cargado: {df.count()} registros")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Renombrar columnas (eliminar puntos)

# COMMAND ----------

# Renombrar columnas que contengan puntos
for c in df.columns:
    if "." in c:
        df = df.withColumnRenamed(c, c.replace(".", "_"))

# Ver columnas después de renombrar
print("📋 Columnas del dataset:")
for col in df.columns:
    print(f"  - {col}")

# COMMAND ----------

# Ver esquema
df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Análisis Exploratorio Inicial

# COMMAND ----------

# Estadísticas descriptivas
display(df.describe())

# COMMAND ----------

# Ver primeros registros
display(df.limit(20))

# COMMAND ----------

# Contar registros totales
total_records = df.count()
print(f"📊 Total de registros: {total_records:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Análisis de Valores Nulos

# COMMAND ----------

# Contar valores nulos por columna
from pyspark.sql.functions import col, count, when, isnan

null_counts = df.select([
    count(when(col(c).isNull() | isnan(c), c)).alias(c) 
    if df.schema[c].dataType in [DoubleType(), FloatType()] 
    else count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
])

# Convertir a formato legible
null_df = null_counts.toPandas().T
null_df.columns = ['null_count']
null_df['null_percentage'] = (null_df['null_count'] / total_records * 100).round(2)
null_df = null_df.sort_values('null_count', ascending=False)

print("❌ Valores nulos por columna:")
print(null_df[null_df['null_count'] > 0])

# COMMAND ----------

# Visualizar porcentaje de nulos
display(
    spark.createDataFrame(
        [(col, int(count), float(pct)) for col, count, pct in 
         zip(null_df.index, null_df['null_count'], null_df['null_percentage']) if count > 0],
        ['columna', 'null_count', 'null_percentage']
    ).orderBy(F.desc('null_percentage'))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Crear columna de fecha completa (timestamp)

# COMMAND ----------

# Combinar year, month, day, hour en un timestamp
df = df.withColumn(
    'timestamp',
    F.to_timestamp(
        F.concat_ws('-', 
            F.col('year').cast('string'),
            F.lpad(F.col('month').cast('string'), 2, '0'),
            F.lpad(F.col('day').cast('string'), 2, '0'),
            F.lpad(F.col('hour').cast('string'), 2, '0')
        ),
        'yyyy-MM-dd-HH'
    )
)

# Verificar
df.select('year', 'month', 'day', 'hour', 'timestamp').show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Limpieza de Datos - Estrategia

# COMMAND ----------

# MAGIC %md
# MAGIC ### Estrategia de limpieza:
# MAGIC 
# MAGIC 1. **PM2.5 (variable objetivo):** Eliminar registros con nulos (crítico para el modelo)
# MAGIC 2. **Variables meteorológicas:** Imputar con la media o forward fill
# MAGIC 3. **Otras variables:** Evaluar caso por caso
# MAGIC 4. **Duplicados:** Verificar y eliminar si existen
# MAGIC 5. **Outliers:** Detectar valores extremos pero mantenerlos (pueden ser reales)

# COMMAND ----------

# Ver registros antes de limpieza
print(f"📊 Registros antes de limpieza: {df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Eliminar registros donde PM2.5 es nulo

# COMMAND ----------

# Filtrar registros donde pm2_5 no es nulo
df_clean = df.filter(col('pm2_5').isNotNull())

print(f"✅ Registros después de eliminar nulos en PM2.5: {df_clean.count()}")
print(f"❌ Registros eliminados: {df.count() - df_clean.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Imputar valores nulos en variables meteorológicas

# COMMAND ----------

# Método alternativo: Calcular la media manualmente e imputar con Spark SQL
# Columnas a imputar (variables meteorológicas)
columns_to_impute = ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']

# Calcular la media de cada columna
print("📊 Calculando medias para imputación:")
means = {}
for column in columns_to_impute:
    mean_value = df_clean.select(F.mean(col(column))).first()[0]
    means[column] = mean_value
    print(f"  {column}: {mean_value:.2f}")

# Imputar valores nulos con la media
df_imputed = df_clean
for column in columns_to_impute:
    df_imputed = df_imputed.withColumn(
        column,
        F.when(col(column).isNull(), F.lit(means[column])).otherwise(col(column))
    )

print("\n✅ Valores nulos imputados con la media")

# COMMAND ----------

# Verificar que no hay nulos en las columnas imputadas
null_check = df_imputed.select([
    count(when(col(c).isNull(), c)).alias(c) 
    for c in columns_to_impute
])

display(null_check)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Manejo de la columna categórica 'cbwd' (dirección del viento)

# COMMAND ----------

# Ver valores únicos de cbwd (combined wind direction)
print("🧭 Direcciones del viento:")
df_imputed.groupBy('cbwd').count().orderBy(F.desc('count')).show()

# COMMAND ----------

# Llenar valores nulos en cbwd con 'Unknown' o la moda
mode_cbwd = df_imputed.groupBy('cbwd').count().orderBy(F.desc('count')).first()[0]

df_imputed = df_imputed.withColumn(
    'cbwd',
    F.when(col('cbwd').isNull(), F.lit('cv')).otherwise(col('cbwd'))
)

print(f"✅ Valores nulos en 'cbwd' reemplazados")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Detectar y analizar outliers

# COMMAND ----------

# Calcular estadísticas para PM2.5
pm25_stats = df_imputed.select('pm2_5').describe().collect()

print("📊 Estadísticas de PM2.5:")
for stat in pm25_stats:
    print(f"  {stat['summary']}: {stat['pm2_5']}")

# COMMAND ----------

# Calcular percentiles y detectar outliers
quantiles = df_imputed.approxQuantile('pm2_5', [0.01, 0.25, 0.5, 0.75, 0.99], 0.01)

q1, median, q3 = quantiles[1], quantiles[2], quantiles[3]
iqr = q3 - q1
lower_bound = q1 - 3 * iqr  # 3*IQR para outliers extremos
upper_bound = q3 + 3 * iqr

print(f"📊 Análisis de outliers (PM2.5):")
print(f"  Q1: {q1:.2f}")
print(f"  Mediana: {median:.2f}")
print(f"  Q3: {q3:.2f}")
print(f"  IQR: {iqr:.2f}")
print(f"  Límite inferior: {lower_bound:.2f}")
print(f"  Límite superior: {upper_bound:.2f}")

outliers = df_imputed.filter((col('pm2_5') < lower_bound) | (col('pm2_5') > upper_bound))
print(f"\n⚠️ Outliers extremos: {outliers.count()} registros ({outliers.count()/df_imputed.count()*100:.2f}%)")

# COMMAND ----------

# Visualizar distribución de PM2.5
display(df_imputed.select('pm2_5').summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Crear categorías de calidad del aire (AQI)

# COMMAND ----------

# Crear categorías basadas en estándares de PM2.5
# https://www.airnow.gov/aqi/aqi-basics/
df_clean_final = df_imputed.withColumn(
    'aqi_category',
    F.when(col('pm2_5') <= 12, 'Good')
    .when(col('pm2_5') <= 35.4, 'Moderate')
    .when(col('pm2_5') <= 55.4, 'Unhealthy for Sensitive Groups')
    .when(col('pm2_5') <= 150.4, 'Unhealthy')
    .when(col('pm2_5') <= 250.4, 'Very Unhealthy')
    .otherwise('Hazardous')
)

# Crear también el valor numérico del AQI
df_clean_final = df_clean_final.withColumn(
    'aqi',
    F.when(col('pm2_5') <= 12, col('pm2_5') * 4.17)
    .when(col('pm2_5') <= 35.4, 50 + (col('pm2_5') - 12) * 2.13)
    .when(col('pm2_5') <= 55.4, 100 + (col('pm2_5') - 35.4) * 2.5)
    .when(col('pm2_5') <= 150.4, 150 + (col('pm2_5') - 55.4) * 1.05)
    .when(col('pm2_5') <= 250.4, 200 + (col('pm2_5') - 150.4) * 0.5)
    .otherwise(300 + (col('pm2_5') - 250.4) * 0.4)
)

# Ver distribución de categorías
print("📊 Distribución por categoría de calidad del aire:")
df_clean_final.groupBy('aqi_category').count().orderBy(F.desc('count')).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Resumen de la limpieza

# COMMAND ----------

print("="*70)
print("📊 RESUMEN DE LIMPIEZA DE DATOS")
print("="*70)
print(f"Registros originales:      {df.count():>10,}")
print(f"Registros después limpieza: {df_clean_final.count():>10,}")
print(f"Registros eliminados:       {df.count() - df_clean_final.count():>10,}")
print(f"Porcentaje retenido:        {df_clean_final.count()/df.count()*100:>10.2f}%")
print("="*70)
print("\n✅ Acciones realizadas:")
print("  1. ✅ Renombrado de columnas (eliminados puntos)")
print("  2. ✅ Creada columna timestamp")
print("  3. ✅ Eliminados registros con PM2.5 nulo")
print("  4. ✅ Imputados valores nulos en variables meteorológicas")
print("  5. ✅ Manejados valores nulos en dirección del viento")
print("  6. ✅ Creadas categorías AQI")
print("  7. ✅ Calculado índice AQI numérico")
print("="*70)

# COMMAND ----------

# Ver esquema final
df_clean_final.printSchema()

# COMMAND ----------

# Ver muestra de datos limpios
display(df_clean_final.select(
    'timestamp', 'pm2_5', 'TEMP', 'PRES', 'DEWP', 'cbwd', 'aqi', 'aqi_category'
).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Guardar datos limpios en el catálogo

# COMMAND ----------

# Asegurar que estamos en la base de datos correcta
spark.sql("CREATE DATABASE IF NOT EXISTS air_quality_project")
spark.sql("USE air_quality_project")

# Guardar tabla limpia
df_clean_final.write.mode("overwrite").saveAsTable("air_quality_clean")

print("✅ Tabla 'air_quality_clean' guardada exitosamente!")
print(f"📊 Total de registros: {spark.table('air_quality_clean').count():,}")

# COMMAND ----------

# Verificar que se guardó correctamente
print("\n📋 Tablas en la base de datos:")
spark.sql("SHOW TABLES").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Siguiente Paso
# MAGIC 
# MAGIC Datos limpios y listos! El próximo notebook será:
# MAGIC - **Feature Engineering & Transformations**
# MAGIC   - Crear variables temporales (día de la semana, mes, estación)
# MAGIC   - Calcular promedios móviles
# MAGIC   - Codificar variables categóricas
# MAGIC   - Normalizar/escalar features
# MAGIC   - Preparar datos para modelado
