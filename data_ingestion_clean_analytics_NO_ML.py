# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline de Datos - SIN Machine Learning
# MAGIC
# MAGIC **Este notebook ejecuta TODO el pipeline de datos sin usar MLlib:**
# MAGIC 1. ✅ Ingesta de datos
# MAGIC 2. ✅ Limpieza de datos
# MAGIC 3. ✅ Feature Engineering
# MAGIC 4. ✅ Guardar datos procesados
# MAGIC 5. ❌ NO entrena modelos (evita problemas de permisos)
# MAGIC
# MAGIC **Resultado:** Tabla `air_quality_features` lista para usar
# MAGIC
# MAGIC **Para entrenar modelos:** Usa el archivo `05_model_training_SKLEARN.py`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Descarga de Datos

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
import requests
import pandas as pd
from io import StringIO

# URL del dataset de Beijing PM2.5
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"

print("📥 Descargando dataset de calidad del aire de Beijing...")

# Descargar el archivo
response = requests.get(url)
if response.status_code == 200:
    print("✅ Dataset descargado exitosamente!")
    
    # Convertir a Pandas DataFrame
    df_pandas = pd.read_csv(StringIO(response.text))
    
    # Convertir a Spark DataFrame
    df = spark.createDataFrame(df_pandas)
    
    print(f"\n📊 Registros: {df.count():,}")
    print(f"📊 Columnas: {len(df.columns)}")
else:
    print(f"❌ Error al descargar: {response.status_code}")

# COMMAND ----------

# Renombrar columna con punto
for c in df.columns:
    if "." in c:
        df = df.withColumnRenamed(c, c.replace(".", "_"))

print("✅ Columnas renombradas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Limpieza de Datos

# COMMAND ----------

# Crear timestamp
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

print("✅ Timestamp creado")

# COMMAND ----------

# Filtrar registros donde pm2_5 no es nulo
df_clean = df.filter(F.col('pm2_5').isNotNull())

print(f"✅ Registros limpios: {df_clean.count():,}")

# COMMAND ----------

# Imputar valores nulos con la media
columns_to_impute = ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']

means = {}
for column in columns_to_impute:
    mean_value = df_clean.select(F.mean(F.col(column))).first()[0]
    means[column] = mean_value
    df_clean = df_clean.withColumn(
        column,
        F.when(F.col(column).isNull(), F.lit(mean_value)).otherwise(F.col(column))
    )

print("✅ Valores nulos imputados")

# COMMAND ----------

# Llenar valores nulos en cbwd
df_clean = df_clean.withColumn(
    'cbwd',
    F.when(F.col('cbwd').isNull(), F.lit('cv')).otherwise(F.col('cbwd'))
)

print("✅ Dirección del viento limpia")

# COMMAND ----------

# Crear categorías AQI
df_clean = df_clean.withColumn(
    'aqi_category',
    F.when(F.col('pm2_5') <= 12, 'Good')
    .when(F.col('pm2_5') <= 35.4, 'Moderate')
    .when(F.col('pm2_5') <= 55.4, 'Unhealthy for Sensitive Groups')
    .when(F.col('pm2_5') <= 150.4, 'Unhealthy')
    .when(F.col('pm2_5') <= 250.4, 'Very Unhealthy')
    .otherwise('Hazardous')
)

# Crear AQI numérico
df_clean = df_clean.withColumn(
    'aqi',
    F.when(F.col('pm2_5') <= 12, F.col('pm2_5') * 4.17)
    .when(F.col('pm2_5') <= 35.4, 50 + (F.col('pm2_5') - 12) * 2.13)
    .when(F.col('pm2_5') <= 55.4, 100 + (F.col('pm2_5') - 35.4) * 2.5)
    .when(F.col('pm2_5') <= 150.4, 150 + (F.col('pm2_5') - 55.4) * 1.05)
    .when(F.col('pm2_5') <= 250.4, 200 + (F.col('pm2_5') - 150.4) * 0.5)
    .otherwise(300 + (F.col('pm2_5') - 250.4) * 0.4)
)

print("✅ Categorías AQI creadas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature Engineering

# COMMAND ----------

# Features temporales
df_temporal = df_clean.withColumn('day_of_week', F.dayofweek('timestamp'))
df_temporal = df_temporal.withColumn('day_of_month', F.dayofmonth('timestamp'))
df_temporal = df_temporal.withColumn('week_of_year', F.weekofyear('timestamp'))

df_temporal = df_temporal.withColumn(
    'season',
    F.when(F.col('month').isin(12, 1, 2), 'Winter')
    .when(F.col('month').isin(3, 4, 5), 'Spring')
    .when(F.col('month').isin(6, 7, 8), 'Summer')
    .otherwise('Fall')
)

df_temporal = df_temporal.withColumn(
    'is_weekend',
    F.when(F.col('day_of_week').isin(1, 7), 1).otherwise(0)
)

df_temporal = df_temporal.withColumn(
    'is_rush_hour',
    F.when(F.col('hour').isin(7, 8, 9, 18, 19, 20), 1).otherwise(0)
)

df_temporal = df_temporal.withColumn(
    'is_business_hours',
    F.when(
        (F.col('hour').between(8, 18)) & (~F.col('day_of_week').isin(1, 7)),
        1
    ).otherwise(0)
)

print("✅ Variables temporales creadas")

# COMMAND ----------

# Promedios móviles
from pyspark.sql.window import Window

window_3h = Window.orderBy('timestamp').rowsBetween(-2, 0)
window_6h = Window.orderBy('timestamp').rowsBetween(-5, 0)
window_12h = Window.orderBy('timestamp').rowsBetween(-11, 0)
window_24h = Window.orderBy('timestamp').rowsBetween(-23, 0)

df_rolling = df_temporal.withColumn('pm25_rolling_3h', F.avg('pm2_5').over(window_3h))
df_rolling = df_rolling.withColumn('pm25_rolling_6h', F.avg('pm2_5').over(window_6h))
df_rolling = df_rolling.withColumn('pm25_rolling_12h', F.avg('pm2_5').over(window_12h))
df_rolling = df_rolling.withColumn('pm25_rolling_24h', F.avg('pm2_5').over(window_24h))

df_rolling = df_rolling.withColumn('temp_rolling_6h', F.avg('TEMP').over(window_6h))
df_rolling = df_rolling.withColumn('temp_rolling_24h', F.avg('TEMP').over(window_24h))

df_rolling = df_rolling.withColumn('pres_rolling_6h', F.avg('PRES').over(window_6h))
df_rolling = df_rolling.withColumn('pres_rolling_24h', F.avg('PRES').over(window_24h))

print("✅ Promedios móviles creados")

# COMMAND ----------

# Variables lag
window_lag = Window.orderBy('timestamp')

df_lag = df_rolling.withColumn('pm25_lag_1h', F.lag('pm2_5', 1).over(window_lag))
df_lag = df_lag.withColumn('pm25_lag_3h', F.lag('pm2_5', 3).over(window_lag))
df_lag = df_lag.withColumn('pm25_lag_6h', F.lag('pm2_5', 6).over(window_lag))
df_lag = df_lag.withColumn('pm25_lag_12h', F.lag('pm2_5', 12).over(window_lag))
df_lag = df_lag.withColumn('pm25_lag_24h', F.lag('pm2_5', 24).over(window_lag))

df_lag = df_lag.withColumn('temp_lag_6h', F.lag('TEMP', 6).over(window_lag))
df_lag = df_lag.withColumn('temp_lag_24h', F.lag('TEMP', 24).over(window_lag))
df_lag = df_lag.withColumn('pres_lag_6h', F.lag('PRES', 6).over(window_lag))

df_lag = df_lag.withColumn('pm25_diff_1h', F.col('pm2_5') - F.col('pm25_lag_1h'))
df_lag = df_lag.withColumn('pm25_diff_24h', F.col('pm2_5') - F.col('pm25_lag_24h'))

print("✅ Variables lag creadas")

# COMMAND ----------

# Ratios e interacciones
df_interact = df_lag.withColumn(
    'humidity_ratio',
    F.when(F.col('TEMP') != 0, F.col('DEWP') / F.col('TEMP')).otherwise(0)
)

df_interact = df_interact.withColumn(
    'temp_pres_interaction',
    F.col('TEMP') * F.col('PRES')
)

df_interact = df_interact.withColumn(
    'temp_deviation',
    F.col('TEMP') - F.col('temp_rolling_24h')
)

df_interact = df_interact.withColumn(
    'pm25_trend',
    F.when(F.col('pm25_rolling_24h') != 0, 
           F.col('pm25_rolling_3h') / F.col('pm25_rolling_24h')
    ).otherwise(1)
)

df_interact = df_interact.withColumn(
    'wind_total',
    F.col('Iws') + F.col('Is') + F.col('Ir')
)

print("✅ Interacciones creadas")

# COMMAND ----------

# One-Hot Encoding manual
df_encoded = df_interact

for direction in ['NE', 'NW', 'SE', 'cv']:
    df_encoded = df_encoded.withColumn(
        f'wind_{direction}',
        F.when(F.col('cbwd') == direction, 1).otherwise(0)
    )

for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    df_encoded = df_encoded.withColumn(
        f'season_{season}',
        F.when(F.col('season') == season, 1).otherwise(0)
    )

print("✅ One-Hot Encoding completado")

# COMMAND ----------

# Eliminar nulos generados por lags
df_final = df_encoded.filter(
    F.col('pm25_lag_24h').isNotNull() & 
    F.col('pm25_rolling_24h').isNotNull()
)

print(f"✅ Dataset final: {df_final.count():,} registros")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Guardar Datos Procesados

# COMMAND ----------

# Seleccionar columnas finales
feature_cols = [
    # Variable objetivo
    'pm2_5',
    'aqi',
    'aqi_category',
    
    # Timestamp
    'timestamp',
    
    # Variables meteorológicas originales
    'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir',
    
    # Variables temporales
    'hour', 'day_of_week', 'day_of_month', 'week_of_year',
    'is_weekend', 'is_rush_hour', 'is_business_hours',
    
    # Promedios móviles
    'pm25_rolling_3h', 'pm25_rolling_6h', 'pm25_rolling_12h', 'pm25_rolling_24h',
    'temp_rolling_6h', 'temp_rolling_24h',
    'pres_rolling_6h', 'pres_rolling_24h',
    
    # Variables lag
    'pm25_lag_1h', 'pm25_lag_3h', 'pm25_lag_6h', 'pm25_lag_12h', 'pm25_lag_24h',
    'temp_lag_6h', 'temp_lag_24h', 'pres_lag_6h',
    'pm25_diff_1h', 'pm25_diff_24h',
    
    # Interacciones
    'humidity_ratio', 'temp_pres_interaction', 'temp_deviation', 
    'pm25_trend', 'wind_total',
    
    # One-hot encoding
    'wind_NE', 'wind_NW', 'wind_SE', 'wind_cv',
    'season_Winter', 'season_Spring', 'season_Summer', 'season_Fall'
]

df_model_ready = df_final.select(feature_cols)

print(f"✅ Dataset final preparado:")
print(f"   Registros: {df_model_ready.count():,}")
print(f"   Features: {len(df_model_ready.columns)}")

# COMMAND ----------

# Crear base de datos y guardar tabla
spark.sql("CREATE DATABASE IF NOT EXISTS air_quality_project")
spark.sql("USE air_quality_project")

df_model_ready.write.mode("overwrite").saveAsTable("air_quality_features")

print("✅ Tabla 'air_quality_features' guardada exitosamente!")
print(f"📊 Registros: {spark.table('air_quality_features').count():,}")
print(f"📊 Columnas: {len(spark.table('air_quality_features').columns)}")

# COMMAND ----------

# Verificar tablas
print("\n📋 Tablas en la base de datos:")
spark.sql("SHOW TABLES").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Pipeline Completado
# MAGIC
# MAGIC **Datos Listos:**
# MAGIC - Tabla: `air_quality_features`
# MAGIC - Registros: ~41,000
# MAGIC - Features: 44 columnas
# MAGIC
# MAGIC **Próximos Pasos:**
# MAGIC 1. Para entrenar con **Sklearn**: Usa `05_model_training_SKLEARN.py`
# MAGIC 2. Para análisis exploratorio: Consulta la tabla directamente
# MAGIC 3. Para exportar: `df = spark.table("air_quality_features").toPandas()`
# MAGIC
# MAGIC **Ventaja:** Este pipeline NO usa MLlib, así que NO tendrás problemas de permisos.
