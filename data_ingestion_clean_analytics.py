# Databricks notebook source
# MAGIC %md
# MAGIC # Datasets de Calidad del Aire
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### **Beijing PM2.5 Data**
# MAGIC - **Fuente:** UCI Machine Learning Repository
# MAGIC - **Link:** https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
# MAGIC - **Tamaño:** ~43K registros
# MAGIC - **Variables:** PM2.5 + datos meteorológicos
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Descargar desde URL directa
# MAGIC
# MAGIC Vamos a usar el dataset de Beijing PM2.5 (UCI) que es público y no requiere autenticación

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *

# COMMAND ----------

# Importar librerías necesarias
import requests
import pandas as pd
from io import StringIO

# URL del dataset de Beijing PM2.5
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"

print("Descargando dataset de calidad del aire de Beijing...")

# Descargar el archivo
response = requests.get(url)
if response.status_code == 200:
    print("Dataset descargado exitosamente!")
    
    # Convertir a Pandas DataFrame
    df_pandas = pd.read_csv(StringIO(response.text))
    
    # Convertir a Spark DataFrame
    df = spark.createDataFrame(df_pandas)
    
    print(f"\nRegistros: {df.count()}")
    print(f"Columnas: {len(df.columns)}")
    print("\nPrimeras filas:")
    df.show(5)
else:
    print(f"Error al descargar: {response.status_code}")

# COMMAND ----------

# Información del dataset
df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Se cambia el nombre de la columna para evitar error de spark en la columna "pm2.5" 

# COMMAND ----------

for c in df.columns:
    if "." in c:
        df = df.withColumnRenamed(c, c.replace(".", "_"))

display(
    df.describe(
        *[col for col in df.columns]
    )
)

# COMMAND ----------

# Contar registros totales
total_records = df.count()
print(f"Total de registros: {total_records:,}")

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

print("Valores nulos por columna:")
print(null_df[null_df['null_count'] > 0])

# COMMAND ----------

# Visualizamos porcentajes de nulos
display(
    spark.createDataFrame(
        [(col, int(count), float(pct)) for col, count, pct in 
         zip(null_df.index, null_df['null_count'], null_df['null_percentage']) if count > 0],
        ['columna', 'null_count', 'null_percentage']
    ).orderBy(F.desc('null_percentage'))
)

# COMMAND ----------

# Combinamos year, month, day, hour en un timestamp
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

# Filtramos registros donde pm2_5 no es nulo
df_clean = df.filter(col('pm2_5').isNotNull())

print(f"Registros después de eliminar nulos en PM2.5: {df_clean.count()}")
print(f"Registros eliminados: {df.count() - df_clean.count()}")

# COMMAND ----------

display(df_clean)

# COMMAND ----------

columns_to_impute = ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']

# Calcular la media de cada columna
print("Se calculan medias para imputación:")
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

print("\nValores nulos imputados con la media")

# COMMAND ----------

# Verificar que no hay nulos en las columnas imputadas
null_check = df_imputed.select([
    count(when(col(c).isNull(), c)).alias(c) 
    for c in columns_to_impute
])

display(null_check)

# COMMAND ----------

print("Direcciones del viento:")
df_imputed.groupBy('cbwd').count().orderBy(F.desc('count')).show()

# COMMAND ----------

# Llenar valores nulos en cbwd con 'Unknown' o la moda
mode_cbwd = df_imputed.groupBy('cbwd').count().orderBy(F.desc('count')).first()[0]

df_imputed = df_imputed.withColumn(
    'cbwd',
    F.when(col('cbwd').isNull(), F.lit('cv')).otherwise(col('cbwd'))
)

print(f"Valores nulos en 'cbwd' reemplazados")

# COMMAND ----------

# Calcular estadísticas para PM2.5
pm25_stats = df_imputed.select('pm2_5').describe().collect()

print("Estadísticas de PM2_5:")
for stat in pm25_stats:
    print(f"  {stat['summary']}: {stat['pm2_5']}")

# COMMAND ----------

# Calculamos percentiles y detectar outliers
quantiles = df_imputed.approxQuantile('pm2_5', [0.01, 0.25, 0.5, 0.75, 0.99], 0.01)

q1, median, q3 = quantiles[1], quantiles[2], quantiles[3]
iqr = q3 - q1
lower_bound = q1 - 3 * iqr  # 3*IQR para outliers extremos
upper_bound = q3 + 3 * iqr

print(f" Análisis de outliers (PM2.5):")
print(f"  Q1: {q1:.2f}")
print(f"  Mediana: {median:.2f}")
print(f"  Q3: {q3:.2f}")
print(f"  IQR: {iqr:.2f}")
print(f"  Límite inferior: {lower_bound:.2f}")
print(f"  Límite superior: {upper_bound:.2f}")

outliers = df_imputed.filter((col('pm2_5') < lower_bound) | (col('pm2_5') > upper_bound))
print(f"\n Outliers extremos: {outliers.count()} registros ({outliers.count()/df_imputed.count()*100:.2f}%)")



# COMMAND ----------

# Visualizar distribución de PM2.5
display(df_imputed.select('pm2_5').summary())

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## ¿Qué se hizo?
# MAGIC
# MAGIC - Se creó una nueva columna llamada **aqi_category** (categórica/texto).
# MAGIC - Se clasificó cada registro según los niveles de PM2.5, basándose en los estándares de la EPA (Environmental Protection Agency de EE.UU.).
# MAGIC - La lógica es similar a un IF-ELSE en cascada.
# MAGIC
# MAGIC | PM2.5 (μg/m³)      | Categoría                              | Significado                                      |
# MAGIC |--------------------|----------------------------------------|--------------------------------------------------|
# MAGIC | 0 - 12             | Good                                   | Aire limpio, sin riesgos                         |
# MAGIC | 12.1 - 35.4        | Moderate                               | Aceptable, pocos riesgos                         |
# MAGIC | 35.5 - 55.4        | Unhealthy for Sensitive Groups         | Personas sensibles pueden tener problemas        |
# MAGIC | 55.5 - 150.4       | Unhealthy                              | Todos pueden empezar a tener problemas           |
# MAGIC | 150.5 - 250.4      | Very Unhealthy                         | Alerta de salud                                  |
# MAGIC | > 250.4            | Hazardous                              | ¡Emergencia! Muy peligroso                       |
# MAGIC

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## ¿Qué hace?
# MAGIC
# MAGIC - Crea una nueva columna llamada **aqi** (numérica).
# MAGIC - Convierte el valor de PM2.5 a un índice estandarizado de 0-500.
# MAGIC - Usa una fórmula de interpolación lineal por tramos.
# MAGIC
# MAGIC ## ¿Por qué es necesario?
# MAGIC
# MAGIC - PM2.5 puede tener valores muy variados (0.5, 50, 200, 500...).
# MAGIC - El AQI normaliza estos valores a una escala estándar (0-500) que es más fácil de entender.
# MAGIC - Permite comparar diferentes contaminantes en la misma escala.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Fórmula de Conversión PM2.5 → AQI
# MAGIC
# MAGIC La fórmula general es:
# MAGIC
# MAGIC \[
# MAGIC \text{AQI} = \frac{I_{high} - I_{low}}{C_{high} - C_{low}} \times (C - C_{low}) + I_{low}
# MAGIC \]
# MAGIC
# MAGIC Donde:
# MAGIC
# MAGIC - **C** = Concentración de PM2.5 (tu valor)
# MAGIC - **C_low**, **C_high** = Límites del rango de concentración
# MAGIC - **I_low**, **I_high** = Límites del rango de AQI correspondiente

# COMMAND ----------

print("Distribución por categoría de calidad del aire:")
df_clean_final.groupBy('aqi_category').count().orderBy(F.desc('count')).show()

# COMMAND ----------

# Ver esquema final
df_clean_final.printSchema()

# COMMAND ----------

# Ver muestra de datos limpios
display(df_clean_final.select(
    'timestamp', 'pm2_5', 'TEMP', 'PRES', 'DEWP', 'cbwd', 'aqi', 'aqi_category'
).limit(20))

# COMMAND ----------

spark.sql("CREATE DATABASE IF NOT EXISTS air_quality_project")
spark.sql("USE air_quality_project")

# Guardar tabla limpia
df_clean_final.write.mode("overwrite").saveAsTable("air_quality_clean")

print("Tabla 'air_quality_clean' guardada exitosamente!")
print(f"Total de registros: {spark.table('air_quality_clean').count():,}")


# Verificar que se guardó correctamente
print("\n📋 Tablas en la base de datos:")
spark.sql("SHOW TABLES").show()

# COMMAND ----------

## Cargamos datos limpios
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# Usar la base de datos del proyecto
spark.sql("USE air_quality_project")

# Cargar tabla limpia
df = spark.table("air_quality_clean")

print(f"Datos cargados: {df.count():,} registros")
print(f"Columnas: {len(df.columns)}")

# COMMAND ----------

# Ver esquema
df.printSchema()

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ### Extraer características temporales del timestamp
# MAGIC
# MAGIC Las variables temporales son importantes porque la contaminación tiene patrones:
# MAGIC - **Hora del día**: Más contaminación en horas pico
# MAGIC - **Día de la semana**: Diferencias entre días laborables y fines de semana
# MAGIC - **Mes/Estación**: Variaciones estacionales
# MAGIC - **Es hora pico**: Variable binaria para horas de tráfico intenso

# COMMAND ----------

# Extraer características temporales
df_temporal = df.withColumn('day_of_week', F.dayofweek('timestamp'))  # 1=Domingo, 7=Sábado
df_temporal = df_temporal.withColumn('day_of_month', F.dayofmonth('timestamp'))
df_temporal = df_temporal.withColumn('week_of_year', F.weekofyear('timestamp'))

# Crear variable de estación del año (basada en el mes)
df_temporal = df_temporal.withColumn(
    'season',
    F.when(F.col('month').isin(12, 1, 2), 'Winter')
    .when(F.col('month').isin(3, 4, 5), 'Spring')
    .when(F.col('month').isin(6, 7, 8), 'Summer')
    .otherwise('Fall')
)

# Variable binaria: es fin de semana
df_temporal = df_temporal.withColumn(
    'is_weekend',
    F.when(F.col('day_of_week').isin(1, 7), 1).otherwise(0)
)

# Variable binaria: es hora pico (7-9am, 6-8pm)
df_temporal = df_temporal.withColumn(
    'is_rush_hour',
    F.when(F.col('hour').isin(7, 8, 9, 18, 19, 20), 1).otherwise(0)
)

# Variable binaria: es horario laboral (8am-6pm, lunes-viernes)
df_temporal = df_temporal.withColumn(
    'is_business_hours',
    F.when(
        (F.col('hour').between(8, 18)) & (~F.col('day_of_week').isin(1, 7)),
        1
    ).otherwise(0)
)

print(" Variables temporales creadas:")
print("  - day_of_week (1-7)")
print("  - day_of_month (1-31)")
print("  - week_of_year (1-52)")
print("  - season (Winter/Spring/Summer/Fall)")
print("  - is_weekend (0/1)")
print("  - is_rush_hour (0/1)")
print("  - is_business_hours (0/1)")

# COMMAND ----------

# Verificar las nuevas columnas
display(df_temporal.select(
    'timestamp', 'hour', 'day_of_week', 'season', 
    'is_weekend', 'is_rush_hour', 'is_business_hours', 'pm2_5'
).limit(20))

# COMMAND ----------

# Ver distribución de contaminación por hora del día
display(
    df_temporal.groupBy('hour')
    .agg(
        F.avg('pm2_5').alias('avg_pm25'),
        F.count('*').alias('count')
    )
    .orderBy('hour')
)

# COMMAND ----------

# Ver distribución por estación
display(
    df_temporal.groupBy('season')
    .agg(
        F.avg('pm2_5').alias('avg_pm25'),
        F.avg('TEMP').alias('avg_temp'),
        F.count('*').alias('count')
    )
    .orderBy('avg_pm25', ascending=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC Arriba pudimos comprobar que la menor temperatura se produce en invierno

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC # ¿Por qué promedios móviles?
# MAGIC   
# MAGIC   Los promedios móviles capturan **tendencias temporales**:
# MAGIC   - Suavizan fluctuaciones de corto plazo
# MAGIC   - Capturan patrones de contaminación persistente
# MAGIC   - Ayudan a predecir valores futuros basándose en el historial reciente
# MAGIC

# COMMAND ----------

# Definir ventana temporal (ordenada por timestamp)
window_3h = Window.orderBy('timestamp').rowsBetween(-2, 0)  # 3 horas (actual + 2 anteriores)
window_6h = Window.orderBy('timestamp').rowsBetween(-5, 0)  # 6 horas
window_12h = Window.orderBy('timestamp').rowsBetween(-11, 0)  # 12 horas
window_24h = Window.orderBy('timestamp').rowsBetween(-23, 0)  # 24 horas

# Calcular promedios móviles de PM2.5
df_rolling = df_temporal.withColumn('pm25_rolling_3h', F.avg('pm2_5').over(window_3h))
df_rolling = df_rolling.withColumn('pm25_rolling_6h', F.avg('pm2_5').over(window_6h))
df_rolling = df_rolling.withColumn('pm25_rolling_12h', F.avg('pm2_5').over(window_12h))
df_rolling = df_rolling.withColumn('pm25_rolling_24h', F.avg('pm2_5').over(window_24h))

# Promedios móviles de temperatura (puede afectar la contaminación)
df_rolling = df_rolling.withColumn('temp_rolling_6h', F.avg('TEMP').over(window_6h))
df_rolling = df_rolling.withColumn('temp_rolling_24h', F.avg('TEMP').over(window_24h))

# Promedios móviles de presión
df_rolling = df_rolling.withColumn('pres_rolling_6h', F.avg('PRES').over(window_6h))
df_rolling = df_rolling.withColumn('pres_rolling_24h', F.avg('PRES').over(window_24h))

print("Promedios móviles creados:")
print("  - pm25_rolling_3h, 6h, 12h, 24h")
print("  - temp_rolling_6h, 24h")
print("  - pres_rolling_6h, 24h")

# COMMAND ----------

# Visualizar promedios móviles
display(df_rolling.select(
    'timestamp', 'pm2_5', 
    'pm25_rolling_3h', 'pm25_rolling_6h', 
    'pm25_rolling_12h', 'pm25_rolling_24h'
).limit(50))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ¿Por qué variables lag?
# MAGIC  
# MAGIC  Los valores anteriores son **predictores poderosos**:
# MAGIC  - La contaminación actual depende de la contaminación reciente
# MAGIC  - Capturan autocorrelación temporal
# MAGIC  - Esenciales para series temporales

# COMMAND ----------

# Definir ventanas para lags
window_lag = Window.orderBy('timestamp')

# Lags de PM2.5 (1, 3, 6, 12, 24 horas atrás)
df_lag = df_rolling.withColumn('pm25_lag_1h', F.lag('pm2_5', 1).over(window_lag))
df_lag = df_lag.withColumn('pm25_lag_3h', F.lag('pm2_5', 3).over(window_lag))
df_lag = df_lag.withColumn('pm25_lag_6h', F.lag('pm2_5', 6).over(window_lag))
df_lag = df_lag.withColumn('pm25_lag_12h', F.lag('pm2_5', 12).over(window_lag))
df_lag = df_lag.withColumn('pm25_lag_24h', F.lag('pm2_5', 24).over(window_lag))

# Lags de temperatura
df_lag = df_lag.withColumn('temp_lag_6h', F.lag('TEMP', 6).over(window_lag))
df_lag = df_lag.withColumn('temp_lag_24h', F.lag('TEMP', 24).over(window_lag))

# Lags de presión
df_lag = df_lag.withColumn('pres_lag_6h', F.lag('PRES', 6).over(window_lag))

# Diferencia con valor anterior (cambio en PM2.5)
df_lag = df_lag.withColumn('pm25_diff_1h', F.col('pm2_5') - F.col('pm25_lag_1h'))
df_lag = df_lag.withColumn('pm25_diff_24h', F.col('pm2_5') - F.col('pm25_lag_24h'))

print("Variables lag creadas:")
print("  - pm25_lag: 1h, 3h, 6h, 12h, 24h")
print("  - temp_lag: 6h, 24h")
print("  - pres_lag: 6h")
print("  - pm25_diff: 1h, 24h (cambios/deltas)")

# COMMAND ----------

# Visualizar lags
display(df_lag.select(
    'timestamp', 'pm2_5', 
    'pm25_lag_1h', 'pm25_lag_6h', 'pm25_lag_24h',
    'pm25_diff_1h', 'pm25_diff_24h'
).limit(50))

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ### ¿Por qué ratios e interacciones?
# MAGIC
# MAGIC Las interacciones capturan **efectos combinados**:
# MAGIC - Temperatura alta + presión baja = más contaminación
# MAGIC - Humedad alta puede afectar la dispersión de contaminantes
# MAGIC - Ratios normalizan valores relativos

# COMMAND ----------

# Ratio de humedad (DEWP/TEMP)
# Valores más altos = mayor humedad relativa
df_interact = df_lag.withColumn(
    'humidity_ratio',
    F.when(F.col('TEMP') != 0, F.col('DEWP') / F.col('TEMP')).otherwise(0)
)

# Interacción temperatura * presión
df_interact = df_interact.withColumn(
    'temp_pres_interaction',
    F.col('TEMP') * F.col('PRES')
)

# Diferencial de temperatura (actual vs promedio 24h)
df_interact = df_interact.withColumn(
    'temp_deviation',
    F.col('TEMP') - F.col('temp_rolling_24h')
)

# Tendencia de PM2.5 (rolling 3h vs rolling 24h)
df_interact = df_interact.withColumn(
    'pm25_trend',
    F.when(F.col('pm25_rolling_24h') != 0, 
           F.col('pm25_rolling_3h') / F.col('pm25_rolling_24h')
    ).otherwise(1)
)

# Velocidad del viento total (suma de componentes)
df_interact = df_interact.withColumn(
    'wind_total',
    F.col('Iws') + F.col('Is') + F.col('Ir')
)

print("Ratios e interacciones creadas:")
print("  - humidity_ratio (DEWP/TEMP)")
print("  - temp_pres_interaction")
print("  - temp_deviation (vs promedio 24h)")
print("  - pm25_trend (rolling 3h / rolling 24h)")
print("  - wind_total (suma de componentes)")

# COMMAND ----------

# Visualizar interacciones
display(df_interact.select(
    'timestamp', 'pm2_5', 'TEMP', 'PRES', 'DEWP',
    'humidity_ratio', 'temp_pres_interaction', 
    'temp_deviation', 'pm25_trend'
).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ### One-Hot Encoding para variables categóras
# MAGIC
# MAGIC Necesario para algoritmos de ML que requieren entrada numérca:
# MAGIC - `cbwd` (dirección del viento): 4 valores → 4 columnas binaias
# MAGIC - `season`: 4 estaciones → 4 columnas binaias
# MAGIC - `aqi_category`: 6 categorías → 6 columnas binarias (para clasificación)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder

# Método 1: One-Hot Encoding manual con Spark SQL (más simple)
# Para dirección del viento (cbwd)
df_encoded = df_interact
for direction in ['NE', 'NW', 'SE', 'cv']:  # cv = calm/variable
    df_encoded = df_encoded.withColumn(
        f'wind_{direction}',
        F.when(F.col('cbwd') == direction, 1).otherwise(0)
    )

# Para estaciones
for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    df_encoded = df_encoded.withColumn(
        f'season_{season}',
        F.when(F.col('season') == season, 1).otherwise(0)
    )

# Para categorías AQI (útil para clasificación)
for category in ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 
                 'Unhealthy', 'Very Unhealthy', 'Hazardous']:
    col_name = category.replace(' ', '_').replace('for_', '').replace('Groups', '').lower()
    df_encoded = df_encoded.withColumn(
        f'aqi_{col_name}',
        F.when(F.col('aqi_category') == category, 1).otherwise(0)
    )

print("Variables categóricas codificadas (One-Hot):")
print("  - wind_NE, wind_NW, wind_SE, wind_cv")
print("  - season_Winter, season_Spring, season_Summer, season_Fall")
print("  - aqi_good, aqi_moderate, aqi_unhealthy_sensitive, aqi_unhealthy, aqi_very_unhealthy, aqi_hazardous")

# COMMAND ----------

# Verificar encoding
display(df_encoded.select(
    'cbwd', 'wind_NE', 'wind_NW', 'wind_SE', 'wind_cv',
    'season', 'season_Winter', 'season_Spring', 'season_Summer', 'season_Fall'
).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Manejo de Valores Nulos Generados
# MAGIC
# MAGIC
# MAGIC
# MAGIC Las operaciones de lag y rolling generan nulos al inicio de la serie temporal.
# MAGIC
# MAGIC **Estrategia**: Eliminar las primeras 24 horas (suficiente para todos los lags)

# COMMAND ----------

print("Valores nulos antes de limpieza:")
null_cols = ['pm25_lag_24h', 'pm25_rolling_24h', 'temp_rolling_24h']
for col_name in null_cols:
    null_count = df_encoded.filter(F.col(col_name).isNull()).count()
    print(f"  {col_name}: {null_count}")


# Eliminar filas con nulos en las variables lag críticas
df_final = df_encoded.filter(
    F.col('pm25_lag_24h').isNotNull() & 
    F.col('pm25_rolling_24h').isNotNull()
)

print(f"\n Registros después de eliminar nulos: {df_final.count():,}")
print(f"Registros eliminados: {df_encoded.count() - df_final.count():,}")

# COMMAND ----------

# Ver lista completa de columnas
print("\n Columnas finales del dataset:")
for i, col in enumerate(df_final.columns, 1):
    print(f"{i:3}. {col}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Seleccionar features relevantes para el modelo
# MAGIC
# MAGIC Eliminaremos columnas redundantes o no necesarias:
# MAGIC - Identificadores (No, year, month, day, hour - ya tenemos timestamp)
# MAGIC - Columnas originales categóricas (ya están codificadas)
# MAGIC - Columnas intermedias

# COMMAND ----------

# Features a INCLUIR en el modelo
feature_cols = [
    # Variable objetivo
    'pm2_5',
    'aqi',
    'aqi_category',
    
    # Timestamp (para ordenamiento y splits temporales)
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

# Seleccionar solo las columnas necesarias
df_model_ready = df_final.select(feature_cols)

print(f"Dataset listo para modelado:")
print(f"   Registros: {df_model_ready.count():,}")
print(f"   Features: {len(df_model_ready.columns)}")


# COMMAND ----------

# Ver muestra final
display(df_model_ready.limit(20))

# COMMAND ----------

# Estadísticas finales
display(df_model_ready.describe())

# COMMAND ----------

# Guardar en el catálogo
df_model_ready.write.mode("overwrite").saveAsTable("air_quality_features")

print("Tabla 'air_quality_features' guardada exitosamente!")
print(f"Registros: {spark.table('air_quality_features').count():,}")
print(f"Columnas: {len(spark.table('air_quality_features').columns)}")


# Verificar tablas en la base de datos
print("\nTablas en la base de datos 'air_quality_project':")
spark.sql("SHOW TABLES").show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Training & Evaluation con MLflow
# MAGIC
# MAGIC Este notebook entrena modelos de Machine Learning usando Spark MLlib y rastrea experimentos con MLflow.
# MAGIC
# MAGIC ## Modelos a entrenar:
# MAGIC 1. **Random Forest Regressor** - Predicción de PM2.5 continuo
# MAGIC 2. **Gradient Boosted Trees Regressor** - Predicción de PM2.5
# MAGIC 3. **Random Forest Classifier** - Clasificación de categorías AQI
# MAGIC
# MAGIC ## Métricas:
# MAGIC - Regresión: RMSE, MAE, R²
# MAGIC - Clasificación: Accuracy, F1-Score, Matriz de confusión

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import mlflow
import mlflow.spark

# Configurar experimento de MLflow
mlflow.set_registry_uri("databricks")
mlflow.set_experiment('/Users/mbritezigle@alumni.unav.es/air_quality_prediction')

print(f"Librerías importadas")
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

# COMMAND ----------

## Cargar Dataset con Features


# Usar la base de datos del proyecto
spark.sql("USE air_quality_project")

# Cargar tabla con features
df = spark.table("air_quality_features")

print(f"Dataset cargado: {df.count():,} registros")
print(f"Columnas: {len(df.columns)}")


# COMMAND ----------

# Definir columnas de features (excluir target y columnas no numéricas)
feature_columns = [
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

print(f"Total de features para el modelo: {len(feature_columns)}")

# COMMAND ----------

try:
    from pyspark.ml.linalg import Vectors, VectorUDT
    MLLIB_AVAILABLE = True
except ImportError:
    MLLIB_AVAILABLE = False

if MLLIB_AVAILABLE:
    from pyspark.sql.functions import udf

    def create_feature_vector(*args):
        return Vectors.dense(args)

    vector_udf = udf(
        create_feature_vector,
        VectorUDT()
    )

    df_assembled = df.withColumn(
        "features",
        vector_udf(*[F.col(c) for c in feature_columns])
    )

    print("Features ensambladas con MLlib")

# COMMAND ----------

display(df_assembled.select('timestamp', 'pm2_5', 'aqi', 'features').limit(5))


# COMMAND ----------

# Cast timestamp to long (seconds since epoch)
df_with_long = df_assembled.withColumn(
    "timestamp_long",
    F.col("timestamp").cast("long")
)

# Calculate 80th percentile on the numeric column
split_long = df_with_long.approxQuantile(
    "timestamp_long",
    [0.8],
    0.01
)[0]

# Convert split_long back to timestamp
split_date = F.from_unixtime(F.lit(split_long)).cast("timestamp")

print(f"Fecha de corte: {split_long}")

# Split temporal
train_df = df_with_long.filter(F.col("timestamp_long") < split_long)
test_df = df_with_long.filter(F.col("timestamp_long") >= split_long)

print(f"\nSplit realizado:")
print(f"   Train: {train_df.count():,} registros ({train_df.count()/df_with_long.count()*100:.1f}%)")
print(f"   Test:  {test_df.count():,} registros ({test_df.count()/df_with_long.count()*100:.1f}%)")

# COMMAND ----------

import mlflow
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# Iniciar run de MLflow
import mlflow
import mlflow.spark

# Iniciar run de MLflow
with mlflow.start_run(run_name="RandomForest_Regressor_v1") as run:
    
    # Parámetros del modelo
    max_depth = 10
    num_trees = 100
    max_bins = 32
    
    # Log de parámetros
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("num_trees", num_trees)
    mlflow.log_param("max_bins", max_bins)
    mlflow.log_param("features_count", len(feature_columns))
    mlflow.log_param("train_count", train_df.count())
    mlflow.log_param("test_count", test_df.count())
    
    # Crear modelo
    rf_regressor = RandomForestRegressor(
        featuresCol="features",
        labelCol="pm2_5",
        maxDepth=max_depth,
        numTrees=num_trees,
        maxBins=max_bins,
        seed=42
    )
    
    print("🌲 Entrenando Random Forest Regressor...")
    
    # Entrenar
    rf_model = rf_regressor.fit(train_df)

    print("Modelo entrenado!")
    
    # Predicciones
    train_predictions = rf_model.transform(train_df)
    test_predictions = rf_model.transform(test_df)
    
    # Evaluadores
    rmse_evaluator = RegressionEvaluator(labelCol="pm2_5", predictionCol="prediction", metricName="rmse")
    mae_evaluator = RegressionEvaluator(labelCol="pm2_5", predictionCol="prediction", metricName="mae")
    r2_evaluator = RegressionEvaluator(labelCol="pm2_5", predictionCol="prediction", metricName="r2")
    
    # Métricas en train
    train_rmse = rmse_evaluator.evaluate(train_predictions)
    train_mae = mae_evaluator.evaluate(train_predictions)
    train_r2 = r2_evaluator.evaluate(train_predictions)
    
    # Métricas en test
    test_rmse = rmse_evaluator.evaluate(test_predictions)
    test_mae = mae_evaluator.evaluate(test_predictions)
    test_r2 = r2_evaluator.evaluate(test_predictions)
    
    # Log de métricas
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("test_r2", test_r2)
    
    # Feature importance
    feature_importance = rf_model.featureImportances.toArray()
    
    # Log del modelo
    mlflow.spark.log_model(rf_model, "random_forest_model")
    
    print("\n" + "="*70)
    print("RESULTADOS - RANDOM FOREST REGRESSOR")
    print("="*70)
    print(f"TRAIN:")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  R²:   {train_r2:.4f}")
    print(f"\nTEST:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    print("="*70)
    
    # Guardar run_id para referencia
    rf_run_id = run.info.run_id
    print(f"\n✅ Experimento registrado en MLflow: {rf_run_id}")

# COMMAND ----------




# COMMAND ----------

# Ver predicciones
display(test_predictions.select('timestamp', 'pm2_5', 'prediction', 'aqi').limit(50))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Importance

# COMMAND ----------

import pandas as pd
import numpy as np

# Crear DataFrame con feature importance
importance_data = list(zip(feature_columns, rf_model.featureImportances.toArray()))
importance_df = pd.DataFrame(importance_data, columns=['feature', 'importance'])
importance_df = importance_df.sort_values('importance', ascending=False).head(20)

print("🎯 Top 20 Features más importantes:")
print(importance_df.to_string(index=False))

# COMMAND ----------

# Visualizar feature importance
display(spark.createDataFrame(importance_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Modelo 2: Gradient Boosted Trees Regressor

# COMMAND ----------

# Iniciar run de MLflow
with mlflow.start_run(run_name="GBT_Regressor_v1") as run:
    
    # Parámetros del modelo
    max_depth = 5
    max_iter = 50
    max_bins = 32
    
    # Log de parámetros
    mlflow.log_param("model_type", "GBTRegressor")
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("max_bins", max_bins)
    mlflow.log_param("features_count", len(feature_columns))
    mlflow.log_param("train_count", train_df.count())
    mlflow.log_param("test_count", test_df.count())
    
    # Crear modelo
    gbt_regressor = GBTRegressor(
        featuresCol="features",
        labelCol="pm2_5",
        maxDepth=max_depth,
        maxIter=max_iter,
        maxBins=max_bins,
        seed=42
    )
    
    print("🚀 Entrenando Gradient Boosted Trees Regressor...")
    
    # Entrenar
    gbt_model = gbt_regressor.fit(train_df)

    print("Modelo entrenado!")

    # Predicciones
    train_predictions = gbt_model.transform(train_df)
    test_predictions = gbt_model.transform(test_df)
    
    # Métricas en train
    train_rmse = rmse_evaluator.evaluate(train_predictions)
    train_mae = mae_evaluator.evaluate(train_predictions)
    train_r2 = r2_evaluator.evaluate(train_predictions)
    
    # Métricas en test
    test_rmse = rmse_evaluator.evaluate(test_predictions)
    test_mae = mae_evaluator.evaluate(test_predictions)
    test_r2 = r2_evaluator.evaluate(test_predictions)
    
    # Log de métricas
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("test_r2", test_r2)
    
    # Log del modelo
    mlflow.spark.log_model(gbt_model, "gbt_model")
    
    print("\n" + "="*70)
    print("RESULTADOS - GRADIENT BOOSTED TREES REGRESSOR")
    print("="*70)
    print(f"TRAIN:")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  R²:   {train_r2:.4f}")
    print(f"\nTEST:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    print("="*70)
    
    gbt_run_id = run.info.run_id
    print(f"\nExperimento registrado en MLflow: {gbt_run_id}")

# COMMAND ----------

# Ver predicciones GBT
display(test_predictions.select('timestamp', 'pm2_5', 'prediction', 'aqi').limit(50))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Modelo 3: Random Forest Classifier (Categorías AQI)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clasificación multiclase de categorías de calidad del aire

# COMMAND ----------

# Preparar labels: convertir aqi_category a índice numérico
indexer = StringIndexer(inputCol="aqi_category", outputCol="label")
df_indexed = indexer.fit(df_assembled).transform(df_assembled)

# Split temporal
train_clf = df_indexed.filter(F.col('timestamp') < split_date)
test_clf = df_indexed.filter(F.col('timestamp') >= split_date)

print(f"Datos preparados para clasificación")
print(f"   Clases: {df_indexed.select('aqi_category').distinct().count()}")

# COMMAND ----------

# Ver mapeo de categorías
display(df_indexed.select('aqi_category', 'label').distinct().orderBy('label'))

# COMMAND ----------

# Iniciar run de MLflow
with mlflow.start_run(run_name="RandomForest_Classifier_v1") as run:
    
    # Parámetros del modelo
    max_depth = 10
    num_trees = 100
    max_bins = 32
    
    # Log de parámetros
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("num_trees", num_trees)
    mlflow.log_param("max_bins", max_bins)
    mlflow.log_param("num_classes", df_indexed.select('label').distinct().count())
    mlflow.log_param("features_count", len(feature_columns))
    
    # Crear modelo
    rf_classifier = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        maxDepth=max_depth,
        numTrees=num_trees,
        maxBins=max_bins,
        seed=42
    )
    
    print("🌲 Entrenando Random Forest Classifier...")
    
    # Entrenar
    rf_clf_model = rf_classifier.fit(train_clf)
    
    print("Modelo entrenado!")
    
    # Predicciones
    train_predictions = rf_clf_model.transform(train_clf)
    test_predictions = rf_clf_model.transform(test_clf)
    
    # Evaluadores
    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
    )
    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedRecall"
    )
    
    # Métricas en train
    train_accuracy = accuracy_evaluator.evaluate(train_predictions)
    train_f1 = f1_evaluator.evaluate(train_predictions)
    train_precision = precision_evaluator.evaluate(train_predictions)
    train_recall = recall_evaluator.evaluate(train_predictions)
    
    # Métricas en test
    test_accuracy = accuracy_evaluator.evaluate(test_predictions)
    test_f1 = f1_evaluator.evaluate(test_predictions)
    test_precision = precision_evaluator.evaluate(test_predictions)
    test_recall = recall_evaluator.evaluate(test_predictions)
    
    # Log de métricas
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("train_f1", train_f1)
    mlflow.log_metric("train_precision", train_precision)
    mlflow.log_metric("train_recall", train_recall)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    
    # Log del modelo
    mlflow.spark.log_model(rf_clf_model, "random_forest_classifier")
    
    print("\n" + "="*70)
    print("RESULTADOS - RANDOM FOREST CLASSIFIER")
    print("="*70)
    print(f"TRAIN:")
    print(f"  Accuracy:  {train_accuracy:.4f}")
    print(f"  F1-Score:  {train_f1:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  Recall:    {train_recall:.4f}")
    print(f"\nTEST:")
    print(f"  Accuracy:  {test_accuracy:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print("="*70)
    
    rf_clf_run_id = run.info.run_id
    print(f"\nExperimento registrado en MLflow: {rf_clf_run_id}")

# COMMAND ----------

# Ver predicciones de clasificación
display(test_predictions.select('timestamp', 'aqi_category', 'label', 'prediction', 'probability').limit(50))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Matriz de Confusión

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

# Obtener predicciones y labels
predictions_and_labels = test_predictions.select(['prediction', 'label']).rdd

# Crear métricas multiclase
metrics = MulticlassMetrics(predictions_and_labels.map(lambda x: (float(x[0]), float(x[1]))))

# Matriz de confusión
confusion_matrix = metrics.confusionMatrix().toArray()

print("Matriz de Confusión:")
print(confusion_matrix)

# COMMAND ----------

# Crear DataFrame de la matriz de confusión para mejor visualización
categories = ['Good', 'Moderate', 'Unhealthy Sensitive', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
conf_matrix_data = []

for i in range(len(confusion_matrix)):
    for j in range(len(confusion_matrix[i])):
        conf_matrix_data.append({
            'Actual': categories[i] if i < len(categories) else f'Class_{i}',
            'Predicted': categories[j] if j < len(categories) else f'Class_{j}',
            'Count': int(confusion_matrix[i][j])
        })

conf_df = spark.createDataFrame(conf_matrix_data)
display(conf_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Comparación de Modelos

# COMMAND ----------

print("="*70)
print("COMPARACIÓN DE MODELOS")
print("="*70)
print("\nREGRESIÓN (Predicción de PM2.5):")
print("-" * 70)
print(f"{'Modelo':<30} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
print("-" * 70)
print(f"{'Random Forest Regressor':<30} {test_rmse:.4f}     {test_mae:.4f}     {test_r2:.4f}")
print("-" * 70)
print("\nCLASIFICACIÓN (Categorías AQI):")
print("-" * 70)
print(f"{'Modelo':<30} {'Accuracy':<10} {'F1-Score':<10}")
print("-" * 70)
print(f"{'Random Forest Classifier':<30} {test_accuracy:.4f}     {test_f1:.4f}")
print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Guardar Mejores Modelos

# COMMAND ----------

# Guardar predicciones del mejor modelo de regresión
best_predictions = rf_model.transform(df_assembled)
best_predictions.select('timestamp', 'pm2_5', 'prediction', 'aqi', 'aqi_category').write.mode("overwrite").saveAsTable("air_quality_predictions")

print("Predicciones guardadas en 'air_quality_predictions'")

# COMMAND ----------

# Verificar tablas finales
print("\nTablas en la base de datos:")
spark.sql("SHOW TABLES").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen Final

# COMMAND ----------

print("="*70)
print("PROYECTO COMPLETO - RESUMEN")
print("="*70)
print("\nDATOS:")
print(f"   - Registros procesados: ~41,700")
print(f"   - Features creadas: 44")
print(f"   - Train/Test split: 80/20 temporal")
print("\nMODELOS ENTRENADOS:")
print("   1. Random Forest Regressor (PM2.5)")
print("   2. Gradient Boosted Trees Regressor (PM2.5)")
print("   3. Random Forest Classifier (Categorías AQI)")
print("\nMLFLOW:")
print("   - 3 experimentos registrados")
print("   - Parámetros, métricas y modelos guardados")
print("   - Feature importance documentada")
print("\nTABLAS CREADAS:")
print("   - air_quality_clean")
print("   - air_quality_features")
print("   - air_quality_predictions")
print("\nPRÓXIMOS PASOS SUGERIDOS:")
print("   - Ajustar hiperparámetros (Grid Search)")
print("   - Probar modelos adicionales (XGBoost, LSTM)")
print("   - Implementar predicción en tiempo real")
print("   - Crear dashboard de visualización")
print("   - Desplegar modelo en producción")
print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Discusión y Mejoras
# MAGIC 
# MAGIC ### Resultados Obtenidos:
# MAGIC 
# MAGIC **Fortalezas:**
# MAGIC - ✅ R² alto indica que el modelo captura bien la variabilidad
# MAGIC - ✅ Variables lag y rolling son los features más importantes
# MAGIC - ✅ El modelo generaliza bien (métricas similares en train/test)
# MAGIC - ✅ Clasificación tiene buena accuracy para categorías críticas
# MAGIC 
# MAGIC **Debilidades:**
# MAGIC - ⚠️ RMSE puede ser alto para valores extremos de contaminación
# MAGIC - ⚠️ Desbalance de clases en categorías AQI
# MAGIC - ⚠️ Solo usamos datos de una ubicación (Beijing)
# MAGIC 
# MAGIC ### 🚀 Mejoras Propuestas:
# MAGIC 
# MAGIC **1. Hiperparámetros:**
# MAGIC - Implementar Grid Search o Bayesian Optimization
# MAGIC - Probar diferentes combinaciones de max_depth, num_trees
# MAGIC - Ajustar min_instances_per_node para evitar overfitting
# MAGIC 
# MAGIC **2. Features:**
# MAGIC - Agregar datos de tráfico vehicular
# MAGIC - Incluir datos de industrias cercanas
# MAGIC - Promedios móviles más largos (48h, 72h)
# MAGIC - Features de Fourier para capturar estacionalidad
# MAGIC 
# MAGIC **3. Modelos:**
# MAGIC - Probar XGBoost (mejor que GBT en muchos casos)
# MAGIC - LSTM/RNN para capturar mejor las series temporales
# MAGIC - Ensemble de múltiples modelos (stacking)
# MAGIC 
# MAGIC **4. Datos:**
# MAGIC - Incorporar múltiples ciudades para generalización
# MAGIC - Más años de datos históricos
# MAGIC - Datos de eventos especiales (festivales, construcciones)
# MAGIC 
# MAGIC **5. Producción:**
# MAGIC - Implementar pipeline de predicción en tiempo real
# MAGIC - Sistema de alertas cuando AQI > umbral
# MAGIC - API REST para servir predicciones
# MAGIC - Reentrenamiento automático con datos nuevos
# MAGIC 
# MAGIC **6. Evaluación:**
# MAGIC - Validación cruzada temporal (Time Series Cross-Validation)
# MAGIC - Análisis de errores por rangos de PM2.5
# MAGIC - Evaluación en condiciones extremas
# MAGIC - Comparación con modelos baseline (media móvil simple)
