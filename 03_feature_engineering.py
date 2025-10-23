# Databricks notebook source
# MAGIC %md
# MAGIC # ⚙️ Feature Engineering & Transformations
# MAGIC 
# MAGIC Este notebook crea características (features) significativas para mejorar el rendimiento del modelo de predicción de calidad del aire.
# MAGIC 
# MAGIC ## Transformaciones a realizar:
# MAGIC 1. Variables temporales (día de la semana, mes, estación, hora)
# MAGIC 2. Promedios móviles (rolling averages)
# MAGIC 3. Variables lag (valores previos)
# MAGIC 4. Ratios e interacciones entre variables
# MAGIC 5. Codificación de variables categóricas
# MAGIC 6. Normalización/escalado de features

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Cargar datos limpios

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# Usar la base de datos del proyecto
spark.sql("USE air_quality_project")

# Cargar tabla limpia
df = spark.table("air_quality_clean")

print(f"✅ Datos cargados: {df.count():,} registros")
print(f"📋 Columnas: {len(df.columns)}")

# COMMAND ----------

# Ver esquema
df.printSchema()

# COMMAND ----------

# Ver muestra
display(df.select('timestamp', 'pm2_5', 'TEMP', 'PRES', 'DEWP', 'cbwd', 'aqi', 'aqi_category').limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Variables Temporales

# COMMAND ----------

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

print("✅ Variables temporales creadas:")
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
# MAGIC ## 3. Promedios Móviles (Rolling Averages)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ¿Por qué promedios móviles?
# MAGIC 
# MAGIC Los promedios móviles capturan **tendencias temporales**:
# MAGIC - Suavizan fluctuaciones de corto plazo
# MAGIC - Capturan patrones de contaminación persistente
# MAGIC - Ayudan a predecir valores futuros basándose en el historial reciente

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

print("✅ Promedios móviles creados:")
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
# MAGIC ## 4. Variables Lag (Valores Previos)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ¿Por qué variables lag?
# MAGIC 
# MAGIC Los valores anteriores son **predictores poderosos**:
# MAGIC - La contaminación actual depende de la contaminación reciente
# MAGIC - Capturan autocorrelación temporal
# MAGIC - Esenciales para series temporales

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

print("✅ Variables lag creadas:")
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

# MAGIC %md
# MAGIC ## 5. Ratios e Interacciones

# COMMAND ----------

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

print("✅ Ratios e interacciones creadas:")
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
# MAGIC ## 6. Codificación de Variables Categóricas

# COMMAND ----------

# MAGIC %md
# MAGIC ### One-Hot Encoding para variables categóricas
# MAGIC 
# MAGIC Necesario para algoritmos de ML que requieren entrada numérica:
# MAGIC - `cbwd` (dirección del viento): 4 valores → 4 columnas binarias
# MAGIC - `season`: 4 estaciones → 4 columnas binarias
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

print("✅ Variables categóricas codificadas (One-Hot):")
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
# MAGIC ## 7. Manejo de Valores Nulos Generados

# COMMAND ----------

# MAGIC %md
# MAGIC Las operaciones de lag y rolling generan nulos al inicio de la serie temporal.
# MAGIC **Estrategia**: Eliminar las primeras 24 horas (suficiente para todos los lags)

# COMMAND ----------

# Contar nulos antes
print("❌ Valores nulos antes de limpieza:")
null_cols = ['pm25_lag_24h', 'pm25_rolling_24h', 'temp_rolling_24h']
for col_name in null_cols:
    null_count = df_encoded.filter(F.col(col_name).isNull()).count()
    print(f"  {col_name}: {null_count}")

# COMMAND ----------

# Eliminar filas con nulos en las variables lag críticas
df_final = df_encoded.filter(
    F.col('pm25_lag_24h').isNotNull() & 
    F.col('pm25_rolling_24h').isNotNull()
)

print(f"\n✅ Registros después de eliminar nulos: {df_final.count():,}")
print(f"❌ Registros eliminados: {df_encoded.count() - df_final.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Resumen de Features Creadas

# COMMAND ----------

print("="*70)
print("📊 RESUMEN DE FEATURE ENGINEERING")
print("="*70)
print(f"Registros finales:          {df_final.count():>10,}")
print(f"Total de columnas:          {len(df_final.columns):>10}")
print("="*70)
print("\n✅ Categorías de features creadas:")
print("\n1️⃣ TEMPORALES (7 features):")
print("   - day_of_week, day_of_month, week_of_year")
print("   - season, is_weekend, is_rush_hour, is_business_hours")
print("\n2️⃣ PROMEDIOS MÓVILES (8 features):")
print("   - pm25_rolling: 3h, 6h, 12h, 24h")
print("   - temp_rolling: 6h, 24h")
print("   - pres_rolling: 6h, 24h")
print("\n3️⃣ VARIABLES LAG (10 features):")
print("   - pm25_lag: 1h, 3h, 6h, 12h, 24h")
print("   - temp_lag: 6h, 24h")
print("   - pres_lag: 6h")
print("   - pm25_diff: 1h, 24h")
print("\n4️⃣ INTERACCIONES (5 features):")
print("   - humidity_ratio, temp_pres_interaction")
print("   - temp_deviation, pm25_trend, wind_total")
print("\n5️⃣ ONE-HOT ENCODING (14 features):")
print("   - wind_direction: 4 columnas")
print("   - season: 4 columnas")
print("   - aqi_category: 6 columnas")
print("\n📈 TOTAL NUEVAS FEATURES: 44")
print("="*70)

# COMMAND ----------

# Ver lista completa de columnas
print("\n📋 Columnas finales del dataset:")
for i, col in enumerate(df_final.columns, 1):
    print(f"{i:3}. {col}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Selección de Features para Modelado

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

print(f"✅ Dataset listo para modelado:")
print(f"   Registros: {df_model_ready.count():,}")
print(f"   Features: {len(df_model_ready.columns)}")

# COMMAND ----------

# Ver muestra final
display(df_model_ready.limit(20))

# COMMAND ----------

# Estadísticas finales
display(df_model_ready.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Guardar Dataset con Features

# COMMAND ----------

# Guardar en el catálogo
df_model_ready.write.mode("overwrite").saveAsTable("air_quality_features")

print("✅ Tabla 'air_quality_features' guardada exitosamente!")
print(f"📊 Registros: {spark.table('air_quality_features').count():,}")
print(f"📋 Columnas: {len(spark.table('air_quality_features').columns)}")

# COMMAND ----------

# Verificar tablas en la base de datos
print("\n📚 Tablas en la base de datos 'air_quality_project':")
spark.sql("SHOW TABLES").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Siguiente Paso
# MAGIC 
# MAGIC ¡Features creadas y listas! El próximo notebook será:
# MAGIC - **Model Training & Evaluation**
# MAGIC   - Split temporal train/test
# MAGIC   - Entrenamiento de modelos (RandomForest, GBT)
# MAGIC   - Evaluación de métricas (RMSE, MAE, R²)
# MAGIC   - Feature importance
# MAGIC   - Integración con MLflow
