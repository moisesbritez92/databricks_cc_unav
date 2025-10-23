# Databricks notebook source
# MAGIC %md
# MAGIC # 🤖 Model Training & Evaluation - Versión Simplificada
# MAGIC 
# MAGIC **NOTA IMPORTANTE**: Este notebook requiere un **Databricks Runtime ML** para funcionar completamente.
# MAGIC 
# MAGIC Si ves errores de MLlib, necesitas:
# MAGIC 1. Crear un cluster con Runtime **13.3 LTS ML** (no el estándar)
# MAGIC 2. Conectar este notebook a ese cluster
# MAGIC 
# MAGIC Esta versión simplificada demuestra el pipeline sin entrenar modelos reales.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚠️ Verificar Entorno

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
import mlflow

print("✅ PySpark disponible")
print(f"📊 MLflow Tracking URI: {mlflow.get_tracking_uri()}")

# Verificar si MLlib está disponible
try:
    from pyspark.ml.regression import RandomForestRegressor
    print("✅ MLlib está disponible - Puedes ejecutar el notebook completo")
    MLLIB_OK = True
except Exception as e:
    print(f"\n❌ MLlib NO está disponible: {e}")
    print("\n🔧 SOLUCIÓN:")
    print("   1. Ve a 'Compute' en Databricks")
    print("   2. Crea un nuevo cluster")
    print("   3. Runtime: 13.3 LTS ML (debe terminar en 'ML')")
    print("   4. Conecta este notebook al nuevo cluster")
    print("\n⚠️ Ejecutando versión simplificada sin entrenamiento...")
    MLLIB_OK = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Cargar Dataset

# COMMAND ----------

spark.sql("USE air_quality_project")
df = spark.table("air_quality_features")

print(f"✅ Dataset cargado: {df.count():,} registros")
print(f"📋 Columnas: {len(df.columns)}")

# COMMAND ----------

display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Análisis Exploratorio

# COMMAND ----------

# Estadísticas de la variable objetivo
print("📊 Estadísticas de PM2.5:")
df.select('pm2_5').describe().show()

print("\n📊 Distribución de categorías AQI:")
df.groupBy('aqi_category').count().orderBy(F.desc('count')).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Preparación de Features

# COMMAND ----------

feature_columns = [
    'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir',
    'hour', 'day_of_week', 'day_of_month', 'week_of_year',
    'is_weekend', 'is_rush_hour', 'is_business_hours',
    'pm25_rolling_3h', 'pm25_rolling_6h', 'pm25_rolling_12h', 'pm25_rolling_24h',
    'temp_rolling_6h', 'temp_rolling_24h',
    'pres_rolling_6h', 'pres_rolling_24h',
    'pm25_lag_1h', 'pm25_lag_3h', 'pm25_lag_6h', 'pm25_lag_12h', 'pm25_lag_24h',
    'temp_lag_6h', 'temp_lag_24h', 'pres_lag_6h',
    'pm25_diff_1h', 'pm25_diff_24h',
    'humidity_ratio', 'temp_pres_interaction', 'temp_deviation', 
    'pm25_trend', 'wind_total',
    'wind_NE', 'wind_NW', 'wind_SE', 'wind_cv',
    'season_Winter', 'season_Spring', 'season_Summer', 'season_Fall'
]

print(f"📊 Features seleccionadas: {len(feature_columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Split Temporal

# COMMAND ----------

# Split 80/20 temporal
split_date = df.approxQuantile('timestamp', [0.8], 0.01)[0]

train_df = df.filter(F.col('timestamp') < split_date)
test_df = df.filter(F.col('timestamp') >= split_date)

print(f"📅 Fecha de corte: {split_date}")
print(f"✅ Train: {train_df.count():,} registros ({train_df.count()/df.count()*100:.1f}%)")
print(f"✅ Test:  {test_df.count():,} registros ({test_df.count()/df.count()*100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Modelo Simplificado (Baseline)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Como MLlib no está disponible, usamos un modelo baseline simple
# MAGIC 
# MAGIC **Modelo Baseline**: Predecir con el promedio móvil de 24h
# MAGIC 
# MAGIC Este es un buen benchmark para comparar modelos más complejos.

# COMMAND ----------

# Crear predicción baseline usando rolling average 24h
test_baseline = test_df.withColumn('prediction', F.col('pm25_rolling_24h'))

# Calcular métricas manualmente
from pyspark.sql.functions import sqrt, avg, abs as spark_abs, pow as spark_pow

# RMSE
rmse = test_baseline.select(
    sqrt(avg(spark_pow(F.col('pm2_5') - F.col('prediction'), 2)))
).first()[0]

# MAE
mae = test_baseline.select(
    avg(spark_abs(F.col('pm2_5') - F.col('prediction')))
).first()[0]

# R² (simplificado)
mean_pm25 = test_baseline.select(avg('pm2_5')).first()[0]
ss_res = test_baseline.select(
    avg(spark_pow(F.col('pm2_5') - F.col('prediction'), 2))
).first()[0]
ss_tot = test_baseline.select(
    avg(spark_pow(F.col('pm2_5') - mean_pm25, 2))
).first()[0]
r2 = 1 - (ss_res / ss_tot)

print("="*70)
print("📊 BASELINE MODEL (Rolling Average 24h)")
print("="*70)
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")
print("="*70)

# COMMAND ----------

# Ver predicciones
display(test_baseline.select('timestamp', 'pm2_5', 'prediction', 'aqi_category').limit(50))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Simulación de Experimento MLflow

# COMMAND ----------

# Configurar experimento
mlflow.set_experiment("/Users/shared/air_quality_prediction")

# Registrar modelo baseline en MLflow
with mlflow.start_run(run_name="Baseline_RollingAvg24h") as run:
    
    # Log parámetros
    mlflow.log_param("model_type", "Baseline_RollingAverage")
    mlflow.log_param("window_hours", 24)
    mlflow.log_param("features_count", 1)
    mlflow.log_param("train_count", train_df.count())
    mlflow.log_param("test_count", test_df.count())
    
    # Log métricas
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_r2", r2)
    
    print(f"✅ Experimento baseline registrado: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Análisis de Correlaciones

# COMMAND ----------

# Calcular correlaciones de las top features con PM2.5
correlations = []
for col in ['pm25_lag_1h', 'pm25_lag_24h', 'pm25_rolling_24h', 'TEMP', 'PRES', 'DEWP']:
    if col in df.columns:
        corr = df.stat.corr('pm2_5', col)
        correlations.append((col, corr))

# Ordenar por correlación absoluta
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("📊 Correlaciones con PM2.5:")
print("-" * 50)
for feat, corr in correlations:
    print(f"{feat:20s}: {corr:7.4f}")
print("-" * 50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Guardar Predicciones

# COMMAND ----------

# Guardar predicciones baseline
test_baseline.select(
    'timestamp', 'pm2_5', 'prediction', 'aqi', 'aqi_category'
).write.mode("overwrite").saveAsTable("air_quality_predictions_baseline")

print("✅ Predicciones baseline guardadas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📋 Resumen del Proyecto

# COMMAND ----------

print("="*70)
print("🎉 PROYECTO DE CALIDAD DEL AIRE - RESUMEN")
print("="*70)
print("\n✅ COMPLETADO:")
print("   1. ✅ Data Ingestion (Beijing PM2.5 dataset)")
print("   2. ✅ Data Cleaning (~41,700 registros)")
print("   3. ✅ Feature Engineering (44 features)")
print("   4. ✅ Modelo Baseline (Rolling Average)")
print("   5. ✅ MLflow Tracking (experimento registrado)")
print("\n📊 MÉTRICAS BASELINE:")
print(f"   - RMSE: {rmse:.4f}")
print(f"   - MAE:  {mae:.4f}")
print(f"   - R²:   {r2:.4f}")
print("\n⚠️ PARA COMPLETAR:")
print("   - Cambiar a Databricks Runtime ML")
print("   - Entrenar Random Forest Regressor")
print("   - Entrenar Gradient Boosted Trees")
print("   - Entrenar Random Forest Classifier")
print("   - Comparar con baseline")
print("\n🔧 PRÓXIMOS PASOS:")
print("   1. Crear cluster con Runtime 13.3 LTS ML")
print("   2. Ejecutar notebook 04_model_training_mlflow.py completo")
print("   3. Comparar modelos en MLflow UI")
print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎓 Discusión y Mejoras
# MAGIC 
# MAGIC ### 📊 Resultados del Baseline:
# MAGIC 
# MAGIC El modelo baseline (promedio móvil 24h) nos da una referencia importante:
# MAGIC - **RMSE**: Error cuadrático medio - valores bajos son mejores
# MAGIC - **MAE**: Error absoluto medio - más interpretable que RMSE
# MAGIC - **R²**: Proporción de varianza explicada (0-1, mayor es mejor)
# MAGIC 
# MAGIC ### 🎯 Expectativas para Modelos ML:
# MAGIC 
# MAGIC Los modelos de ML (Random Forest, GBT) deberían superar significativamente al baseline:
# MAGIC - **RMSE** esperado: 15-25% mejor que baseline
# MAGIC - **R²** esperado: > 0.85 (baseline típicamente ~0.70-0.75)
# MAGIC 
# MAGIC ### 💡 Insights del Dataset:
# MAGIC 
# MAGIC **Variables más importantes (por correlación):**
# MAGIC 1. `pm25_lag_1h` - Valor de la hora anterior
# MAGIC 2. `pm25_rolling_24h` - Promedio móvil 24h
# MAGIC 3. `pm25_lag_24h` - Valor del mismo momento ayer
# MAGIC 
# MAGIC Esto confirma que la contaminación tiene **fuerte autocorrelación temporal**.
# MAGIC 
# MAGIC ### 🚀 Mejoras Propuestas:
# MAGIC 
# MAGIC **1. Modelos Más Complejos:**
# MAGIC - XGBoost (mejor que GBT en muchos casos)
# MAGIC - LSTM para series temporales
# MAGIC - Ensemble (combinar múltiples modelos)
# MAGIC 
# MAGIC **2. Hiperparámetros:**
# MAGIC - Grid Search para optimizar parámetros
# MAGIC - Cross-validation temporal
# MAGIC - Early stopping para evitar overfitting
# MAGIC 
# MAGIC **3. Features Adicionales:**
# MAGIC - Datos de tráfico vehicular
# MAGIC - Actividad industrial cercana
# MAGIC - Eventos especiales (festivales, construcción)
# MAGIC - Features de Fourier para estacionalidad
# MAGIC 
# MAGIC **4. Datos:**
# MAGIC - Múltiples ciudades para generalización
# MAGIC - Más años de histórico
# MAGIC - Validación en diferentes contextos
# MAGIC 
# MAGIC **5. Producción:**
# MAGIC - API REST para predicciones en tiempo real
# MAGIC - Sistema de alertas (AQI > umbral)
# MAGIC - Dashboard interactivo
# MAGIC - Reentrenamiento automático
# MAGIC 
# MAGIC ### 📈 Impacto Esperado:
# MAGIC 
# MAGIC Con modelos ML completos esperamos:
# MAGIC - **Mejora del 20-30%** vs baseline en RMSE
# MAGIC - **R² > 0.85** (excelente para series temporales)
# MAGIC - **Accuracy > 90%** en clasificación de categorías AQI
# MAGIC - **Predicción útil** hasta 6-12 horas adelante
# MAGIC 
# MAGIC ### ⚠️ Limitaciones Actuales:
# MAGIC 
# MAGIC - Solo una ubicación (Beijing) - sesgo geográfico
# MAGIC - Periodo limitado (2010-2014)
# MAGIC - No incluye eventos extremos (COVID, olimpiadas)
# MAGIC - Variables meteorológicas básicas
# MAGIC 
# MAGIC ### ✅ Conclusión:
# MAGIC 
# MAGIC Este proyecto demuestra un **pipeline completo de ML** para series temporales:
# MAGIC 1. ✅ Ingesta y limpieza robusta
# MAGIC 2. ✅ Feature engineering exhaustivo
# MAGIC 3. ✅ Modelo baseline documentado
# MAGIC 4. ✅ Infraestructura MLflow lista
# MAGIC 5. ⏳ Modelos ML pendientes (requiere Runtime ML)
# MAGIC 
# MAGIC **Con Runtime ML, el proyecto alcanzará su potencial completo.**
