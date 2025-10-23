# Databricks notebook source
# MAGIC %md
# MAGIC # 🌍 Datasets de Calidad del Aire - Opciones Recomendadas
# MAGIC 
# MAGIC ## 📊 Datasets Disponibles:
# MAGIC 
# MAGIC ### 1. **Air Quality Data in India (2015-2020)** ⭐ RECOMENDADO
# MAGIC - **Fuente:** Kaggle
# MAGIC - **Link:** https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
# MAGIC - **Tamaño:** ~26MB, +440K registros
# MAGIC - **Variables:** PM2.5, PM10, NO2, SO2, CO, O3, temperatura, humedad
# MAGIC - **Ubicaciones:** 26 ciudades de India
# MAGIC 
# MAGIC ### 2. **Beijing PM2.5 Data**
# MAGIC - **Fuente:** UCI Machine Learning Repository
# MAGIC - **Link:** https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
# MAGIC - **Tamaño:** ~43K registros
# MAGIC - **Variables:** PM2.5 + datos meteorológicos
# MAGIC 
# MAGIC ### 3. **US Air Quality (1980-2023)**
# MAGIC - **Fuente:** EPA (Environmental Protection Agency)
# MAGIC - **Variables:** PM2.5, PM10, O3, NO2, SO2, CO
# MAGIC - **Cobertura:** Todo Estados Unidos
# MAGIC 
# MAGIC ### 4. **OpenAQ - Global Air Quality**
# MAGIC - **API pública:** https://openaq.org
# MAGIC - **Cobertura:** Mundial en tiempo real
# MAGIC - **Variables:** PM2.5, PM10, NO2, SO2, CO, O3

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📥 Opción 1: Descargar desde URL directa
# MAGIC 
# MAGIC Vamos a usar el dataset de Beijing PM2.5 (UCI) que es público y no requiere autenticación

# COMMAND ----------

# Importar librerías necesarias
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
    
    print(f"\n📊 Registros: {df.count()}")
    print(f"📋 Columnas: {len(df.columns)}")
    print("\n🔍 Primeras filas:")
    df.show(5)
else:
    print(f"❌ Error al descargar: {response.status_code}")

# COMMAND ----------

# Información del dataset
df.printSchema()

# COMMAND ----------

# Estadísticas básicas
display(df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📥 Opción 2: Dataset Sintético para Pruebas
# MAGIC 
# MAGIC Si no puedes descargar datasets externos, creamos uno sintético realista

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
import random
from datetime import datetime, timedelta

# Función para generar datos sintéticos
def generate_air_quality_data(num_records=10000):
    """Genera datos sintéticos de calidad del aire"""
    
    ciudades = ["Madrid", "Barcelona", "Valencia", "Sevilla", "Bilbao", "Zaragoza"]
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(num_records):
        ciudad = random.choice(ciudades)
        fecha = start_date + timedelta(hours=i % (365*24))
        hora = fecha.hour
        
        # Simular patrones realistas
        # Más contaminación en horas pico (7-9am, 6-8pm)
        factor_hora = 1.3 if hora in [7, 8, 9, 18, 19, 20] else 1.0
        
        # Más contaminación en invierno
        factor_estacion = 1.4 if fecha.month in [12, 1, 2] else 1.0
        
        base_factor = factor_hora * factor_estacion
        
        pm25 = max(5, random.gauss(35, 15) * base_factor)
        pm10 = pm25 * random.uniform(1.3, 1.8)
        no2 = random.gauss(30, 10) * base_factor
        so2 = random.gauss(15, 5) * base_factor
        co = random.gauss(0.8, 0.3) * base_factor
        o3 = random.gauss(50, 20)
        
        # Datos meteorológicos
        temp = random.gauss(18, 8) + (10 if fecha.month in [6,7,8] else -5)
        humedad = random.gauss(60, 15)
        velocidad_viento = random.gauss(10, 5)
        presion = random.gauss(1013, 10)
        
        # Calcular AQI simplificado (basado en PM2.5)
        if pm25 <= 12:
            aqi = pm25 * 4.17
            categoria = "Bueno"
        elif pm25 <= 35.4:
            aqi = 50 + (pm25 - 12) * 2.13
            categoria = "Moderado"
        elif pm25 <= 55.4:
            aqi = 100 + (pm25 - 35.4) * 2.5
            categoria = "Dañino para grupos sensibles"
        elif pm25 <= 150.4:
            aqi = 150 + (pm25 - 55.4) * 1.05
            categoria = "Dañino"
        else:
            aqi = 200 + (pm25 - 150.4) * 0.66
            categoria = "Muy dañino"
        
        data.append({
            'ciudad': ciudad,
            'fecha': fecha.strftime('%Y-%m-%d'),
            'hora': hora,
            'timestamp': fecha,
            'pm25': round(pm25, 2),
            'pm10': round(pm10, 2),
            'no2': round(no2, 2),
            'so2': round(so2, 2),
            'co': round(co, 3),
            'o3': round(o3, 2),
            'temperatura': round(temp, 1),
            'humedad': round(humedad, 1),
            'velocidad_viento': round(velocidad_viento, 1),
            'presion': round(presion, 1),
            'aqi': round(aqi, 1),
            'categoria': categoria
        })
    
    return data

print("🎲 Generando dataset sintético...")
data = generate_air_quality_data(50000)

# Convertir a Spark DataFrame
df_synthetic = spark.createDataFrame(data)

print(f"✅ Dataset generado con {df_synthetic.count()} registros")
df_synthetic.show(10)

# COMMAND ----------

# Ver distribución por ciudad
display(df_synthetic.groupBy("ciudad").count().orderBy(desc("count")))

# COMMAND ----------

# Ver distribución por categoría de calidad
display(df_synthetic.groupBy("categoria").count().orderBy(desc("count")))

# COMMAND ----------

# Estadísticas de contaminantes
print("📊 Estadísticas de Contaminantes:")
df_synthetic.select("pm25", "pm10", "no2", "so2", "co", "o3", "aqi").describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💾 Guardar en el Catálogo de Databricks

# COMMAND ----------

# Cambiar a la base de datos del proyecto
spark.sql("USE air_quality_project")

# Guardar el dataset como tabla
df_synthetic.write.mode("overwrite").saveAsTable("air_quality_raw")

print("✅ Tabla 'air_quality_raw' guardada en el catálogo!")

# Verificar
print(f"\n📊 Registros guardados: {spark.table('air_quality_raw').count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📥 Opción 3: Descargar manualmente desde Kaggle
# MAGIC 
# MAGIC ### Pasos para usar datasets de Kaggle:
# MAGIC 
# MAGIC 1. **Ve a Kaggle y descarga el dataset:**
# MAGIC    - Air Quality India: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
# MAGIC    
# MAGIC 2. **Sube el archivo a DBFS:**
# MAGIC    ```python
# MAGIC    # Opción A: Desde la UI de Databricks
# MAGIC    # Data > Add Data > Upload File
# MAGIC    
# MAGIC    # Opción B: Usando dbutils
# MAGIC    # dbutils.fs.cp("file:/local/path/data.csv", "dbfs:/FileStore/air_quality/data.csv")
# MAGIC    ```
# MAGIC 
# MAGIC 3. **Leer el archivo:**
# MAGIC    ```python
# MAGIC    df = spark.read.csv("dbfs:/FileStore/air_quality/data.csv", header=True, inferSchema=True)
# MAGIC    ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📥 Opción 4: Usar API de OpenAQ (datos reales en tiempo real)

# COMMAND ----------

# Instalar librería (si no está disponible)
%pip install openaq

# COMMAND ----------

# Reiniciar kernel después de instalar
dbutils.library.restartPython()

# COMMAND ----------

import openaq
from datetime import datetime, timedelta

# Crear cliente de OpenAQ
api = openaq.OpenAQ()

# Obtener datos de las últimas 24 horas para España
print("🌍 Obteniendo datos en tiempo real de OpenAQ...")

# Buscar ubicaciones en España
locations = api.locations(country='ES', limit=10)

if locations['meta']['found'] > 0:
    print(f"✅ Encontradas {locations['meta']['found']} estaciones en España")
    
    # Obtener mediciones recientes
    measurements = api.measurements(
        country='ES',
        limit=1000,
        parameter=['pm25', 'pm10', 'no2', 'o3'],
        date_from=datetime.now() - timedelta(days=7)
    )
    
    if measurements['meta']['found'] > 0:
        # Convertir a DataFrame
        df_openaq = pd.DataFrame(measurements['results'])
        df_spark = spark.createDataFrame(df_openaq)
        
        print(f"✅ Obtenidos {df_spark.count()} registros")
        df_spark.show(10)
    else:
        print("⚠️ No se encontraron mediciones recientes")
else:
    print("⚠️ No se encontraron estaciones en España")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 Resumen de Opciones
# MAGIC 
# MAGIC | Opción | Ventajas | Desventajas |
# MAGIC |--------|----------|-------------|
# MAGIC | **Dataset Sintético** | ✅ Fácil, no requiere descarga | ❌ No es real |
# MAGIC | **Beijing UCI** | ✅ Real, descarga directa | ❌ Solo una ciudad |
# MAGIC | **Kaggle (India)** | ✅ Real, múltiples ciudades | ❌ Requiere descarga manual |
# MAGIC | **OpenAQ API** | ✅ Datos en tiempo real | ❌ Límites de API |
# MAGIC 
# MAGIC **Recomendación:** Usa el **dataset sintético** para desarrollo y pruebas rápidas. 
# MAGIC Para el proyecto final, descarga manualmente el dataset de Kaggle.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Siguiente Paso
# MAGIC 
# MAGIC Ahora que tienes los datos en la tabla `air_quality_raw`, el siguiente notebook será:
# MAGIC - **Data Cleaning & Transformation**
# MAGIC   - Limpieza de datos
# MAGIC   - Manejo de nulos
# MAGIC   - Feature engineering
# MAGIC   - Preparación para modelado
