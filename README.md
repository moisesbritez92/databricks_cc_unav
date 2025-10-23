# 🌍 Proyecto Databricks: Predicción de Calidad del Aire

## 📋 Descripción del Proyecto

Sistema de predicción de calidad del aire utilizando **Spark MLlib** y **MLflow** en Databricks. El proyecto analiza datos de contaminantes atmosféricos (PM2.5, PM10, NO2, SO2, CO, O3) junto con variables meteorológicas para predecir el índice de calidad del aire (AQI).

---

## 🎯 Objetivos

1. **Data Ingestion & Cleaning**: Cargar y limpiar datos de calidad del aire
2. **Feature Engineering**: Crear variables significativas para el modelo
3. **Model Training**: Entrenar modelos de ML con Spark MLlib
4. **MLflow Tracking**: Registrar experimentos, métricas y artefactos
5. **Evaluation & Discussion**: Analizar resultados y proponer mejoras

---

## 📊 Dataset

**Fuente**: Beijing PM2.5 Data (UCI Machine Learning Repository)
- **URL**: https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
- **Período**: 2010-2014 (5 años)
- **Registros**: ~43,000 mediciones horarias
- **Ubicación**: Beijing, China

### Variables del Dataset

#### Contaminantes:
- `pm2_5`: Partículas finas (μg/m³) - **Variable objetivo**
- Variables meteorológicas disponibles

#### Meteorología:
- `DEWP`: Punto de rocío (°C)
- `TEMP`: Temperatura (°C)
- `PRES`: Presión atmosférica (hPa)
- `cbwd`: Dirección del viento (categórica)
- `Iws`: Velocidad del viento acumulada (m/s)
- `Is`: Nieve acumulada (horas)
- `Ir`: Lluvia acumulada (horas)

#### Temporales:
- `year`, `month`, `day`, `hour`
- `timestamp`: Fecha/hora completa (creada)

---

## 🗂️ Estructura del Proyecto

```
Databricks/
│
├── 01_data_ingestion.py          # Descarga y carga de datos
├── 02_data_cleaning.py            # Limpieza y EDA ✅ COMPLETADO
├── 03_feature_engineering.py      # Transformaciones y features (próximo)
├── 04_model_training.py           # Entrenamiento de modelos (pendiente)
├── 05_mlflow_tracking.py          # Tracking con MLflow (pendiente)
├── 06_evaluation_discussion.py   # Evaluación y análisis (pendiente)
│
├── test_connection.py             # Notebook de prueba inicial
├── catalogo_databricks.py         # Exploración del catálogo
└── README.md                      # Este archivo
```

---

## 🧹 Progreso Actual: Data Cleaning (Completado)

### ✅ Acciones Realizadas

#### 1. **Renombrado de Columnas**
- Eliminados puntos (`.`) de nombres de columnas
- Ejemplo: `pm2.5` → `pm2_5`

#### 2. **Creación de Timestamp**
```python
timestamp = F.to_timestamp(
    F.concat_ws('-', year, month, day, hour),
    'yyyy-MM-dd-HH'
)
```

#### 3. **Manejo de Valores Nulos**

| Columna | Estrategia |
|---------|------------|
| `pm2_5` (objetivo) | **Eliminar** registros (crítico) |
| Variables meteorológicas | **Imputar** con la media |
| `cbwd` (dirección viento) | Rellenar con moda ('cv') |

**Código de imputación:**
```python
# Calcular media
means = {col: df.select(F.mean(col)).first()[0] for col in columns_to_impute}

# Imputar
df_imputed = df_clean
for column in columns_to_impute:
    df_imputed = df_imputed.withColumn(
        column,
        F.when(col(column).isNull(), F.lit(means[column])).otherwise(col(column))
    )
```

#### 4. **Análisis de Outliers**
- Detectados outliers extremos usando IQR (Interquartile Range)
- **Decisión**: Mantener outliers (pueden ser eventos reales de alta contaminación)

#### 5. **Creación de Categorías AQI**

Basado en estándares de la EPA (Environmental Protection Agency):

| PM2.5 (μg/m³) | AQI Range | Categoría | Descripción |
|---------------|-----------|-----------|-------------|
| 0 - 12.0 | 0 - 50 | **Good** | Aire limpio |
| 12.1 - 35.4 | 51 - 100 | **Moderate** | Aceptable |
| 35.5 - 55.4 | 101 - 150 | **Unhealthy for Sensitive Groups** | Riesgo para sensibles |
| 55.5 - 150.4 | 151 - 200 | **Unhealthy** | Todos afectados |
| 150.5 - 250.4 | 201 - 300 | **Very Unhealthy** | Alerta de salud |
| > 250.4 | 301+ | **Hazardous** | Emergencia |

#### 6. **Cálculo del AQI Numérico**

Fórmula de conversión PM2.5 → AQI:

$$
AQI = I_{low} + \frac{I_{high} - I_{low}}{C_{high} - C_{low}} \times (C - C_{low})
$$

**Ejemplo**: Si PM2.5 = 25 μg/m³
```
Rango: 12.1 - 35.4 (Moderate)
AQI = 50 + (25 - 12) × 2.13
AQI = 50 + 27.69 = 77.69 ≈ 78
```

**Implementación en código:**
```python
df_clean_final = df_imputed.withColumn(
    'aqi',
    F.when(col('pm2_5') <= 12, col('pm2_5') * 4.17)
    .when(col('pm2_5') <= 35.4, 50 + (col('pm2_5') - 12) * 2.13)
    .when(col('pm2_5') <= 55.4, 100 + (col('pm2_5') - 35.4) * 2.5)
    .when(col('pm2_5') <= 150.4, 150 + (col('pm2_5') - 55.4) * 1.05)
    .when(col('pm2_5') <= 250.4, 200 + (col('pm2_5') - 150.4) * 0.5)
    .otherwise(300 + (col('pm2_5') - 250.4) * 0.4)
)
```

### 📊 Resultados de Limpieza

```
================================================================
                    RESUMEN DE LIMPIEZA DE DATOS
================================================================
Registros originales:           43,824
Registros después limpieza:     41,757
Registros eliminados:            2,067
Porcentaje retenido:            95.28%
================================================================
```

### 💾 Tabla Generada

**Base de datos**: `air_quality_project`  
**Tabla**: `air_quality_clean`

**Columnas finales**:
- Todas las originales + limpias
- `timestamp` (datetime)
- `aqi` (double) - Índice numérico 0-500
- `aqi_category` (string) - Categoría textual

---

## 🔧 Configuración de Databricks

### Extensión de VS Code
1. Instalar extensión "Databricks"
2. Configurar conexión:
   - **Host**: `https://<tu-instancia>.cloud.databricks.com`
   - **Token**: Personal Access Token (PAT)

### Cluster Recomendado
```yaml
Cluster Name: proyecto-calidad-aire
Mode: Single Node
Runtime: Databricks Runtime 13.3 LTS ML
Node Type: Standard_DS3_v2
Auto-termination: 30 minutes
```

### Catálogo de Databricks

#### Comandos útiles:
```python
# Listar bases de datos
spark.catalog.listDatabases()

# Cambiar base de datos
spark.sql("USE air_quality_project")

# Listar tablas
spark.catalog.listTables()

# Leer tabla
df = spark.table("air_quality_clean")
```

#### SQL:
```sql
-- Ver bases de datos
SHOW DATABASES;

-- Usar base de datos
USE air_quality_project;

-- Ver tablas
SHOW TABLES;

-- Consultar tabla
SELECT * FROM air_quality_clean LIMIT 10;
```

---

## 🚀 Próximos Pasos

### 3️⃣ Feature Engineering (En desarrollo)
- [ ] Variables temporales (día de la semana, mes, estación)
- [ ] Promedios móviles (3h, 6h, 24h)
- [ ] Lags de variables (t-1, t-3, t-6)
- [ ] Codificación de variables categóricas (One-Hot Encoding)
- [ ] Normalización/escalado de features
- [ ] Interacciones entre variables

### 4️⃣ Model Training
- [ ] Split train/test (temporal)
- [ ] Modelos a probar:
  - **Regresión**: Random Forest, Gradient Boosted Trees
  - **Clasificación**: Random Forest Classifier (categorías AQI)
  - **Clustering**: K-Means (patrones temporales)
- [ ] Evaluación de métricas

### 5️⃣ MLflow Tracking
- [ ] Configurar experimentos
- [ ] Logging de:
  - Parámetros del modelo
  - Métricas (RMSE, MAE, R², Accuracy, F1)
  - Artefactos (plots, modelos)
- [ ] Comparación de modelos

### 6️⃣ Evaluation & Discussion
- [ ] Análisis de feature importance
- [ ] Curvas de aprendizaje
- [ ] Análisis de errores
- [ ] Discusión de mejoras

---

## 📚 Referencias

- **Dataset**: [Beijing PM2.5 - UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)
- **AQI Standards**: [EPA AQI Basics](https://www.airnow.gov/aqi/aqi-basics/)
- **Databricks Docs**: [Databricks Documentation](https://docs.databricks.com/)
- **Spark MLlib**: [MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- **MLflow**: [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

---

## 🛠️ Tecnologías Utilizadas

- **Databricks**: Plataforma de análisis
- **Apache Spark**: Procesamiento distribuido
- **PySpark**: API de Python para Spark
- **Spark MLlib**: Machine Learning en Spark
- **MLflow**: Tracking de experimentos
- **Python**: Pandas, NumPy, Matplotlib
- **VS Code**: Editor con extensión de Databricks

---

## 👤 Autor

**Proyecto de Cloud Computing - Databricks**  
Predicción de Calidad del Aire con Machine Learning

---

## 📝 Notas Técnicas

### Problemas Resueltos

#### Error: `Imputer` no permitido
**Problema**:
```
py4j.security.Py4JSecurityException: Constructor public org.apache.spark.ml.feature.Imputer
```

**Solución**: Usar funciones nativas de Spark
```python
# En lugar de MLlib Imputer
means = {col: df.select(F.mean(col)).first()[0] for col in columns}
df = df.withColumn(col, F.when(F.col(col).isNull(), means[col]).otherwise(F.col(col)))
```

#### Error: Columnas con puntos
**Problema**: Nombres de columnas con `.` causan errores
**Solución**:
```python
for c in df.columns:
    if "." in c:
        df = df.withColumnRenamed(c, c.replace(".", "_"))
```

---

## 📊 Estado del Proyecto

```
[████████████████░░░░░░░░░░░░] 40% Completado

✅ Configuración de Databricks
✅ Exploración del catálogo
✅ Data Ingestion
✅ Data Cleaning & EDA
🔄 Feature Engineering (en progreso)
⏳ Model Training
⏳ MLflow Tracking
⏳ Evaluation & Discussion
```

---

**Última actualización**: 20 de Octubre, 2025
