# 🚀 Cómo conectar VS Code al cluster ML

## Paso 1: Abrir configuración de Databricks

1. En VS Code, presiona `Ctrl+Shift+P`
2. Escribe: `Databricks: Configure Cluster`
3. Selecciona tu workspace

## Paso 2: Seleccionar el nuevo cluster

- Aparecerá una lista de clusters
- Selecciona: **air-quality-ml-cluster** (o el nombre que le pusiste)
- Espera a que diga "Connected" ✅

## Paso 3: Verificar que funciona MLlib

Ejecuta este código de prueba:

```python
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler

print("✅ MLlib imports funcionan!")

# Crear un VectorAssembler de prueba
assembler = VectorAssembler(inputCols=["feature1"], outputCol="features")
print("✅ VectorAssembler creado exitosamente!")

# Crear un RandomForestRegressor de prueba
rf = RandomForestRegressor(featuresCol="features", labelCol="label")
print("✅ RandomForestRegressor creado exitosamente!")

print("\n🎉 ¡MLlib está funcionando correctamente!")
```

Si ves los ✅ sin errores, **MLlib está listo**.

---

## 📋 Diferencias entre clusters:

| Característica | Runtime Standard | Runtime ML |
|----------------|------------------|------------|
| MLlib básico | ✅ | ✅ |
| **MLlib completo** | ❌ Bloqueado | ✅ Funciona |
| VectorAssembler | ❌ | ✅ |
| RandomForest | ❌ | ✅ |
| GBT | ❌ | ✅ |
| scikit-learn | ✅ (manual) | ✅ Pre-instalado |
| TensorFlow | ❌ | ✅ Pre-instalado |
| PyTorch | ❌ | ✅ Pre-instalado |

---

## 🎯 Archivo a ejecutar con MLlib:

Una vez conectado al cluster ML, ejecuta:

```
data_ingestion_clean_analytics.py
```

Este archivo SÍ usa MLlib (RandomForestRegressor de pyspark.ml).

---

## ⚠️ Importante:

- **Runtime ML consume más recursos** (y puede costar más)
- **Configura auto-terminación** (30 min de inactividad)
- **Detén el cluster** cuando no lo uses:
  ```
  Databricks → Compute → Tu cluster → Terminate
  ```

---

## 🔄 Cambiar entre clusters:

Puedes tener **ambos clusters** y cambiar según necesites:

- **Cluster Standard** → Para `05_model_training_SKLEARN.py` (sklearn)
- **Cluster ML** → Para `data_ingestion_clean_analytics.py` (MLlib)

En VS Code:
```
Ctrl+Shift+P → Databricks: Configure Cluster → Selecciona el que necesites
```
