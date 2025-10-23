# 🚨 Solución al Error de MLlib Restringido

## Problema
```
py4j.security.Py4JSecurityException: Constructor public org.apache.spark.ml.feature.VectorAssembler is not whitelisted
```

Este error ocurre porque tu cluster usa un **Databricks Runtime estándar** que restringe clases de MLlib por seguridad.

---

## ✅ Solución: Cambiar a Databricks Runtime ML

### Opción 1: Crear un nuevo cluster con Runtime ML (RECOMENDADO)

1. **En Databricks, ve a Compute (o Clusters)**
2. **Click en "Create Cluster"**
3. **Configuración recomendada:**
   ```
   Cluster name: proyecto-calidad-aire-ml
   Cluster mode: Single Node
   Databricks runtime: 13.3 LTS ML (o superior)
                      ↑↑↑ IMPORTANTE: Debe ser "ML"
   Node type: Standard_DS3_v2
   Terminate after: 30 minutes
   ```

4. **Click "Create Cluster"**
5. **Espera 3-5 minutos** mientras se inicia
6. **En tus notebooks, selecciona este nuevo cluster**

---

### Opción 2: Modificar tu cluster existente

1. **Ve a tu cluster actual**
2. **Click en "Edit"**
3. **En "Databricks Runtime Version":**
   - Cambia de: `13.3 LTS` 
   - A: `13.3 LTS ML` ← Nota la "ML"
4. **Click "Confirm"**
5. **Reinicia el cluster**

---

## 🔍 Verificar que funciona

Ejecuta esta celda en un notebook conectado al nuevo cluster:

```python
# Test MLlib availability
try:
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import RandomForestRegressor
    print("✅ MLlib está disponible y funcionando")
    print("✅ Puedes ejecutar el proyecto sin problemas")
except Exception as e:
    print(f"❌ Error: {e}")
    print("⚠️ Verifica que estás usando Databricks Runtime ML")
```

---

## 📋 Diferencias entre Runtime estándar y ML

| Característica | Runtime Estándar | Runtime ML |
|----------------|------------------|------------|
| Spark/PySpark | ✅ Sí | ✅ Sí |
| SQL | ✅ Sí | ✅ Sí |
| **MLlib** | ⚠️ Restringido | ✅ **Completo** |
| **MLflow** | ⚠️ Básico | ✅ **Pre-instalado** |
| scikit-learn | ❌ No | ✅ **Pre-instalado** |
| TensorFlow | ❌ No | ✅ Pre-instalado |
| PyTorch | ❌ No | ✅ Pre-instalado |
| XGBoost | ❌ No | ✅ Pre-instalado |

---

## ⚡ Si NO puedes cambiar el cluster

Si no tienes permisos para crear/modificar clusters:

### Alternativa temporal (menos ideal):

1. **Contacta al administrador** de tu workspace
2. **Solicita acceso** a un cluster con Runtime ML
3. **O solicita** que habiliten las clases de MLlib en tu cluster actual

### Mientras tanto, puedes:

- Completar los notebooks 01, 02, 03 (no requieren MLlib)
- El notebook 04 (training) sí requiere Runtime ML

---

## 💡 ¿Por qué Runtime ML es mejor para este proyecto?

1. ✅ **MLlib sin restricciones** - Todas las clases disponibles
2. ✅ **MLflow pre-instalado** - Tracking listo para usar
3. ✅ **Librerías ML incluidas** - scikit-learn, XGBoost, etc.
4. ✅ **Optimizado para ML** - Mejor performance en entrenamientos
5. ✅ **No requiere instalaciones** - Todo pre-configurado

---

## 🎯 Resumen de Acción

**PASO 1**: Crear cluster nuevo con **Databricks Runtime 13.3 LTS ML**

**PASO 2**: Conectar tus notebooks a ese cluster

**PASO 3**: Ejecutar notebooks en orden:
   - 01_data_ingestion.py
   - 02_data_cleaning.py
   - 03_feature_engineering.py
   - 04_model_training_mlflow.py ← Ahora funcionará

**PASO 4**: ¡Disfrutar de MLflow tracking y modelos entrenados!

---

## ❓ Preguntas Frecuentes

**Q: ¿Cuesta más un cluster ML?**
A: No, el costo es similar. Solo usa recursos cuando está activo.

**Q: ¿Puedo usar el mismo cluster para todo?**
A: Sí, un cluster ML puede ejecutar notebooks estándar también.

**Q: ¿Necesito reinstalar librerías?**
A: No, en Runtime ML ya están todas instaladas.

**Q: ¿Funciona con la versión gratuita de Databricks?**
A: Sí, Databricks Community Edition incluye acceso a Runtime ML.

---

## 📞 Soporte Adicional

Si sigues teniendo problemas:

1. Verifica que la versión de Runtime termine en "ML"
2. Reinicia el cluster completamente
3. Verifica que MLflow esté disponible: `import mlflow`
4. Contacta a soporte de Databricks

---

✅ **Con Databricks Runtime ML, tu proyecto funcionará perfectamente sin modificar el código.**
