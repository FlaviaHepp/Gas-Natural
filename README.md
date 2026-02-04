# Análisis y predicción del consumo de gas natural

Este proyecto analiza la evolución histórica del **consumo de gas natural**, combinando **análisis exploratorio de datos (EDA)**, **series temporales** y **modelos de machine learning** para comprender patrones de consumo y evaluar la capacidad de predicción futura.

El trabajo integra datos temporales, geográficos y operativos, con un enfoque orientado a **energía, planificación y forecasting**.

---

## 🌍 Contexto del problema

El gas natural es un recurso energético estratégico.  
Comprender su consumo permite:

- mejorar la planificación energética
- detectar patrones estacionales
- analizar diferencias regionales y por proceso
- construir modelos predictivos de demanda

Este proyecto aborda el problema desde una perspectiva **data-driven**, combinando análisis descriptivo y predictivo.

---

## 🎯 Objetivos

- Analizar tendencias temporales del consumo de gas natural
- Identificar patrones estacionales y ciclos
- Comparar consumo por área geográfica y por proceso
- Preparar datos para modelos predictivos
- Entrenar y evaluar un modelo de ML para predicción de consumo

---

## 📊 Dataset

El dataset contiene información histórica de consumo de gas natural con múltiples dimensiones.

### Variables principales
- `year`, `month`
- `value` – consumo de gas natural
- `area-name` – área geográfica
- `process-name` – tipo de proceso
- `product-name`
- Otras variables categóricas relacionadas con el sistema energético

Los datos se cargan desde el archivo `data.csv`.

---

## 🧹 Limpieza y preparación de datos

- Análisis de valores faltantes
- Imputación:
  - valores numéricos → mediana
  - variables categóricas → moda
- Conversión de variables temporales a formato datetime
- Codificación de variables categóricas mediante **Label Encoding**
- Creación de dataset preprocesado para modelado

---

## 🔍 Análisis exploratorio (EDA)

### Estadísticas descriptivas
- Medidas resumen para variables numéricas
- Identificación de outliers y rangos de consumo

### Análisis temporal
- Consumo promedio mensual
- Tendencia mensual y anual de consumo
- Series temporales agregadas
- Visualización de picos y caídas de consumo

### Descomposición estacional
- Tendencia
- Estacionalidad
- Residuos  
(usando `seasonal_decompose`)

---

## 📍 Análisis por segmentos

- Consumo total por **área geográfica**
- Identificación de:
  - área con mayor consumo
  - área con menor consumo
- Consumo total por **proceso**
- Comparación visual entre áreas y procesos

---

## 🤖 Modelado predictivo

### Enfoque de Machine Learning
- **Tipo de problema:** Regresión
- **Modelo:** XGBoost Regressor
- **Features:**
  - variables temporales
  - variables geográficas
  - variables de proceso y producto
- **Target:** consumo (`value`)

### Evaluación del modelo
- RMSE
- MAPE
- R²
- Comparación visual: valores reales vs. predicciones

---

## 📈 Resultados

- El modelo XGBoost captura patrones no lineales del consumo
- Buen ajuste entre valores reales y predichos
- Se observan patrones estacionales claros en el consumo mensual
- Existen diferencias significativas por área y proceso

---

## 🛠️ Tecnologías utilizadas

- **Python**
- **pandas, numpy**
- **matplotlib, seaborn**
- **statsmodels**
- **scikit-learn**
- **XGBoost**

---

## 📂 Estructura del repositorio

├── data.csv
├── Análisis de gas natural.py
├── README.md


---

## 🚀 Próximos pasos

- Feature engineering específico para series temporales
- Validación temporal (train/test split por fecha)
- Optimización de hiperparámetros del modelo
- Comparación con modelos clásicos (ARIMA / SARIMAX / Prophet)
- Interpretabilidad del modelo (feature importance, SHAP)
- Forecasting a largo plazo

---

## 👤 Autor

**Flavia Hepp**  
Data Scientist / Energy Analytics en formación  
