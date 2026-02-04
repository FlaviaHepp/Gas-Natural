# Gas-Natural
Análisis de consumo de gas natural

Este conjunto de datos contiene datos mensuales de consumo de gas natural para los Estados Unidos desde enero de 2014 hasta enero de 2024 , desglosados ​​por estado, sector (residencial, comercial, industrial, combustible para vehículos y energía eléctrica) y proceso de consumo específico. Los datos provienen de la Administración de Información Energética (EIA) de EE. UU.

Cada fila representa el valor del consumo de gas natural para un estado, sector y proceso específico. La columna "valor" proporciona la cantidad de consumo en millones de pies cúbicos (MMcf). Faltan algunos valores, probablemente debido a que no hay datos disponibles.

**Las columnas clave son:**
*duoárea:* abreviatura del estado
*nombre-área:* nombre del estado
*producto:* Producto energético (todas las filas tienen "EPG0" para Gas Natural)
*sector:* Sector de consumo (por ejemplo, "VRS" para residencial, "VCS" para comercial)
*proceso:* Proceso de consumo específico dentro del sector
*valor:* Consumo mensual en millones de pies cúbicos (MMcf)

**Desarrollo:** 
Análisis detallado del consumo de gas natural en Estados Unidos, desglosado por estado, sector y proceso. Incluyó la identificación de tendencias estacionales, análisis de patrones de consumo y predicciones basadas en series temporales para soportar decisiones estratégicas en políticas energéticas.
Herramientas clave: Python, pandas, numpy, matplotlib, seaborn, statsmodels, XGBoost.

**Resultados clave:**
Identificación de áreas y sectores con mayor consumo energético.
Análisis de tendencias estacionales y anuales del consumo de gas natural.
Predicción precisa del consumo con un modelo de regresión (XGBRegressor) obteniendo una puntuación R2R^2R2 de X% y un RMSE de Y.
*Habilidades aplicadas:* Análisis exploratorio de datos, modelado de series temporales, evaluación de modelos, visualización avanzada y manejo de datos faltantes.


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
