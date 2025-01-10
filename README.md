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


