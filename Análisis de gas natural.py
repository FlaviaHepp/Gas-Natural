

"""El gas natural es un combustible fósil versátil y ampliamente utilizado que desempeña un papel crucial en el panorama energético de Estados 
Unidos. Es una fuente de energía primaria para calefacción, cocina y generación de electricidad en entornos residenciales, comerciales e 
industriales. Este conjunto de datos contiene datos mensuales de consumo de gas natural para los Estados Unidos desde enero de 2014 hasta enero 
de 2024 , desglosados ​​por estado, sector (residencial, comercial, industrial, combustible para vehículos y energía eléctrica) y proceso de 
consumo específico. Los datos provienen de la Administración de Información Energética (EIA) de EE. UU.

Cada fila representa el valor del consumo de gas natural para un estado, sector y proceso específico. La columna "valor" proporciona la cantidad 
de consumo en millones de pies cúbicos (MMcf). Faltan algunos valores, probablemente debido a que no hay datos disponibles.

Las columnas clave son:

duoárea: abreviatura del estado
nombre-área: nombre del estado
producto: Producto energético (todas las filas tienen "EPG0" para Gas Natural)
sector: Sector de consumo (por ejemplo, "VRS" para residencial, "VCS" para comercial)
proceso: Proceso de consumo específico dentro del sector
valor: Consumo mensual en millones de pies cúbicos (MMcf)
Este conjunto de datos granulares permite un análisis detallado de los patrones de consumo de gas natural en todos los estados y sectores. Podría usarse para comparar el consumo entre estados, identificar los sectores consumidores más grandes en cada estado, rastrear las tendencias de consumo estacional y más. Los datos pueden ser de interés para analistas de energía, empresas de servicios públicos, formuladores de políticas y otras personas que investigan el uso del gas natural."""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor


df = pd.read_csv('data.csv')

# Seleccionar columnas numéricas para estadísticas resumidas
numerical_cols = df.select_dtypes(include=[np.number]).columns

# Calcular estadísticas resumidas para columnas numéricas
summary_statistics = df[numerical_cols].describe()

# Mostrando estadísticas resumidas
print("Estadísticas resumidas de consumo de gas natural:\n")
print(summary_statistics)

# Analizar las tendencias a lo largo del tiempo agregando datos mensualmente a lo largo de todos los años
monthly_consumption = df.groupby('month')['value'].mean()
plt.figure(figsize=(10, 6))
monthly_consumption.plot(kind='bar')
plt.title('Consumo Promedio Mensual de Gas Natural\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Mes\n')
plt.ylabel('Consumo medio\n')
plt.xticks(rotation=360)
plt.show()

# Convierta 'año' y 'mes' a un formato de fecha y hora para facilitar el análisis de series temporales
df['date'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1))

# Establecer 'fecha' como índice
df.set_index('date', inplace=True)

# Agrupar por fecha y sumar el consumo total de cada mes
monthly_consumption = df.groupby(pd.Grouper(freq='M'))['value'].sum()

# Trazar la tendencia del consumo de gas natural a lo largo del tiempo.
plt.figure(figsize=(14, 7))
monthly_consumption.plot(title='Tendencia del consumo de gas natural a lo largo del tiempo\n')
plt.xlabel('Fecha\n')
plt.ylabel('Consumo total\n')
plt.show()

# Calcular el consumo medio de cada año.
yearly_consumption = df.groupby('year')['value'].mean()

# Trazar la tendencia del consumo promedio de gas natural por año
plt.figure(figsize=(10, 6))
yearly_consumption.plot(kind='bar')
plt.title('Consumo Promedio Anual de Gas Natural\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Año\n')
plt.ylabel('Consumo medio\n')
plt.xticks(rotation=360)
plt.show()

# Descomposición estacional para analizar tendencias, estacionalidad y residuos.
decomposition = seasonal_decompose(monthly_consumption, model='additive')

# Trazar los componentes descompuestos de la serie temporal.
fig = decomposition.plot()
fig.set_size_inches(14, 7)
plt.show()

# Agrupar datos por zona y calcular el consumo total de cada zona
area_consumption = df.groupby('area-name')['value'].sum().sort_values()

# Identificar la zona de mayor y menor consumo

# Encuentre el área con mayor consumo (excluyendo EE. UU.)
highest_area_consumption = area_consumption.drop('U.S.').idxmax()
lowest_area_consumption = area_consumption.idxmin()

# Encuentre el área con mayor consumo (excluyendo EE. UU.)
highest_consumption = area_consumption.drop('U.S.').idxmax()

print(f"Zona con mayor consumo de gas natural: {highest_area_consumption} - {area_consumption.max()}")
print(f"Zona con menor consumo de gas natural: {lowest_area_consumption} - {area_consumption.min()}")

# Agrupar datos por proceso y calcular el consumo total de cada proceso
process_consumption = df.groupby('process-name')['value'].sum().sort_values()

# Identificar el proceso de mayor y menor consumo
highest_process_consumption = process_consumption.idxmax()
lowest_process_consumption = process_consumption.idxmin()

print(f"Proceso con mayor consumo de gas natural: {highest_process_consumption} - {process_consumption.max()}")
print(f"Proceso con menor consumo de gas natural: {lowest_process_consumption} - {process_consumption.min()}")

# Visualización del consumo del área (excluyendo EE. UU.)
plt.figure(figsize=(14, 7))
area_consumption.drop('U.S.').plot(kind='bar')  # Excluir 'EE.UU.' usando .drop()
plt.title('Consumo de gas natural por área (excluyendo EE. UU.)\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Área\n')
plt.ylabel('Consumo total\n')
plt.show()

# Visualizando el consumo del proceso
plt.figure(figsize=(14, 7))
process_consumption.plot(kind='bar')
plt.title('Consumo de Gas Natural por Proceso\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Proceso\n')
plt.ylabel('Consumo total\n')
plt.xticks(rotation=45)
plt.show()

print("Estadísticas resumidas de consumo de gas natural:")
print(summary_statistics)
print("\nZona con mayor consumo de gas natural: {} - {}".format(highest_area_consumption, area_consumption.max()))
print("Zona con menor consumo de gas natural: {} - {}".format(lowest_area_consumption, area_consumption.min()))
print("Proceso con mayor consumo de gas natural: {} - {}".format(highest_process_consumption, process_consumption.max()))
print("Proceso con menor consumo de gas natural: {} - {}".format(lowest_process_consumption, process_consumption.min()))

# Manejo de valores faltantes en el conjunto de datos
# Primero, verifique si faltan valores en cada columna
missing_values = df.isnull().sum()
print("Valores faltantes por columna antes de la imputación:")
print(missing_values)

# Para columnas numéricas, impute los valores faltantes con la mediana de la columna
for col in numerical_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# Para columnas categóricas, impute los valores faltantes con la moda (valor más frecuente) de la columna
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

# Verifique nuevamente los valores faltantes para asegurarse de que todos hayan sido manejados
missing_values_after = df.isnull().sum()
print("\nValores faltantes por columna después de la imputación:")
print(missing_values_after)

# Copiar el DataFrame para preprocesamiento
df_preprocessed = df.copy()

# Inicializar LabelEncoder
label_encoder = LabelEncoder()

# Lista de columnas categóricas para codificar
categorical_cols = ['duoarea', 'area-name', 'product', 'product-name', 'process', 'process-name', 'series', 'series-description', 'units']

# Aplicar codificación de etiquetas a cada columna categórica
for col in categorical_cols:
    df_preprocessed[col] = label_encoder.fit_transform(df_preprocessed[col])

# Mostrar las primeras filas del DataFrame preprocesado para verificar los cambios
print(df_preprocessed.head())

# Preparar el conjunto de datos para el pronóstico de series temporales
# Seleccionar características relevantes para el modelo.
features = ['year', 'month', 'duoarea', 'area-name', 'product', 'product-name', 'process', 'process-name', 'series', 'series-description', 'units']
target = 'value'

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(df_preprocessed[features], df_preprocessed[target], test_size=0.2, random_state=42)

# Inicializando y entrenando el modelo regresor XGBoost
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
predictions = model.predict(X_test)

# Evaluación del rendimiento del modelo.
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Error cuadrático medio (RMSE) en el conjunto de prueba: {rmse}")

# Previsión del consumo futuro de gas natural
# Nota: Para realizar pronósticos reales, necesitará valores futuros de las características. Aquí, estamos demostrando la evaluación del modelo.

# Calcular el error porcentual absoluto medio (MAPE)
mape = mean_absolute_percentage_error(y_test, predictions)

print(f"Error porcentual absoluto medio (MAPE) en el conjunto de prueba: {mape}")

# calcular la puntuación r2
r2 = r2_score(y_test, predictions)
print(f"Puntuación R2 en el conjunto de pruebas: {r2}")

# Visualización del rendimiento del modelo: consumo real frente a consumo previsto
plt.figure(figsize=(14, 7))
plt.scatter(y_test, predictions, alpha=0.5, color = "fuchsia")
plt.title('Consumo de gas natural real vs. previsto\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Consumo real\n')
plt.ylabel('Consumo previsto\n')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal', color = "white")
plt.legend()
plt.show()

df['month'].unique()

df['year'].unique()

df['product-name'].value_counts()

sns.lineplot(data=df,x='month',y='value',marker='^', color= 'fuchsia')
plt.show()

monthly_consumption = df.groupby('month')['value'].sum().reset_index()
print(monthly_consumption)

sns.lineplot(data=monthly_consumption,x='month',y='value',marker='o',color='red')
plt.show()

year_consumption = df.groupby('year')['value'].sum().reset_index()
print(year_consumption)

sns.lineplot(data=year_consumption,x='year',y='value', color= 'limegreen', marker = "v")
plt.show()