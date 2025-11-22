# iris
# Proyecto de Clasificación de Iris — Streamlit + RandomForest  
**Autores:**  
- Melanny Doncel (10919)  
- Jose Diaz (10895)

## Descripción del Proyecto
Este proyecto implementa una aplicación interactiva desarrollada en **Python** utilizando **Streamlit**.  
La aplicación permite:

- Cargar y visualizar el dataset Iris.
- Entrenar un modelo de clasificación usando **RandomForestClassifier**.
- Mostrar métricas de rendimiento del modelo.
- Realizar predicciones a partir de valores ingresados por el usuario.
- Explorar los datos mediante visualizaciones interactivas 2D y 3D.

El dataset utilizado es el Iris Dataset incluido en scikit-learn.

## Estructura del Proyecto

1. Exploración de Datos

Visualización del dataset.

Estadísticas descriptivas.

Gráficas 3D interactivas con Plotly.

Histogramas y scatter matrix.

2. Entrenamiento del Modelo

Entrenamiento mediante RandomForestClassifier.

Generación automática del archivo iris_rf_model.joblib.

Cálculo de métricas como accuracy y validación cruzada.

3. Predicción Interactiva

El usuario ingresa las medidas del iris.

El modelo predice la especie.

Se agrega el punto en la gráfica 3D.

