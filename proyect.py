# Proyect.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import os

MODEL_PATH = "iris_rf_model.joblib"

@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df['species'] = df['target'].map({i: name for i, name in enumerate(iris.target_names)})
    df = df.rename(columns={'sepal length (cm)':'sepal_length',
                            'sepal width (cm)':'sepal_width',
                            'petal length (cm)':'petal_length',
                            'petal width (cm)':'petal_width'})
    return df, iris

def train_and_save_model(df):
    X = df[['sepal_length','sepal_width','petal_length','petal_width']].values
    y = df['species'].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # metrics
    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='macro'),
        "recall": recall_score(y_test, y_pred, average='macro'),
        "f1": f1_score(y_test, y_pred, average='macro'),
        "classification_report": classification_report(y_test, y_pred, target_names=le.inverse_transform([0,1,2]))
    }

    # cross-val
    cv_scores = cross_val_score(clf, X, y_enc, cv=5, scoring='accuracy')

    # save
    joblib.dump({"model": clf, "label_encoder": le}, MODEL_PATH)
    return metrics, cv_scores

def load_model():
    data = joblib.load(MODEL_PATH)
    return data['model'], data['label_encoder']

def predict_single(model, le, features):
    pred = model.predict(np.array(features).reshape(1,-1))
    return le.inverse_transform(pred)[0]

# --- Streamlit layout ---
st.set_page_config(page_title="Iris Species Classification", layout="wide")
st.title("Iris Species Classification — Proyecto")

df, iris = load_data()

# Sidebar: entrenamiento / info
with st.sidebar:
    st.header("Configuración")
    if st.button("(Re)Entrenar modelo"):
        with st.spinner("Entrenando modelo..."):
            metrics, cv_scores = train_and_save_model(df)
        st.success("Modelo entrenado y guardado.")
        st.write("Accuracy (test):", metrics["accuracy"])
        st.write("CV accuracy (5-fold):", cv_scores.mean())
    st.markdown("---")
    st.write("Nº muestras:", len(df))
    st.write("Clases:", df['species'].unique())
    st.markdown("Repositorio y entregables: Añadir link en README.")

# Main: show dataset and visuals
st.subheader("Exploración de datos")
col1, col2 = st.columns([2,1])

with col1:
    st.dataframe(df[['sepal_length','sepal_width','petal_length','petal_width','species']])

with col2:
    st.write("Estadísticas")
    st.write(df.describe())

st.subheader("Visualizaciones")
tab1, tab2 = st.tabs(["Scatter 3D", "Histogramas & Pares"])

with tab1:
    x_axis = st.selectbox("Eje X", ['sepal_length','sepal_width','petal_length','petal_width'], index=2)
    y_axis = st.selectbox("Eje Y", ['sepal_length','sepal_width','petal_length','petal_width'], index=3)
    z_axis = st.selectbox("Eje Z", ['sepal_length','sepal_width','petal_length','petal_width'], index=0)
    fig3d = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color='species', symbol='species',
                         hover_data=['sepal_length','sepal_width','petal_length','petal_width'])
    st.plotly_chart(fig3d, use_container_width=True)

with tab2:
    feature = st.selectbox("Histograma de:", ['sepal_length','sepal_width','petal_length','petal_width'])
    fig_hist = px.histogram(df, x=feature, color='species', barmode='overlay')
    st.plotly_chart(fig_hist, use_container_width=True)
    st.write("Scatter matrix (pares):")
    fig_matrix = px.scatter_matrix(df, dimensions=['sepal_length','sepal_width','petal_length','petal_width'],
                                   color='species')
    st.plotly_chart(fig_matrix, use_container_width=True)

# Load or train model on first run
if os.path.exists(MODEL_PATH):
    model, le = load_model()
else:
    with st.spinner("Entrenando modelo por primera vez..."):
        metrics, cv_scores = train_and_save_model(df)
    model, le = load_model()

# Show model metrics
st.subheader("Métricas del modelo")
X = df[['sepal_length','sepal_width','petal_length','petal_width']].values
y = le.transform(df['species'])
# simple evaluation using cross_val_score for display
st.write("Accuracy (CV 5-fold):", float(np.mean(cross_val_score(model, X, y, cv=5, scoring='accuracy'))))

# Predict panel
st.subheader("Panel de predicción — ingresa medidas")
c1, c2, c3, c4 = st.columns(4)
with c1:
    sepal_length = st.number_input("Sepal length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
with c2:
    sepal_width = st.number_input("Sepal width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
with c3:
    petal_length = st.number_input("Petal length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
with c4:
    petal_width = st.number_input("Petal width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

if st.button("Predecir especie"):
    feat = [sepal_length, sepal_width, petal_length, petal_width]
    species_pred = predict_single(model, le, feat)
    st.success(f"Especie predicha: {species_pred}")

    # add point to 3D scatter (recreate fig)
    fig3d_new = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color='species', symbol='species',
                             hover_data=['sepal_length','sepal_width','petal_length','petal_width'])
    # append new point
    new_df = pd.DataFrame([{
        x_axis: feat[['sepal_length','sepal_width','petal_length','petal_width'].index(x_axis)],
        y_axis: feat[['sepal_length','sepal_width','petal_length','petal_width'].index(y_axis)],
        z_axis: feat[['sepal_length','sepal_width','petal_length','petal_width'].index(z_axis)],
        'species': 'Nuevo (pred: ' + species_pred + ')'
    }])
    fig3d_new.add_scatter3d(x=new_df[x_axis], y=new_df[y_axis], z=new_df[z_axis],
                            mode='markers', marker=dict(size=6, symbol='x'), name='Nuevo')
    st.plotly_chart(fig3d_new, use_container_width=True)

st.markdown("---")
st.write("Informe: seguir el workflow en README. Fuente de datos: UCI / Kaggle.")
