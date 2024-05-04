import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Cargar el modelo entrenado
with open('models/classification_model.pkl', 'rb') as file:
    model = pickle.load(file) # Tuvo que haber creado con la misma versión de s

# Encabezado de la app
st.write("""
# Clasificador de flores Iris
Este es un clasificador de flores Iris creado con un modelo de Machine Learning.
""")
# Sidebar
st.sidebar.header('Datos de entrada')

def user_input_features():
    sepal_length = st.sidebar.number_input('Largo del sépalo (cm)', value=5.0)
    sepal_width = st.sidebar.number_input('Ancho del sépalo (cm)', value=3.0)
    petal_length = st.sidebar.number_input('Largo del pétalo (cm)', value=1.0)
    petal_width = st.sidebar.number_input('Ancho del pétalo (cm)', value=0.5)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Mostrar los datos de entrada
st.subheader('Datos de entrada')
df.columns = ['Largo del sépalo (cm)', 'Ancho del sépalo (cm)', 'Largo del pétalo (cm)', 'Ancho del pétalo (cm)']
st.dataframe(df, hide_index=True)

# Realizar la predicción
pred = model.predict(df.values)
pred_proba = model.predict_proba(df.values)

st.subheader('Predicción')
left_column, right_column = st.columns([0.4, 0.6])
with left_column:
    # Mostrar la predicción
    st.write("Tipo de flor iris:")
    iris_species = np.array(['Setosa', 'Versicolor', 'Virginica'])
    st.info((iris_species[pred][0]))

    # Mostrar la probabilidad
    pred_proba_df = pd.DataFrame(pred_proba, columns=iris_species)

    # Cambia los valores a porcentaje texto con el formato adecuado valor * 100 y el signo de porcentaje
    pred_proba_df = pred_proba_df * 100
    pred_proba_df = pred_proba_df.astype(int).astype(str) + '%'
    
    st.dataframe(pred_proba_df, hide_index=True)

with right_column:
    # Show an image depending on the prediction
    _, image_column, _ = st.columns([.1,.6,.1])
    with image_column:
        st.image('static/images/iris-' + iris_species[pred][0].astype(str).lower() + '.png', width=200)

# Código para ejecutar la app
# streamlit run app.py

# Ejemplos de datos de entrada para cada especie de flor Iris
# setosa = [5.1, 3.5, 1.4, 0.2]
# versicolor = [7.0, 3.2, 4.7, 1.4]
# virginica = [6.3, 3.3, 6.0, 2.5]