import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Cargar el modelo entrenado
with open('models/clustering_model.pkl', 'rb') as file:
    model = pickle.load(file) # Tuvo que haber creado con la misma versión de s

# Encabezado de la app
st.write("""
# Segmento de clientes
Este es un estimador del segmento del cliente usando un modelo de Machine Learning.
""")
# Sidebar
st.sidebar.header('Datos de entrada')

default_values = {
    'Gender':'Female', 
    'Age': 35, 
    'Annual Income (k$)': 90, 
    'Spending Score (1-100)': 10
}

def user_input_features():    
    data = {}
    for key, value in default_values.items():
        if key == 'Gender':
            gender = st.sidebar.selectbox(f'Enter {key}:', ['Male', 'Female'], index=1)
            input_value = 0 if gender == 'Male' else 1
        else:
            input_value = st.sidebar.number_input(f'Enter {key}:', value=value)
        data[key] = input_value   
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Mostrar los datos de entrada
st.subheader('Datos de entrada')
df.columns = default_values.keys()
df_temp = df.copy()
df_temp['Gender'] = df['Gender'].apply(lambda x: 'Male' if x == 0 else 'Female')
st.dataframe(df_temp, hide_index=True)

# Realizar la predicción
pred = model.predict(df.values)

st.subheader('Predicción')
left_column, right_column = st.columns([0.4, 0.6])
with left_column:
    # Mostrar la predicción
    st.write("El segmento estimado es:")
    st.info(f"{pred[0]}")

with right_column:
    # Show an image depending on the prediction
    pass

# Código para ejecutar la app
# streamlit run app.py

# Ejemplos de datos de entrada para cada especie de flor Iris
# setosa = [5.1, 3.5, 1.4, 0.2]
# versicolor = [7.0, 3.2, 4.7, 1.4]
# virginica = [6.3, 3.3, 6.0, 2.5]