import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Cargar el modelo entrenado
with open('models/regression_model.pkl', 'rb') as file:
    model = pickle.load(file) # Tuvo que haber creado con la misma versión de s

# Encabezado de la app
st.write("""
# Predicción de precio de casas
Este es un estimador de precios de casas creado con un modelo de Machine Learning.
""")
# Sidebar
st.sidebar.header('Datos de entrada')

default_values = {'MedInc': 3.84, 
                  'HouseAge': 52.0, 
                  'AveRooms': 6.28, 
                  'AveBedrms': 1.08, 
                  'Population': 565, 
                  'AveOccup': 2.18, 
                  'Latitude': 37.85, 
                  'Longitude': -122.25}
def user_input_features():    
    data = {}
    for key, value in default_values.items():
        input_value = st.sidebar.number_input(f'Enter {key}:', value=value)
        data[key] = input_value   
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Mostrar los datos de entrada
st.subheader('Datos de entrada')
df.columns = default_values.keys()
st.dataframe(df, hide_index=True)

# Realizar la predicción
pred = model.predict(df.values)

st.subheader('Predicción')
left_column, right_column = st.columns([0.4, 0.6])
with left_column:
    # Mostrar la predicción
    st.write("El precio estimado de la casa es:")
    st.info(f"${pred[0]*100000:,.2f}")

with right_column:
    # Show an image depending on the prediction
    pass

# Código para ejecutar la app
# streamlit run app.py

# Ejemplos de datos de entrada para cada especie de flor Iris
# setosa = [5.1, 3.5, 1.4, 0.2]
# versicolor = [7.0, 3.2, 4.7, 1.4]
# virginica = [6.3, 3.3, 6.0, 2.5]