"""
Streamlit App para Predicción de Ictus
======================================

Interfaz interactiva que carga un modelo entrenado (modelo.pkl) 
y permite realizar predicciones ingresando edad y nivel de glucosa.
"""

import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Configurar página
st.set_page_config(page_title="Predicción de Ictus", layout="centered")

# Título y descripción
st.title("🏥 Predicción de Ictus")
st.write("""
Esta aplicación utiliza un modelo de Machine Learning (Random Forest) 
entrenado en datos clínicos para predecir la probabilidad de ictus.
""")

# Verificar si el modelo existe
if not os.path.exists("modelo.pkl"):
    st.error("""
    ⚠️ Archivo 'modelo.pkl' no encontrado.
    
    Por favor, ejecuta primero el notebook `main.ipynb` para entrenar y guardar el modelo.
    """)
    st.stop()

# Cargar el modelo
try:
    model = joblib.load("modelo.pkl")
    st.success("✅ Modelo cargado correctamente")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {e}")
    st.stop()

# Interfaz de entrada de datos
st.subheader("Ingrese los datos del paciente:")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input(
            "Edad (años)",
            min_value=0.0,
            max_value=120.0,
            value=50.0,
            step=1.0
        )
    
    with col2:
        glucose = st.number_input(
            "Nivel de glucosa promedio",
            min_value=0.0,
            max_value=300.0,
            value=100.0,
            step=1.0
        )
    
    # Botón para predecir
    submitted = st.form_submit_button("🔮 Realizar Predicción")

# Realizar predicción
if submitted:
    try:
        # Nota: el modelo espera todas las features. 
        # Aquí preparamos datos mínimos; ajusta según tus features reales.
        # Para uso completo, necesitarías incluir todas las features entrenadas.
        
        # Crear array con los datos (usa edad y glucosa como ejemplo)
        # En producción, deberías incluir TODAS las features del modelo
        input_data = np.array([[age, glucose, 0, 0, 0, 0, 0, 0, 0]])
        
        # Realizar predicción
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Mostrar resultados
        st.subheader("Resultado de la Predicción:")
        
        if prediction == 1:
            st.warning(f"⚠️ **Riesgo ALTO de Ictus** (Probabilidad: {prediction_proba[1]:.2%})")
        else:
            st.success(f"✅ **Riesgo BAJO de Ictus** (Probabilidad: {prediction_proba[1]:.2%})")
        
        # Mostrar probabilidades detalladas
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sin Ictus", f"{prediction_proba[0]:.2%}")
        with col2:
            st.metric("Con Ictus", f"{prediction_proba[1]:.2%}")
    
    except ValueError as e:
        st.error(f"""
        ❌ Error en la predicción: {e}
        
        **Nota**: El modelo requiere todas las features de entrenamiento.
        Actualmente se están usando solo edad y glucosa como ejemplo.
        
        Para usar el modelo completo, ajusta este script con todas las features
        que se utilizaron durante el entrenamiento en `main.ipynb`.
        """)
    except Exception as e:
        st.error(f"❌ Error inesperado: {e}")

# Información adicional
st.divider()
st.info("""
**ℹ️ Información Importante:**
- Este modelo fue entrenado con un dataset específico de accidentes cerebrovasculares.
- Las predicciones son solo para propósitos educativos y de demostración.
- **NO utilices esta aplicación para diagnósticos médicos reales.**
- Para diagnósticos médicos, consulta siempre con profesionales de la salud.
""")
