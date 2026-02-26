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
    # Datos demográficos y clínicos
    st.write("**Información Clínica y Demográfica**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Edad (años)", min_value=0.0, max_value=120.0, value=50.0, step=1.0)
    with col2:
        bmi = st.number_input("IMC (kg/m²)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    with col3:
        glucose = st.number_input("Glucosa promedio", min_value=0.0, max_value=300.0, value=100.0, step=1.0)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        hypertension = st.selectbox("¿Hipertensión?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
    with col2:
        heart_disease = st.selectbox("¿Enfermedad cardíaca?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
    with col3:
        gender_male = st.selectbox("Género", [0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
    
    # Estado civil
    st.write("**Estado Civil y Trabajo**")
    col1, col2 = st.columns(2)
    with col1:
        ever_married = st.selectbox("¿Alguna vez casado?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
    with col2:
        work_type = st.selectbox(
            "Tipo de trabajo",
            ["Govt_job", "Private", "Self-employed", "children"],
            format_func=lambda x: {"Govt_job": "Sector público", "Private": "Privado", "Self-employed": "Autónomo", "children": "Dependiente"}[x]
        )
    
    # Tipo de residencia y hábitos
    st.write("**Residencia y Hábitos**")
    col1, col2 = st.columns(2)
    with col1:
        residence_urban = st.selectbox("Tipo de residencia", [0, 1], format_func=lambda x: "Rural" if x == 0 else "Urbana")
    with col2:
        smoking_status = st.selectbox(
            "Estado de fumador",
            ["never smoked", "formerly smoked", "smokes", "Unknown"],
            format_func=lambda x: {"never smoked": "Nunca ha fumado", "formerly smoked": "Fumador anterior", "smokes": "Fuma actualmente", "Unknown": "Desconocido"}[x]
        )
    
    # Botón para predecir
    submitted = st.form_submit_button("🔮 Realizar Predicción")

# Realizar predicción
if submitted:
    try:
        # Construir vector con todas las features en el orden correcto
        # Features: age, hypertension, heart_disease, avg_glucose_level, bmi, 
        #           gender_Male, ever_married_Yes, work_type_Private, work_type_Self-employed, 
        #           work_type_children, Residence_type_Urban, smoking_status_formerly smoked, 
        #           smoking_status_never smoked, smoking_status_smokes
        
        # Dummies para work_type
        work_type_private = 1 if work_type == "Private" else 0
        work_type_self_employed = 1 if work_type == "Self-employed" else 0
        work_type_children = 1 if work_type == "children" else 0
        
        # Dummies para smoking_status
        smoking_formerly = 1 if smoking_status == "formerly smoked" else 0
        smoking_never = 1 if smoking_status == "never smoked" else 0
        smoking_smokes = 1 if smoking_status == "smokes" else 0
        
        # Crear array con todos los features en orden
        input_data = np.array([[
            age,
            hypertension,
            heart_disease,
            glucose,
            bmi,
            gender_male,
            ever_married,
            work_type_private,
            work_type_self_employed,
            work_type_children,
            residence_urban,
            smoking_formerly,
            smoking_never,
            smoking_smokes
        ]])
        
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
        st.error(f"❌ Error en la predicción: {e}")
    except Exception as e:
        st.error(f"❌ Error inesperado: {e}")

# Información adicional
st.divider()
st.info("""
**ℹ️ Información Importante:**
- Este modelo fue entrenado con un dataset específico de accidentes cerebrovasculares.
- Las predicciones se basan en 14 features clínicas y demográficas.
- Las predicciones son solo para propósitos educativos y de demostración.
- **NO utilices esta aplicación para diagnósticos médicos reales.**
- Para diagnósticos médicos, consulta siempre con profesionales de la salud.

**Features utilizadas en el modelo (14):**
age, hypertension, heart_disease, avg_glucose_level, bmi, gender, ever_married, 
work_type, residence_type, smoking_status
""")
