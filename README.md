# 🧠 Predicción de Ictus con Streamlit y Docker

Aplicación web desarrollada con **Streamlit** para la predicción de riesgo de ictus.  
El proyecto está contenerizado usando **Docker** para facilitar su ejecución en cualquier entorno.

---

## 🚀 Requisitos

- Tener instalado **Docker Desktop**
- Tener Docker en ejecución

Desde la carpeta donde se encuentra el Dockerfile, ejecutar:

docker build -t prediccion-ictus .
2️⃣ Ejecutar el contenedor
docker run -p 8501:8501 prediccion-ictus
3️⃣ Abrir la aplicación

Abrir el navegador y acceder a:

http://localhost:8501

Si todo está correcto, se mostrará la aplicación web.
Aquí tienes el texto actualizado incluyendo el uso de Docker de forma profesional y coherente con el resto del documento:

---

# 🧠 Predicción de Ictus (Stroke) — Proyecto de Aprendizaje Supervisado

## 📌 Descripción del Proyecto

Este proyecto tiene como objetivo predecir si un paciente está en riesgo de sufrir un **ictus (stroke)** a partir de variables demográficas y clínicas.

Se trata de un problema de **clasificación binaria supervisada**, donde el modelo debe identificar correctamente a los pacientes con mayor riesgo.

---

## 🎯 Problema de Negocio

El ictus es una de las principales causas de muerte y discapacidad a nivel mundial. Detectar pacientes en riesgo de forma temprana puede ayudar a tomar medidas preventivas.

El dataset presenta un fuerte desbalanceo:

* 95% → No ictus
* 5% → Ictus

Por lo tanto, el objetivo principal del modelo no es maximizar la accuracy, sino:

> 🔎 Maximizar el **Recall de la clase positiva (stroke = 1)**
> En problemas médicos, es más grave un falso negativo que un falso positivo.

---

## 📊 Descripción del Dataset

**Variable objetivo:**

* `stroke` → 0 (No ictus), 1 (Ictus)

**Variables numéricas:**

* `age`
* `avg_glucose_level`
* `bmi`

**Variables binarias:**

* `hypertension`
* `heart_disease`

**Variables categóricas:**

* `gender`
* `ever_married`
* `work_type`
* `Residence_type`
* `smoking_status`

---

## 🔍 Análisis Exploratorio de Datos (EDA)

Principales hallazgos:

* Los pacientes con ictus tienden a tener mayor edad.
* Los niveles elevados de glucosa se asocian con mayor incidencia de ictus.
* La hipertensión y las enfermedades cardíacas muestran una relación clara con la variable objetivo.
* El dataset está fuertemente desbalanceado, lo que requiere técnicas especiales de modelado.

---

## 🧹 Preprocesamiento de Datos

Se realizaron las siguientes transformaciones:

* Eliminación de categorías poco representativas (si aplica).
* Imputación de valores nulos en `bmi` utilizando la mediana.
* Codificación One-Hot para variables categóricas.
* Separación en conjunto de entrenamiento y prueba con `stratify`.
* Manejo del desbalanceo mediante:

  * `class_weight="balanced"`
  * Técnica de sobremuestreo SMOTE.

---

## 🤖 Modelos Implementados

### 1️⃣ Regresión Logística (Modelo Base)

Modelo inicial para establecer una línea base de rendimiento.

### 2️⃣ Regresión Logística con balanceo

Uso de `class_weight` para mejorar el recall de la clase minoritaria.

### 3️⃣ Random Forest

Modelo basado en árboles para capturar relaciones no lineales.

### 4️⃣ Optimización de Hiperparámetros

Uso de `GridSearchCV` con validación cruzada, optimizando la métrica **recall**.

---

## 📈 Evaluación del Modelo

Métricas utilizadas:

* Matriz de confusión
* Precision
* Recall (métrica principal)
* F1-score
* ROC-AUC
* Validación cruzada

Se priorizó mejorar la detección de casos positivos (ictus), minimizando falsos negativos.

---

## 🏆 Selección del Modelo Final

El modelo final seleccionado:

* Mejora significativamente el recall en la clase minoritaria.
* Mantiene un buen equilibrio entre precision y recall.
* Presenta bajo riesgo de sobreajuste.

Las variables más importantes fueron:

* Edad
* Nivel promedio de glucosa
* Hipertensión
* Enfermedad cardíaca

---

## 🖥 Aplicación

Se desarrolló una aplicación en **Streamlit** que permite:

* Introducir datos de un paciente
* Obtener predicción de riesgo
* Visualizar probabilidad estimada

Además, el proyecto fue **contenedorizado con Docker**, lo que permite:

* Garantizar la reproducibilidad del entorno
* Facilitar el despliegue en distintos sistemas
* Simplificar la puesta en producción
* Estandarizar dependencias y versiones


---

## 🛠 Tecnologías Utilizadas

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Matplotlib / Seaborn
* Streamlit
* Docker

---

## 📚 Aprendizajes Clave

* Manejo de datasets desbalanceados
* Importancia del recall en modelos médicos
* Interpretación de importancia de variables
* Validación cruzada y ajuste de hiperparámetros
* Desarrollo de un flujo completo de Machine Learning
* Contenerización y despliegue con Docker

---

## 👤 Autor

**Enrique**
Data Analyst | Machine Learning

---


