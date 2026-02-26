# Aprendizaje Supervisado - Predicción de Ictus

Resumen rápido
- Dataset: `stroke_dataset.csv` (datos clínicos y demográficos).
- Objetivo: explorar los datos (EDA), entrenar modelos supervisados y exponer una interfaz sencilla con Streamlit para realizar predicciones.

## Exploratory Data Analysis (EDA)

- **Carga y revisión inicial**: se inspecciona el dataframe con `df.info()`, `df.describe()` y `df.isnull().sum()` para detectar problemas y tipos de datos.
- **Distribución de la variable objetivo**: se calcula `value_counts()` y la proporción de clases para detectar desequilibrio (95% sin ictus, 5% con ictus).
- **Limpieza básica**: se eliminan filas con `gender == 'Other'` para simplificar el análisis.
- **Codificación**: variables categóricas convertidas a dummies con `pd.get_dummies(..., drop_first=True)`.
- **Visualizaciones principales**:
  - Boxplots de `age` y `avg_glucose_level` por clase `stroke`.
  - `countplot` de `hypertension` por clase.
  - `heatmap` de correlaciones entre variables numéricas.
- **Desequilibrio de clases**: se aplica SMOTE en el conjunto de entrenamiento para balancear las clases antes de reentrenar modelos.

## Modelado

- Se prueban modelos de **Regresión Logística** y **Random Forest**.
- **Evaluación** mediante métricas como recall, matriz de confusión, reporte de clasificación y ROC AUC.
- **Validación cruzada** (5-fold) y búsqueda de hiperparámetros (`GridSearchCV`) usando `recall` como métrica principal.
- El mejor modelo final se guarda en `modelo.pkl` (archivo serializado con `joblib`) para uso por la interfaz.

## Cómo ejecutar

### 1. Clonar el repositorio y situarse en la carpeta del proyecto:

```powershell
git clone https://github.com/Bootcamp-Data-Analyst/AprendizajeSupervisado_Enrique.git
cd AprendizajeSupervisado_Enrique
```

### 2. Crear y activar un entorno virtual e instalar dependencias:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requeriments.txt
```

### 3. Ejecutar el notebook para reproducir el EDA y entrenar modelos:

```powershell
jupyter notebook main.ipynb
```
O abrir en VS Code y ejecutar las celdas secuencialmente.

### 4. Ejecutar la app Streamlit:

Se debe extraer la sección Streamlit del notebook a un script `.py`:

```powershell
jupyter nbconvert --to script main.ipynb
# Editar main.py para dejar solo la parte de Streamlit y guardarlo como app.py
streamlit run app.py
```

## Notas importantes

- El archivo `modelo.pkl` debe existir en la carpeta del proyecto para que la app Streamlit pueda cargarlo.
- `streamlit run` no ejecuta notebooks directamente; por eso es necesario extraer la parte de Streamlit a un script `.py`.
- Si el notebook no tiene `modelo.pkl`, ejecuta la última celda que lo genera.

## Dependencias principales

Ver `requeriments.txt` para la lista completa. Principales:
- pandas, numpy, scikit-learn, scipy
- matplotlib, seaborn (visualización)
- optuna (optimización)
- streamlit (interfaz web)
- imbalanced-learn (SMOTE)