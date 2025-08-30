# =============================
# 📦 Chargement des librairies
# =============================
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd
import joblib
from io import BytesIO
import os
import sys

# =============================
# 📂 Gestion des chemins
# =============================
BASE_DIR = os.path.dirname(__file__)

# Ajouter le dossier pipelines au PYTHONPATH
sys.path.append(os.path.join(BASE_DIR, 'pipelines'))

# =============================
# 🔧 Imports internes
# =============================
from preprocess import DataCleaningPipeline
from train import DepositTermPredictor

# =============================
# 📦 Chargement du modèle
# =============================
model_path = os.path.join(BASE_DIR, 'models', 'deposit_model.pkl')
model = joblib.load(model_path)

# =============================
# 🔑 Définition des features
# =============================
numerical_features = ['age', 'campaign', 'pdays', 'previous', 'euribor3m', 'cons.price.idx', 'cons.conf.idx']
onehot_features = ['job', 'marital', 'default', 'housing', 'loan', 'contact', 'poutcome']
ordinal_features = ['education']
cyclic_features = ['month', 'day_of_week']

education_order = [
    'missing', 'illiterate', 'basic.4y', 'basic.6y', 'basic.9y',
    'high.school', 'professional.course', 'university.degree'
]

# =============================
# 🚀 Initialisation API
# =============================
app = FastAPI(title="API Dépôt à Terme")

@app.get("/")
def read_root():
    return {"message": "API pour prédire la souscription à un dépôt à terme"}

# =============================
# 🔮 Endpoint de prédiction
# =============================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Vérification du format
    if not file.filename.endswith(".csv"):
        return {"error": "Le fichier doit être un CSV"}

    # Lecture du CSV uploadé
    contents = await file.read()
    df = pd.read_csv(BytesIO(contents), sep=';')

    # --- Nettoyage / transformation ---
    cleaning_pipeline = DataCleaningPipeline(
        min_age=18,
        max_duration=300,
        max_calls=5,
        output_dir="temp"
    )
    cleaned_df = cleaning_pipeline.fit_transform(df)

    # Mapping mois et jours
    mois_mapping = {
        "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
        "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12
    }
    jour_mapping = {"mon":1,"tue":2,"wed":3,"thu":4,"fri":5}
    cleaned_df['month'] = cleaned_df['month'].map(mois_mapping)
    cleaned_df['day_of_week'] = cleaned_df['day_of_week'].map(jour_mapping)

    # Features indicatrices dynamiques
    indicator_features = [col for col in cleaned_df.columns if col.startswith('missing') or col.startswith('outlier')]
    if 'never_contacted' in cleaned_df.columns:
        indicator_features.append('never_contacted')
    if 'outlier_duration' in indicator_features:
        indicator_features.remove('outlier_duration')

    # Préparation des features
    X = cleaned_df[numerical_features + onehot_features + ordinal_features + cyclic_features + indicator_features]

    # --- Prédiction ---
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Ajout au DataFrame original (df) pour garder toutes les colonnes
    df['prediction'] = predictions
    df['score'] = probabilities

    # --- Génération CSV ---
    stream = BytesIO()
    df.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"}
    )
