"""
preprocess.py
    Pipeline de nettoyage et de prétraitement des données selon les règles de gestion définies.
    Etapes principales : 
        - Gestion des valeurs manquantes implicites
        - Exclusion des mineurs
        - Gestion des outliers pour les durées et les appels
        - Gestion des variables catégorielles
        - Imputation des valeurs manquantes
        - Export des données invalides
"""
# =============================
# 📦 Chargement des librairies
# =============================
import os    # gérer les chemins et créer les dossiers
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin       # créer un pipeline scikit-learn

class DataCleaningPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, min_age=18, max_duration=300, max_calls=5, 
                 output_dir="../data/outputs"):
        self.min_age = min_age             # age minimum pour conserver un client
        self.max_duration = max_duration   # durée maximale des appels
        self.max_calls = max_calls         # nombre maximal de contacts pour campaign/previous
        self.output_dir = output_dir       # répertoire où seront enregistrés les fichiers exportés

        # Création du dossier de sortie s'il n'existe pas
        os.makedirs(self.output_dir, exist_ok=True)

        # Définition des chemins complets pour les fichiers de sortie
        self.minors_file = os.path.join(self.output_dir, "mineurs.csv")
        self.invalid_date_file = os.path.join(self.output_dir, "invalid_month_day.csv")

    # Pas d'apprentissage dans ce pipeline, donc on renvoie self
    def fit(self, X, y=None):
        return self

    # Copie du DataFrame pour éviter de modifier l'original
    def transform(self, X):
        df = X.copy()

        # 1️⃣ Remplacer les 'unknown' par 'missing' et créer des colonnes indicatrices
        object_cols = df.select_dtypes(include='object').columns.tolist()  # colonnes de type object
        for col in object_cols:
            if 'unknown' in df[col].values:
                indicator_col = f'missing_{col}'             # nommer la colonne indicatrice
                df[indicator_col] = (df[col] == 'unknown')   # créer une colonne indicatrice pour signaler la présence de 'unknown'
            df[col] = df[col].replace('unknown', 'missing')  # remplacer 'unknown' par 'missing' 
        
        # 2️⃣ Extraire les mineurs et les exclure
        minors = df[df['age'] < self.min_age]                # extraire les mineurs 
        if not minors.empty:
            minors.to_csv(self.minors_file, index=False)     # exporter les mineurs dans un fichier csv

        df = df[df['age'] >= self.min_age]                   # exclure les mineurs du DataFrame

        # 3️⃣ Gestion de la durée d'appel
        df['outlier_duration'] = df['duration'] > self.max_duration                 # créer un flag pour les outliers
        df.loc[df['duration'] > self.max_duration, 'duration'] = self.max_duration  # limiter les valeurs à max_duration

        # 4️⃣ Gestion du nombre d'appels pour 'campaign' et 'previous'
        for col in ['campaign', 'previous']:                                        
            outlier_col = f'outlier_{col}'                            # nommer le flag
            df[outlier_col] = df[col] > self.max_calls                # créer une colonne flag pour les outliers
            df.loc[df[col] > self.max_calls, col] = self.max_calls    # limiter les valeurs à max_calls

        # 5️⃣ Gestion de 'pdays' 
        df['never_contacted'] = df['pdays'] == 999                    # créer un flag pour les clients jamais contactés
        df['pdays'] = df['pdays'].replace(999, np.nan)                # remplacer 999 par NaN pour l'imputation

        # 6️⃣ Gestion de 'poutcome'
        df['poutcome'] = df['poutcome'].replace('nonexistent', 'no_previous_contact')  # remplacer 'nonexistent' par 'no_previous_contact'

        # 7️⃣ Contrôle de la validité des jours et des mois
        valid_days = ['mon', 'tue', 'wed', 'thu', 'fri']                      # définir les jours valides
        df.loc[~df['day_of_week'].isin(valid_days), 'day_of_week'] = np.nan   # renvoyer NaN pour les jours invalides

        valid_months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',             # définir les mois valides
                        'jul', 'aug', 'sep', 'oct', 'nov', 'dec']                  
        df.loc[~df['month'].isin(valid_months), 'month'] = np.nan             # renvoyer NaN pour les mois invalides

        invalid_dates = df[df['month'].isna() | df['day_of_week'].isna()]     # vérifier s'il y a eu des lignes invalides
        if not invalid_dates.empty:
            invalid_dates.to_csv(self.invalid_date_file, index=False)         # exporter les lignes invalides

        # 8️⃣ Conversion des colonnes object en catégorie
        for col in object_cols:
            df[col] = df[col].astype('category')                              # convertir les colonnes object en catégorie pour économiser la mémoire

        # 9️⃣ Imputation des valeurs manquantes
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()     # définir les colonnes numériques
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())                        # remplacer les valeurs manquantes des colonnes numériques par la médiane

        for col in object_cols:
            df[col] = df[col].fillna(df[col].mode()[0])                       # remplacer les valeurs manquantes des colonnes catégorielles par le mode

        return df  # retourner le DataFrame transformé