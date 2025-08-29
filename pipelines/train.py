import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer, LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, log_loss, brier_score_loss

# Ajout du chemin ../utils pour importer les fonctions
sys.path.append('../')

# Importation des fonctions utilitaires
from utils import find_threshold_with_recall_priority

# --- Fonction globale pour encoder les variables cycliques ---
def sin_cos_encode_global(X, max_vals):
    """
    Encode les colonnes cycliques en sin/cos.
    X : array-like (n_samples, n_features)
    max_vals : liste des valeurs max pour chaque feature (ex: [12,5])
    """
    X = np.array(X, dtype=float)
    result = []
    for i in range(X.shape[1]):
        X_sin = np.sin(2 * np.pi * X[:, i] / max_vals[i])
        X_cos = np.cos(2 * np.pi * X[:, i] / max_vals[i])
        result.append(X_sin)
        result.append(X_cos)
    return np.column_stack(result)

# --- Classe CyclicEncoder picklable ---
class CyclicEncoder(FunctionTransformer):
    def __init__(self, max_vals, feature_names):
        """
        Encodeur cyclique pour variables périodiques.
        max_vals : liste des valeurs max par colonne (ex: [12,5])
        feature_names : noms des colonnes originales
        """
        self.max_vals = max_vals
        self.feature_names = feature_names
        super().__init__(
            func=sin_cos_encode_global,
            validate=True,
            kw_args={'max_vals': self.max_vals}  # on passe max_vals à la fonction globale
        )

    def get_feature_names_out(self, input_features=None):
        """
        Génère automatiquement les noms des colonnes sin/cos.
        Exemple : 'month' → 'month_sin', 'month_cos'
        """
        names = []
        for name in self.feature_names:
            names.extend([f"{name}_sin", f"{name}_cos"])
        return np.array(names)


# Classe principale
class DepositTermPredictor:
    def __init__(self,
                 numerical_features,
                 onehot_features,
                 ordinal_features,
                 cyclic_features,
                 indicator_features,
                 ordinal_categories=None,
                 model_dir='../models',
                 random_state=42):
        self.numerical_features = numerical_features
        self.onehot_features = onehot_features
        self.ordinal_features = ordinal_features
        self.cyclic_features = cyclic_features
        self.indicator_features = indicator_features
        self.ordinal_categories = ordinal_categories
        self.model_dir = model_dir
        self.random_state = random_state
        
        self.pipeline = None
        self.grid_search = None
        self.lb = LabelBinarizer()
        self.best_threshold = None
        
        # Crée le dossier si inexistant
        os.makedirs(self.model_dir, exist_ok=True)

    def preprocess_target(self, y_raw):
        """Encodage binaire de la target"""
        return self.lb.fit_transform(y_raw).ravel()

    def build_pipeline(self):
        """Construit le pipeline complet avec preprocessing et modèle"""
        # Encodage sin/cos pour variables cycliques
        sin_cos_transformer = CyclicEncoder(max_vals=[12,5], feature_names=self.cyclic_features)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('onehot', OneHotEncoder(handle_unknown='ignore'), self.onehot_features),
                ('ordinal', OrdinalEncoder(categories=[self.ordinal_categories]), self.ordinal_features),
                ('cyclic', sin_cos_transformer, self.cyclic_features),
                ('ind', 'passthrough', self.indicator_features)
            ]
        )
        
        # Pipeline complet
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                penalty='l1',
                solver='liblinear',
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state
            ))
        ])
        
    def fit(self, X_train, y_train, param_grid=None, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2):
        """Entraîne le pipeline avec GridSearchCV"""
        if param_grid is None:
            param_grid = {'classifier__C': [0.01, 0.1, 1, 10]}
        
        self.grid_search = GridSearchCV(
            self.pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        start_time = time.time()
        self.grid_search.fit(X_train, y_train)
        end_time = time.time()
        print(f"Durée d'entraînement : {end_time - start_time:.2f} secondes")
        
        # Récupère le pipeline entraîné
        self.pipeline = self.grid_search.best_estimator_
        print(f"Meilleur hyperparamètre C : {self.grid_search.best_params_['classifier__C']}")
    
    def predict_proba(self, X):
        """Retourne les probabilités positives"""
        return self.pipeline.predict_proba(X)[:,1]
    
    def predict(self, X, threshold=0.5):
        """Prédit en utilisant un seuil (0.5 par défaut)"""
        y_proba = self.predict_proba(X)
        return (y_proba >= threshold).astype(int)
    
    def find_best_threshold(self, y_true, y_proba, min_precision=0.35):
        """Trouve le seuil optimal selon la fonction utilitaire"""
        best = find_threshold_with_recall_priority(y_true, y_proba, min_precision=min_precision)
        self.best_threshold = best['threshold']
        return best
    
    def evaluate(self, X_test, y_test):
        """Évaluation complète du modèle"""
        y_proba = self.predict_proba(X_test)
        if self.best_threshold is None:
            threshold = 0.5
        else:
            threshold = self.best_threshold
        y_pred = (y_proba >= threshold).astype(int)
        
        print(classification_report(y_test, y_pred, digits=3))
        print("Log-Loss:", log_loss(y_test, y_proba))
        print("Brier Score:", brier_score_loss(y_test, y_proba))
        
    def save_model(self, filename='deposit_model.pkl'):
        """Sauvegarde le pipeline complet dans le dossier ../models"""
        filepath = os.path.join(self.model_dir, filename)
        joblib.dump(self.pipeline, filepath)
        print(f"Modèle sauvegardé dans : {filepath}")