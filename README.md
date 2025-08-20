<h1 align="center">🏦 Prédiction de la souscription d'un dépôt à terme </h1>

<p align="center">
  <img src="https://github.com/komiadok/termdepositforecast/blob/main/cover_image.jpg" width="550"><br>
  <a href="https://fr.freepik.com/">📸 Freepik</a>
</p>

---

## 📌 Objectifs

L’objectif de ce projet est de développer un modèle de prédiction de la souscription d’un dépôt à terme auprès des clients d’une banque, en suivant une approche rigoureuse de data science appliquée au secteur bancaire. Le projet se déroule en plusieurs étapes clés :

1. **Exploration et préparation des données**
   * Collecte et compréhension des données clients et transactionnelles.
   * Détection et gestion des anomalies : valeurs manquantes, incohérentes ou aberrantes.
   * Réalisation d'un **data cleaning** structuré afin de garantir la qualité des données avant l'analyse et la modélisation.
2. **Analyse exploratoire des données (EDA)**
   * Identification des tendances et des patterns significatifs au sein du dataset.
   * Visualisation des distributions, corrélations et comportements des variables clés.
   * Réalisation de **tests statistiques** adaptés (tests t de Student, tests du chi², etc.) pour comprendre les relations entre les variables.
3. **Modélisation prédictive**
   * Construction et entraînement de modèles de machine learning adaptés à la prédiction binaire (souscription yes/no)
   * Évaluation et scoring des modèles à l'aide de métriques pertinentes (AUC, précision, rappel, F1-score).
   * Sélection du modèle final optimisé pour la performance et la robustesse.
4. Déploiement et intégration
   * Implémentation d'une **API** permettant de réaliser des prédictions sur de nouvelles données clients.
   * Documentation et mise en place d'une architecture reproductible pour l'utilisation et la maintenance du modèle en production

---

## 📚 Données

Ce jeu de données contient des données liées aux campagnes marketing direct d'une institution bancaire portugaise. Les campagnes étaient basés sur des appels téléphoniques et elles ont été réalisées entre mai 2008 et novembre 2010. 
* **Source** : [Bank Marketing Dataset — UCI](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
* **Volume** : 41 188 lignes et 21 colonnes

| Champs         | Description                                                                | Type      |
|----------------|----------------------------------------------------------------------------|-----------|
| age            | Âge du client                                                              | Numérique |
| job            | Profession du client                                                       | Texte     |
| marital        | Situation matrimoniale du client                                           | Texte     |
| education      | Niveau d'études du client                                                  | Texte     |
| default        | Le client a-t-il un défaut de paiement de crédit ?                         | Texte     |
| housing        | Le client a-t-il un prêt immobilier ?                                      | Texte     |
| loan           | Le client a-t-il un prêt personnel ?                                       | Texte     |
| contact        | Moyen de communication du client                                           | Texte     |
| month          | Mois du dernier contact avec le client                                     | Texte     |
| day_of_week    | Jour du dernier contact avec le client                                     | Texte     |
| duration       | Durée du dernier contact avec le client                                    | Numérique |
| campaign       | Nombre de contacts effectués avec le client durant la campagne             | Numérique |
| pdays          | Nombre de jours écoulés depuis le dernier contact avec le client           | Numérique |
| previous       | Nombre de contacts effectués avec le client lors de la précédente campagne | Numérique |
| poutcome       | Résultat de la précédente campagne marketing avec le client                | Texte     |
| emp.var.rate   | Taux de variation de l'emploi                                              | Numérique |
| cons.price.idx | Indice des prix à la consommation                                          | Numérique |
| cons.conf.idx  | Indice de confiance des consommateurs                                      | Numérique |
| euribor3m      | Taux Euribor à 3 mois                                                      | Numérique |
| nr.employed    | Nombre de personnes actives                                                | Numérique |
| y              | Le client a-t-il souscrit à un dépôt à terme ?                             | Texte     |  

---

## 🧰 Environnement technique

### 📋 Prérequis 

* Installer [Visual Studio Code](https://code.visualstudio.com/)
  > Télécharger la version correspondante à ton OS (Windows / Mac / Linux)
* Installer [Miniconda](https://www.anaconda.com/download/)
  > Entrer son email et choisir la distribution de Miniconda adaptée.<br>
  > S'assurer que `(base)` apparaît devant le chemin du disque dur après installation.

### 💻 Langage et environnement 

* Python : langage utilisé pour les analyses de données et la modélisation prédictive.
* GitHub : plateforme utilisée pour le versionnage du code et le stockage en ligne du projet.
* Jupyter Notebook (via Miniconda) : environnement interactif utilisé pour l’exploration et la visualisation des données.
* VS Code : environnement de développement utilisé pour le développement et le déploiement des pipelines.

### 📦 Librairies Python utilisées

* `pandas` : manipulation de données
* `numpy` : traitement numérique
* `matplotlib` et `seaborn` : visualisations
* `scipy` : donctions statistiques et tests (t-test, chi², etc.)
* `scikit-learn` : pipelines, algorithmes d'apprentissage automatique, métriques, etc.
* `category_encoders` : encodages catégoriels avancés
* `xgboost` : algorithme d'apprentissage automatique (performant pour classification binaire)
* `SHAP` et `eli5` : interprétabilité des modèles
* `fastapi` : création d'une API
* `joblib`: sauvegarde et chargement du modèle entraîné
* `imbalanced-learn (imblearn.over_sampling)` : gestion du déséquilibre des classes
* `pydantic` : validation stricte des données d'entrée dans l'API
* `typing` : annotations de types pour un code plus robuste et lisible

---

## 📂 Organisation du projet

```
termdepositforecast/
│
├── data/                                       # Données du projet     
│   └── raw/                                    # Données brutes
│       └── bank-additional-full.csv            # Dataset principal pour l'analyse et l'entraînement
│       └── bank-additional.csv                 # Dataset utilisé pour tester l'API
│   └── outputs/                                # Données de sortie
│       └── mineurs.csv                         # Extraction des mineurs du jeu de données 
├── notebook/                                   # Notebooks pour exploration et analyses
│   └── exploration.ipynb                       # Exploration des données
│   └── eda.ipynb                               # Analyse exploratoire des données
├── pipelines/                                  # Pipelines pour le prétraitement et l'entraînement des modèles
│   └── preprocess.py                           # Pipeline de prétraitement des données
├── utils.py                                    # Fonctions utilitaires réutilisables
├── environment.yml                             # Dépendances
├── cover_image.jpg                             # Image de couverture du projet
└── README.md                                   # Documentation du projet
```
