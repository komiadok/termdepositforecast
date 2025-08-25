"""
utils.py
    Contient les fonctions utiles pour le projet de prédiction des souscriptions d'un dépôt à terme
    Inclut des fonctions pour : 
        - les visualisations, 
        - le chargement, 
        - le nettoyage, 
        - le prétraitement,
        - l'entraînement des modèles
        - l'évaluation des modèles
        - l'implémentation d'API
"""
# =============================
# 📦 Chargement des librairies
# =============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, levene, probplot, skew, mannwhitneyu, chi2_contingency 
# =============================
# 📊 Visualisations
# =============================

def plot_horizontal_bar(ax, df, column, title):
    """
    Description:
        Trace un barplot horizontal avec annotations pour visualiser la proportion de chaque modalité d'une variable catégorielle.

    Arguments:
        ax : l'axe sur lequel tracer le graphique.
        df : le DataFrame contenant les données.
        column (str) : la colonne catégorielle à représenter.
        title (str) : le titre du graphique.

    Retourne:
        Un barplot horizontal avec des annotations
    """
    
    # Calcul des proportions (%) de chaque modalité
    counts = df[column].value_counts(normalize=True) 
    counts = counts.sort_values(ascending=False)       # tri décroissant pour avoir les fréquentes en haut

    # Création d'un DataFrame exploitable par Seaborn
    df_plot = counts.reset_index()             # convertir en DataFrame
    df_plot.columns = [column, 'proportion']   # renommer les colonnes
    
    # Création d'une palette "Greens" : du vert foncé (valeurs élevées) au vert clair (valeurs faibles)
    colors = sns.color_palette("Greens", n_colors=len(df_plot))

    # Création du barplot horizontal
    sns.barplot(
        data=df_plot,
        y=column,         # les catégories sur l'axe Y
        x='proportion',   # les proportions sur l'axe X
        hue='proportion', # couleur selon la proportion
        palette=colors,   # définir la palette      
        legend=False,     # supprimer la légende
        ax=ax             # axe sur lequel dessiner
    )

    # Ajout des proportions en pourcentage à droite des barres
    for p in ax.patches:
        ax.annotate(
            f"{p.get_width()*100:.1f}%",            # texte formaté en pourcentage
            (p.get_x() + p.get_width() + 0.001,     # décaler légèrement vers la droite
             p.get_y() + p.get_height()/2),         # centrer verticalement la barre
            ha='left', va='center', fontsize=10     # aligner le texte
        )

    # Personnalisation finale
    ax.set_title(title, fontsize=12, fontweight='bold')      # titre du graphique avec style gras et taille de police
    ax.set_xlabel("")                                        # supprimer le label de l'axe X
    ax.set_ylabel("")                                        # supprimer le label de l'axe Y
    ax.set_xticks([])                                        # supprimer les valeurs affichées sur l'axe X


def plot_pie_chart(df, column, title, explode_cats=None):
    """
    Description:
        Affiche un camembert (pie chart) représentant les proportions
        d'une variable catégorielle, avec légende en bas et possibilité
        de détacher certaines modalités.

    Arguments:
        df : DataFrame contenant les données
        column (str) : nom de la colonne catégorielle à représenter
        title (str) : titre du graphique
        explode_cats (list, optionnel) : liste des modalités à détacher

    Retourne:
        Un pie chart avec des annotations
    """

    # Calcul des proportions (%) et tri décroissant
    counts = df[column].value_counts(normalize=True).sort_values(ascending=False) * 100

    # Définition du explode
    if explode_cats is None:
        explode_cats = []
    explode = [
        0.1 if cat.lower() in [c.lower() for c in explode_cats] else 0
        for cat in counts.index
    ]

    # Création du pie chart
    plt.subplots(figsize=(6, 6))
    
    wedges, texts, autotexts = plt.pie(
        counts,
        labels=None,                                        # pas d'étiquettes directement sur les parts
        autopct='%1.1f%%',                                  # affiche les pourcentages
        pctdistance=1.15,                                   # décale les % à l'extérieur des parts
        startangle=90,                                      # départ du camembert en haut
        counterclock=False,                                 # sens horaire
        colors=plt.cm.Set2.colors,                          # palette douce et qualitative
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},  # bord blanc entre les parts
        explode=explode,                                    # détachement des parts spécifiées
        shadow=True                                         # ombre pour effet 3D léger
    )

    # Style des pourcentages : couleur, taille et graisse
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')

    # Ajout de la légende à droite du camembert
    plt.legend(
        wedges,                    # objets correspondants aux parts
        counts.index,              # étiquettes correspondant aux modalités
        title=column,              # titre de la légende
        loc="center left",         # positionnement à droite centré verticalement
        bbox_to_anchor=(1, 0.5),   # décalage exact par rapport à la figure
        fontsize=10,               # taille de la police
        frameon=True,              # active le cadre
        framealpha=1,              # opacité du cadre
        edgecolor='black',         # couleur du contour
        facecolor='white'          # couleur de fond
    )

    # Titre du graphique
    plt.title(title, fontsize=16, fontweight='bold', pad=20)

    # Affichage
    plt.show()

def plot_subscription_rate(ax, df, column, title, orientation="horizontal"):
    """
    Description:
        Trace un barplot du taux de souscription (y=1) par catégorie.

    Arguments:
        df : le dataframe contenant les données.
        column (str) : la colonne catégorielle utilisée pour grouper.
        ax : l'axe sur lequel dessiner le graphique. 
        title (str) : le titre du graphique.
        orientation (str) : "horizontal" ou "vertical", direction des barres.

    Retourne:
        Un barplot avec des annotations
    """
    
    # Calcul du taux de souscription par catégorie
    rates = (
        df.groupby(column, observed=True)['y']
          .mean()  # calcule la moyenne
          .reset_index(name='subscription_rate')
          .sort_values('subscription_rate', ascending=False)
    )
    
    # Palette de couleurs : du vert foncé (taux élevés) au vert clair (taux faibles)
    colors = sns.color_palette("Greens", n_colors=len(rates))

    # Création du barplot horizontal
    if orientation == "horizontal":
        sns.barplot(
            data=rates,
            y=column,
            x='subscription_rate',
            hue='subscription_rate',
            palette=colors,
            legend=False,
            ax=ax,
            order=rates[column]  # trier les barres
        )
        
        # Ajout des taux en pourcentage à droite des barres
        for p in ax.patches:
            ax.annotate(
                f"{p.get_width()*100:.1f}%",
                (p.get_x() + p.get_width() + 0.001, p.get_y() + p.get_height()/2),
                ha='left', va='center', fontsize=10
            )
        
        # Personnalisation finale
        ax.set_xlabel("")   # supprimer label X
        ax.set_ylabel("")   # supprimer label Y
        ax.set_xticks([])   # supprimer ticks X

    # Création du barplot vertical
    elif orientation == "vertical":
        sns.barplot(
            data=rates,
            x=column,
            y='subscription_rate',
            hue='subscription_rate',
            palette=colors,
            legend=False,
            ax=ax,
            order=rates[column]  # trier les barres
        )
        
        # Ajout des taux en pourcentage au-dessus des barres
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height()*100:.1f}%",
                (p.get_x() + p.get_width()/2, p.get_height() + 0.005),
                ha='center', va='bottom', fontsize=10
            )
        
        # Personnalisation finale
        ax.set_xlabel("")   # supprimer label X
        ax.set_ylabel("")   # supprimer label Y
        ax.set_yticks([])   # supprimer ticks X
        
    else:
        raise ValueError("orientation doit être 'horizontal' ou 'vertical'")

    # Définition du titre
    ax.set_title(title, fontsize=12, fontweight='bold')

def plot_scatterplot(ax, df, x_var, y_var, title):
    """
    Description:
        Trace un scatter plot + une ligne pointillée reliant les points

    Arguments:
        ax : Axe sur lequel tracer.
        df : Données contenant les variables.
        x_var (str) : Nom de la colonne pour l'axe des X.
        y_var (str) : Nom de la colonne pour l'axe des Y.
        title (str) : Titre du graphique.

    Retourne:
        Un scatterplot avec des lignes en pointillés
    """

    # Scatter plot
    sns.scatterplot(
        data=df,
        x=x_var,
        y=y_var,
        s=50,              # taille des points
        color='seagreen',  # couleur des points
        ax=ax
    )

    # Ligne pointillée
    sns.lineplot(
        data=df,
        x=x_var,
        y=y_var,
        color='seagreen',
        linestyle='--',    # style de ligne 
        marker=None,       # pas de marqueur sur la ligne
        ax=ax
    )

    # Titre
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Nettoyage axes
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='x', rotation=45)   # Rotation des étiquettes de l’axe X
    ax.grid(False)                          # Suppression de la grille


# =============================
# 🧪 Tests statistiques
# =============================

def chi2_analysis(data, col_cat, col_target):
    """
    Description:
        Réalise une analyse complète d'association entre une variable catégorielle (col_cat) et une variable binaire/catégorielle (col_target).
            1. Construit le tableau de contingence
            2. Applique le test du Chi2 d'indépendance
            3. Évalue la taille de l'effet (Cramér's V)
            4. Analyse les résidus standardisés
    
    Arguments:
        data : pd.DataFrame
            Jeu de données
        col_cat : str
            Nom de la variable catégorielle (ex: "job")
        col_target : str
            Nom de la variable qualitative cible (ex: "y")

    Retourne:
        dict : contenant les résultats du test, la taille d'effet et les résidus
    """

    # 1️⃣ Tableau de contingence
    contingency = pd.crosstab(data[col_cat], data[col_target])
    
    # 2️⃣ Vérification des conditions de validité
    N = contingency.sum().sum()
    if N < 40:
        return {"message": f"⚠️ Effectif total insuffisant (N={N}) pour réaliser le test Chi2."}
    
    chi2, p, dof, expected = chi2_contingency(contingency)
    
    # Vérification des effectifs attendus
    prop_low = (expected < 5).sum() / expected.size
    if np.any(expected < 1) or prop_low > 0.2:
        return {"message": "⚠️ Conditions de validité non respectées : "
                           f"{(expected < 1).sum()} cellules < 1, "
                           f"{prop_low*100:.1f}% des cellules < 5."}
    
    # 3️⃣ Taille de l’effet : Cramér's V
    cramers_v = None
    resid_df = None
    if p < 0.05:
        cramers_v = np.sqrt(chi2 / (N * (min(contingency.shape) - 1)))
        
        # 4️⃣ Résidus standardisés
        resid = (contingency - expected) / np.sqrt(expected)
        resid_df = pd.DataFrame(resid, 
                                index=contingency.index, 
                                columns=contingency.columns)

    # 5️⃣ Interprétation automatique
    interpretation = "➡️ Les variables sont "
    if p < 0.05:
        interpretation += f"associées (p-value={p:.4f} < 0.05)."
    else:
        interpretation += f"indépendantes (p-value={p:.4f} ≥ 0.05)."
    
    # Résumé clair
    results = {
        "tableau_contingence": contingency,
        "chi2": chi2,
        "ddl": dof,
        "p_value": p,
        "cramers_v": cramers_v,
        "residus_standardises": resid_df,
        "interpretation": interpretation
    }
    
    return results

def mean_difference_test(df, columns, target='y', plot=True, skew_alert=1):
    """
    Description:
        Analyse statistique d'une ou plusieurs colonnes quantitatives en fonction d'une variable binaire.

    Arguments:
        df (pd.DataFrame): DataFrame contenant les données
        columns (str or list): Nom(s) des colonnes numériques à tester
        target (str): Nom de la colonne binaire pour les groupes
        plot (bool): Affiche histogrammes et QQ-plots si True
        skew_alert (float): seuil d'asymétrie pour alerte sur distribution fortement biaisée

    Retourne:
        Affiche tests t (Student/Welch) ou Mann-Whitney U, Cohen's d, skew, histogrammes et QQ-plots.
    """

    # Gestion d'une seule colonne 
    if isinstance(columns, str):
        columns = [columns]

    # Boucle sur les colonnes
    for col in columns:
        print(f"\n🔹 Analyse pour '{col}'")
        
        # Séparation des données selon le groupe cible (0 ou 1)
        group0 = df[df[target] == 'no'][col]
        group1 = df[df[target] == 'yes'][col]
        
        # Calcul de l'asymétrie (skewness) des deux groupes
        skew0 = skew(group0)
        skew1 = skew(group1)

        # Visualisation (histogrammes et QQ-plots)
        if plot:
            # Histogrammes avec densité KDE
            fig, axes = plt.subplots(1, 2, figsize=(10,4))
            sns.histplot(group0, kde=True, ax=axes[0], color='skyblue')
            axes[0].set_title(f"{col} (y=no) | skew={skew0:.2f}")
            sns.histplot(group1, kde=True, ax=axes[1], color='salmon')
            axes[1].set_title(f"{col} (y=yes) | skew={skew1:.2f}")
            plt.tight_layout()
            plt.show()

            # QQ-plots pour vérifier la normalité
            fig, axes = plt.subplots(1, 2, figsize=(10,4))
            probplot(group0, dist="norm", plot=axes[0])
            axes[0].set_title(f"QQ-plot {col} (y=no)")
            probplot(group1, dist="norm", plot=axes[1])
            axes[1].set_title(f"QQ-plot {col} (y=yes)")
            plt.tight_layout()
            plt.show()
        
        # Choix du test statistique en fonction de l'asymétrie
        if abs(skew0) > skew_alert or abs(skew1) > skew_alert:
            # Test non paramétrique Mann-Whitney U
            stat, p_value = mannwhitneyu(group0, group1, alternative='two-sided')
            test_type = "Mann-Whitney U (non paramétrique)"
            cohen_d = np.nan  # non défini pour Mann-Whitney
            print(f"⚠️ Distribution fortement asymétrique. Utilisation du test Mann-Whitney U")
        else:
            # Test de Levene pour vérifier l'homogénéité des variances
            levene_test = levene(group0, group1)
            equal_var = levene_test.pvalue > 0.05 # variances égales si p>0.05

            # Test t de Student ou Welch selon l'homogénéité
            stat, p_value = ttest_ind(group0, group1, equal_var=equal_var)
            test_type = "Student (variances égales)" if equal_var else "Welch (variances inégales)"
            
            # Calcul de la taille de l'effet (Cohen's d)
            mean_diff = np.mean(group1) - np.mean(group0)
            if equal_var:
                # Pooled standard deviation si variances égales
                pooled_std = np.sqrt(((len(group1)-1)*np.var(group1, ddof=1) + (len(group0)-1)*np.var(group0, ddof=1))
                                     / (len(group1) + len(group0) - 2))
            else:
                # Moyenne des variances si inégales (approche simplifiée pour Cohen's d)
                pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group0, ddof=1)) / 2)
            cohen_d = mean_diff / pooled_std
        
        # Affichage des résultats
        print(f"🔍 Test choisi : {test_type}")
        print(f"Statistique = {stat:.3f}, p-value = {p_value:.4f}")
        if not np.isnan(cohen_d):
            print(f"Taille de l'effet (Cohen's d) = {cohen_d:.2f}")
        print(f"Asymétrie (skew) : y=0 -> {skew0:.2f}, y=1 -> {skew1:.2f}")

        # Interprétation de la p-value
        if p_value < 0.05:
            print(f"✅ Différence significative pour '{col}' entre {target}=no et {target}=yes")
        else:
            print(f"⚠️ Aucune différence significative pour '{col}'")

def spearman_corr_matrix(data, numeric_cols, figsize=(10,5), annot=True, cmap="coolwarm"):
    """
    Description:
        Calcule et affiche la matrice de corrélation de Spearman pour un ensemble de colonnes quantitatives.

    Arguments:
        data : pd.DataFrame
            Jeu de données contenant les variables
        numeric_cols : list
            Liste des colonnes quantitatives à inclure
        figsize : tuple
            Taille de la figure matplotlib
        annot : bool
            Affiche les coefficients sur la heatmap si True
        cmap : str
            Palette de couleurs pour la heatmap

    Retourne:
        corr_matrix : pd.DataFrame
            Matrice de corrélation de Spearman
    """

    # Sélection des colonnes quantitatives
    df_numeric = data[numeric_cols]

    # Calcul de la corrélation de Spearman
    corr_matrix = df_numeric.corr(method='spearman')

    # Affichage graphique
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, fmt=".2f", cbar=True, square=True, linewidths=0.5)
    plt.title("Matrice de corrélation de Spearman", fontsize=14)
    plt.show()

    return corr_matrix