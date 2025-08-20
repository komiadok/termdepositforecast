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
import matplotlib.pyplot as plt
import seaborn as sns

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