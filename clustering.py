"""Logique de clustering partagée par l'appli Streamlit (app.py).

Chaque jeu de données est décrit par une entrée de DATASETS. La fonction
`preparer` entraîne KMeans et la CAH et renvoie tout ce dont l'appli a besoin.
La fonction `predire` affecte un nouvel individu au centroïde le plus proche
(même principe que le déploiement du TP « Segmentation des clients »).
"""
import os

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import Normalizer, StandardScaler

SEED = 42

# --------------------------------------------------------------------------- #
# Chargement des jeux de données
# --------------------------------------------------------------------------- #
WINE_COLS = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
             'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
             'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315',
             'Proline']

_BASE_BC = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
            'compactness', 'concavity', 'concave points', 'symmetry',
            'fractal dimension']
BC_COLS = [f'{p}_{b}' for p in ['mean', 'se', 'worst'] for b in _BASE_BC]


def charger_wine(dossier):
    df = pd.read_csv(os.path.join(dossier, 'wine.data'), header=None,
                     names=['Class'] + WINE_COLS)
    y = df.pop('Class')
    df.index = [f'Vin{i + 1}' for i in range(len(df))]
    y.index = df.index
    return df, y


def charger_breast_cancer(dossier):
    df = pd.read_csv(os.path.join(dossier, 'wdbc.data'), header=None,
                     names=['ID', 'Diagnosis'] + BC_COLS)
    y = df['Diagnosis']
    df = df.drop(columns=['ID', 'Diagnosis'])
    df.index = [f'Echantillon{i + 1}' for i in range(len(df))]
    y.index = df.index
    return df, y


def charger_grains(dossier):
    # Même fichier que dans le notebook des graines de blé
    df = pd.read_csv(os.path.join(dossier, 'wheat_seeds_dataset.csv'))
    df.index = [f'Graine{i + 1}' for i in range(len(df))]
    return df, None


def charger_telecom(dossier):
    # Même fichier que dans le TP « Segmentation des clients »
    df = pd.read_csv(os.path.join(dossier, 'Customer_dataset.csv'))
    # Le fichier contient des doubles espaces dans certains noms ("Call  Failure")
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
    df.index = [f'Client{i + 1}' for i in range(len(df))]
    return df, None


# --------------------------------------------------------------------------- #
# Configuration de chaque jeu de données
#   echelle : 'standard' (StandardScaler), 'normalize' (chaque ligne ramenée à une
#             norme 1, comme dans le TP télécom) ou None (données brutes)
#   modele_defaut : modèle présélectionné dans l'appli (le meilleur d'après le notebook)
# --------------------------------------------------------------------------- #
DATASETS = {
    'Graines de blé': dict(
        charger=charger_grains,
        echelle=None,         # comme dans le notebook des graines (données brutes)
        k_kmeans=4, k_cah=4, liaison_cah='average',   # comme le notebook Customer du prof
        cle='area A',
        source='https://archive.ics.uci.edu/dataset/236/seeds',
        description=("Graines de blé décrites par 7 variables morphologiques "
                     "(aire, périmètre, compacité, longueur, largeur, coefficient "
                     "d'asymétrie, longueur du sillon)."),
        noms={},              # noms par défaut : « Classe 0 », « Classe 1 »…
    ),
    'Clients télécom': dict(
        charger=charger_telecom,
        echelle='normalize',  # comme dans le TP télécom
        k_kmeans=3, k_cah=2, liaison_cah='average',
        modele_defaut='CAH',  # le TP déploie la CAH
        cle='Seconds of Use',
        source='https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset',
        description=("Clients d'une entreprise de télécommunications décrits par 12 "
                     "variables de comportement (appels échoués, plaintes, durée "
                     "d'abonnement, secondes d'utilisation, SMS, âge, valeur du "
                     "client…). La CAH (liaison moyenne, 2 classes) sépare les clients "
                     "quasi inactifs des clients actifs."),
        noms={'CAH': {0: 'Client non fidèle (quasi inactif)',
                      1: 'Client fidèle (actif)'},
              'KMeans': {0: 'Client quasi inactif',
                         1: 'Client orienté SMS, forte valeur',
                         2: "Client orienté appels, très actif"}},
    ),
    'Vins': dict(
        charger=charger_wine,
        echelle='standard',
        k_kmeans=3, k_cah=3,
        cle='Proline',
        source='https://archive.ics.uci.edu/dataset/109/wine',
        description=("178 vins italiens issus de 3 cultivars, décrits par 13 "
                     "mesures d'analyse chimique (alcool, acide malique, "
                     "flavonoïdes, proline…). La colonne de classe (cultivar) n'est "
                     "pas utilisée pour le clustering, seulement pour comparer."),
        noms={0: "Vins légers : alcool et proline faibles, couleur peu intense",
              1: "Vins très colorés : acide malique élevé, flavonoïdes et teinte faibles",
              2: "Vins corsés : alcool, proline et flavonoïdes élevés"},
    ),
    'Cancer du sein': dict(
        charger=charger_breast_cancer,
        echelle='standard',
        k_kmeans=2, k_cah=2,
        cle='worst_area',
        source='https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic',
        description=("569 prélèvements de masses mammaires, décrits par 30 mesures "
                     "des noyaux cellulaires (moyenne, erreur standard et pire valeur "
                     "de 10 caractéristiques). Le diagnostic (B/M) n'est pas utilisé "
                     "pour le clustering, seulement pour comparer. "
                     "Démonstration pédagogique : ne remplace pas un avis médical."),
        noms={0: "Profil plutôt bénin : noyaux petits et peu concaves",
              1: "Profil plutôt malin : noyaux grands et très concaves"},
    ),
}


# --------------------------------------------------------------------------- #
# Outils
# --------------------------------------------------------------------------- #
def ordonner(df, partition, variable):
    """Renumérote les classes par moyenne croissante de `variable`.

    Les numéros produits par KMeans ou la CAH sont arbitraires ; cet ordre les
    rend stables d'une exécution à l'autre (et identiques entre notebook et appli).
    """
    moyennes = df.groupby(np.asarray(partition))[variable].mean().sort_values()
    ordre = {ancienne: nouvelle for nouvelle, ancienne in enumerate(moyennes.index)}
    return np.array([ordre[c] for c in partition])


def nom_classe(cfg, classe, methode=None):
    """Nom lisible d'une classe. `noms` peut être {classe: nom} (valable pour les deux
    modèles) ou {'KMeans': {...}, 'CAH': {...}} (noms propres à chaque modèle)."""
    noms = cfg['noms']
    if methode in noms:
        noms = noms[methode]
    elif any(isinstance(v, dict) for v in noms.values()):
        noms = {}
    return noms.get(classe, f'Classe {classe}')


def a_des_noms(cfg, methode):
    return any(nom_classe(cfg, c, methode) != f'Classe {c}' for c in range(10))


def preparer(nom, dossier):
    """Charge un jeu de données et entraîne KMeans + CAH."""
    cfg = DATASETS[nom]
    df, y = cfg['charger'](dossier)

    scaler = {'standard': StandardScaler, 'normalize': Normalizer}.get(cfg['echelle'])
    scaler = scaler() if scaler else None
    x = scaler.fit_transform(df.values) if scaler else df.values.astype(float)

    pca = PCA(n_components=2)
    df_pca = pd.DataFrame(pca.fit_transform(x), columns=['PC1', 'PC2'], index=df.index)

    kmeans = KMeans(n_clusters=cfg['k_kmeans'], init='random', max_iter=500,
                    n_init=10, random_state=SEED).fit(x)
    cah = AgglomerativeClustering(n_clusters=cfg['k_cah'],
                                 linkage=cfg.get('liaison_cah', 'ward')).fit(x)

    partitions = {
        'KMeans': ordonner(df, kmeans.labels_, cfg['cle']),
        'CAH': ordonner(df, cah.labels_, cfg['cle']),
    }
    centroides = {
        m: np.array([x[p == c].mean(axis=0) for c in np.unique(p)])
        for m, p in partitions.items()
    }
    silhouettes = {m: silhouette_score(x, p) for m, p in partitions.items()}

    return dict(
        cfg=cfg, df=df, y=y, x=x, scaler=scaler, pca=pca, df_pca=df_pca,
        partitions=partitions, centroides=centroides, silhouettes=silhouettes,
        ari_kmeans_cah=adjusted_rand_score(partitions['KMeans'], partitions['CAH']),
    )


def predire(modele, valeurs, methode):
    """Classe du centroïde le plus proche pour un nouvel individu.

    `valeurs` : liste dans l'ordre des colonnes de `modele['df']`.
    """
    v = np.asarray(valeurs, dtype=float).reshape(1, -1)
    if modele['scaler'] is not None:
        v = modele['scaler'].transform(v)
    distances = np.linalg.norm(modele['centroides'][methode] - v, axis=1)
    return int(distances.argmin())
