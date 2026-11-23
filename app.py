"""Appli Streamlit : segmentation par clustering (KMeans et CAH) sur 4 jeux de données.

Lancer avec :  streamlit run app.py
"""
import html
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from clustering import DATASETS, a_des_noms, nom_classe, predire, preparer

DOSSIER = os.path.dirname(os.path.abspath(__file__))
EXEMPLE_MEDIANE = 'Valeurs médianes'
METHODES = ['KMeans', 'CAH']

# Une couleur par classe, la même dans tous les graphiques et toutes les cartes
PALETTE = ['#4F46E5', '#F59E0B', '#10B981', '#EF4444', '#06B6D4', '#8B5CF6',
           '#EC4899', '#84CC16']

st.set_page_config(page_title='Segmentation par clustering', page_icon='🧩',
                   layout='wide')

# --------------------------------------------------------------------------- #
# Style (couleurs semi-transparentes : lisible en thème clair comme sombre)
# --------------------------------------------------------------------------- #
CSS = """
<style>
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
h2, h3 { letter-spacing: -0.01em; }

/* Bandeau d'en-tête */
.hero {
  background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #06B6D4 120%);
  border-radius: 20px; padding: 28px 32px; color: #fff; margin-bottom: 1.2rem;
  box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
}
.hero .kicker { font-size: 0.78rem; letter-spacing: 0.14em; text-transform: uppercase;
  opacity: 0.85; font-weight: 600; }
.hero h1 { color: #fff; font-size: 2.1rem; margin: 0.2rem 0 0.5rem 0; padding: 0; }
.hero p { color: rgba(255,255,255,0.92); font-size: 1rem; line-height: 1.55;
  margin: 0 0 0.9rem 0; max-width: 62rem; }
.chip { display: inline-block; background: rgba(255,255,255,0.18); color: #fff;
  border: 1px solid rgba(255,255,255,0.35); border-radius: 999px; padding: 4px 13px;
  font-size: 0.82rem; margin: 0 8px 6px 0; font-weight: 500; }
.chip a { color: #fff; text-decoration: underline; }

/* Cartes d'indicateurs */
.kpi { border: 1px solid rgba(128,128,128,0.25); background: rgba(128,128,128,0.07);
  border-radius: 16px; padding: 16px 18px; }
.kpi .label { font-size: 0.8rem; opacity: 0.7; font-weight: 500; }
.kpi .value { font-size: 1.9rem; font-weight: 700; line-height: 1.25; }
.kpi .note { font-size: 0.78rem; opacity: 0.65; }
.kpi.best { border: 2px solid #10B981; background: rgba(16,185,129,0.09); }
.badge { display: inline-block; background: #10B981; color: #fff; border-radius: 999px;
  font-size: 0.68rem; padding: 2px 9px; margin-left: 6px; vertical-align: middle;
  font-weight: 600; letter-spacing: 0.03em; }

/* Cartes de classes */
.classe { border: 1px solid rgba(128,128,128,0.25); background: rgba(128,128,128,0.06);
  border-left-width: 6px; border-radius: 14px; padding: 14px 18px; margin-bottom: 12px; }
.classe .titre { font-weight: 700; font-size: 1.02rem; }
.classe .effectif { float: right; font-size: 0.85rem; opacity: 0.75; }
.classe .nom { margin: 4px 0 6px 0; font-size: 0.95rem; }
.classe .traits { font-size: 0.84rem; opacity: 0.8; line-height: 1.6; }
.hausse { color: #10B981; font-weight: 600; }
.baisse { color: #EF4444; font-weight: 600; }

/* Résultat de la prédiction */
.resultat { border-radius: 18px; padding: 22px 26px; margin: 8px 0 18px 0; color: #fff; }
.resultat .petit { font-size: 0.8rem; letter-spacing: 0.12em; text-transform: uppercase;
  opacity: 0.9; font-weight: 600; }
.resultat .grand { font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
.resultat .methode { font-size: 0.85rem; opacity: 0.9; margin-top: 6px; }

/* Onglets */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 8px 18px;
  font-weight: 600; }

/* Barre latérale */
section[data-testid="stSidebar"] .brand { font-size: 1.35rem; font-weight: 800;
  letter-spacing: -0.01em; }
section[data-testid="stSidebar"] .sous-titre { font-size: 0.82rem; opacity: 0.7;
  margin-bottom: 0.8rem; }
</style>
"""
def bloc(contenu):
    """Affiche du HTML (sur une seule ligne : le Markdown casse le HTML aux lignes vides)."""
    st.markdown(' '.join(ligne.strip() for ligne in contenu.splitlines()),
                unsafe_allow_html=True)


bloc(CSS)


def couleur(classe):
    return PALETTE[int(classe) % len(PALETTE)]


def style_figure(fig, legende=True):
    fig.update_layout(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', showlegend=legende,
                      margin=dict(l=10, r=10, t=40, b=10), legend_title_text='',
                      font=dict(size=13))
    return fig


def graphique_barres(partition):
    effectifs = pd.Series([f'Classe {c}' for c in partition]).value_counts().sort_index()
    cmap = {f'Classe {c}': couleur(c) for c in set(partition)}
    fig = px.bar(x=effectifs.index, y=effectifs.values, color=effectifs.index,
                 color_discrete_map=cmap, text_auto=True,
                 labels={'x': '', 'y': "Nombre d'individus"})
    fig.update_traces(marker_line_width=0)
    fig.update_layout(title='Effectif de chaque classe')
    return style_figure(fig, legende=False)


def graphique_acp(df_pca, partition, titre):
    etiquettes = [f'Classe {c}' for c in partition]
    cmap = {f'Classe {c}': couleur(c) for c in set(partition)}
    fig = px.scatter(df_pca, x='PC1', y='PC2', color=etiquettes,
                     color_discrete_map=cmap,
                     category_orders={'color': sorted(set(etiquettes))},
                     labels={'color': 'Classe'}, opacity=0.8)
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
    fig.update_layout(title=titre)
    return style_figure(fig)


@st.cache_resource(show_spinner='Entraînement des modèles…')
def charger_modele(nom):
    return preparer(nom, DOSSIER)


# --------------------------------------------------------------------------- #
# Barre latérale : choix du jeu de données
# --------------------------------------------------------------------------- #
with st.sidebar:
    bloc("""<div class="brand">🧩 Clustering</div>
            <div class="sous-titre">KMeans et classification ascendante hiérarchique
            (CAH) sur 4 jeux de données</div>""")
    nom = st.radio('Jeu de données', list(DATASETS))
    st.divider()
    st.caption("Chaque page présente les données, les classes trouvées par KMeans et "
               "par la CAH, et permet d'affecter un nouvel individu à une classe.")

cfg = DATASETS[nom]

try:
    m = charger_modele(nom)
except FileNotFoundError:
    st.error("Fichier de données introuvable : place-le dans le même dossier que "
             "app.py (wine.data, wdbc.data, wheat_seeds_dataset.csv ou Customer_dataset.csv).")
    st.stop()

df = m['df']
DEFAUT = cfg.get('modele_defaut', 'KMeans')
sil_km, sil_cah = m['silhouettes']['KMeans'], m['silhouettes']['CAH']
meilleur = 'KMeans' if sil_km >= sil_cah else 'CAH'

# --------------------------------------------------------------------------- #
# En-tête
# --------------------------------------------------------------------------- #
bloc(f"""<div class="hero">
  <div class="kicker">Segmentation par clustering</div>
  <h1>{html.escape(nom)}</h1>
  <p>{html.escape(cfg['description'])}</p>
  <span class="chip">{df.shape[0]} individus</span>
  <span class="chip">{df.shape[1]} variables</span>
  <span class="chip">KMeans : {cfg['k_kmeans']} classes</span>
  <span class="chip">CAH : {cfg['k_cah']} classes</span>
  <span class="chip"><a href="{html.escape(cfg['source'])}" target="_blank">Source UCI ↗</a></span>
</div>""")

tab_donnees, tab_classes, tab_pred = st.tabs(
    ['📊 Données', '🧬 Classes (KMeans / CAH)', '🎯 Affecter un nouvel individu'])

# --------------------------------------------------------------------------- #
# Onglet 1 : données
# --------------------------------------------------------------------------- #
with tab_donnees:
    st.subheader('Aperçu')
    st.dataframe(df.head(10))
    st.subheader('Statistiques descriptives')
    st.dataframe(df.describe().T)
    if cfg['echelle'] == 'standard':
        st.info("Les variables sont **standardisées** (moyenne 0, écart-type 1) avant le "
                "clustering, car leurs échelles sont très différentes.")
    elif cfg['echelle'] == 'normalize':
        st.info("Chaque individu est **normalisé** (vecteur de norme 1) avant le "
                "clustering, comme dans le TP.")

# --------------------------------------------------------------------------- #
# Onglet 2 : classes
# --------------------------------------------------------------------------- #
with tab_classes:
    # Indicateurs de qualité
    k1, k2, k3 = st.columns(3)
    with k1:
        bloc(f"""<div class="kpi {'best' if meilleur == 'KMeans' else ''}">
          <div class="label">Silhouette KMeans
          {'<span class="badge">MEILLEUR</span>' if meilleur == 'KMeans' else ''}</div>
          <div class="value">{sil_km:.3f}</div>
          <div class="note">Plus c'est proche de 1, mieux les classes sont séparées</div>
        </div>""")
    with k2:
        bloc(f"""<div class="kpi {'best' if meilleur == 'CAH' else ''}">
          <div class="label">Silhouette CAH
          {'<span class="badge">MEILLEUR</span>' if meilleur == 'CAH' else ''}</div>
          <div class="value">{sil_cah:.3f}</div>
          <div class="note">Plus c'est proche de 1, mieux les classes sont séparées</div>
        </div>""")
    with k3:
        bloc(f"""<div class="kpi">
          <div class="label">Accord KMeans / CAH</div>
          <div class="value">{m['ari_kmeans_cah']:.3f}</div>
          <div class="note">Indice de Rand ajusté (1 = partitions identiques)</div>
        </div>""")

    st.write('')
    methode = st.radio('Modèle affiché', METHODES, index=METHODES.index(DEFAUT),
                       horizontal=True, key=f'{nom}_methode')
    partition = m['partitions'][methode]

    g1, g2 = st.columns([2, 3])
    with g1:
        st.plotly_chart(graphique_barres(partition))
    with g2:
        st.plotly_chart(graphique_acp(m['df_pca'], partition,
                                      'Classes dans le plan de l\'ACP'))

    # Une carte par classe : effectif, nom, variables qui la distinguent le plus
    st.subheader('Les classes en un coup d\'œil')
    moyennes = df.groupby(partition).mean()
    ecarts = ((moyennes - df.mean()) / df.std()).fillna(0)
    for c in sorted(set(partition)):
        n = int((partition == c).sum())
        z = ecarts.loc[c].sort_values()
        hautes = ', '.join(html.escape(v) for v in z[::-1].index[:3] if z[v] > 0.2)
        basses = ', '.join(html.escape(v) for v in z.index[:3] if z[v] < -0.2)
        nom_c = nom_classe(cfg, c, methode)
        ligne_nom = (f'<div class="nom">{html.escape(nom_c)}</div>'
                     if a_des_noms(cfg, methode) else '')
        traits = ''
        if hautes:
            traits += f'<span class="hausse">▲ Plus élevé :</span> {hautes}<br>'
        if basses:
            traits += f'<span class="baisse">▼ Plus bas :</span> {basses}'
        bloc(f"""<div class="classe" style="border-left-color:{couleur(c)}">
          <span class="effectif">{n} individus ({n / len(partition):.0%})</span>
          <div class="titre" style="color:{couleur(c)}">Classe {c}</div>
          {ligne_nom}
          <div class="traits">{traits}</div>
        </div>""")

    with st.expander('Profil moyen détaillé de chaque classe'):
        profil = df.groupby(partition).mean().round(3)
        profil.index = [f'Classe {c}' for c in profil.index]
        st.dataframe(profil.T)

    if m['y'] is not None:
        with st.expander('Comparaison avec la classe réelle du jeu de données'):
            etiquettes = pd.Series([f'Classe {c}' for c in partition], index=df.index,
                                   name='Classe trouvée')
            st.dataframe(pd.crosstab(etiquettes, m['y']))

# --------------------------------------------------------------------------- #
# Onglet 3 : prédiction pour un nouvel individu
# --------------------------------------------------------------------------- #
with tab_pred:
    st.write("Saisis les caractéristiques d'un nouvel individu : il est affecté à la "
             "classe dont le centroïde est le plus proche.")
    p1, p2 = st.columns([1, 2])
    with p1:
        methode_pred = st.radio('Modèle utilisé', METHODES,
                                index=METHODES.index(DEFAUT), horizontal=True,
                                key=f'{nom}_methode_pred')
    with p2:
        exemple = st.selectbox('Pré-remplir avec',
                               [EXEMPLE_MEDIANE] + list(df.index[:10]),
                               key=f'{nom}_exemple')

    valeurs = []
    with st.expander('Caractéristiques de l\'individu', expanded=True):
        colonnes = st.columns(3)
        for i, col in enumerate(df.columns):
            if exemple == EXEMPLE_MEDIANE:
                defaut = float(df[col].median())
            else:
                defaut = float(df.loc[exemple, col])
            pas = float(df[col].std()) / 20 or 0.01
            with colonnes[i % 3]:
                # La clé contient l'exemple choisi : changer d'exemple remet les valeurs
                valeurs.append(st.number_input(col, value=defaut, step=pas,
                                               format='%.5f',
                                               key=f'{nom}_{exemple}_{col}'))

    if st.button('Prédire la classe', type='primary', key=f'{nom}_predire'):
        classe = predire(m, valeurs, methode_pred)
        bloc(f"""<div class="resultat" style="background:{couleur(classe)}">
          <div class="petit">Résultat · Classe {classe}</div>
          <div class="grand">{html.escape(nom_classe(cfg, classe, methode_pred))}</div>
          <div class="methode">Modèle utilisé : {methode_pred}</div>
        </div>""")

        # Position du nouvel individu dans le plan de l'ACP
        v = pd.DataFrame([valeurs], columns=df.columns).values
        if m['scaler'] is not None:
            v = m['scaler'].transform(v)
        pc = m['pca'].transform(v)[0]
        fig = graphique_acp(m['df_pca'], m['partitions'][methode_pred],
                            'Position du nouvel individu dans le plan de l\'ACP')
        fig.add_scatter(x=[pc[0]], y=[pc[1]], mode='markers', name='Nouvel individu',
                        marker=dict(size=18, symbol='star', color='black',
                                    line=dict(width=1.5, color='white')))
        st.plotly_chart(fig)