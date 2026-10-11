"""Application Streamlit simple : résultats du projet + section IA générative (LLM).
Lancer depuis la racine du projet :  streamlit run app/streamlit_app.py
(Il faut d'abord avoir exécuté les scripts python/01 à 07 : l'application ne fait que LIRE figures/, results/ et llm/.)"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from figures_info import FIGURES  # noqa: E402

FIG, RES, LLM = ROOT / "figures", ROOT / "results", ROOT / "llm"
st.set_page_config(page_title="KO & PEP — Séries temporelles", layout="wide")
st.title("Coca-Cola (KO) & PepsiCo (PEP) — Séries temporelles, 2019-2024")

page = st.sidebar.radio("Navigation", ["Résumé", "Graphiques et interprétation", "Tableaux de résultats", "IA générative (LLM)"])


def read_csv(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


if page == "Résumé":
    st.write((RES / "source.txt").read_text(encoding="utf-8") if (RES / "source.txt").exists()
             else "Exécutez d'abord python/01_donnees_eda.py.")
    kr = RES / "key_results.json"
    if kr.exists():
        st.subheader("Résultats clés")
        st.json(json.loads(kr.read_text()))

elif page == "Graphiques et interprétation":
    sections = sorted({f["section"] for f in FIGURES})
    choix = st.selectbox("Section du cahier des charges", ["Toutes"] + sections)
    for f in FIGURES:
        if choix != "Toutes" and f["section"] != choix:
            continue
        st.subheader(f["titre"])
        st.caption(f"Section {f['section']} — fichier figures/{f['f']} — produit par python/{f['script']}")
        if (FIG / f["f"]).exists():
            st.image(str(FIG / f["f"]), use_container_width=True)
        else:
            st.warning("Figure absente : exécutez le script indiqué.")
        st.markdown(f"**Ce que montre le graphique :** {f['montre']}")
        st.markdown(f"**Comment l'interpréter :** {f['regles']}")
        st.divider()

elif page == "Tableaux de résultats":
    tables = {"Stationnarité (ADF/KPSS)": "stationnarite.csv", "Statistiques descriptives": "stats_descriptives.csv",
              "Diagnostic ARIMA/SARIMA": "diag_arima.csv", "Diagnostic VAR": "diag_var.csv",
              "Sélection GARCH": "garch_selection.csv", "Diagnostic GARCH": "diag_garch.csv",
              "Hyperparamètres LSTM/GRU": "dl_hyperparams.csv", "Validation croisée walk-forward": "dl_cv_walkforward.csv",
              "COMPARAISON FINALE (MSE, RMSE, MAE, MAPE)": "metrics.csv", "Diebold-Mariano vs baseline": "dm_vs_naive.csv",
              "Diebold-Mariano (p-values par paires)": "dm_pvalues.csv"}
    for titre, fn in tables.items():
        df = read_csv(fn)
        st.subheader(titre)
        st.dataframe(df) if df is not None else st.info(f"{fn} absent : exécuter les scripts.")

else:
    st.header("Section 5 — IA générative et LLM")
    st.info("Méthode : (1) vous rédigez votre interprétation experte AVANT ; (2) vous collez chaque prompt dans un LLM ; "
            "(3) vous collez la réponse INTÉGRALE ici ; (4) vous vérifiez chaque affirmation chiffrée (hallucinations statistiques).")
    tasks = {"1. Hypothèses de modèles": "01_hypotheses_modeles", "2. Vulgarisation pour un décideur": "02_vulgarisation",
             "3. Recommandations simulées + limites": "03_recommandations"}
    tabs = st.tabs(list(tasks) + ["Mon interprétation", "Comparaison humain / LLM"])
    for tab, (titre, name) in zip(tabs, tasks.items()):
        with tab:
            pf, rf = LLM / "prompts" / f"{name}.md", LLM / "reponses" / f"{name}.md"
            st.subheader("Prompt envoyé au LLM")
            st.code(pf.read_text(encoding="utf-8") if pf.exists() else "Exécutez python/07_llm_prompts.py", language="markdown")
            st.subheader("Réponse du LLM (à coller telle quelle)")
            txt = st.text_area("Réponse", rf.read_text(encoding="utf-8") if rf.exists() else "", height=300, key=name)
            if st.button("Enregistrer la réponse", key="b" + name):
                rf.parent.mkdir(parents=True, exist_ok=True)
                rf.write_text(txt, encoding="utf-8")
                st.success("Enregistré dans llm/reponses/")
    for tab, fn in [(tabs[3], "interpretation_humaine.md"), (tabs[4], "comparaison_critique.md")]:
        with tab:
            p = LLM / fn
            txt = st.text_area(fn, p.read_text(encoding="utf-8") if p.exists() else "", height=450, key=fn)
            if st.button("Enregistrer", key="b" + fn):
                p.write_text(txt, encoding="utf-8")
                st.success(f"Enregistré dans llm/{fn}")
    st.warning("Attention à l'hallucination statistique : p-values inventées, causalité non vérifiée. "
               "Toute affirmation du LLM doit être comparée à vos tableaux de résultats.")
