# %% [markdown]
# # 07 — Section 5 (IA générative) : génération des prompts à partir de VOS résultats
# Ce script ne parle pas à un LLM : il prépare les prompts. Vous les collez dans ChatGPT / Claude / Gemini,
# puis vous collez la réponse dans llm/reponses/ (ou directement dans l'application Streamlit).

# %%
from common import *

key = read_json("key_results.json")
def csv_txt(name, nd=4):
    p = RES_DIR / name
    return pd.read_csv(p).round(nd).to_string(index=False) if p.exists() else "(fichier absent : exécuter les scripts précédents)"

desc = pd.read_csv(RES_DIR / "stats_descriptives.csv", index_col=0).round(5).to_string() if (RES_DIR / "stats_descriptives.csv").exists() else "(absent)"
CTX = "Contexte : séries journalières Coca-Cola (KO) et PepsiCo (PEP), 2019-2024 (Yahoo Finance, prix ajustés), split chronologique 90/10.\n"

# %%
P1 = CTX + f"""
Tu es un expert en séries temporelles. Voici les statistiques descriptives des rendements log (échantillon d'apprentissage) :
{desc}

Résultats des tests de stationnarité (ADF : H0 = racine unitaire ; KPSS : H0 = stationnaire) :
{csv_txt('stationnarite.csv')}

TÂCHE : propose une liste ordonnée d'hypothèses de modèles à tester (ARIMA, SARIMA, VAR, cointégration/VECM, GARCH, réseaux
récurrents...). Pour chaque modèle, justifie par les chiffres ci-dessus et indique quel diagnostic le validerait ou l'invaliderait.
N'invente aucun chiffre : si une information manque, dis-le.
"""

P2 = CTX + f"""
Tu t'adresses à un décideur NON statisticien (directeur financier). Explique simplement, sans jargon, les résultats suivants.

Tableau de comparaison des modèles de prévision à 10 jours (prix de KO en USD ; RMSE/MAE en dollars, MAPE en %) :
{csv_txt('metrics.csv')}

Test de Diebold-Mariano contre la baseline naïve :
{csv_txt('dm_vs_naive.csv')}

Résultats clés : {json.dumps({k: key.get(k) for k in ['granger_PEP_to_KO_p','granger_KO_to_PEP_p','engle_granger_p','johansen_rank','garch_best','garch_persistence','garch_half_life_days','best_arima','best_rmse_model']}, ensure_ascii=False)}

Diagnostics ARIMA :
{csv_txt('diag_arima.csv')}
Diagnostics VAR :
{csv_txt('diag_var.csv')}
Diagnostics GARCH :
{csv_txt('diag_garch.csv')}
Validation walk-forward des réseaux :
{csv_txt('dl_cv_walkforward.csv')}

TÂCHE : (1) ce que les résultats disent de KO et PEP, (2) ce que signifient cointégration, Granger et volatilité conditionnelle
en langage courant, (3) le modèle à retenir et pourquoi. N'invente aucune p-value ni aucun chiffre absent du texte ci-dessus.
Signale explicitement tout diagnostic rejeté, toute divergence de rang de cointégration et toute absence de différence significative au test DM.
"""

P3 = CTX + f"""
À partir des performances ci-dessous, formule des recommandations d'investissement SIMULÉES (exercice pédagogique) sur KO et PEP :
{csv_txt('metrics.csv')}
Volatilité prévue par le GARCH sur les 10 prochains jours (écart-type journalier en %) :
{csv_txt('garch_vol_forecast.csv')}
Comparaison statistique des prévisions (DM contre naïf) :
{csv_txt('dm_vs_naive.csv')}

TÂCHE : (1) donne tes recommandations ; (2) TOI-MÊME, liste ensuite explicitement les limites, risques et hypothèses fragiles de ces
recommandations (efficience des marchés, coûts de transaction, instabilité des paramètres, risque de modèle, surapprentissage...).
Il s'agit uniquement d'un exercice pédagogique, pas d'un conseil financier personnalisé. Ne présente pas une meilleure métrique descriptive
comme un avantage statistiquement significatif si le test DM ne le confirme pas.
"""

for name, txt in [("01_hypotheses_modeles", P1), ("02_vulgarisation", P2), ("03_recommandations", P3)]:
    (LLM_DIR / "prompts" / f"{name}.md").write_text(txt.strip() + "\n", encoding="utf-8")
    rep = LLM_DIR / "reponses" / f"{name}.md"
    if not rep.exists():
        rep.write_text("(Collez ici la réponse INTÉGRALE du LLM, sans la modifier.)\n", encoding="utf-8")

# Create starter documents only for a new project; never erase the student's analysis or critique.
human = LLM_DIR / "interpretation_humaine.md"
if not human.exists():
    human.write_text(
        "# Mon interprétation experte\n\n"
        "## Stationnarité et choix de d\n\n## Modèle ARIMA retenu et diagnostic des résidus\n\n"
        "## VAR, Granger, cointégration / VECM\n\n## Volatilité conditionnelle (GARCH)\n\n"
        "## LSTM/GRU vs modèles classiques\n\n## Meilleur modèle et pourquoi (performance, interprétabilité, coût)\n",
        encoding="utf-8")

critique = LLM_DIR / "comparaison_critique.md"
if not critique.exists():
    critique.write_text(
        "# Comparaison humain / LLM — détection des erreurs (hallucinations statistiques)\n\n"
        "| Affirmation du LLM | Vérification (mes résultats / script) | Verdict | Correction |\n"
        "|---|---|---|---|\n| | | | |\n\n"
        "## Où l'interprétation du LLM converge avec la mienne\n\n"
        "## Où le LLM se trompe, simplifie ou invente une causalité\n\n"
        "## Ce que le LLM m'a fait gagner (temps, clarté) et ce qu'il n'a pas su faire\n",
        encoding="utf-8")
print("Prompts générés dans llm/prompts/ — copiez-les dans un LLM, puis collez les réponses dans llm/reponses/.")
