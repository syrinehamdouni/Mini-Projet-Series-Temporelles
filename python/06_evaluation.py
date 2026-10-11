# %% [markdown]
# # 06 — Évaluation comparative et test de Diebold-Mariano (section 6)

# %%
from common import *

px = load_prices()
n_train, origs = origins(len(px))
KO = px["KO"].values
actual = np.array([KO[o:o + H] for o in origs])             # (n_blocs, H) prix observés du test
ORDER = ["naive", "ARIMA_manuel", "ARIMA_auto", "SARIMA", "VAR", "VECM", "LSTM", "GRU"]
fc = {m: pd.read_csv(RES_DIR / f"fc_{m}.csv").values for m in ORDER if (RES_DIR / f"fc_{m}.csv").exists()}
print("Modèles évalués :", list(fc), f"| {len(origs)} blocs × {H} pas = {actual.size} prévisions chacun")

# %%
# Tableau récapitulatif UNIQUE : mêmes observations, mêmes métriques pour tous les modèles.
COMPLEXITE = {"naive": "Aucune", "ARIMA_manuel": "Faible", "ARIMA_auto": "Faible", "SARIMA": "Moyenne", "VAR": "Moyenne",
              "VECM": "Moyenne", "LSTM": "Élevée", "GRU": "Élevée"}
INTERPRET = {"naive": "Totale", "ARIMA_manuel": "Élevée", "ARIMA_auto": "Élevée", "SARIMA": "Élevée", "VAR": "Élevée (Granger)",
             "VECM": "Élevée (équilibre LT)", "LSTM": "Faible (boîte noire)", "GRU": "Faible (boîte noire)"}
tim = read_json("timings.json")
lstm_t = {"LSTM": [v for k, v in tim.items() if k.startswith("LSTM_")], "GRU": [v for k, v in tim.items() if k.startswith("GRU_")]}
rows = []
for m, f in fc.items():
    t = tim.get(m) if m in tim else (np.mean(lstm_t[m]) if m in lstm_t and lstm_t[m] else np.nan)
    rows.append({"modele": m, **metrics(actual.ravel(), f.ravel()), "temps_fit_s": t,
                 "complexite": COMPLEXITE[m], "interpretabilite": INTERPRET[m]})
tab = pd.DataFrame(rows).sort_values("RMSE")
tab.to_csv(RES_DIR / "metrics.csv", index=False)
print("\n", tab.round(4).to_string(index=False))

# %%
# Test de Diebold-Mariano : une différence de RMSE n'est utile que si elle est statistiquement significative.
err = {m: (actual - f).ravel() for m, f in fc.items()}
names = list(fc)
pmat = pd.DataFrame(np.nan, index=names, columns=names)
dm_rows = []
for a in names:
    for b in names:
        if a != b:
            pmat.loc[a, b] = dm_test(err[a], err[b])[1]
    if a != "naive":
        stat, p = dm_test(err["naive"], err[a])          # stat > 0 : le modèle 'a' bat la baseline
        dm_rows.append({"modele": a, "DM_stat_vs_naive": stat, "p_value": p,
                        "conclusion": ("meilleur que la baseline" if stat > 0 else "moins bon que la baseline") + (
                            " — SIGNIFICATIF (5 %)" if p < 0.05 else " — NON significatif")})
pmat.to_csv(RES_DIR / "dm_pvalues.csv")
dmv = pd.DataFrame(dm_rows)
dmv.to_csv(RES_DIR / "dm_vs_naive.csv", index=False)
print("\nDiebold-Mariano contre la baseline naïve (perte quadratique) :\n", dmv.round(4).to_string(index=False))

# %%
# Conclusion automatique, à reformuler dans le rapport en tenant compte de l'interprétabilité et du coût.
best = tab.iloc[0]["modele"]
print(f"\nMeilleur RMSE : {best} ({tab.iloc[0]['RMSE']:.4f} USD) ; baseline naïve : {tab.loc[tab.modele == 'naive', 'RMSE'].iloc[0]:.4f} USD")
sig = dmv[dmv.p_value < 0.05]
print("Modèles significativement différents de la baseline :", list(sig.modele) or "AUCUN (cas fréquent sur actions liquides : marché quasi efficient)")
log_json("key_results.json", best_rmse_model=best, models_sig_vs_naive=list(sig.modele))

# %%
# Figures : prévisions recollées sur tout le test, barres de RMSE, carte des p-values DM.
dates_blocks = np.concatenate([px.index[o:o + H] for o in origs])
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates_blocks, actual.ravel(), "k", lw=1.8, label="observé")
for m, f in fc.items():
    for k in range(len(origs)):
        ax.plot(px.index[origs[k]:origs[k] + H], f[k], lw=0.9, color=plt.cm.tab10(ORDER.index(m)), label=m if k == 0 else None)
ax.legend(ncol=4, fontsize=8); ax.set_title(f"Prévisions à {H} jours sur l'échantillon test (blocs de {H} jours, KO en USD)")
savefig(fig, "fig17_previsions_test.png")

fig, ax = plt.subplots(figsize=(9, 4))
t = tab.sort_values("RMSE")
ax.barh(t.modele, t.RMSE, color="steelblue"); ax.invert_yaxis(); ax.set_xlabel("RMSE (USD)"); ax.set_title("RMSE par modèle (plus petit = meilleur)")
savefig(fig, "fig18_metriques.png")

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(pmat.values.astype(float), vmin=0, vmax=0.2, cmap="RdYlGn_r")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
for i in range(len(names)):
    for j in range(len(names)):
        if i != j:
            ax.text(j, i, f"{pmat.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, label="p-value DM (rouge < 0.05 : différence significative)"); ax.set_title("Test de Diebold-Mariano — p-values par paires")
savefig(fig, "fig19_dm_pvalues.png")
