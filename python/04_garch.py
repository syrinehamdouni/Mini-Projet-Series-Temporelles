# %% [markdown]
# # 04 — Test ARCH-LM et modèles GARCH (section 3.3)
# Série : rendements journaliers de KO (en %), échantillon d'apprentissage.

# %%
from common import *
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

F = make_features(load_prices())
n_train, _ = origins(len(F))
r = (F["ret_KO"].iloc[1:n_train] * 100)          # en % : l'optimiseur de GARCH converge mieux à cette échelle

# %%
# 1) Faut-il un GARCH ? Test ARCH-LM sur les résidus du meilleur ARIMA (script 02) et sur les rendements.
res_arima = pd.read_csv(RES_DIR / "resid_best_arima.csv")["resid"].values
p_arch_arima = het_arch(res_arima, nlags=10)[1]
p_arch_ret = het_arch(r.values - r.mean(), nlags=10)[1]
print("ARCH-LM résidus du meilleur ARIMA :", verdict(p_arch_arima, "variance conditionnelle constante"))
print("ARCH-LM rendements KO             :", verdict(p_arch_ret, "variance conditionnelle constante"))
print("=> si H0 rejetée : la variance varie dans le temps, un GARCH est justifié." )

# %%
# 2) Sélection : GARCH(1,1) de référence comparé à des ordres supérieurs, lois normale et Student-t (BIC).
rows, fits = [], {}
for (p, q) in [(1, 1), (1, 2), (2, 1), (2, 2)]:
    for dist in ["normal", "t"]:
        f = arch_model(r, mean="Constant", vol="GARCH", p=p, q=q, dist=dist).fit(disp="off")
        fits[(p, q, dist)] = f
        rows.append({"modele": f"GARCH({p},{q})", "loi": dist, "logL": f.loglikelihood, "AIC": f.aic, "BIC": f.bic})
sel = pd.DataFrame(rows)
sel.to_csv(RES_DIR / "garch_selection.csv", index=False)
print("\n", sel.round(2).to_string(index=False))
best_row = sel.loc[sel["BIC"].idxmin()]
key_best = (int(best_row.modele[6]), int(best_row.modele[8]), best_row["loi"])
print("\nMeilleur selon BIC :", best_row.modele, best_row["loi"])

# %%
# 3) Diagnostic des résidus STANDARDISÉS : doivent redevenir sans autocorrélation et homoscédastiques.
def diag_garch(f, label):
    z = np.asarray(f.std_resid, dtype=float); z = z[~np.isnan(z)]
    lb_z = acorr_ljungbox(z, lags=[10, 20], return_df=True)["lb_pvalue"]
    lb_z2 = acorr_ljungbox(z ** 2, lags=[10, 20], return_df=True)["lb_pvalue"]   # z² : variance résiduelle ?
    d = {"modele": label, "LB10_z_p": lb_z.iloc[0], "LB20_z_p": lb_z.iloc[1], "LB10_z2_p": lb_z2.iloc[0],
         "LB20_z2_p": lb_z2.iloc[1], "ARCH10_p": het_arch(z, nlags=10)[1], "JB_p": stats.jarque_bera(z)[1]}
    print(f"\n--- Diagnostic résidus standardisés : {label}")
    print("  Ljung-Box z   :", verdict(d["LB10_z_p"], "pas d'autocorrélation"))
    print("  Ljung-Box z²  :", verdict(d["LB10_z2_p"], "pas d'autocorrélation de la variance"))
    print("  ARCH-LM z     :", verdict(d["ARCH10_p"], "homoscédasticité retrouvée"))
    print("  Jarque-Bera z :", verdict(d["JB_p"], "résidus standardisés normaux"))
    return d

f11 = fits[(1, 1, "normal")]
f_best = fits[key_best]
dg = pd.DataFrame([diag_garch(f11, "GARCH(1,1) normal"), diag_garch(f_best, f"{best_row.modele} {best_row['loi']} (meilleur BIC)")])
dg.to_csv(RES_DIR / "diag_garch.csv", index=False)

# %%
# 4) Persistance et demi-vie des chocs de volatilité (interprétation économique du modèle retenu).
par = f_best.params
persist = sum(v for k, v in par.items() if k.startswith(("alpha", "beta")))
half = np.log(0.5) / np.log(persist) if 0 < persist < 1 else np.inf
print(f"\nPersistance (Σα+Σβ) = {persist:.4f} -> demi-vie estimée d'un choc de volatilité ~ {half:.1f} jours")
vol_prev = np.sqrt(f_best.forecast(horizon=H).variance.iloc[-1].values)
pd.DataFrame({"horizon": range(1, H + 1), "vol_journaliere_%": vol_prev}).to_csv(RES_DIR / "garch_vol_forecast.csv", index=False)
log_json("key_results.json", garch_best=f"{best_row.modele} {best_row['loi']}", garch_persistence=float(persist),
         garch_half_life_days=float(half), arch_lm_arima_p=float(p_arch_arima), arch_lm_returns_p=float(p_arch_ret),
         garch_z2_lb10_p=float(dg["LB10_z2_p"].iloc[1]), garch_arch_z_p=float(dg["ARCH10_p"].iloc[1]))

# %%
# 5) Volatilité conditionnelle dans le temps, reliée à des événements connus (à vérifier sur le graphique).
vol = pd.Series(np.asarray(f_best.conditional_volatility), index=F.index[1:n_train])
fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
ax[0].plot(r.index if hasattr(r, "index") else vol.index, r.values, lw=0.5); ax[0].set_title("Rendements KO (%)")
ax[1].plot(vol, lw=0.9); ax[1].set_title(f"Volatilité conditionnelle estimée — {best_row.modele} {best_row['loi']} (écart-type journalier, %)")
for date, txt in [("2020-03-16", "Krach COVID-19"), ("2022-03-16", "1re hausse des taux Fed")]:
    for a in ax:
        a.axvline(pd.Timestamp(date), color="r", ls="--", lw=0.8)
    ax[1].text(pd.Timestamp(date), vol.max() * 0.9, " " + txt, color="r", rotation=90, va="top")
savefig(fig, "fig13_garch_volatilite.png")

z = np.asarray(f_best.std_resid, dtype=float); z = z[~np.isnan(z)]
plot_diag(z, f"Résidus standardisés — {best_row.modele} {best_row['loi']}", "fig14_garch_residus.png")
