# %% [markdown]
# # 03 — VAR, causalité de Granger, cointégration et VECM (section 3.2)
# Variables : KO et PEP. VAR sur les rendements (stationnaires), cointégration sur les log-prix (I(1)).

# %%
from common import *
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank
from statsmodels.tsa.vector_ar.vecm import select_order as vecm_select_order

F = make_features(load_prices())
n_train, origs = origins(len(F))
logKO = F["log_KO"].values
ret = F[["ret_KO", "ret_PEP"]]
ret_tr = ret.iloc[1:n_train].reset_index(drop=True)          # rendements stationnaires (d=1 sur les log-prix)

# %% [markdown]
# ## VAR sur les rendements

# %%
# Ordre choisi par BIC (plus parcimonieux que l'AIC sur ~1300 observations) ; le tableau montre aussi l'AIC.
sel = VAR(ret_tr).select_order(10)
print(sel.summary())
p_var = max(1, sel.selected_orders["bic"])
with Timer("VAR"):
    var_res = VAR(ret_tr).fit(p_var)
print(f"\nVAR({p_var}) estimé | stable : {var_res.is_stable()} (racines hors du cercle unité = OK)")

# Causalité de Granger dans les deux sens (test F). H0 : la variable 'causante' n'aide pas à prévoir l'autre.
g1 = var_res.test_causality("ret_KO", ["ret_PEP"], kind="f")
g2 = var_res.test_causality("ret_PEP", ["ret_KO"], kind="f")
print("\nGranger PEP -> KO :", verdict(g1.pvalue, "PEP ne cause pas KO au sens de Granger"))
print("Granger KO -> PEP :", verdict(g2.pvalue, "KO ne cause pas PEP au sens de Granger"))
print("(Granger = pouvoir prédictif, pas une causalité économique.)")

# Diagnostic des résidus du VAR : portemanteau multivarié, normalité, puis tests par équation.
white, norm = var_res.test_whiteness(nlags=20), var_res.test_normality()
print("Portemanteau (autocorrélation des résidus) :", verdict(white.pvalue, "résidus = bruit blanc"))
print("Normalité (Jarque-Bera multivarié)         :", verdict(norm.pvalue, "résidus normaux"))
diag_rows = []
for col in ["ret_KO", "ret_PEP"]:
    dg = diag_tests(var_res.resid[col].values, model_df=p_var)
    print_diag(f"VAR équation {col}", dg)
    diag_rows.append({"equation": col, **dg})
pd.DataFrame(diag_rows).to_csv(RES_DIR / "diag_var.csv", index=False)
plot_diag(var_res.resid["ret_KO"].values, f"Diagnostic des résidus — VAR({p_var}), équation KO", "fig11_resid_VAR_KO.png")

# %% [markdown]
# ## Cointégration sur les log-prix

# %%
lv = F[["log_KO", "log_PEP"]].reset_index(drop=True)
lv_tr = lv.iloc[:n_train]
# Engle-Granger (2 séries) : H0 = PAS de cointégration.
eg_t, eg_p, _ = coint(lv_tr["log_KO"], lv_tr["log_PEP"], trend="c", autolag="aic")
print("Engle-Granger :", verdict(eg_p, "pas de cointégration"))
# Johansen (trace) : rang de cointégration r. Le nombre de retards vient du critère BIC sur les niveaux.
k_diff = max(1, vecm_select_order(lv_tr, maxlags=10, deterministic="co").bic)
rank = select_coint_rank(lv_tr, det_order=0, k_ar_diff=k_diff, method="trace", signif=0.05)
print(f"Johansen (trace, 5 %) : rang de cointégration r = {rank.rank} (k_ar_diff = {k_diff})")
print(rank.summary())

# Spread de long terme KO - b·PEP (régression des niveaux) : doit osciller autour d'une moyenne si cointégration.
b, c = np.polyfit(lv_tr["log_PEP"], lv_tr["log_KO"], 1)
spread = lv["log_KO"] - b * lv["log_PEP"] - c
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(F.index, spread, lw=0.8); ax.axhline(0, color="k"); ax.axvline(F.index[n_train], color="r", ls="--")
ax.set_title(f"Écart de long terme  log(KO) − {b:.2f}·log(PEP) − {c:.2f}  (pointillé rouge = début du test)")
savefig(fig, "fig12_spread_cointegration.png")
log_json("key_results.json", var_order=int(p_var), granger_PEP_to_KO_p=float(g1.pvalue), granger_KO_to_PEP_p=float(g2.pvalue),
         var_whiteness_p=float(white.pvalue), var_stable=bool(var_res.is_stable()),
         engle_granger_p=float(eg_p), johansen_rank=int(rank.rank), vecm_lags_diff=int(k_diff),
         coint_beta_PEP=float(b))

# %% [markdown]
# ## VECM rank 1 si Engle-Granger soutient la cointégration entre les deux séries I(1).

# %%
vecm_ok = eg_p < 0.05
if vecm_ok:
    vecm_rank = 1
    with Timer("VECM"):
        vecm = VECM(lv_tr, k_ar_diff=k_diff, coint_rank=vecm_rank, deterministic="ci").fit()
    alpha, beta, pa = vecm.alpha[:, 0], vecm.beta[:, 0], vecm.pvalues_alpha[:, 0]
    print("\nVECM : relation de long terme normalisée sur la 1re variable (log_KO) :")
    print(f"  beta (log_KO, log_PEP) = {np.round(beta, 4)}")
    print(f"  alpha (vitesse d'ajustement) : KO = {alpha[0]:.4f} (p={pa[0]:.3f}) | PEP = {alpha[1]:.4f} (p={pa[1]:.3f})")
    for nm, al, pv in [("KO", alpha[0], pa[0]), ("PEP", alpha[1], pa[1])]:
        if al < 0 and pv < 0.05:
            print(f"  -> {nm} corrige l'écart : environ {abs(al)*100:.1f} % de l'écart disparaît chaque jour, "
                  f"demi-vie ≈ {np.log(0.5)/np.log(1+al):.0f} jours de bourse.")
        else:
            print(f"  -> {nm} ne corrige pas significativement l'écart (alpha non significatif ou de signe non correcteur).")
    log_json("key_results.json", vecm_fit_rank=vecm_rank, vecm_deterministic="ci",
             vecm_alpha_KO=float(alpha[0]), vecm_alpha_PEP=float(alpha[1]),
             vecm_alpha_KO_p=float(pa[0]), vecm_alpha_PEP_p=float(pa[1]))
    if rank.rank == 2:
        print("ATTENTION : Engle-Granger soutient le rang 1, mais Johansen Python indique le rang plein 2 ; "
              "le VECM rank 1 est conditionnel à Engle-Granger et la spécification doit être discutée.")
else:
    print("\nEngle-Granger ne détecte pas de cointégration au seuil de 5 % -> pas de VECM.")

# %% [markdown]
# ## Prévisions glissantes h=10 (VAR et VECM) pour la comparaison finale

# %%
def var_forecast(o):
    r = ret.iloc[1:o].reset_index(drop=True)
    rf = VAR(r).fit(p_var).forecast(r.values[-p_var:], steps=H)[:, 0]      # rendements prévus de KO
    return np.exp(logKO[o - 1] + np.cumsum(rf))                              # on recompose le prix

save_forecasts("VAR", np.array([var_forecast(o) for o in origs]))
if vecm_ok:
    save_forecasts("VECM", np.array([
        np.exp(VECM(lv.iloc[:o], k_ar_diff=k_diff, coint_rank=vecm_rank, deterministic="ci").fit().predict(steps=H)[:, 0])
        for o in origs]))
print("Prévisions glissantes OK : VAR" + (", VECM" if vecm_ok else ""))
