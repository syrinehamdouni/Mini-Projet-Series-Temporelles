# %% [markdown]
# # 02 — ARIMA / SARIMA avec diagnostic complet (section 3.1) + prévisions glissantes h=10
# Série modélisée : log(Adj Close) de KO.

# %%
from common import *
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

F = make_features(load_prices())
n_train, origs = origins(len(F))
logKO = F["log_KO"].values
y_tr = logKO[:n_train]
d = read_json("key_results.json").get("d_ordre", 1) or 1   # d issu des tests ADF/KPSS (script 01)

# %% [markdown]
# ## Identification à partir de l'ACF/PACF (approche manuelle)

# %%
# Règle de lecture : p = dernier retard significatif de la PACF, q = dernier retard significatif de l'ACF
# (seuil ±1.96/√n, 5 premiers retards). C'est un POINT DE DÉPART à valider sur fig04 ; on peut le forcer :
ORDRE_FORCE = None   # ex. (1, 1, 1) pour imposer un ordre après lecture des graphiques
dy = np.diff(y_tr, n=d)
seuil = 1.96 / np.sqrt(len(dy))
a, pc = acf(dy, nlags=5, fft=False)[1:], pacf(dy, nlags=5)[1:]
p_man = max([k + 1 for k, v in enumerate(pc) if abs(v) > seuil], default=0)
q_man = max([k + 1 for k, v in enumerate(a) if abs(v) > seuil], default=0)
order_man = ORDRE_FORCE or (p_man, d, q_man)
print(f"Seuil de significativité ±{seuil:.3f} | ACF(1..5)={np.round(a, 3)} | PACF(1..5)={np.round(pc, 3)}")
print("Ordre manuel (ACF/PACF) :", order_man)

# %% [markdown]
# ## Recherche automatique (pmdarima, sinon grille AIC de secours) et SARIMA

# %%
def grille_aic(y, d, max_p=3, max_q=3):
    """Secours si pmdarima n'est pas disponible : on minimise l'AIC sur une petite grille (p,d,q)."""
    best = None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                r = ARIMA(y, order=(p, d, q)).fit()
            except (ValueError, np.linalg.LinAlgError):
                continue
            if best is None or r.aic < best[0]:
                best = (r.aic, (p, d, q))
    return best[1]

try:
    import pmdarima as pm
    order_auto = pm.auto_arima(y_tr, d=d, seasonal=False, information_criterion="aic", stepwise=True,
                               suppress_warnings=True, error_action="ignore").order
    sar = pm.auto_arima(y_tr, d=d, seasonal=True, m=5, D=0, information_criterion="aic", stepwise=True,
                        suppress_warnings=True, error_action="ignore")
    order_sar, seas_sar = sar.order, sar.seasonal_order
    print("auto_arima (pmdarima) utilisé")
except ImportError as e:
    print(f"pmdarima indisponible ({type(e).__name__}) -> grille AIC de secours")
    order_auto = grille_aic(y_tr, d)
    cands = [(1, 0, 0, 5), (0, 0, 1, 5), (1, 0, 1, 5)]   # saisonnalité hebdomadaire m=5 jours de bourse
    order_sar = order_auto
    seas_sar = min(cands, key=lambda s: ARIMA(y_tr, order=order_auto, seasonal_order=s).fit().aic)
print("Ordre auto :", order_auto, "| SARIMA :", order_sar, seas_sar)
# m=5 car en données boursières journalières le seul cycle court plausible est hebdomadaire ;
# la saisonnalité annuelle est testée par la STL (script 01).

NULL = (0, 0, 0, 0)
MODELES = {"ARIMA_manuel": (order_man, NULL), "ARIMA_auto": (order_auto, NULL), "SARIMA": (order_sar, seas_sar)}
print("Comparaison manuel vs auto :", "IDENTIQUES" if order_man == order_auto else "DIFFÉRENTS (à commenter dans le rapport)")

# %% [markdown]
# ## Estimation et diagnostic complet des résidus

# %%
rows, resids = [], {}
for name, (order, seas) in MODELES.items():
    with Timer(name):
        res = ARIMA(y_tr, order=order, seasonal_order=seas).fit()
    skip = order[1] + seas[1] * seas[3]            # on retire les 1ers résidus (initialisation de la différenciation)
    r = res.resid[max(skip, 1):]
    k = order[0] + order[2] + seas[0] + seas[2]    # nb de paramètres ARMA : corrige les degrés de liberté de Ljung-Box
    dg = diag_tests(r, model_df=k)
    print_diag(f"{name} {order}{seas if seas != NULL else ''}", dg)
    plot_diag(r, f"Diagnostic des résidus — {name}", f"fig0{7 + list(MODELES).index(name)}_resid_{name}.png")
    rows.append({"modele": name, "ordre": str(order), "saisonnier": str(seas), "AIC": res.aic, "BIC": res.bic, **dg})
    resids[name] = (r, order, seas)
cmp_ = pd.DataFrame(rows)
cmp_.to_csv(RES_DIR / "diag_arima.csv", index=False)
print("\n", cmp_[["modele", "ordre", "saisonnier", "AIC", "BIC", "LB10_p", "JB_p", "ARCH10_p"]].round(4).to_string(index=False))

# Meilleur modèle (AIC) : ses résidus servent au test ARCH-LM de la section 3.3.
best = cmp_.loc[cmp_["AIC"].idxmin(), "modele"]
log_json("key_results.json", best_arima=best, best_arima_order=str(MODELES[best][0]), best_arima_seasonal=str(MODELES[best][1]),
         arima_manual_order=str(order_man), arima_auto_order=str(order_auto),
         arima_arch_p=float(cmp_.loc[cmp_.modele == best, "ARCH10_p"].iloc[0]),
         arima_lb10_p=float(cmp_.loc[cmp_.modele == best, "LB10_p"].iloc[0]))
pd.Series(resids[best][0]).to_csv(RES_DIR / "resid_best_arima.csv", index=False, header=["resid"])
print("Meilleur ARIMA/SARIMA (AIC) :", best)

# %% [markdown]
# ## Prévision h=10 avec intervalle de confiance (premier bloc du test)

# %%
order_b, seas_b = MODELES[best]
res_b = ARIMA(y_tr, order=order_b, seasonal_order=seas_b).fit()
fc = res_b.get_forecast(H)
mean, ci = fc.predicted_mean, fc.conf_int(alpha=0.05)
dates = F.index
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(dates[n_train - 60:n_train], np.exp(y_tr[-60:]), label="train (60 derniers jours)")
ax.plot(dates[n_train:n_train + H], np.exp(logKO[n_train:n_train + H]), "k", label="observé (test)")
ax.plot(dates[n_train:n_train + H], np.exp(mean), "r--", label=f"prévision {best}")
ax.fill_between(dates[n_train:n_train + H], np.exp(ci[:, 0]), np.exp(ci[:, 1]), color="r", alpha=0.2, label="IC 95 %")
ax.legend(); ax.set_title(f"Prévision à {H} jours — {best} (KO, USD)")
savefig(fig, "fig10_prevision_arima.png")

# %% [markdown]
# ## Prévisions glissantes (walk-forward) pour la comparaison finale (section 6)
# Pour chaque bloc de 10 jours du test : on réestime sur tout l'historique connu (ordre fixé), on prévoit 10 pas.

# %%
def rolling(fit_forecast):
    return np.array([np.exp(fit_forecast(logKO[:o])) for o in origs])

save_forecasts("naive", rolling(lambda h: np.repeat(h[-1], H)))   # baseline marche aléatoire : dernier prix connu
for name, (order, seas) in MODELES.items():
    save_forecasts(name, rolling(lambda h, o=order, s=seas: ARIMA(h, order=o, seasonal_order=s).fit().forecast(H)))
    print("  prévisions glissantes OK :", name)
