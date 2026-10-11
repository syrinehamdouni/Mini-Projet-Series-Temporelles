# %% [markdown]
# # 01 — Données et analyse exploratoire (sections 1 et 2)
# Source : Yahoo Finance, Coca-Cola (KO) et PepsiCo (PEP), journalier, 2019-2024.

# %%
from common import *
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

key_results_path = RES_DIR / "key_results.json"
key_results_path.write_text("{}\n", encoding="utf-8")
print_versions()

# %% [markdown]
# ## 1. Importation des données

# %%
# Téléchargement une seule fois puis cache en CSV : le notebook reste reproductible même hors-ligne.
def telecharger():
    import yfinance as yf

    raw = yf.download(
        TICKERS,
        start=START,
        end=END_EXCL,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=False,
        timeout=30
    )

    if raw.empty:
        raise RuntimeError(
            "Yahoo Finance n'a retourné aucune donnée. Réessayez."
        )

    for t in TICKERS:
        if t not in raw.columns.get_level_values(0):
            raise RuntimeError(f"Données absentes pour {t}")

        df = raw[t].dropna(how="all")

        if df.empty:
            raise RuntimeError(f"Aucune donnée valide pour {t}")

        df.to_csv(DATA_DIR / f"{t}.csv")
        print(f"{t}: {len(df)} lignes sauvegardées.")

if not all((DATA_DIR / f"{t}.csv").exists() for t in TICKERS):
    try:
        telecharger()
    except Exception as e:
        raise SystemExit(f"Téléchargement impossible ({e}). Téléchargez KO et PEP à la main sur "
                         "finance.yahoo.com (Historical Data, 2019-2024) et placez KO.csv / PEP.csv dans data/.")

# %%
# Série temporelle native : DatetimeIndex trié, prix ajustés des dividendes (Adj Close).
px = load_raw("KO")  # aperçu du vecteur multidimensionnel complet
print(px.head(), "\n")
prices = load_prices()
F = make_features(prices)
n = len(F)
n_train = int(n * TRAIN_RATIO)          # split CHRONOLOGIQUE : jamais aléatoire sur une série temporelle
train, test = F.iloc[:n_train], F.iloc[n_train:]
print(f"{n} jours | train : {train.index[0].date()} -> {train.index[-1].date()} ({len(train)}) "
      f"| test : {test.index[0].date()} -> {test.index[-1].date()} ({len(test)})")

# Documentation de la source (exigée par la consigne 1.3)
(RES_DIR / "source.txt").write_text(
    "Source : Yahoo Finance (finance.yahoo.com) via yfinance, tickers KO et PEP, fréquence journalière.\n"
    f"Période : {prices.index[0].date()} -> {prices.index[-1].date()} ({n} jours de bourse).\n"
    "Variables brutes : Open, High, Low, Close, Adj Close, Volume.\n"
    "Transformation du fournisseur : 'Adj Close' est ajusté des dividendes et des splits ; 'Close' ne l'est que des splits.\n"
    "Série modélisée : log(Adj Close). Indicateurs dérivés : rendement log, moyenne mobile 20 j, volatilité réalisée 21 j.\n"
    f"Split chronologique : train 90 % ({len(train)} jours), test 10 % ({len(test)} jours).\n", encoding="utf-8")
pd.concat({t: load_raw(t).describe().T for t in TICKERS}).to_csv(RES_DIR / "describe_raw.csv")

# %% [markdown]
# ## 2. Analyse exploratoire

# %%
# Prix bruts (échelles différentes) puis rebasés base 100 pour comparer les dynamiques.
fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
prices.plot(ax=ax[0]); ax[0].set_title("Prix ajustés (USD) — échelles différentes")
(prices / prices.iloc[0] * 100).plot(ax=ax[1]); ax[1].set_title("Prix rebasés (base 100 au 01/01/2019)")
for a in ax:
    a.axvline(test.index[0], color="k", ls="--", lw=1)  # frontière train / test
ax[1].text(test.index[0], ax[1].get_ylim()[1] * 0.95, " test", va="top")
savefig(fig, "fig01_prix.png")

# Indicateurs dérivés : moyenne mobile, rendements, volatilité réalisée.
fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
ax[0].plot(prices["KO"], lw=0.8, label="KO"); ax[0].plot(F["ma20_KO"], label="MM 20 j"); ax[0].legend(); ax[0].set_title("KO et moyenne mobile 20 j")
F[["ret_KO", "ret_PEP"]].plot(ax=ax[1], lw=0.6); ax[1].set_title("Rendements logarithmiques")
F[["vol21_KO", "vol21_PEP"]].plot(ax=ax[2]); ax[2].set_title("Volatilité réalisée 21 j (annualisée) — clusters de volatilité")
savefig(fig, "fig02_indicateurs.png")

# Statistiques descriptives sur le TRAIN (utilisées aussi pour le prompt LLM, section 5).
desc = pd.DataFrame({c: {"moyenne": train[c].mean(), "variance": train[c].var(), "ecart_type": train[c].std(),
                         "skewness": train[c].skew(), "kurtosis_excess": train[c].kurt(),
                         "min": train[c].min(), "max": train[c].max(),
                         "JB_p": stats.jarque_bera(train[c].dropna())[1]} for c in ["ret_KO", "ret_PEP"]})
desc.to_csv(RES_DIR / "stats_descriptives.csv")
print("\nStatistiques descriptives des rendements (train) :\n", desc.round(6))
print("Corrélation des rendements KO/PEP :", round(train["ret_KO"].corr(train["ret_PEP"]), 3))

# %%
# ACF/PACF sur la série brute puis différenciée : l'ACF lente du niveau suggère une racine unitaire.
y = train["log_KO"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(y, lags=40, ax=ax[0], title="ACF log(KO) — série brute"); plot_pacf(y, lags=40, ax=ax[1], title="PACF log(KO) — série brute")
savefig(fig, "fig03_acf_pacf_niveau.png")
dy = y.diff().dropna()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(dy, lags=40, ax=ax[0], title="ACF Δlog(KO) — série différenciée"); plot_pacf(dy, lags=40, ax=ax[1], title="PACF Δlog(KO) — série différenciée")
savefig(fig, "fig04_acf_pacf_diff.png")

# %%
# Décomposition STL robuste, période 252 (≈ 1 an de bourse) : sépare tendance, saisonnalité annuelle, reste.
stl = STL(pd.Series(y.values), period=252, robust=True).fit()
fig = stl.plot(); fig.set_size_inches(11, 8); savefig(fig, "fig05_stl_KO.png")
force_tendance = max(0, 1 - np.var(stl.resid) / np.var(stl.trend + stl.resid))
force_saison = max(0, 1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid))
pd.Series({"force_tendance": force_tendance, "force_saisonnalite": force_saison}).to_csv(RES_DIR / "stl_forces.csv", header=["valeur"])
print(f"\nSTL : force de la tendance = {force_tendance:.3f} ; force de la saisonnalité = {force_saison:.3f} (0 = nulle, 1 = très forte)")

# %%
# Valeurs aberrantes : règle IQR (1.5) calibrée sur le train, appliquée à tout l'échantillon.
def iqr_flags(x, ref, k=1.5):
    q1, q3 = ref.quantile([0.25, 0.75]); i = q3 - q1
    return (x < q1 - k * i) | (x > q3 + k * i)

flags = pd.DataFrame({f"flag_{t}": iqr_flags(F[f"ret_{t}"], train[f"ret_{t}"]) for t in TICKERS})
flags.loc[flags.index[0]] = False  # le 1er rendement est NaN
out = pd.concat([F[["ret_KO", "ret_PEP"]], flags], axis=1)
out[out[["flag_KO", "flag_PEP"]].any(axis=1)].to_csv(RES_DIR / "outliers.csv")
print("\nValeurs aberrantes (IQR) :", flags.sum().to_dict())
print("5 plus gros |rendements| KO :\n", F["ret_KO"].abs().nlargest(5).round(4))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].boxplot([train["ret_KO"].dropna(), train["ret_PEP"].dropna()]); ax[0].set_xticklabels(TICKERS); ax[0].set_title("Boxplot des rendements (train)")
ax[1].plot(F["ret_KO"], lw=0.5); ax[1].scatter(F.index[flags["flag_KO"]], F["ret_KO"][flags["flag_KO"]], c="r", s=8, label="aberrant (IQR)")
ax[1].legend(); ax[1].set_title("Rendements KO et points aberrants")
savefig(fig, "fig06_outliers.png")
# Traitement : conserver les chocs de marché et enregistrer les flags pour l'audit.
# Les flags ne sont pas utilisées comme variables de modèle dans cette version.

# %%
# Stationnarité : ADF (H0 = racine unitaire) ET KPSS (H0 = stationnaire) -> hypothèses nulles opposées.
rows = []
for name, s in [("log_KO", train["log_KO"]), ("log_PEP", train["log_PEP"]), ("ret_KO", train["ret_KO"]), ("ret_PEP", train["ret_PEP"])]:
    s = s.dropna()
    adf_p = adfuller(s, regression="c", autolag="AIC")[1]
    kpss_p = kpss(s, regression="c", nlags="auto")[1]
    if adf_p < 0.05 and kpss_p >= 0.05:
        concl = "stationnaire"
    elif adf_p >= 0.05 and kpss_p < 0.05:
        concl = "NON stationnaire (racine unitaire)"
    else:
        concl = "ambigu (les deux tests divergent)"
    rows.append({"serie": name, "ADF_p": adf_p, "KPSS_p": kpss_p, "conclusion": concl})
stat = pd.DataFrame(rows)
stat.to_csv(RES_DIR / "stationnarite.csv", index=False)
print("\n", stat.round(4).to_string(index=False))
d_ordre = 1 if (stat.loc[stat.serie == "log_KO", "conclusion"].iloc[0].startswith("NON")
                and stat.loc[stat.serie == "ret_KO", "conclusion"].iloc[0] == "stationnaire") else 0
print(f"\n>>> Conclusion : ordre de différenciation d = {d_ordre} pour log(KO) "
      "(niveau non stationnaire, rendement stationnaire => d=1)." if d_ordre else
      "\n>>> Les tests ne concluent pas proprement à d=1 : à discuter dans le rapport (voir tableau).")
log_json("key_results.json", d_ordre=d_ordre, n_train=n_train, n_test=len(test),
         corr_rendements=float(train["ret_KO"].corr(train["ret_PEP"])), force_tendance=float(force_tendance),
         force_saisonnalite=float(force_saison), n_aberrants_KO=int(flags["flag_KO"].sum()))
