"""Fonctions et constantes partagées par tous les scripts (une seule source de vérité).

Pourquoi ce fichier : le split 90/10, la graine, les métriques et le test de Diebold-Mariano
doivent être IDENTIQUES pour tous les modèles, sinon la comparaison n'est pas équitable.
"""
import json
import os
import random
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas de fenêtre : les figures sont sauvegardées dans figures/
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------- paramètres globaux
SEED = 42                       # graine fixe -> résultats reproductibles
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
warnings.filterwarnings("ignore")  # on garde la sortie lisible (les tests sont affichés explicitement)

TICKERS = ["KO", "PEP"]         # Coca-Cola et PepsiCo : deux actifs corrélés (consigne 1.1)
START, END_EXCL = "2019-01-01", "2025-01-01"   # 2019-2024 (end exclusif dans yfinance)
H = 10                          # horizon de prévision (consigne : 5 à 10 pas)
TRAIN_RATIO = 0.9               # split chronologique 90 % / 10 % (consigne 1.3)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, FIG_DIR, RES_DIR, LLM_DIR = ROOT / "data", ROOT / "figures", ROOT / "results", ROOT / "llm"
for _d in (DATA_DIR, FIG_DIR, RES_DIR, LLM_DIR):
    _d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})


# ---------------------------------------------------------------- données
def load_raw(ticker):
    """Lit le CSV Yahoo complet (Open, High, Low, Close, Adj Close, Volume)."""
    return pd.read_csv(DATA_DIR / f"{ticker}.csv", index_col=0, parse_dates=True).sort_index()


def load_prices():
    """Prix ajustés (Adj Close) des deux actifs, alignés sur les mêmes dates.
    Adj Close = ajusté des dividendes et splits -> rendement total correct (à documenter)."""
    px = pd.DataFrame({t: load_raw(t)["Adj Close"] for t in TICKERS}).dropna()
    px.index = pd.DatetimeIndex(px.index)
    return px


def make_features(px):
    """Log-prix, rendements log, moyenne mobile 20 j et volatilité réalisée 21 j (annualisée)."""
    f = pd.DataFrame(index=px.index)
    for t in TICKERS:
        f[f"log_{t}"] = np.log(px[t])                               # log pour stabiliser la variance
        f[f"ret_{t}"] = f[f"log_{t}"].diff()                        # rendement logarithmique
        f[f"ma20_{t}"] = px[t].rolling(20).mean()                   # moyenne mobile (indicateur technique)
        f[f"vol21_{t}"] = f[f"ret_{t}"].rolling(21).std() * np.sqrt(252)  # volatilité réalisée
    return f


def origins(n, h=H, ratio=TRAIN_RATIO):
    """Origines de prévision : blocs de h jours consécutifs, non chevauchants, dans l'échantillon test.
    Une origine o signifie : on connaît les données jusqu'à la position o-1 et on prévoit o..o+h-1."""
    n_train = int(n * ratio)
    return n_train, [n_train + k * h for k in range(((n - n_train) // h))]


# ---------------------------------------------------------------- sorties
def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=130)
    plt.close(fig)
    print(f"  [figure] figures/{name}")


def save_forecasts(name, arr):
    """arr : (n_blocs, H) prévisions de PRIX de KO. Format commun lu par 06_evaluation.py."""
    pd.DataFrame(arr, columns=[f"h{i+1}" for i in range(arr.shape[1])]).to_csv(RES_DIR / f"fc_{name}.csv", index=False)


def log_json(fname, **kv):
    """Ajoute des clés à un fichier JSON de résultats (key_results.json, timings.json)."""
    p = RES_DIR / fname
    d = json.loads(p.read_text()) if p.exists() else {}
    d.update(kv)
    p.write_text(json.dumps(d, indent=2, default=float))


def read_json(fname):
    p = RES_DIR / fname
    return json.loads(p.read_text()) if p.exists() else {}


class Timer:
    """Mesure le temps de calcul d'un modèle (critère 'coût de calcul' de la section 6)."""
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.time()
    def __exit__(self, *a):
        log_json("timings.json", **{self.name: round(time.time() - self.t0, 2)})


# ---------------------------------------------------------------- métriques et tests
def metrics(y, yhat):
    e = np.asarray(y) - np.asarray(yhat)
    return {"MSE": np.mean(e ** 2), "RMSE": np.sqrt(np.mean(e ** 2)),
            "MAE": np.mean(np.abs(e)), "MAPE": np.mean(np.abs(e / np.asarray(y))) * 100}


def dm_test(e1, e2, h=H):
    """Test de Diebold-Mariano (perte quadratique) avec variance de Newey-West (poids de Bartlett,
    lag h-1) et correction de Harvey-Leybourne-Newbold. H0 : même précision.
    Statistique > 0 => le modèle 2 est meilleur que le modèle 1."""
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    T = len(d)
    dbar = d.mean()
    var = np.var(d)
    for k in range(1, h):
        var += 2 * (1 - k / h) * np.cov(d[k:], d[:-k], ddof=0)[0, 1]
    var /= T
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var) * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return dm, 2 * (1 - stats.t.cdf(abs(dm), df=T - 1))


def verdict(p, h0, alpha=0.05):
    """Phrase d'interprétation automatique d'un test (jamais afficher un p-value sans l'interpréter)."""
    return (f"p={p:.4f} < {alpha} -> on REJETTE H0 ({h0})" if p < alpha
            else f"p={p:.4f} >= {alpha} -> on NE rejette PAS H0 ({h0})")


def diag_tests(resid, model_df=0):
    """Diagnostic complet de résidus : Ljung-Box, Jarque-Bera, Shapiro-Wilk, ARCH-LM."""
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    r = np.asarray(resid, dtype=float)
    r = r[~np.isnan(r)]
    lb_lags = [max(10, model_df + 1), max(20, model_df + 2)]
    lb = acorr_ljungbox(r, lags=lb_lags, model_df=model_df, return_df=True)
    out = {"LB10_lag": lb_lags[0], "LB10_p": float(lb["lb_pvalue"].iloc[0]),
           "LB20_lag": lb_lags[1], "LB20_p": float(lb["lb_pvalue"].iloc[1]),
           "JB_p": float(stats.jarque_bera(r)[1]), "Shapiro_p": float(stats.shapiro(r)[1]),
           "ARCH10_p": float(het_arch(r, nlags=10)[1]),
           "skew": float(stats.skew(r)), "kurtosis_excess": float(stats.kurtosis(r))}
    return out


def print_diag(name, d):
    print(f"\n--- Diagnostic résidus : {name}")
    print(f"  Ljung-Box({d['LB10_lag']}) :", verdict(d["LB10_p"], "résidus indépendants"))
    print(f"  Ljung-Box({d['LB20_lag']}) :", verdict(d["LB20_p"], "résidus indépendants"))
    print("  Jarque-Bera   :", verdict(d["JB_p"], "résidus normaux"))
    print("  Shapiro-Wilk  :", verdict(d["Shapiro_p"], "résidus normaux"))
    print("  ARCH-LM(10)   :", verdict(d["ARCH10_p"], "variance conditionnelle constante"))


def plot_diag(resid, title, fname):
    """4 graphiques de diagnostic : résidus, ACF, histogramme, QQ-plot."""
    from statsmodels.graphics.tsaplots import plot_acf
    r = np.asarray(resid, dtype=float)
    r = r[~np.isnan(r)]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(r, lw=0.7); ax[0, 0].set_title("Résidus dans le temps")
    plot_acf(r, lags=30, ax=ax[0, 1], title="ACF des résidus")
    ax[1, 0].hist(r, bins=50, density=True, alpha=0.7)
    xs = np.linspace(r.min(), r.max(), 200)
    ax[1, 0].plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), "r"); ax[1, 0].set_title("Histogramme vs loi normale")
    stats.probplot(r, dist="norm", plot=ax[1, 1]); ax[1, 1].set_title("QQ-plot")
    fig.suptitle(title)
    savefig(fig, fname)


def print_versions():
    import importlib
    import platform
    print("Python", platform.python_version(), "| seed =", SEED)
    for m in ["numpy", "pandas", "scipy", "statsmodels", "arch", "pmdarima", "sklearn", "tensorflow", "yfinance"]:
        try:
            print(f"  {m:12s}", importlib.import_module(m).__version__)
        except Exception:
            print(f"  {m:12s} (non installé)")
