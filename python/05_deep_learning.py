# %% [markdown]
# # 05 — LSTM et GRU avec validation croisée temporelle (section 4.1)
# Implémenté en Python/Keras uniquement (socle avancé : un seul langage autorisé).
# Entrée : fenêtre glissante de L jours de [rendement KO, rendement PEP, volatilité 21 j KO].
# Sortie : les H=10 prochains rendements de KO (prévision directe multi-pas), recomposés en prix.

# %%
from common import *
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.utils.set_random_seed(SEED)   # graine fixée pour numpy, python et TensorFlow

F = make_features(load_prices())
n_train, origs = origins(len(F))
FEAT = ["ret_KO", "ret_PEP", "vol21_KO"]
Fv, ret, logKO = F[FEAT].values, F["ret_KO"].values, F["log_KO"].values
NF, I0 = len(FEAT), 21                       # I0 : 1re ligne sans NaN (la volatilité 21 j démarre à 21)

# %% [markdown]
# ## Fenêtres glissantes

# %%
def make_windows(L, ends):
    """X : L jours jusqu'à t inclus ; Y : les H rendements suivants. Construction 100 % causale."""
    X = np.stack([Fv[t - L + 1:t + 1] for t in ends])
    Y = np.stack([ret[t + 1:t + 1 + H] for t in ends])
    return X, Y

def train_ends(L):
    return list(range(I0 + L - 1, n_train - 1 - H + 1))   # la cible reste dans le train (aucune fuite vers le test)

def build(kind, L, dropout):
    keras.backend.clear_session()
    cell = layers.LSTM if kind == "LSTM" else layers.GRU
    m = keras.Sequential([layers.Input((L, NF)), cell(32, return_sequences=True), layers.Dropout(dropout),
                          cell(16), layers.Dropout(dropout), layers.Dense(H)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m

def fit(kind, L, dropout, Xtr, Ytr, Xva, Yva, epochs=40):
    """Les échelles sont ajustées sur le train du pli ; l'arrêt précoce utilise sa validation chronologique."""
    sc = StandardScaler().fit(Xtr.reshape(-1, NF))
    target_scale = float(np.std(Ytr))
    if not np.isfinite(target_scale) or target_scale == 0:
        raise ValueError("L'échelle des rendements cibles doit être finie et strictement positive.")
    f = lambda X: sc.transform(X.reshape(-1, NF)).reshape(X.shape)
    m = build(kind, L, dropout)
    h = m.fit(f(Xtr), Ytr / target_scale, validation_data=(f(Xva), Yva / target_scale),
              epochs=epochs, batch_size=32, verbose=0, shuffle=False,
              callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    return m, sc, target_scale, h

def split_val(n, frac=0.2):
    """Validation = fin de la période d'apprentissage, séparée par un embargo de H jours (cibles chevauchantes)."""
    cut = int(n * (1 - frac))
    return slice(0, cut - H), slice(cut, n)

# %% [markdown]
# ## Choix des hyperparamètres (grille réduite, sélection sur la perte de validation)

# %%
# Petit réseau (2 couches, 32/16 unités) car ~1300 fenêtres seulement : un gros réseau surapprendrait.
# On compare 2 tailles de fenêtre (10 j ≈ 2 semaines, 30 j ≈ 1,5 mois) et 2 niveaux de dropout.
rows, models, hists = [], {}, {}
for kind in ["LSTM", "GRU"]:
    for L in [10, 30]:
        for dr in [0.1, 0.3]:
            X, Y = make_windows(L, train_ends(L))
            a, b = split_val(len(X))
            with Timer(f"{kind}_L{L}_d{dr}"):
                m, sc, target_scale, h = fit(kind, L, dr, X[a], Y[a], X[b], Y[b])
            val_mse = min(h.history["val_loss"]) * target_scale**2
            best_epoch = int(np.argmin(h.history["val_loss"])) + 1
            rows.append({"modele": kind, "fenetre": L, "dropout": dr, "val_mse": val_mse, "epochs": best_epoch})
            hists[(kind, L, dr)] = h
            print(f"  {kind} L={L} dropout={dr} -> MSE validation={val_mse:.8f}")
hp = pd.DataFrame(rows)
hp.to_csv(RES_DIR / "dl_hyperparams.csv", index=False)
best_cfg = {k: tuple(hp[hp.modele == k].sort_values("val_mse").iloc[0][["fenetre", "dropout"]]) for k in ["LSTM", "GRU"]}
best_cfg = {k: (int(v[0]), float(v[1])) for k, v in best_cfg.items()}
print("\nMeilleures configurations :", best_cfg)

# Courbes de perte train/validation du modèle retenu (consigne : documenter l'entraînement).
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a_, kind in zip(ax, ["LSTM", "GRU"]):
    L, dr = best_cfg[kind]; h = hists[(kind, L, dr)]
    a_.plot(h.history["loss"], label="train"); a_.plot(h.history["val_loss"], label="validation")
    a_.set_title(f"{kind} (fenêtre {L}, dropout {dr}) — perte MSE"); a_.set_xlabel("epoch"); a_.legend()
savefig(fig, "fig15_dl_pertes.png")

# Refit final models with every training observation; epoch counts come only from the validation split.
for kind in ["LSTM", "GRU"]:
    L, dr = best_cfg[kind]
    X, Y = make_windows(L, train_ends(L))
    sc = StandardScaler().fit(X.reshape(-1, NF))
    target_scale = float(np.std(Y))
    m = build(kind, L, dr)
    X_scaled = sc.transform(X.reshape(-1, NF)).reshape(X.shape)
    epochs = int(np.argmin(hists[(kind, L, dr)].history["val_loss"])) + 1
    m.fit(X_scaled, Y / target_scale, epochs=epochs, batch_size=32, shuffle=False, verbose=0)
    models[(kind, L, dr)] = (m, sc, target_scale)

# %% [markdown]
# ## Validation croisée temporelle (walk-forward, fenêtre extensible) — jamais de k-fold classique

# %%
# Chaque pli choisit ses hyperparamètres/époques uniquement dans son passé, avec un embargo de H jours.
# Métrique : RMSE sur le rendement cumulé à 10 jours, comparée à la baseline 'rendement nul' (= marche aléatoire).
cv_rows = []
base_origins = np.asarray(train_ends(10))
X_base, Y_base = make_windows(10, base_origins)
for kind in ["LSTM", "GRU"]:
    for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits=5, gap=H).split(X_base), 1):
        last_train_origin = base_origins[tr[-1]]
        test_origins = base_origins[te]
        candidates = []
        for candidate_L in [10, 30]:
            candidate_origins = np.asarray(train_ends(candidate_L))
            X_candidate, Y_candidate = make_windows(candidate_L, candidate_origins)
            eligible = np.flatnonzero(candidate_origins <= last_train_origin)
            X_fold_train, Y_fold_train = X_candidate[eligible], Y_candidate[eligible]
            for candidate_dropout in [0.1, 0.3]:
                a, b = split_val(len(X_fold_train), 0.2)
                _, _, _, hist_candidate = fit(
                    kind, candidate_L, candidate_dropout,
                    X_fold_train[a], Y_fold_train[a], X_fold_train[b], Y_fold_train[b], epochs=40)
                best_epoch = int(np.argmin(hist_candidate.history["val_loss"])) + 1
                val_mse = min(hist_candidate.history["val_loss"]) * float(np.std(Y_fold_train[a]))**2
                candidates.append((val_mse, candidate_L,
                                   candidate_dropout, best_epoch))
        _, fold_L, fold_dropout, fold_epochs = min(candidates, key=lambda item: item[0])
        fold_origins = np.asarray(train_ends(fold_L))
        X_fold, Y_fold = make_windows(fold_L, fold_origins)
        fold_train = np.flatnonzero(fold_origins <= last_train_origin)
        fold_test = np.searchsorted(fold_origins, test_origins)
        if not np.array_equal(fold_origins[fold_test], test_origins):
            raise RuntimeError("Walk-forward origins differ between the tuning and test windows.")
        X_train, Y_train = X_fold[fold_train], Y_fold[fold_train]
        fold_scaler = StandardScaler().fit(X_train.reshape(-1, NF))
        fold_scale = float(np.std(Y_train))
        if not np.isfinite(fold_scale) or fold_scale == 0:
            raise ValueError("L'échelle d'un pli doit être finie et strictement positive.")
        X_train_scaled = fold_scaler.transform(X_train.reshape(-1, NF)).reshape(X_train.shape)
        model = build(kind, fold_L, fold_dropout)
        model.fit(X_train_scaled, Y_train / fold_scale, epochs=fold_epochs, batch_size=32,
                  shuffle=False, verbose=0)
        X_test = X_fold[fold_test]
        Y_test = Y_fold[fold_test]
        X_test_scaled = fold_scaler.transform(X_test.reshape(-1, NF)).reshape(X_test.shape)
        pred = model.predict(X_test_scaled, verbose=0) * fold_scale
        rmse = np.sqrt(np.mean((pred.sum(1) - Y_test.sum(1)) ** 2))
        rmse_naif = np.sqrt(np.mean(Y_test.sum(1) ** 2))
        cv_rows.append({"modele": kind, "pli": k, "fenetre": fold_L, "dropout": fold_dropout,
                        "RMSE_cum10j": rmse, "RMSE_naif": rmse_naif})
        print(f"  {kind} pli {k} : RMSE={rmse:.4f} | naïf={rmse_naif:.4f}")
cv = pd.DataFrame(cv_rows)
cv.to_csv(RES_DIR / "dl_cv_walkforward.csv", index=False)
print("\nMoyenne par modèle :\n", cv.groupby("modele")[["RMSE_cum10j", "RMSE_naif"]].agg(["mean", "std"]).round(4))
fig, ax = plt.subplots(figsize=(9, 4))
w = 0.25
for i, kind in enumerate(["LSTM", "GRU"]):
    s = cv[cv.modele == kind]; ax.bar(s.pli + (i - 0.5) * w, s.RMSE_cum10j, w, label=kind)
ax.plot(cv[cv.modele == "LSTM"].pli, cv[cv.modele == "LSTM"].RMSE_naif, "k--o", label="baseline naïve")
ax.set_xlabel("pli (walk-forward)"); ax.set_ylabel("RMSE rendement cumulé 10 j"); ax.legend(); ax.set_title("Validation croisée temporelle")
savefig(fig, "fig16_dl_cv_walkforward.png")
log_json("key_results.json", dl_best_cfg=str(best_cfg), dl_cv_mean=cv.groupby("modele")["RMSE_cum10j"].mean().to_dict(),
         dl_cv_naive_mean=float(cv["RMSE_naif"].mean()))

# %% [markdown]
# ## Prévisions h=10 sur les mêmes blocs de test que les modèles économétriques

# %%
# Les réseaux sont entraînés UNE fois (train) ; les modèles économétriques sont ré-estimés à chaque bloc.
# Cette asymétrie (coût de calcul) est signalée dans les limites du rapport.
for kind in ["LSTM", "GRU"]:
    L, dr = best_cfg[kind]
    m, sc, target_scale = models[(kind, L, dr)]
    X = np.stack([Fv[o - L:o] for o in origs])                      # fenêtre qui finit en o-1 (dernier jour connu)
    pred = m.predict(sc.transform(X.reshape(-1, NF)).reshape(X.shape), verbose=0) * target_scale
    save_forecasts(kind, np.exp(logKO[np.array(origs) - 1][:, None] + np.cumsum(pred, axis=1)))
    print("  prévisions OK :", kind)
