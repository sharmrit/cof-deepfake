"""
All 12 fusion methods from the TIFS paper.

Proposed (6 correlation-based):
  1. COF         — Direct correlation maximization
  2. L1-COF      — COF + L1 sparsity + dual-split correlation
  3. Meta-Ens.   — Stacked generalization over 3 base methods
  4. 2M-Ens.     — Average of Logistic + COF
  5. Hier-Fus.   — Theory-guided group fusion
  6. SC-Weight   — Squared-correlation weighting (zero training)

Baselines (6):
  7. Logistic     — Logistic regression on sources
  8. Ridge        — Ridge regression on sources
  9. Random Forest
  10. Averaging   — Uniform weights
  11. Neural      — Small MLP
  12. Max         — Maximum uncertainty
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, List, Tuple


def _safe_corr(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0, 1.0
    return pearsonr(x, y)


# =========================================================================
# 1. COF — Direct Correlation Maximization
# =========================================================================

def cof_fusion(U_train, e_train, U_test, e_test, n_restarts=20, **kw):
    K = U_train.shape[1]

    def neg_corr(w):
        f = U_train @ w
        if np.std(f) < 1e-12:
            return 1.0
        return -pearsonr(f, e_train)[0]

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0, 1)] * K
    best_w, best_val = None, np.inf

    for i in range(n_restarts):
        w0 = np.ones(K) / K if i == 0 else np.random.dirichlet(np.ones(K))
        try:
            res = minimize(neg_corr, w0, method="SLSQP", bounds=bounds,
                          constraints=constraints, options={"maxiter": 1000, "ftol": 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_w = res.x
        except Exception:
            continue

    if best_w is None:
        best_w = np.ones(K) / K

    fused_test = U_test @ best_w
    corr, pval = _safe_corr(fused_test, e_test)
    return {"name": "COF", "correlation": corr, "p_value": pval,
            "weights": best_w, "fused": fused_test}


# =========================================================================
# 2. L1-COF — Sparsity + Dual-Split Correlation (Eq. 9)
# =========================================================================

def l1_cof_fusion(U_train, e_train, U_test, e_test, lam=0.1,
                  n_restarts=20, **kw):
    K = U_train.shape[1]
    # Split train into two halves for dual correlation
    n = len(e_train)
    half = n // 2
    U1, e1 = U_train[:half], e_train[:half]
    U2, e2 = U_train[half:], e_train[half:]

    def objective(w):
        f1 = U1 @ w
        f2 = U2 @ w
        c1 = pearsonr(f1, e1)[0] if np.std(f1) > 1e-12 else 0
        c2 = pearsonr(f2, e2)[0] if np.std(f2) > 1e-12 else 0
        return -(0.5 * c1 + 0.5 * c2) + lam * np.sum(np.abs(w))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0, 1)] * K
    best_w, best_val = None, np.inf

    for i in range(n_restarts):
        w0 = np.ones(K) / K if i == 0 else np.random.dirichlet(np.ones(K))
        try:
            res = minimize(objective, w0, method="SLSQP", bounds=bounds,
                          constraints=constraints, options={"maxiter": 1000})
            if res.fun < best_val:
                best_val = res.fun
                best_w = res.x
        except Exception:
            continue

    if best_w is None:
        best_w = np.ones(K) / K

    fused_test = U_test @ best_w
    corr, pval = _safe_corr(fused_test, e_test)
    return {"name": "L1-COF", "correlation": corr, "p_value": pval,
            "weights": best_w, "fused": fused_test}


# =========================================================================
# 3. Meta-Ensemble — Stacked Generalization (Eq. 10)
# =========================================================================

def meta_ensemble_fusion(U_train, e_train, U_test, e_test, n_restarts=20, **kw):
    # Three base methods: Logistic, Ridge, COF
    base_preds_train = []
    base_preds_test = []
    base_corrs = []

    # Logistic
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(U_train, (e_train > 0).astype(int))
    pred_train_lr = lr.predict_proba(U_train)[:, 1]
    pred_test_lr = lr.predict_proba(U_test)[:, 1]
    c_lr = _safe_corr(pred_train_lr, e_train)[0]
    base_preds_train.append(pred_train_lr)
    base_preds_test.append(pred_test_lr)
    base_corrs.append(c_lr)

    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(U_train, e_train)
    pred_train_ridge = ridge.predict(U_train)
    pred_test_ridge = ridge.predict(U_test)
    c_ridge = _safe_corr(pred_train_ridge, e_train)[0]
    base_preds_train.append(pred_train_ridge)
    base_preds_test.append(pred_test_ridge)
    base_corrs.append(c_ridge)

    # COF
    cof_res = cof_fusion(U_train, e_train, U_train, e_train, n_restarts=n_restarts)
    pred_train_cof = cof_res["fused"]
    pred_test_cof = U_test @ cof_res["weights"]
    c_cof = _safe_corr(pred_train_cof, e_train)[0]
    base_preds_train.append(pred_train_cof)
    base_preds_test.append(pred_test_cof)
    base_corrs.append(c_cof)

    # Squared-correlation meta-weights (Eq. 10)
    sq_corrs = np.array([max(c, 0) ** 2 for c in base_corrs])
    if sq_corrs.sum() > 1e-12:
        meta_w = sq_corrs / sq_corrs.sum()
    else:
        meta_w = np.ones(3) / 3

    fused_test = sum(w * p for w, p in zip(meta_w, base_preds_test))
    corr, pval = _safe_corr(fused_test, e_test)
    return {"name": "Meta-Ens.", "correlation": corr, "p_value": pval,
            "meta_weights": meta_w.tolist(), "fused": fused_test}


# =========================================================================
# 4. Two-Method Ensemble (2M-Ens.)
# =========================================================================

def two_method_ensemble(U_train, e_train, U_test, e_test, n_restarts=20, **kw):
    # Logistic
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(U_train, (e_train > 0).astype(int))
    pred_test_lr = lr.predict_proba(U_test)[:, 1]

    # COF
    cof_res = cof_fusion(U_train, e_train, U_train, e_train, n_restarts=n_restarts)
    pred_test_cof = U_test @ cof_res["weights"]

    fused_test = 0.5 * pred_test_lr + 0.5 * pred_test_cof
    corr, pval = _safe_corr(fused_test, e_test)
    return {"name": "2M-Ens.", "correlation": corr, "p_value": pval,
            "fused": fused_test}


# =========================================================================
# 5. Hierarchical Fusion (Hier-Fus.)
# =========================================================================

def hierarchical_fusion(U_train, e_train, U_test, e_test, n_restarts=20, **kw):
    K = U_train.shape[1]

    # Groups: Bayesian {epistemic, aleatoric}, Prediction {calibration, conformal}, Distribution {distributional}
    groups = [[0, 1], [2, 3], [4]] if K >= 5 else [[0, 1], [2, 3]]

    # Intra-group: equal weights -> group scores
    group_train = []
    group_test = []
    for g in groups:
        gt = U_train[:, g].mean(axis=1)
        gs = U_test[:, g].mean(axis=1)
        group_train.append(gt)
        group_test.append(gs)

    G_train = np.column_stack(group_train)
    G_test = np.column_stack(group_test)
    n_groups = len(groups)

    # Inter-group: optimize weights
    def neg_corr(w):
        f = G_train @ w
        if np.std(f) < 1e-12:
            return 1.0
        return -pearsonr(f, e_train)[0]

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0, 1)] * n_groups
    best_w, best_val = None, np.inf

    for i in range(n_restarts):
        w0 = np.ones(n_groups) / n_groups if i == 0 else np.random.dirichlet(np.ones(n_groups))
        try:
            res = minimize(neg_corr, w0, method="SLSQP", bounds=bounds,
                          constraints=constraints, options={"maxiter": 500})
            if res.fun < best_val:
                best_val = res.fun
                best_w = res.x
        except Exception:
            continue

    if best_w is None:
        best_w = np.ones(n_groups) / n_groups

    fused_test = G_test @ best_w
    corr, pval = _safe_corr(fused_test, e_test)
    return {"name": "Hier-Fus.", "correlation": corr, "p_value": pval,
            "group_weights": best_w.tolist(), "fused": fused_test}


# =========================================================================
# 6. SC-Weight — Squared Correlation Weighting (Eq. 11, zero training)
# =========================================================================

def sc_weight_fusion(U_train, e_train, U_test, e_test, **kw):
    K = U_train.shape[1]
    corrs = []
    for i in range(K):
        c, _ = _safe_corr(U_train[:, i], e_train)
        corrs.append(max(c, 0))

    sq = np.array(corrs) ** 2
    if sq.sum() > 1e-12:
        weights = sq / sq.sum()
    else:
        weights = np.ones(K) / K

    fused_test = U_test @ weights
    corr, pval = _safe_corr(fused_test, e_test)
    return {"name": "SC-Weight", "correlation": corr, "p_value": pval,
            "weights": weights, "fused": fused_test}


# =========================================================================
# 7. Logistic Regression
# =========================================================================

def logistic_fusion(U_train, e_train, U_test, e_test, **kw):
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(U_train, (e_train > 0).astype(int))
    fused_test = lr.predict_proba(U_test)[:, 1]
    corr, pval = _safe_corr(fused_test, e_test)
    return {"name": "Logistic", "correlation": corr, "p_value": pval,
            "fused": fused_test}


# =========================================================================
# 8. Ridge Regression
# =========================================================================

def ridge_fusion(U_train, e_train, U_test, e_test, **kw):
    ridge = Ridge(alpha=1.0)
    ridge.fit(U_train, e_train)
    fused_test = ridge.predict(U_test)
    corr, pval = _safe_corr(fused_test, e_test)
    return {"name": "Ridge", "correlation": corr, "p_value": pval,
            "fused": fused_test}


# =========================================================================
# 9. Random Forest
# =========================================================================

def random_forest_fusion(U_train, e_train, U_test, e_test, **kw):
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(U_train, e_train)
    fused_test = rf.predict(U_test)
    corr, pval = _safe_corr(fused_test, e_test)
    return {"name": "Random Forest", "correlation": corr, "p_value": pval,
            "fused": fused_test}


# =========================================================================
# 10. Uniform Averaging
# =========================================================================

def averaging_fusion(U_train, e_train, U_test, e_test, **kw):
    fused_test = U_test.mean(axis=1)
    corr, pval = _safe_corr(fused_test, e_test)
    return {"name": "Averaging", "correlation": corr, "p_value": pval,
            "fused": fused_test}


# =========================================================================
# 11. Neural (MLP)
# =========================================================================

def neural_fusion(U_train, e_train, U_test, e_test, **kw):
    scaler = StandardScaler()
    U_tr_s = scaler.fit_transform(U_train)
    U_te_s = scaler.transform(U_test)

    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500,
                       random_state=42, early_stopping=True,
                       validation_fraction=0.2)
    mlp.fit(U_tr_s, e_train)
    fused_test = mlp.predict(U_te_s)
    corr, pval = _safe_corr(fused_test, e_test)
    return {"name": "Neural", "correlation": corr, "p_value": pval,
            "fused": fused_test}


# =========================================================================
# 12. Maximum Uncertainty
# =========================================================================

def max_fusion(U_train, e_train, U_test, e_test, **kw):
    fused_test = U_test.max(axis=1)
    corr, pval = _safe_corr(fused_test, e_test)
    return {"name": "Max", "correlation": corr, "p_value": pval,
            "fused": fused_test}


# =========================================================================
# Registry: Run all 12 methods
# =========================================================================

ALL_METHODS = [
    ("Meta-Ens.", meta_ensemble_fusion),
    ("2M-Ens.", two_method_ensemble),
    ("COF", cof_fusion),
    ("L1-COF", l1_cof_fusion),
    ("Logistic", logistic_fusion),
    ("SC-Weight", sc_weight_fusion),
    ("Ridge", ridge_fusion),
    ("Hier-Fus.", hierarchical_fusion),
    ("Random Forest", random_forest_fusion),
    ("Averaging", averaging_fusion),
    ("Neural", neural_fusion),
    ("Max", max_fusion),
]


def run_all_12_methods(U_train, e_train, U_test, e_test, n_restarts=20):
    """
    Run all 12 fusion methods from the paper.

    Parameters
    ----------
    U_train : ndarray (N_train, K)
    e_train : ndarray (N_train,)
    U_test : ndarray (N_test, K)
    e_test : ndarray (N_test,)

    Returns
    -------
    results : list of dicts, sorted by correlation descending
    """
    results = []
    for name, func in ALL_METHODS:
        try:
            res = func(U_train, e_train, U_test, e_test, n_restarts=n_restarts)
            results.append(res)
        except Exception as e:
            print("  WARNING: {} failed: {}".format(name, e))
            results.append({"name": name, "correlation": 0.0, "p_value": 1.0})

    results.sort(key=lambda x: -x.get("correlation", 0))
    return results
