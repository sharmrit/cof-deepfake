"""
Compute Table: Forensic Utility Metrics for tab:forensic_utility.
Metrics: Pearson rho, Spearman rho, AUROC (fused score as binary error predictor)
Methods: COF, RF (cross-domain trained), SC-Weight
"""
import numpy as np
import json
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from cof_uq.config import Config, ARCHITECTURES, ARCH_SHORT_NAMES

def safe_pearson(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12: return 0.0
    return pearsonr(x, y)[0]

def safe_spearman(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12: return 0.0
    return spearmanr(x, y)[0]

def safe_auroc(scores, errors, threshold=None):
    if threshold is None:
        threshold = np.mean(errors)
    labels = (errors > threshold).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels): return float('nan')
    return roc_auc_score(labels, scores)

def cof_fusion(U_train, e_train, U_test, n_restarts=20):
    K = U_train.shape[1]
    best_w, best_corr = None, -np.inf
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0, 1)] * K
    for _ in range(n_restarts):
        w0 = np.random.dirichlet(np.ones(K))
        res = minimize(lambda w: -pearsonr(U_train @ w, e_train)[0],
                      w0, method='SLSQP', bounds=bounds, constraints=constraints)
        if res.success and -res.fun > best_corr:
            best_corr = -res.fun
            best_w = res.x
    if best_w is None:
        best_w = np.ones(K) / K
    return U_test @ best_w

def sc_weight_fusion(U_train, e_train, U_test):
    corrs = np.array([safe_pearson(U_train[:, k], e_train) for k in range(U_train.shape[1])])
    w = corrs ** 2
    w = np.maximum(w, 0)
    if w.sum() < 1e-12: w = np.ones(len(w)) / len(w)
    else: w /= w.sum()
    return U_test @ w

def rf_fusion(U_train, e_train, U_test):
    scaler = StandardScaler()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(scaler.fit_transform(U_train), e_train)
    return rf.predict(scaler.transform(U_test))

def main():
    config = Config.from_yaml("configs/tifs.yaml")
    base = Path(config.output_dir) / "uncertainties"

    print("=" * 85)
    print("TABLE: Forensic Utility Metrics (In-domain FF++ and Cross-domain CelebDF)")
    print("=" * 85)

    results = {}
    for domain, label in [("faceforensics", "FF++"), ("celebdf", "CelebDF")]:
        print(f"\n--- {label} ---")
        print(f"{'Arch':<14} {'COF-P':>7} {'COF-S':>7} {'COF-AUC':>8} "
              f"{'RF-P':>7} {'RF-S':>7} {'RF-AUC':>8} "
              f"{'SC-P':>7} {'SC-S':>7} {'SC-AUC':>8}")
        print("-" * 85)

        domain_results = {}
        for arch in ARCHITECTURES:
            src_fp = base / arch / f"faceforensics_seed42.npz"
            tgt_fp = base / arch / f"{domain}_seed42.npz"
            if not src_fp.exists() or not tgt_fp.exists():
                continue
            src = np.load(src_fp)
            tgt = np.load(tgt_fp)
            U_train = src["uncertainties"][:, :5]
            e_train = src["errors"]
            U_test = tgt["uncertainties"][:, :5]
            e_test = tgt["errors"]

            cof_scores = cof_fusion(U_train, e_train, U_test)
            rf_scores = rf_fusion(U_train, e_train, U_test)
            sc_scores = sc_weight_fusion(U_train, e_train, U_test)

            row = {
                "cof": {
                    "pearson": safe_pearson(cof_scores, e_test),
                    "spearman": safe_spearman(cof_scores, e_test),
                    "auroc": safe_auroc(cof_scores, e_test),
                },
                "rf": {
                    "pearson": safe_pearson(rf_scores, e_test),
                    "spearman": safe_spearman(rf_scores, e_test),
                    "auroc": safe_auroc(rf_scores, e_test),
                },
                "sc": {
                    "pearson": safe_pearson(sc_scores, e_test),
                    "spearman": safe_spearman(sc_scores, e_test),
                    "auroc": safe_auroc(sc_scores, e_test),
                },
            }
            name = ARCH_SHORT_NAMES.get(arch, arch)
            print(f"{name:<14} "
                  f"{row['cof']['pearson']:>7.4f} {row['cof']['spearman']:>7.4f} {row['cof']['auroc']:>8.4f} "
                  f"{row['rf']['pearson']:>7.4f} {row['rf']['spearman']:>7.4f} {row['rf']['auroc']:>8.4f} "
                  f"{row['sc']['pearson']:>7.4f} {row['sc']['spearman']:>7.4f} {row['sc']['auroc']:>8.4f}")
            domain_results[arch] = row

        results[domain] = domain_results

    # Save
    save_path = Path(config.output_dir) / "baselines" / "forensic_utility.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {save_path}")

if __name__ == "__main__":
    main()
