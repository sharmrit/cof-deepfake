"""
Hessian Eigenvalue Analysis for COF Optimization Landscape.

Computes the Hessian of the COF objective at the optimum to analyze:
  - Sharpness of the loss landscape (large eigenvalues = sharp)
  - Flatness / robustness (small eigenvalues = flat)
  - Ill-conditioning (ratio of max/min eigenvalue)
  - Mode connectivity and distributional paradox geometry

TIFS Extension: Second-order optimization analysis.
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import approx_fprime
from typing import Dict, Optional, List, Tuple

from ..fusion.cof import CorrelationOptimizedFusion
from ..config import UNCERTAINTY_SOURCES


class HessianAnalyzer:
    """
    Hessian analysis of the COF objective function.

    Computes eigenvalues and eigenvectors of the Hessian matrix
    ∇²L(w*) at the optimized weights w*.

    Parameters
    ----------
    uncertainties : ndarray (N, K)
    errors : ndarray (N,)
    source_names : list of str
    """

    def __init__(
        self,
        uncertainties: np.ndarray,
        errors: np.ndarray,
        source_names: Optional[List[str]] = None,
    ):
        self.U = uncertainties
        self.errors = errors
        self.K = uncertainties.shape[1]
        self.source_names = source_names or list(UNCERTAINTY_SOURCES[: self.K])

    def _objective(self, w: np.ndarray) -> float:
        """Negative Pearson correlation (COF objective)."""
        fused = self.U @ w
        if np.std(fused) < 1e-12:
            return 1.0
        corr, _ = pearsonr(fused, self.errors)
        return -corr

    def _gradient(self, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Numerical gradient via central differences."""
        return approx_fprime(w, self._objective, eps)

    def compute_hessian(
        self, w: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:
        """
        Compute Hessian matrix via finite differences.

        H[i,j] = (∂²L) / (∂wᵢ ∂wⱼ)
        """
        K = len(w)
        H = np.zeros((K, K))

        for i in range(K):
            e_i = np.zeros(K)
            e_i[i] = eps
            for j in range(i, K):
                e_j = np.zeros(K)
                e_j[j] = eps

                # Central difference: (f(w+ei+ej) - f(w+ei-ej) - f(w-ei+ej) + f(w-ei-ej)) / (4*eps^2)
                fpp = self._objective(w + e_i + e_j)
                fpm = self._objective(w + e_i - e_j)
                fmp = self._objective(w - e_i + e_j)
                fmm = self._objective(w - e_i - e_j)

                H[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
                H[j, i] = H[i, j]

        return H

    def compute_analytical_hessian(self, w: np.ndarray) -> np.ndarray:
        """
        Compute Hessian analytically using Pearson correlation derivatives.

        For ρ(Uw, e), the Hessian involves second derivatives of the
        correlation with respect to weights.
        """
        N = self.U.shape[0]
        fused = self.U @ w
        e = self.errors

        # Pearson correlation components
        f_bar = fused.mean()
        e_bar = e.mean()
        f_centered = fused - f_bar
        e_centered = e - e_bar

        cov_fe = np.mean(f_centered * e_centered)
        var_f = np.mean(f_centered ** 2)
        var_e = np.mean(e_centered ** 2)
        std_f = np.sqrt(var_f) if var_f > 1e-12 else 1e-6
        std_e = np.sqrt(var_e) if var_e > 1e-12 else 1e-6

        # ρ = cov_fe / (std_f * std_e)
        rho = cov_fe / (std_f * std_e)

        K = len(w)
        H = np.zeros((K, K))

        # U_centered = U - U.mean(axis=0)
        U_c = self.U - self.U.mean(axis=0)

        for i in range(K):
            for j in range(i, K):
                # ∂²ρ/∂wᵢ∂wⱼ involves terms from product and chain rule
                # d(cov)/dwi = (1/N) * Σ U_ci * e_c
                # d²(cov)/dwi dwj = 0 (linear in w)

                # d(var_f)/dwi = (2/N) * Σ f_c * U_ci
                # d²(var_f)/dwi dwj = (2/N) * Σ U_ci * U_cj

                d2_var = 2 * np.mean(U_c[:, i] * U_c[:, j])
                d_var_i = 2 * np.mean(f_centered * U_c[:, i])
                d_var_j = 2 * np.mean(f_centered * U_c[:, j])
                d_cov_i = np.mean(U_c[:, i] * e_centered)
                d_cov_j = np.mean(U_c[:, j] * e_centered)

                # Quotient rule for ρ = cov / (std_f * std_e)
                # This is complex; use the full second derivative
                denom = std_f * std_e
                term1 = 0  # d²cov/dwi dwj = 0
                term2 = -(d_cov_i * d_var_j + d_cov_j * d_var_i) / (
                    2 * std_f ** 3 * std_e
                )
                term3 = (
                    -cov_fe * d2_var / (2 * std_f ** 3 * std_e)
                )
                term4 = (
                    3 * cov_fe * d_var_i * d_var_j / (4 * std_f ** 5 * std_e)
                )

                H[i, j] = -(term1 + term2 + term3 + term4)  # negative for -ρ
                H[j, i] = H[i, j]

        return H

    def analyze(
        self,
        weights: Optional[np.ndarray] = None,
        use_analytical: bool = True,
        n_restarts: int = 10,
    ) -> Dict:
        """
        Full Hessian analysis at the COF optimum.

        Parameters
        ----------
        weights : ndarray, optional
            Optimized weights. If None, runs COF optimization first.
        use_analytical : bool
            Use analytical Hessian (faster, more accurate) or numerical.

        Returns
        -------
        analysis : dict with eigenvalues, condition number, sharpness, etc.
        """
        if weights is None:
            cof = CorrelationOptimizedFusion(
                k_sources=self.K, n_restarts=n_restarts
            )
            cof.fit(self.U, self.errors, source_names=self.source_names)
            weights = cof.weights_
            opt_corr = cof.result_.correlation
        else:
            fused = self.U @ weights
            opt_corr, _ = pearsonr(fused, self.errors) if np.std(fused) > 1e-12 else (0.0, 1.0)

        # Compute Hessian
        if use_analytical:
            H = self.compute_analytical_hessian(weights)
        else:
            H = self.compute_hessian(weights)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        sorted_idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Landscape metrics
        max_eigenvalue = float(np.max(np.abs(eigenvalues)))
        min_eigenvalue = float(np.min(np.abs(eigenvalues[np.abs(eigenvalues) > 1e-10]))) if np.any(np.abs(eigenvalues) > 1e-10) else 1e-10
        condition_number = max_eigenvalue / min_eigenvalue if min_eigenvalue > 1e-12 else np.inf
        trace = float(np.sum(eigenvalues))
        determinant = float(np.prod(eigenvalues))

        # Sharpness: trace of Hessian (sum of eigenvalues)
        sharpness = float(np.sum(np.abs(eigenvalues)))

        # Flatness ratio: fraction of eigenvalues below threshold
        flatness_ratio = float(np.mean(np.abs(eigenvalues) < 0.01))

        # Positive definiteness check (should be positive at minimum)
        n_positive = int(np.sum(eigenvalues > 0))
        n_negative = int(np.sum(eigenvalues < 0))
        is_minimum = n_negative == 0

        return {
            "weights": weights.tolist(),
            "correlation_at_optimum": opt_corr,
            "eigenvalues": eigenvalues.tolist(),
            "eigenvectors": eigenvectors.tolist(),
            "max_eigenvalue": max_eigenvalue,
            "min_eigenvalue": min_eigenvalue,
            "condition_number": condition_number,
            "trace": trace,
            "determinant": determinant,
            "sharpness": sharpness,
            "flatness_ratio": flatness_ratio,
            "n_positive_eigenvalues": n_positive,
            "n_negative_eigenvalues": n_negative,
            "is_local_minimum": is_minimum,
            "source_names": self.source_names,
        }

    def compare_landscapes(
        self,
        weights_k4: np.ndarray,
        weights_k5: np.ndarray,
        U_k4: np.ndarray,
        U_k5: np.ndarray,
    ) -> Dict:
        """
        Compare optimization landscapes with/without distributional source.
        Supports distributional paradox analysis.
        """
        analyzer_k4 = HessianAnalyzer(U_k4, self.errors, self.source_names[:4])
        analyzer_k5 = HessianAnalyzer(U_k5, self.errors, self.source_names[:5])

        h4 = analyzer_k4.analyze(weights_k4)
        h5 = analyzer_k5.analyze(weights_k5)

        return {
            "k4_landscape": h4,
            "k5_landscape": h5,
            "sharpness_change": h5["sharpness"] - h4["sharpness"],
            "condition_number_change": h5["condition_number"] - h4["condition_number"],
            "correlation_change": h5["correlation_at_optimum"] - h4["correlation_at_optimum"],
        }
