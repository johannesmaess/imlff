"""
Adams-Bashforth extrapolation predictor for warm-starting fixed-point solves.

Keeps a short history of converged fixed points per variable, forms central
finite-difference derivative estimates, and combines them with Adams-Bashforth
coefficients to extrapolate one step forward.

References:
    Hairer, Nørsett, Wanner, "Solving Ordinary Differential Equations I",
    2nd ed., Springer, 1993, Ch. III, §1.
"""
from collections import defaultdict
from math import comb, factorial


# ===========================================================================
# Central finite differences
# ===========================================================================

def finite_differences(history):
    """
    Central finite-difference derivative estimates from equally spaced values.

    Given k+1 values [s_{n-k}, ..., s_{n-1}, s_n] (oldest first), returns k
    estimates [f_0, ..., f_{k-1}] with f_i ≈ s'(t_{n-i}):

        f_0 = s_n - s_{n-1}                      (backward; no s_{n+1})
        f_i = (s_{n-i+1} - s_{n-i-1}) / 2   i≥1  (central, O(h²))
    """
    k = len(history) - 1
    if k == 0:
        return []
    diffs = [history[-1] - history[-2]]
    for i in range(1, k):
        diffs.append((history[-i] - history[-(i + 2)]) / 2)
    return diffs


# ===========================================================================
# Adams-Bashforth coefficients (one step ahead)
# ===========================================================================

def _ab_gammas(order):
    """γ_j = ∫_0^1 σ(σ+1)···(σ+j-1)/j! dσ.  Hairer et al. III.1, Eq. (1.5).

    Values: 1, 1/2, 5/12, 3/8, ...  (∫_0^1 σ^i dσ = 1/(i+1)).
    """
    gammas = []
    for j in range(order):
        if j == 0:
            gammas.append(1.0)
            continue
        # Rising factorial σ(σ+1)···(σ+j-1) as a polynomial in σ.
        poly = [0.0, 1.0]
        for m in range(1, j):
            new = [0.0] * (len(poly) + 1)
            for i, a in enumerate(poly):
                new[i]     += m * a
                new[i + 1] += a
            poly = new
        jfact = factorial(j)
        gammas.append(sum((a / jfact) / (i + 1) for i, a in enumerate(poly)))
    return gammas


def ab_coefficients(order):
    """
    Adams-Bashforth coefficients for a one-step-ahead extrapolation:
        ŝ_{n+1} = s_n + Σ_j b_j f_{n-j},
    with derivative nodes at the grid points ŝ_j = -j.
    Hairer et al. III.1, Eq. (1.4).
    """
    if order == 0:
        return []
    gammas = _ab_gammas(order)
    return [
        sum((-1) ** m * comb(j, m) * gammas[j] for j in range(m, order))
        for m in range(order)
    ]


def build_ab_table(max_degree):
    """{degree: [AB coefficients]} for degrees 0..max_degree."""
    return {k: ab_coefficients(k) for k in range(max_degree + 1)}


# ===========================================================================
# Stateful Adams-Bashforth predictor
# ===========================================================================

class StatefulAdamsBashforth():
    """
    Stateful Adams-Bashforth extrapolation predictor.

    Config maps each variable to its polynomial degree:
        warmstart_configs = {'z': 2, 'g_bar': 3}

    Interface:
        save_state()  — record the converged _star values (call after the corrector)
        get_state()   — return the extrapolated warmstart,
                        dict  name+'0' -> tensor | None
    """

    def __init__(self, warmstart_configs=None, log_prefix=''):
        self.log_prefix = log_prefix
        warmstart_configs = warmstart_configs or {}

        # name -> degree.  Accept a bare int degree or a legacy tuple whose
        # first entry is the degree (any smoothing / fit-degree fields ignored).
        self.warmstart_configs = {
            name: (val[0] if isinstance(val, tuple) else val)
            for name, val in warmstart_configs.items()
        }
        self.degree_configs = dict(self.warmstart_configs)  # backward-compat alias

        # name -> [tensor, ...] (oldest first)
        self.state = defaultdict(list)

        # name -> {degree: [AB coefficients]}
        self._ab_tables = {
            name: build_ab_table(deg)
            for name, deg in self.warmstart_configs.items()
        }

    def _degree_available(self, name):
        """Effective degree given the history collected so far."""
        return min(self.warmstart_configs[name], len(self.state[name]) - 1)

    def get_warmstart_configs_available(self):
        """Available degree per variable given current history."""
        return {name: self._degree_available(name)
                for name in self.warmstart_configs}

    def get_state(self):
        """Extrapolated warmstart: dict  name+'0' -> tensor | None."""
        warmstart = {}
        for name in self.warmstart_configs:
            deg = self._degree_available(name)
            if deg < 0:
                warmstart[name + '0'] = None
                continue
            history = self.state[name]
            coeffs = self._ab_tables[name][deg]
            diffs = finite_differences(history)
            estimate = history[-1].clone()
            for j, b in enumerate(coeffs):
                estimate += b * diffs[j]
            warmstart[name + '0'] = estimate
        return warmstart

    def save_state(self):
        """Record the converged _star values (call after the corrector)."""
        results = getattr(self, self.results_key)
        if not results:
            raise ValueError(f"Could not find {self.results_key} in self.")

        for name, deg in self.warmstart_configs.items():
            name_star = name + '_star'
            assert name_star in results, \
                f"Missing {name_star} in {self.results_key}"
            history = self.state[name]
            history.append(results[name_star].clone().detach())
            # Keep only the entries the degree needs.
            del history[:-(deg + 1)]