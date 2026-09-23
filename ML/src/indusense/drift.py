"""Détection de dérive (modules 31-34) : PSI + KS, une seule règle pour
les deux — le PSI décide, le KS confirme (jamais l'inverse, voir
reports/drift/drift_spec.md).

Le PSI est calculé sur des bins figés sur la RÉFÉRENCE seule (jamais
recalculés sur la fenêtre courante) : sinon deux fenêtres évaluées contre
la même référence ne seraient plus comparables entre elles.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

EPSILON = 1e-4


def reference_bin_edges(reference: pd.Series, bins: int = 10) -> np.ndarray:
    """Bornes de bins (quantiles) figées sur la référence seule."""
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(reference.dropna(), quantiles)
    edges[0], edges[-1] = -np.inf, np.inf
    return np.unique(edges)


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index d'une feature entre une référence figée
    et une fenêtre courante, sur les bins de la référence. `EPSILON`
    évite une division par zéro quand un bin est vide d'un côté (PSI
    infini sur un simple artefact d'échantillonnage sinon)."""
    edges = reference_bin_edges(reference, bins=bins)
    ref_counts, _ = np.histogram(reference.dropna(), bins=edges)
    cur_counts, _ = np.histogram(current.dropna(), bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1) + EPSILON
    cur_pct = cur_counts / max(cur_counts.sum(), 1) + EPSILON

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_pvalue(reference: pd.Series, current: pd.Series) -> float:
    """p-value du test de Kolmogorov-Smirnov à deux échantillons —
    calculé pour chaque feature à chaque fenêtre (même cadence que le
    PSI), mais jamais décisionnel seul : voir drift_spec.md."""
    result = ks_2samp(reference.dropna(), current.dropna())
    return float(result.pvalue)


def drift_table(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """PSI + KS p-value par feature, référence vs fenêtre courante."""
    rows = [
        {
            "feature": feature,
            "psi": psi(reference[feature], current[feature]),
            "ks_pvalue": ks_pvalue(reference[feature], current[feature]),
        }
        for feature in features
    ]
    return pd.DataFrame(rows)
