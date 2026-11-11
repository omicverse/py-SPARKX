"""SPARK-X projection-kernel tests. Reference: xzhoulab/SPARK (GPL-3)."""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2
from ._davies import Davies


def acat(p):
    """Equal-weight Cauchy combination with stable small-p handling."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('p must contain probabilities in [0, 1].')
    if np.any(p == 0) and np.any(p == 1):
        raise ValueError('Cannot combine both exact zero and exact one.')
    if np.any(p == 0):
        return 0.
    if np.any(p == 1):
        return 1.
    terms = np.empty_like(p)
    small = p < 1e-16
    terms[small] = 1 / (np.pi * p[small])
    terms[~small] = np.tan((0.5 - p[~small]) * np.pi)
    value = terms.mean()
    return 1 / (np.pi * value) if value > 1e15 else 0.5 - np.arctan(value) / np.pi


def _by(p):
    n = len(p)
    order = np.argsort(p, kind='stable')
    adjusted = p[order] * n * np.sum(1 / np.arange(1, n + 1)) / np.arange(1, n + 1)
    result = np.empty(n)
    result[order] = np.minimum(np.minimum.accumulate(adjusted[::-1])[::-1], 1.)
    return result


def sparkx(counts, coordinates, covariates=None, *, option='mixture'):
    """Test spatial dependence; counts are genes x spots, coordinates spots x d.

    Supports projection (single) and 11-kernel mixture, with optional full-rank
    covariates. Returns named statistics/pvalues and BY-adjusted combined tests.
    All-zero spots and genes are removed as in SPARK-X. Filter masks are returned.
    Sparse counts remain sparse; no dense spots-by-spots kernel is constructed.

    Tail probabilities preserve SPARK-X's Davies integration defaults and its
    Liu fallback (the projection spectrum has equal nonzero eigenvalues).
    Singular coordinate/covariate designs fail rather than changing the test.
    """
    frame = isinstance(counts, pd.DataFrame)
    if frame and (not counts.index.is_unique or not counts.columns.is_unique):
        raise ValueError('Gene and spot labels must be unique.')
    x = counts.to_numpy() if frame else counts
    x = sparse.csr_matrix(x, dtype=float) if sparse.issparse(x) else np.asarray(x, dtype=float)
    data = x.data if sparse.issparse(x) else x
    loc = np.asarray(coordinates, dtype=float)
    if x.ndim != 2 or not min(x.shape) or not np.isfinite(data).all() or np.any(data < 0):
        raise ValueError('counts must be a finite nonnegative genes-by-spots matrix.')
    if loc.ndim != 2 or loc.shape[0] != x.shape[1] or not loc.shape[1] or not np.isfinite(loc).all():
        raise ValueError('coordinates must have a finite row per input spot.')
    if option not in ('single', 'mixture'):
        raise ValueError('option must be single or mixture.')
    keep_spots = np.asarray(x.sum(axis=0)).ravel() != 0
    x = x[:, keep_spots]; loc = loc[keep_spots]
    keep_genes = np.asarray(x.sum(axis=1)).ravel() != 0
    x = x[keep_genes]
    if min(x.shape) == 0:
        raise ValueError('No nonzero genes/spots remain.')
    n = x.shape[1]
    denom = np.asarray(x.multiply(x).sum(axis=1) if sparse.issparse(x) else (x*x).sum(axis=1)).ravel()
    if covariates is not None:
        cov = np.asarray(covariates, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != len(keep_spots) or not np.isfinite(cov).all():
            raise ValueError('covariates must have a finite row per input spot.')
        cov = cov[keep_spots]
        if not cov.shape[1] or np.linalg.matrix_rank(cov) != cov.shape[1]:
            raise ValueError('Covariates must have full column rank.')
        inv = np.linalg.inv(cov.T @ cov)
        gx = np.asarray(x @ cov)
        residual_energy = denom - np.einsum('ij,ij->i', gx @ inv, gx)
        center = lambda z: z - cov @ (inv @ (cov.T @ z))
    else:
        totals = np.asarray(x.sum(axis=1)).ravel()
        residual_energy = denom - totals**2 / n
        center = lambda z: z - z.mean(axis=0)
    scale = residual_energy / denom
    if np.any(scale <= 1e-14):
        raise ValueError('Some genes have no residual variation; remove constant/confounded genes.')
    kernels = {'projection': loc}
    if option == 'mixture':
        centered = loc - loc.mean(axis=0)
        widths = np.quantile(np.abs(centered), np.arange(1, 6) / 5, axis=0)
        if np.any(widths <= 0):
            raise ValueError('Coordinate quantiles have zero bandwidth.')
        for family in ('gaus', 'cos'):
            for i, width in enumerate(widths):
                kernels[f'{family}{i+1}'] = (np.exp(-centered**2/(2*width**2)) if family == 'gaus'
                                             else np.cos(2*np.pi*centered/width))
    stats, pvals = {}, {}
    for name, raw in kernels.items():
        z = center(raw)
        if np.linalg.matrix_rank(z) != z.shape[1]:
            raise ValueError(f'{name} coordinates are rank deficient after adjustment.')
        gram_inv = np.linalg.inv(z.T @ z)
        product = np.asarray(x @ z)
        statistic = np.einsum('ij,ij->i', product @ gram_inv, product) * n / denom
        stats[name] = statistic
        eigenvalues = np.linalg.eigvalsh(z.T @ (z @ gram_inv))
        probabilities = []
        for value, factor in zip(statistic, scale):
            probability = Davies(np.sort(factor*eigenvalues)[::-1],value).sf()
            if probability <= 0:
                probability = chi2.sf(value/factor,df=z.shape[1])
            probabilities.append(probability)
        pvals[name] = probabilities
    names = counts.index[keep_genes] if frame else np.flatnonzero(keep_genes)
    stats = pd.DataFrame(stats, index=names)
    pvals = pd.DataFrame(pvals, index=names)
    combined = pvals.apply(lambda row: acat(row.to_numpy()), axis=1)
    return {'statistics': stats, 'pvalues': pvals,
            'results': pd.DataFrame({'combinedPval': combined, 'adjustedPval': _by(combined.to_numpy())}, index=names),
            'keep_genes': keep_genes, 'keep_spots': keep_spots}
