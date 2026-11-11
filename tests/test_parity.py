from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
from scipy.stats import spearmanr
from pysparkx import sparkx, acat
DATA=Path(__file__).parents[1]/'data'
@pytest.mark.parametrize('mode',['single','mixture','covariates'])
def test_reference(mode):
    x=np.loadtxt(DATA/'input.csv',delimiter=',')
    loc=np.loadtxt(DATA/'coordinates.csv',delimiter=',')
    cov=np.loadtxt(DATA/'covariates.csv',delimiter=',') if mode=='covariates' else None
    out=sparkx(sparse.csr_matrix(x),loc,cov,option='single' if mode=='single' else 'mixture')
    stat=np.loadtxt(DATA/f'{mode}_stats.csv',delimiter=',',ndmin=2)
    p=np.loadtxt(DATA/f'{mode}_p.csv',delimiter=',',ndmin=2)
    if mode=='single': stat,p=stat.reshape(-1,1),p.reshape(-1,1)
    np.testing.assert_allclose(out['statistics'],stat,atol=1e-8,rtol=1e-8)
    np.testing.assert_allclose(out['pvalues'],p,atol=1e-6,rtol=0)
    ref=np.loadtxt(DATA/f'{mode}_combined.csv',delimiter=',')
    for col in range(2):
        observed=out['results'].iloc[:,col].to_numpy()
        assert spearmanr(observed,ref[:,col]).statistic>=.99
        a=set(np.argsort(observed)[:20]);b=set(np.argsort(ref[:,col])[:20])
        assert len(a&b)/len(a|b)>=.9

def test_dense_sparse_filtering_and_degenerate_inputs():
    x=np.loadtxt(DATA/'input.csv',delimiter=',');loc=np.loadtxt(DATA/'coordinates.csv',delimiter=',')
    a=sparkx(x,loc,option='single');b=sparkx(sparse.csr_matrix(x),loc,option='single')
    np.testing.assert_allclose(a['results'],b['results'],atol=1e-12)
    zero=np.pad(x,((0,1),(0,1))); coords=np.vstack([loc,[0,0]])
    c=sparkx(zero,coords,option='single')
    assert not c['keep_genes'][-1] and not c['keep_spots'][-1]
    np.testing.assert_allclose(c['results'],a['results'],atol=1e-12)
    with pytest.raises(ValueError,match='rank'):sparkx(x,np.ones_like(loc),option='single')
    with pytest.raises(ValueError):acat([0,1])
    assert acat([1e-100,1e-100])==pytest.approx(1e-100)
