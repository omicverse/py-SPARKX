# Attribution

Upstream SPARK 1.1.1, commit a8b4bf27b804604dfda53da42992f100b8e4e727.
Algorithms reconstructed from the upstream source under the accompanying LICENSE.
Upstream author declarations:

```
Package: SPARK
Type: Package
Title: Spatial Pattern Recognition via Kernels
Version: 1.1.1
Date: 2021-02-18
Author: Shiquan Sun, Jiaqiang Zhu, and Xiang Zhou
Maintainer: Jiaqiang Zhu <jiaqiang@umich.edu>
Description: SPARK is an efficient tool for identifying spatial expression patterns. SPARK directly models raw count data generated from various spatial resolved transcriptomic techniques. With a new efficient penalized quasi-likelihood based algorithm, SPARK is scalable to data sets with tens of thousands of genes measured on thousands of samples. Build upon a non-parametric framework, SPARK-X is scalable to large-scale data sets with tens of thousands of genes measured on hundred thousands of samples.
License: GPL-3
Depends: R (>= 3.4.0), methods
Imports: Rcpp (>= 1.0.5),foreach,doParallel,parallel,Matrix,CompQuadForm,matlab,pracma
LinkingTo: Rcpp, RcppArmadillo
RoxygenNote: 7.1.1

```

The central Davies implementation also follows cran/CompQuadForm src/qfc.cpp at f1e110ac54aa0e64dda8cf6254823e801878e036, Algorithm AS 155 by Robert B. Davies (1980). GPL-compatible attribution retained in _davies.py.
