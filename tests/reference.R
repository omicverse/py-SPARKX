library(Matrix)
library(CompQuadForm)
library(parallel)
ref <- commandArgs(TRUE)[1]
source(file.path(ref,'R','sparkx.R'))
# Native helpers only perform sparse sums/means; use Matrix equivalents.
sp_sums_Rcpp <- function(x,rowSums=FALSE) if(rowSums) Matrix::rowSums(x) else Matrix::colSums(x)
sp_means_Rcpp <- function(x,rowSums=FALSE) if(rowSums) Matrix::rowMeans(x) else Matrix::colMeans(x)
set.seed(42)
loc <- matrix(rnorm(120*2),120,2)
x <- matrix(rpois(90*120,4),90,120)
x[1:10,loc[,1]>0] <- x[1:10,loc[,1]>0]+4
cov <- cbind(1,rnorm(120))
write.table(x,'data/input.csv',sep=',',row.names=FALSE,col.names=FALSE)
write.table(loc,'data/coordinates.csv',sep=',',row.names=FALSE,col.names=FALSE)
write.table(cov,'data/covariates.csv',sep=',',row.names=FALSE,col.names=FALSE)
for (mode in c('single','mixture','covariates')) {
    res <- sparkx(Matrix(x,sparse=TRUE),loc,X_in=if(mode=='covariates') cov else NULL,
                  option=if(mode=='single') 'single' else 'mixture',verbose=FALSE)
    write.table(res$stats,paste0('data/',mode,'_stats.csv'),sep=',',row.names=FALSE,col.names=FALSE)
    write.table(res$res_stest,paste0('data/',mode,'_p.csv'),sep=',',row.names=FALSE,col.names=FALSE)
    write.table(res$res_mtest,paste0('data/',mode,'_combined.csv'),sep=',',row.names=FALSE,col.names=FALSE)
}
