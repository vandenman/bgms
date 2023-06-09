#ifndef COMPUTE_C_PARALLEL
#define COMPUTE_C_PARALLEL

#include <Rcpp.h>
using namespace Rcpp;

double compute_c_parallel(
    const int no_persons,
    const int node,
    const int category,
    const double exp_current,
    const NumericMatrix rest_matrix,
    const IntegerVector no_categories,
    const NumericMatrix thresholds,
    NumericVector g,
    NumericVector q
);

#endif // COMPUTE_C_PARALLEL