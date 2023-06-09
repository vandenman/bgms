#ifndef LOG_PSEUDOLIKELIHOOD_RATIO_PARALLEL
#define LOG_PSEUDOLIKELIHOOD_RATIO_PARALLEL

#include <Rcpp.h>
using namespace Rcpp;

double log_pseudolikelihood_ratio_parallel(
  const NumericMatrix interactions,
  const NumericMatrix thresholds,
  const IntegerMatrix observations,
  const IntegerVector no_categories,
  const int           no_persons,
  const int           node1,
  const int           node2,
  const double        proposed_state,
  const double        current_state,
  const NumericMatrix rest_matrix);


#endif // LOG_PSEUDOLIKELIHOOD_RATIO_PARALLEL