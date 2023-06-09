#include <Rcpp.h>
#include <RcppParallel.h>

using namespace Rcpp;
using namespace RcppParallel;

// [[Rcpp::depends(RcppParallel)]]

struct Compute_c_worker : public Worker
{
  // source vector
  const int             no_persons;
  const int             node;
  const int             category;
  const double          exp_current;
  const RMatrix<double> rest_matrix;
  const RVector<int>    no_categories;
  const RMatrix<double> thresholds;

  RVector<double> g;
  RVector<double> q;

  double c_value;

  // constructors
  Compute_c_worker(
    const int no_persons,
    const int node,
    const int category,
    const double exp_current,
    const NumericMatrix rest_matrix,
    const IntegerVector no_categories,
    const NumericMatrix thresholds,
    NumericVector g,
    NumericVector q) :
    no_persons(no_persons),
    node(node),
    category(category),
    exp_current(exp_current),
    rest_matrix(rest_matrix),
    no_categories(no_categories),
    thresholds(thresholds),
    g(g),
    q(q),
    c_value(0)
  {}

  Compute_c_worker(
    const Compute_c_worker& w, Split) :
    no_persons(w.no_persons),
    node(w.node),
    category(w.category),
    exp_current(w.exp_current),
    rest_matrix(w.rest_matrix),
    no_categories(w.no_categories),
    thresholds(w.thresholds),
    g(w.g),
    q(w.q),
    c_value(0)
  {}

  void join(const Compute_c_worker& rhs) {
    c_value += rhs.c_value;
  }

  // accumulate just the element of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {

    double c = 0.0;
    for (std::size_t person = begin; person < end; person++ ) {
      g[person] = 1.0;
      q[person] = 1.0;
      double rest_score = rest_matrix(person, node);
      for(int cat = 0; cat < no_categories[node]; cat++) {
        if(cat != category) {
          g[person] += std::exp(thresholds(node, cat) +
            (cat + 1) * rest_score);
        }
      }
      q[person] = std::exp((category + 1) * rest_score);
      c +=  q[person] / (g[person] + q[person] * exp_current);
    }

    c_value += c;
  }
};

// [[Rcpp::export]]
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
) {
  Compute_c_worker c_worker(
      no_persons,
      node,
      category,
      exp_current,
      rest_matrix,
      no_categories,
      thresholds,
      g,
      q
  );

  parallelReduce(0, no_persons, c_worker);

  return c_worker.c_value;
}