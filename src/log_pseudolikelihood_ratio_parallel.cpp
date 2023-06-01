#include <Rcpp.h>
#include <RcppParallel.h>

using namespace Rcpp;
using namespace RcppParallel;

// [[Rcpp::depends(RcppParallel)]]

/*
//// [[Rcpp::export]]
double compute_ratio_node_orig(
    const RMatrix<double>& thresholds,
    const RVector<int>&    no_categories,
    // const NumericMatrix&   thresholds,
    // const IntegerVector&   no_categories,
    const double           bound,
    const double           rest_score,
    const double           proposed_state,
    const double           current_state,
    const int              obs_score,
    const int              node
) {

  double denominator_prop = std::exp(-bound);
  double denominator_curr = std::exp(-bound);
  int score;
  double exponent;
  for(int category = 0; category < no_categories[node]; category++) {
    score = category + 1;
    exponent = thresholds(node, category) +
      score * rest_score -
      bound;
    denominator_prop +=
      std::exp(exponent + score * obs_score * proposed_state);
    denominator_curr +=
      std::exp(exponent + score * obs_score * current_state);
  }

  return std::log(denominator_curr) - std::log(denominator_prop);

}


//// [[Rcpp::export]]
double compute_ratio_node_new(
    // const RMatrix<double>& thresholds,
    // const RVector<int>&    no_categories,
    // const NumericMatrix&   thresholds,
    // const IntegerVector&   no_categories,
    const double           bound,
    const double           rest_score,
    const double           proposed_state,
    const double           current_state,
    const int              obs_score,
    const int              node1,
    const int person
) {

  double denominator_prop = std::exp(-bound);
  double denominator_curr = denominator_prop;
  int obs_score_times_score = 0;
  double rest_score_times_score = 0.0,
    proposed_state_times_score_times_obs_score = 0.0,
    current_state_times_score_times_obs_score = 0.0;
  double proposed_state_times_obs_score = obs_score * proposed_state,
    current_state_times_obs_score  = obs_score * current_state;

  for (int category = 0; category < no_categories[node]; category++) {

     // score = category + 1;
     // double exponent = thresholds(node, category) + score * rest_score - bound;
     // denominator_prop +=
     // std::exp(exponent + score * obs_score2 * proposed_state);
     // denominator_curr +=
     // std::exp(exponent + score * obs_score2 * current_state);

    rest_score_times_score += rest_score;
    obs_score_times_score  += obs_score;
    proposed_state_times_score_times_obs_score += proposed_state_times_obs_score;
    proposed_state_times_score_times_obs_score += current_state_times_obs_score;
    double exponent = thresholds(node, category) + rest_score_times_score - bound;

    denominator_prop += std::exp(exponent + obs_score_times_score * proposed_state);
    denominator_curr += std::exp(exponent + obs_score_times_score * current_state);
  }

  return std::log(denominator_curr / denominator_prop);

}
*/

struct Log_pseudolikelihood_ratio_worker : public Worker
{
  // source vector
  const RMatrix<double> interactions;
  const RMatrix<double> thresholds;
  const RMatrix<int>    observations;
  const RVector<int>    no_categories;
  const int             no_persons;
  const int             node1;
  const int             node2;
  const double          proposed_state;
  const double          current_state;
  const RMatrix<double> rest_matrix;

  const double          delta_state;

  // accumulated value
  double value;

  // constructors
  Log_pseudolikelihood_ratio_worker(
    const NumericMatrix interactions,
    const NumericMatrix thresholds,
    const IntegerMatrix observations,
    const IntegerVector no_categories,
    const int           no_persons,
    const int           node1,
    const int           node2,
    const double        proposed_state,
    const double        current_state,
    const NumericMatrix rest_matrix) :
    interactions(interactions),
    thresholds(thresholds),
    observations(observations),
    no_categories(no_categories),
    no_persons(no_persons),
    node1(node1),
    node2(node2),
    proposed_state(proposed_state),
    current_state(current_state),
    rest_matrix(rest_matrix),
    delta_state(proposed_state - current_state),
    value(0)
  {}

  Log_pseudolikelihood_ratio_worker(
    const Log_pseudolikelihood_ratio_worker& w, Split) :
    interactions(w.interactions),
    thresholds(w.thresholds),
    observations(w.observations),
    no_categories(w.no_categories),
    no_persons(w.no_persons),
    node1(w.node1),
    node2(w.node2),
    proposed_state(w.proposed_state),
    current_state(w.current_state),
    rest_matrix(w.rest_matrix),
    delta_state(proposed_state - current_state),
    value(0) {}

  // accumulate just the element of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {

    // for(int person = 0; person < no_persons; person++) {
    double rest_score, bound;
    double pseudolikelihood_ratio = 0.0;
    // double denominator_prop, denominator_curr, exponent;
    // int score;//, obs_score1, obs_score2;

    for (std::size_t person = begin; person < end; person++ ) {
      const int obs_score1 = observations(person, node1);
      const int obs_score2 = observations(person, node2);

      // pseudolikelihood numerator
      pseudolikelihood_ratio += 2 * obs_score1 * obs_score2 * delta_state;

      // pseudolikelihood denominators
      pseudolikelihood_ratio += compute_pseudolikelihood_denominator(obs_score2, node1, node2, person);
      pseudolikelihood_ratio += compute_pseudolikelihood_denominator(obs_score1, node2, node1, person);

      // //Node 1 log pseudolikelihood ratio
      // rest_score = rest_matrix(person, node1) -
      //   obs_score2 * interactions(node2, node1);
      //
      // // bound = rest_score > 0 ? no_categories[node1] * rest_score : 0.0;
      // if(rest_score > 0) {
      //   bound = no_categories[node1] * rest_score;
      // } else {
      //   bound = 0.0;
      // }

      // denominator_prop = std::exp(-bound);
      // denominator_curr = std::exp(-bound);
      // for(int category = 0; category < no_categories[node1]; category++) {
      //   score = category + 1;
      //   exponent = thresholds(node1, category) +
      //     score * rest_score -
      //     bound;
      //   denominator_prop +=
      //     std::exp(exponent + score * obs_score2 * proposed_state);
      //   denominator_curr +=
      //     std::exp(exponent + score * obs_score2 * current_state);
      // }
      // pseudolikelihood_ratio -= std::log(denominator_prop);
      // pseudolikelihood_ratio += std::log(denominator_curr);

      //Node 2 log pseudolikelihood ratio
      // rest_score = rest_matrix(person, node2) -
      //   obs_score1 * interactions(node1, node2);
      //
      // if(rest_score > 0) {
      //   bound = no_categories[node2] * rest_score;
      // } else {
      //   bound = 0.0;
      // }
      //
      // pseudolikelihood_ratio += compute_ratio_node_new(
      //   thresholds, no_categories, bound, rest_score,
      //   proposed_state, current_state,
      //   obs_score1, node2
      // );

      // denominator_prop = std::exp(-bound);
      // denominator_curr = std::exp(-bound);
      // for(int category = 0; category < no_categories[node2]; category++) {
      //   score = category + 1;
      //   exponent = thresholds(node2, category) +
      //     score * rest_score -
      //     bound;
      //   denominator_prop +=
      //     std::exp(exponent + score * obs_score1 * proposed_state);
      //   denominator_curr +=
      //     std::exp(exponent + score * obs_score1 * current_state);
      // }
      // pseudolikelihood_ratio -= std::log(denominator_prop);
      // pseudolikelihood_ratio += std::log(denominator_curr);
    }

    value += pseudolikelihood_ratio;

    // value += std::accumulate(input.begin() + begin, input.begin() + end, 0.0);
  }

  // join my value with that of another Sum
  void join(const Log_pseudolikelihood_ratio_worker& rhs) {
    value += rhs.value;
  }

  //// [[Rcpp::export]]
  double compute_pseudolikelihood_denominator(
      const int obs_score2,
      const int node1,
      const int node2,
      const int person
  ) {

    const double rest_score = rest_matrix(person, node1) - obs_score2 * interactions(node2, node1);

    const double bound = rest_score > 0 ? no_categories[node1] * rest_score : 0.0;

    double denominator_prop = std::exp(-bound);
    double denominator_curr = denominator_prop;

    int obs_score_times_score                    = 0;

    double
      rest_score_times_score                     = 0.0,
      proposed_state_times_score_times_obs_score = 0.0,
      current_state_times_score_times_obs_score  = 0.0;

    double
      proposed_state_times_obs_score = obs_score2 * proposed_state,
      current_state_times_obs_score  = obs_score2 * current_state;

    for (int category = 0; category < no_categories[node1]; category++) {

      // score = category + 1;
      // double exponent = thresholds(node, category) + score * rest_score - bound;
      // denominator_prop +=
      //   std::exp(exponent + score * obs_score2 * proposed_state);
      // denominator_curr +=
      //   std::exp(exponent + score * obs_score2 * current_state);

      rest_score_times_score += rest_score;
      obs_score_times_score  += obs_score2;
      proposed_state_times_score_times_obs_score += proposed_state_times_obs_score;
      proposed_state_times_score_times_obs_score += current_state_times_obs_score;
      double exponent = thresholds(node1, category) + rest_score_times_score - bound;

      denominator_prop += std::exp(exponent + obs_score_times_score * proposed_state);
      denominator_curr += std::exp(exponent + obs_score_times_score * current_state);
    }

    return std::log(denominator_curr / denominator_prop);

  }
};

// [[Rcpp::export]]
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
    const NumericMatrix rest_matrix) {

  Log_pseudolikelihood_ratio_worker pseudolikelihood_ratio(
    interactions,
    thresholds,
    observations,
    no_categories,
    no_persons,
    node1,
    node2,
    proposed_state,
    current_state,
    rest_matrix
  );

  parallelReduce(0, no_persons, pseudolikelihood_ratio);

  return pseudolikelihood_ratio.value;

}

/*
#include <Rcpp.h>
#include <RcppParallel.h>

using namespace Rcpp;
using namespace RcppParallel;

struct Sum : public Worker
{
  // source vector
  const RVector<double> input;

  // accumulated value
  double value;

  // constructors
  Sum(const NumericVector input) : input(input), value(0) {}
  Sum(const Sum& sum, Split) : input(sum.input), value(0) {}

  // accumulate just the element of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    value += std::accumulate(input.begin() + begin, input.begin() + end, 0.0);
  }

  // join my value with that of another Sum
  void join(const Sum& rhs) {
    value += rhs.value;
  }
};

// [[Rcpp::export]]
double parallelVectorSum(NumericVector x) {

  // declare the Sum instance
  Sum sum(x);

  // call parallel_reduce to start the work
  parallelReduce(0, x.length(), sum);

  // return the computed sum
  return sum.value;
}

*/