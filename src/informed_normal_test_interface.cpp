#include "models/ggm/zratio_engine.h"
#include "priors/parameter_prior.h"
#include <RcppArmadillo.h>

// [[Rcpp::export]]
Rcpp::List informed_normal_potentials_cpp(Rcpp::List spec, Rcpp::List graphs) {
  auto engine = zratio_engine_from_spec(spec);
  arma::vec potentials(graphs.size());
  for (int k = 0; k < graphs.size(); ++k)
    potentials(k) = engine->informed.potential(Rcpp::as<arma::imat>(graphs[k]));
  return Rcpp::List::create(
      Rcpp::Named("log_z") = potentials,
      Rcpp::Named("hits") = engine->informed.hits,
      Rcpp::Named("evaluations") = engine->informed.evaluations,
      Rcpp::Named("half_width") = engine->informed.largest_half_width);
}

// [[Rcpp::export]]
Rcpp::NumericVector informed_normal_density_cpp(arma::mat mean, arma::mat sd,
                                                double x, int i, int j) {
  InformedNormalPrior prior(mean, sd);
  return Rcpp::NumericVector::create(prior.logp(x, i, j), prior.grad(x, i, j));
}

// [[Rcpp::export]]
arma::mat informed_normal_reference_cpp(const arma::imat &graph,
                                        const arma::mat &mean,
                                        const arma::mat &sd, double rate,
                                        double delta, int i, int j, int draws,
                                        int warmup, int chains, int seed) {
  const size_t p = graph.n_rows;
  arma::mat values(draws, chains), m = -2 * mean, tau = 2 * sd;
  const double beta = rate / 2;

  auto ratio = [&](const arma::mat &C, const arma::uvec &neighbors,
                   const arma::uvec &rest, size_t candidate) {
    const size_t d = neighbors.n_elem;
    double q = 2 * beta * C(candidate, candidate), k = 0;
    if (d) {
      arma::vec scales(d), mu(d);
      for (size_t t = 0; t < d; ++t) {
        scales(t) = tau(rest(neighbors(t)), j);
        mu(t) = m(rest(neighbors(t)), j);
      }
      arma::mat Coo = C.submat(neighbors, neighbors);
      arma::vec column_c = C.col(candidate);
      arma::vec c = column_c.elem(neighbors);
      arma::mat S = arma::diagmat(scales);
      arma::mat B = arma::eye(d, d) + 2 * beta * S * Coo * S;
      arma::mat joint(d + 1, d + 1, arma::fill::zeros);
      joint.submat(0, 0, d - 1, d - 1) = B;
      arma::vec cross = std::sqrt(2 * beta) * (scales % c);
      joint.submat(0, d, d - 1, d) = cross;
      joint.submat(d, 0, d, d - 1) = cross.t();
      joint(d, d) = C(candidate, candidate);
      arma::mat L;
      if (!joint.is_finite() || arma::rcond(joint) < 1e-14 ||
          !arma::chol(L, joint, "lower"))
        throw std::runtime_error(
            "Reference Schur complement is numerically unresolved");
      q = 2 * beta * L(d, d) * L(d, d);
      arma::mat Q = 2 * beta * Coo + arma::diagmat(1 / arma::square(scales));
      arma::vec a;
      if (!arma::solve(a, Q, mu / arma::square(scales),
                       arma::solve_opts::likely_sympd +
                           arma::solve_opts::no_approx) ||
          !a.is_finite())
        throw std::runtime_error("Reference mean solve failed");
      k = 2 * beta * arma::dot(c, a);
    }
    const double mc = m(rest(candidate), j),
                 variance = std::pow(tau(rest(candidate), j), 2);
    const double result = -.5 * std::log1p(q * variance) +
                          (.5 * variance * k * k - mc * k - .5 * q * mc * mc) /
                              (1 + q * variance);
    if (!(q > 0) || !std::isfinite(result))
      throw std::runtime_error(
          "Reference candidate mass is numerically unresolved");
    return result;
  };
  for (int chain = 0; chain < chains; ++chain) {
    SafeRNG rng(seed + chain);
    arma::mat K = arma::eye(p, p) *
                  ((delta + 1) / beta * (.5 + double(chain + 1) / chains));
    for (int iter = 0; iter < draws + warmup; ++iter) {
      for (size_t vertex = 0; vertex < p; ++vertex) {
        arma::uvec rest(p - 1);
        size_t at = 0;
        for (size_t k = 0; k < p; ++k)
          if (k != vertex)
            rest(at++) = k;
        arma::ivec column = graph.col(vertex);
        arma::uvec N = arma::find(column.elem(rest));
        arma::mat C = arma::inv_sympd(K.submat(rest, rest));
        arma::vec b(p - 1, arma::fill::zeros);
        if (!N.is_empty()) {
          arma::vec prec(N.n_elem), mu(N.n_elem);
          for (size_t k = 0; k < N.n_elem; ++k) {
            prec(k) = 1 / std::pow(tau(rest(N(k)), vertex), 2);
            mu(k) = m(rest(N(k)), vertex);
          }
          arma::mat Q = 2 * beta * C.submat(N, N) + arma::diagmat(prec);
          arma::mat L = arma::chol(Q);
          b.elem(N) =
              arma::solve(Q, prec % mu,
                          arma::solve_opts::likely_sympd +
                              arma::solve_opts::no_approx) +
              arma::solve(arma::trimatu(L), arma_rnorm_vec(rng, N.n_elem),
                          arma::solve_opts::no_approx);
        }
        for (size_t k = 0; k < rest.n_elem; ++k)
          K(rest(k), vertex) = K(vertex, rest(k)) = b(k);
        K(vertex, vertex) =
            rgamma(rng, delta + 1, beta) + arma::as_scalar(b.t() * C * b);
      }
      if (iter >= warmup) {
        arma::uvec rest(p - 1);
        size_t at = 0, candidate = 0;
        for (size_t k = 0; k < p; ++k)
          if (k != size_t(j)) {
            if (k == size_t(i))
              candidate = at;
            rest(at++) = k;
          }
        arma::ivec column = graph.col(j);
        arma::uvec N = arma::find(column.elem(rest));
        arma::mat C = arma::inv_sympd(K.submat(rest, rest));
        values(iter - warmup, chain) = std::exp(ratio(C, N, rest, candidate));
      }
    }
  }
  if (!values.is_finite())
    throw std::runtime_error("Nonfinite reference weights");
  return values;
}

#include "models/ggm/gaussian_integrated_mass.h"

// [[Rcpp::export]]
double informed_normal_mass_cpp(double a, double b, double mean,
                                double variance) {
  return gaussian_integrated_log_mass(a, b, mean, variance);
}

// [[Rcpp::export]]
Rcpp::List informed_normal_row_cpp(const arma::mat &A, const arma::vec &mean,
                                   const arma::vec &sd, double beta) {
  InformedNormalizer normalizer;
  return normalizer.inspect_row(A, mean, sd, beta);
}

#include "models/ggm/ggm_gradient.h"
// [[Rcpp::export]]
Rcpp::List informed_normal_gradient_cpp(const arma::vec &theta,
                                        const arma::mat &sufficient, int n,
                                        const arma::imat &graph,
                                        const arma::mat &mean,
                                        const arma::mat &sd, double shape,
                                        double rate) {
  GraphConstraintStructure constraints;
  constraints.build(graph);
  InformedNormalPrior interaction(mean, sd);
  GammaScalePrior diagonal(shape, rate);
  GGMGradientEngine engine;
  engine.rebuild(constraints, n, sufficient, interaction, diagonal);
  auto result = engine.logp_and_gradient(theta);
  return Rcpp::List::create(Rcpp::Named("value") = result.first,
                            Rcpp::Named("gradient") = result.second);
}
