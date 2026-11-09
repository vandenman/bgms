#pragma once
#include "rng/rng_utils.h"
#include <RcppArmadillo.h>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <chrono>
#include <cstring>
#include <map>
#include <stdexcept>

class InformedNormalizerFailure : public std::runtime_error {
  public:
    explicit InformedNormalizerFailure(const std::string &reason)
        : std::runtime_error("Informed normalizer failure: " + reason) {}
};

/** Frozen full-graph SMC potentials in precision coordinates. */
class InformedNormalizer {
    arma::mat mean_, sd_;
    double beta_ = 1, delta_ = 0, tol_ = .001, confidence_ = .99;
    int initial_ = 1024, maximum_ = 1048576, multiplier_ = 4, replicas_ = 32;
    int seed_ = 1, sweeps_ = 1;
    std::map<std::string, double> cache_;
    struct Row {
        arma::mat U, L;
        arma::vec u;
        double logh = 0;
    };
    Row row(const arma::mat &A, const arma::uvec &N, const arma::vec &m, const arma::vec &s) const {
        Row r;
        if (N.is_empty()) {
            r.U.zeros(A.n_rows, 0);
            return r;
        }
        arma::mat E(A.n_rows, N.n_elem, arma::fill::zeros), LA;
        for (size_t k = 0; k < N.n_elem; ++k)
            E(N(k), k) = s(k);
        if (!arma::chol(LA, A, "lower"))
            throw InformedNormalizerFailure("precision Cholesky");
        r.U = arma::solve(arma::trimatl(LA), E, arma::solve_opts::no_approx);
        arma::mat B = arma::eye(N.n_elem, N.n_elem) + 2 * beta_ * r.U.t() * r.U;
        if (!arma::chol(r.L, B, "lower"))
            throw InformedNormalizerFailure("row Cholesky");
        arma::vec v = m / s;
        r.u = arma::solve(arma::trimatu(r.L.t()),
            arma::solve(arma::trimatl(r.L), v, arma::solve_opts::no_approx),
            arma::solve_opts::no_approx);
        r.logh = -arma::sum(arma::log(r.L.diag())) -
                 .5 * (arma::accu(arma::square(r.u - v)) +
                       2 * beta_ * arma::accu(arma::square(r.U * r.u)));
        if (!std::isfinite(r.logh))
            throw InformedNormalizerFailure("nonfinite row mass");
        return r;
    }
    arma::vec draw(const Row &r, SafeRNG &rng) const {
        if (r.u.is_empty())
            return arma::vec();
        arma::vec z = r.u + arma::solve(arma::trimatu(r.L.t()),
            arma_rnorm_vec(rng, r.u.n_elem), arma::solve_opts::no_approx);
        if (!z.is_finite()) throw InformedNormalizerFailure("nonfinite conditional draw");
        return z;
    }
    void rejuvenate(arma::mat &K, const arma::imat &G, SafeRNG &rng) const {
        const size_t p = K.n_rows;
        for (int sweep = 0; sweep < sweeps_; ++sweep)
            for (size_t j = 0; j < p; ++j) {
                arma::uvec rest(p - 1);
                size_t pos = 0;
                for (size_t k = 0; k < p; ++k)
                    if (k != j)
                        rest(pos++) = k;
                arma::ivec column = G.col(j).head(p);
                arma::uvec N = arma::find(column.elem(rest));
                arma::vec m(N.n_elem), s(N.n_elem);
                for (size_t k = 0; k < N.n_elem; ++k) {
                    m(k) = mean_(rest(N(k)), j);
                    s(k) = sd_(rest(N(k)), j);
                }
                Row r = row(K.submat(rest, rest), N, m, s);
                arma::vec z = draw(r, rng), b = s % z;
                K.row(j).zeros();
                K.col(j).zeros();
                for (size_t k = 0; k < N.n_elem; ++k)
                    K(j, rest(N(k))) = K(rest(N(k)), j) = b(k);
                K(j, j) = rgamma(rng, delta_ + 1, beta_) + arma::accu(arma::square(r.U * z));
            }
    }
    double replica(const arma::imat &G, int particles, uint64_t seed) const {
        SafeRNG rng;
        rng.eng.seed(seed);
        std::vector<arma::mat> K(particles);
        double logz = 0;
        for (size_t j = 0; j < G.n_rows; ++j) {
            arma::uvec N = arma::find(G.col(j).head(j));
            arma::vec m(N.n_elem), s(N.n_elem), logw(particles);
            for (size_t k = 0; k < N.n_elem; ++k) {
                m(k) = mean_(N(k), j);
                s(k) = sd_(N(k), j);
            }
            std::vector<Row> rows;
            rows.reserve(particles);
            for (int r = 0; r < particles; ++r) {
                rows.push_back(row(K[r], N, m, s));
                logw(r) = rows.back().logh;
            }
            double maxw = logw.max();
            arma::vec w = arma::exp(logw - maxw);
            logz += maxw + std::log(arma::mean(w));
            w /= arma::sum(w);
            if (j + 1 == G.n_rows)
                break;
            std::vector<arma::mat> next;
            next.reserve(particles);
            double offset = runif(rng) / particles, cdf = w(0);
            int a = 0;
            for (int r = 0; r < particles; ++r) {
                double point = offset + double(r) / particles;
                while (point > cdf && a + 1 < particles)
                    cdf += w(++a);
                arma::mat out(j + 1, j + 1, arma::fill::zeros);
                if (j)
                    out.submat(0, 0, j - 1, j - 1) = K[a];
                arma::vec z = draw(rows[a], rng), b = s % z;
                for (size_t k = 0; k < N.n_elem; ++k)
                    out(N(k), j) = out(j, N(k)) = b(k);
                out(j, j) =
                    rgamma(rng, delta_ + 1, beta_) + arma::accu(arma::square(rows[a].U * z));
                rejuvenate(out, G, rng);
                next.push_back(std::move(out));
            }
            K.swap(next);
        }
        return logz;
    }

  public:
    Rcpp::List inspect_row(const arma::mat& A, const arma::vec& m,
                           const arma::vec& s, double beta) {
        beta_ = beta;
        try {
            Row r = row(A, arma::regspace<arma::uvec>(0, A.n_rows - 1), m, s);
            arma::mat noise = arma::solve(arma::trimatu(r.L.t()),
                arma::eye(s.n_elem, s.n_elem), arma::solve_opts::no_approx);
            arma::mat scaled = arma::diagmat(s) * noise;
            return Rcpp::List::create(Rcpp::Named("log_h") = r.logh,
                Rcpp::Named("mean") = s % r.u,
                Rcpp::Named("covariance") = scaled * scaled.t());
        } catch (const std::exception& e) { throw InformedNormalizerFailure(e.what()); }
    }
    bool active = false;
    long hits = 0, evaluations = 0;
    double largest_half_width = 0, seconds = 0;
    int largest_particles = 0;
    InformedNormalizer() = default;
    explicit InformedNormalizer(const Rcpp::List &s) : active(true) {
        mean_ = -2 * Rcpp::as<arma::mat>(s["mean"]);
        sd_ = 2 * Rcpp::as<arma::mat>(s["sd"]);
        beta_ = Rcpp::as<double>(s["rate"]) / 2;
        delta_ = Rcpp::as<double>(s["delta"]);
        Rcpp::List c(s["control"]);
        initial_ = Rcpp::as<int>(c["particles_initial"]);
        maximum_ = Rcpp::as<int>(c["particles_max"]);
        multiplier_ = Rcpp::as<int>(c["particles_multiplier"]);
        replicas_ = Rcpp::as<int>(c["replicas"]);
        seed_ = Rcpp::as<int>(c["master_seed"]);
        sweeps_ = Rcpp::as<int>(c["rejuvenation_sweeps"]);
        tol_ = Rcpp::as<double>(c["log_z_tol"]);
        confidence_ = Rcpp::as<double>(c["nominal_confidence"]);
    }
    double potential(const arma::imat &G) {
        auto start = std::chrono::steady_clock::now();
        try {
            double value = potential_impl(G);
            seconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            return value;
        } catch (const InformedNormalizerFailure &) {
            throw;
        } catch (const std::exception &e) {
            throw InformedNormalizerFailure(e.what());
        }
    }
    double potential_impl(const arma::imat &G) {
        std::string key;
        uint64_t hash = 14695981039346656037ULL;
        bool empty = true;
        for (size_t j = 1; j < G.n_cols; ++j)
            for (size_t i = 0; i < j; ++i) {
                char x = G(i, j) != 0;
                empty &= !x;
                key.push_back('0' + x);
                hash = (hash ^ x) * 1099511628211ULL;
            }
        auto mix = [&](double value) {
            uint64_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            for (int b = 0; b < 8; ++b)
                hash = (hash ^ ((bits >> (8 * b)) & 255)) * 1099511628211ULL;
        };
        mix(1);
        mix(beta_);
        mix(delta_);
        mix(initial_);
        mix(maximum_);
        mix(multiplier_);
        mix(replicas_);
        mix(sweeps_);
        mix(tol_);
        mix(confidence_);
        for (size_t j = 1; j < G.n_cols; ++j)
            for (size_t i = 0; i < j; ++i) {
                mix(mean_(i, j));
                mix(sd_(i, j));
            }
        if (empty)
            return 0;
        auto found = cache_.find(key);
        if (found != cache_.end()) {
            ++hits;
            return found->second;
        }
        std::vector<int> component(G.n_rows, -1);
        int count = 0;
        for (size_t root = 0; root < G.n_rows; ++root)
            if (component[root] < 0) {
                std::vector<size_t> queue{root};
                component[root] = count;
                for (size_t q = 0; q < queue.size(); ++q)
                    for (size_t k = 0; k < G.n_rows; ++k)
                        if (k != queue[q] && G(queue[q], k) && component[k] < 0) {
                            component[k] = count;
                            queue.push_back(k);
                        }
                ++count;
            }
        if (count > 1) {
            double sum = 0, combined_half_width = 0;
            for (int c = 0; c < count; ++c) {
                std::vector<arma::uword> nodes;
                for (size_t k = 0; k < G.n_rows; ++k)
                    if (component[k] == c)
                        nodes.push_back(k);
                if (nodes.size() < 2)
                    continue;
                arma::uvec idx(nodes);
                InformedNormalizer sub = *this;
                sub.mean_ = mean_.submat(idx, idx);
                sub.sd_ = sd_.submat(idx, idx);
                sub.cache_.clear();
                sub.tol_ = tol_ / count;
                sub.confidence_ = 1.0 - (1.0 - confidence_) / count;
                sub.largest_half_width = 0;
                sub.largest_particles = 0;
                sum += sub.potential(G.submat(idx, idx));
                combined_half_width += sub.largest_half_width;
                largest_particles = std::max(largest_particles, sub.largest_particles);
            }
            largest_half_width = std::max(largest_half_width, combined_half_width);
            cache_[key] = sum;
            ++evaluations;
            return sum;
        }
        if (G.n_rows == 2) {
            double m = mean_(0, 1), tau = sd_(0, 1), error = 0;
            auto integrand = [&](double t) {
                if (t <= 0 || t >= 1)
                    return 0.0;
                double a = t / (1 - t), den = a + 2 * beta_ * tau * tau;
                return std::exp((delta_ + 1) * std::log(beta_) - std::lgamma(delta_ + 1) +
                                delta_ * std::log(a) - beta_ * a -
                                0.5 * std::log1p(2 * beta_ * tau * tau / a) - beta_ * m * m / den -
                                2 * std::log1p(-t));
            };
            double z = boost::math::quadrature::gauss_kronrod<double, 61>::integrate(
                integrand, 0, 1, 15, 1e-10, &error);
            if (z > 0 && std::isfinite(z) && error / z < tol_ * 0.1) {
                cache_[key] = std::log(z);
                ++evaluations;
                return cache_[key];
            }
        }
        int levels = 1;
        for (int n = initial_; n <= maximum_ / multiplier_; n *= multiplier_)
            ++levels;
        double critical = boost::math::quantile(boost::math::students_t(replicas_ - 1),
                                                1 - (1 - confidence_) / (2 * levels));
        for (int n = initial_, level = 0; n <= maximum_; ++level) {
            bool pass = true;
            double estimate = 0, half = 0;
            for (int phase = 0; phase < 2; ++phase) {
                arma::vec logs(replicas_);
                for (int r = 0; r < replicas_; ++r) {
                    uint64_t stream = hash ^ (uint64_t(seed_) * 0x9e3779b97f4a7c15ULL);
                    stream ^=
                        uint64_t(1 + r + replicas_ * (phase + 2 * level)) * 0xbf58476d1ce4e5b9ULL;
                    logs(r) = replica(G, n, stream);
                }
                double mx = logs.max();
                arma::vec values = arma::exp(logs - mx);
                double avg = arma::mean(values),
                       error = critical * arma::stddev(values) / std::sqrt(double(replicas_));
                half = error < avg ? -std::log1p(-error / avg) : INFINITY;
                estimate = mx + std::log(avg);
                if (!std::isfinite(estimate) || half > tol_) {
                    pass = false;
                    break;
                }
            }
            if (pass) {
                largest_particles = std::max(largest_particles, n);
                ++evaluations;
                largest_half_width = std::max(largest_half_width, half);
                cache_[key] = estimate;
                return estimate;
            }
            if (n > maximum_ / multiplier_)
                break;
            n *= multiplier_;
        }
        throw InformedNormalizerFailure("precision budget exhausted for graph " + key);
    }
};
