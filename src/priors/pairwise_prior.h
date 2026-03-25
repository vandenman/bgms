#pragma once

#include <memory>
#include <cmath>
#include <RcppArmadillo.h>
#include "../rng/rng_utils.h"


/**
 * Abstract base class for pairwise (slab) priors on precision off-diagonals.
 *
 * Each concrete subclass provides a log-density evaluation and an optional
 * hyperparameter update step.  The GGMModel holds a unique_ptr<BasePairwisePrior>
 * and calls log_density() wherever the old R::dcauchy() calls lived.
 */
class BasePairwisePrior {
public:
    virtual ~BasePairwisePrior() = default;

    /** Log-density of the prior evaluated at x. */
    virtual double log_density(double x) const = 0;

    /**
     * Sample hyperparameters from their full conditionals.
     *
     * Called once per MCMC sweep. Implementations that have no
     * hyperparameters (e.g. Cauchy) leave this as a no-op.
     *
     * @param precision_matrix  Current precision matrix
     * @param edge_indicators   Current edge indicator matrix (used when
     *                          edge selection is active)
     * @param num_variables     Number of variables p
     * @param rng               Random number generator
     */
    virtual void update_hyperparameters(
        const arma::mat& precision_matrix,
        const arma::imat& edge_indicators,
        int num_variables,
        SafeRNG& rng
    ) {
        (void)precision_matrix;
        (void)edge_indicators;
        (void)num_variables;
        (void)rng;
    }

    /** Current value of the shrinkage parameter (for output storage). */
    virtual double get_lambda() const { return 0.0; }

    /** Whether this prior has a sampled hyperparameter to store. */
    virtual bool has_hyperparameter() const { return false; }

    virtual std::unique_ptr<BasePairwisePrior> clone() const = 0;
};


/**
 * Cauchy slab prior on precision off-diagonal elements.
 *
 * p(omega_ij | scale) = Cauchy(0, scale).
 * No hyperparameters to update.
 */
class CauchyPairwisePrior : public BasePairwisePrior {
public:
    explicit CauchyPairwisePrior(double scale = 2.5)
        : scale_(scale) {}

    double log_density(double x) const override {
        return R::dcauchy(x, 0.0, scale_, true);
    }

    std::unique_ptr<BasePairwisePrior> clone() const override {
        return std::make_unique<CauchyPairwisePrior>(*this);
    }

private:
    double scale_;
};


/**
 * Laplace (double-exponential) slab prior — Bayesian Lasso.
 *
 * p(omega_ij | lambda) = (lambda/2) * exp(-lambda * |omega_ij|)
 *
 * Hyperprior on lambda^2:
 *   lambda^2 ~ Gamma(gamma_shape, gamma_rate)
 *
 * Full conditional:
 *   lambda^2 | Omega ~ Gamma(gamma_shape + p(p-1)/2,
 *                             gamma_rate + sum_{i<j} |omega_ij|)
 * then lambda = sqrt(lambda^2).
 */
class LaplacePairwisePrior : public BasePairwisePrior {
public:
    LaplacePairwisePrior(double gamma_shape = 2.0,
                         double gamma_rate = 0.1)
        : gamma_shape_(gamma_shape),
          gamma_rate_(gamma_rate),
          lambda_(std::sqrt(gamma_shape / gamma_rate))
    {}

    double log_density(double x) const override {
        // log( lambda/2 * exp(-lambda * |x|) )
        //   = log(lambda) - log(2) - lambda * |x|
        return std::log(lambda_) - std::log(2.0) - lambda_ * std::abs(x);
    }

    void update_hyperparameters(
        const arma::mat& precision_matrix,
        const arma::imat& edge_indicators,
        int num_variables,
        SafeRNG& rng
    ) override {
        double sum_abs = 0.0;
        int count = 0;
        for (int i = 0; i < num_variables - 1; ++i) {
            for (int j = i + 1; j < num_variables; ++j) {
                if (edge_indicators(i, j) == 1) {
                    sum_abs += std::abs(precision_matrix(i, j));
                }
                ++count;
            }
        }

        double post_shape = gamma_shape_ + count;
        // R::rgamma uses (shape, scale) parametrization; scale = 1/rate
        double post_rate = gamma_rate_ + sum_abs;
        double lambda_sq = R::rgamma(post_shape, 1.0 / post_rate);
        lambda_ = std::sqrt(lambda_sq);
    }

    double get_lambda() const override { return lambda_; }
    bool has_hyperparameter() const override { return true; }

    std::unique_ptr<BasePairwisePrior> clone() const override {
        return std::make_unique<LaplacePairwisePrior>(*this);
    }

private:
    double gamma_shape_;
    double gamma_rate_;
    double lambda_;
};


/**
 * Normal prior on precision off-diagonal elements.
 *
 * p(omega_ij | sd) = Normal(0, sd).
 * No hyperparameters to update.
 */
class NormalPairwisePrior : public BasePairwisePrior {
public:
    explicit NormalPairwisePrior(double sd = 1.0)
        : sd_(sd) {}

    double log_density(double x) const override {
        return R::dnorm(x, 0.0, sd_, true);
    }

    std::unique_ptr<BasePairwisePrior> clone() const override {
        return std::make_unique<NormalPairwisePrior>(*this);
    }

private:
    double sd_;
};


/**
 * Factory: create a pairwise prior from a type string and hyperparameters.
 *
 * @param type            "cauchy", "blasso", or "normal"
 * @param pairwise_scale  Scale parameter for Cauchy prior / SD for Normal prior
 * @param blasso_shape    Gamma shape for Bayesian Lasso hyperprior
 * @param blasso_rate     Gamma rate for Bayesian Lasso hyperprior
 */
inline std::unique_ptr<BasePairwisePrior> create_pairwise_prior(
    const std::string& type,
    double pairwise_scale = 2.5,
    double blasso_shape = 2.0,
    double blasso_rate = 0.1
) {
    if (type == "blasso") {
        return std::make_unique<LaplacePairwisePrior>(blasso_shape, blasso_rate);
    }
    if (type == "normal") {
        return std::make_unique<NormalPairwisePrior>(pairwise_scale);
    }
    return std::make_unique<CauchyPairwisePrior>(pairwise_scale);
}
