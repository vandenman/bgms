#pragma once

#include <memory>
#include <cmath>
#include <RcppArmadillo.h>


/**
 * Abstract base class for priors on precision matrix diagonal elements.
 *
 * The diagonal of the precision matrix must be positive. Concrete subclasses
 * define a log-density over positive reals. The GGMModel holds a
 * unique_ptr<BaseDiagonalPrior> and calls log_density() in
 * update_diagonal_parameter().
 */
class BaseDiagonalPrior {
public:
    virtual ~BaseDiagonalPrior() = default;

    /** Log-density of the prior evaluated at x (x > 0). */
    virtual double log_density(double x) const = 0;

    virtual std::unique_ptr<BaseDiagonalPrior> clone() const = 0;
};


/**
 * Exponential prior on diagonal precision elements.
 *
 * p(x | rate) = rate * exp(-rate * x),  x > 0.
 *
 * With rate = 1 this is equivalent to the Gamma(1, 1) prior used in the
 * original GGM sampler, since Gamma(shape=1, rate=1) = Exp(1).
 */
class ExponentialDiagonalPrior : public BaseDiagonalPrior {
public:
    explicit ExponentialDiagonalPrior(double rate = 1.0)
        : rate_(rate) {}

    double log_density(double x) const override {
        // R::dexp uses scale = 1/rate
        return R::dexp(x, 1.0 / rate_, true);
    }

    std::unique_ptr<BaseDiagonalPrior> clone() const override {
        return std::make_unique<ExponentialDiagonalPrior>(*this);
    }

private:
    double rate_;
};


/**
 * Factory: create a diagonal prior from a type string and parameter.
 *
 * @param type  "exponential"
 * @param rate  Rate parameter (default 1.0)
 */
inline std::unique_ptr<BaseDiagonalPrior> create_diagonal_prior(
    const std::string& type,
    double rate = 1.0
) {
    // Currently only exponential is supported; extend here for future types
    (void)type;
    return std::make_unique<ExponentialDiagonalPrior>(rate);
}
