#ifndef BGMS_INTERACTION_PRIOR_H
#define BGMS_INTERACTION_PRIOR_H

#include <Rmath.h>
#include <cmath>
#include <string>

// =============================================================================
// InteractionPriorType — enum for pairwise interaction priors
// =============================================================================
enum class InteractionPriorType {
    Cauchy = 0,
    Normal = 1
};

// =============================================================================
// Conversion from R string to enum
// =============================================================================
inline InteractionPriorType interaction_prior_from_string(const std::string& s) {
    if (s == "cauchy") return InteractionPriorType::Cauchy;
    if (s == "normal") return InteractionPriorType::Normal;
    Rf_error("Unknown interaction prior type: '%s'", s.c_str());
    return InteractionPriorType::Cauchy; // unreachable
}

// =============================================================================
// Log-density of the interaction prior at x, given scale
// =============================================================================
inline double interaction_prior_logp(InteractionPriorType type,
                                     double x, double scale) {
    switch (type) {
        case InteractionPriorType::Cauchy:
            return R::dcauchy(x, 0.0, scale, true);
        case InteractionPriorType::Normal:
            return R::dnorm(x, 0.0, scale, true);
    }
    return 0.0; // unreachable
}

// =============================================================================
// Gradient d/dx log p(x) for the interaction prior
// =============================================================================
inline double interaction_prior_grad(InteractionPriorType type,
                                     double x, double scale) {
    switch (type) {
        case InteractionPriorType::Cauchy: {
            // d/dx log Cauchy(x; 0, s) = -2x / (s^2 + x^2)
            double s2 = scale * scale;
            return -2.0 * x / (s2 + x * x);
        }
        case InteractionPriorType::Normal: {
            // d/dx log Normal(x; 0, s) = -x / s^2
            double s2 = scale * scale;
            return -x / s2;
        }
    }
    return 0.0; // unreachable
}


// =============================================================================
// ThresholdPriorType — enum for main-effect / threshold priors
// =============================================================================
enum class ThresholdPriorType {
    BetaPrime = 0,
    Normal = 1
};

// =============================================================================
// Conversion from R string to enum
// =============================================================================
inline ThresholdPriorType threshold_prior_from_string(const std::string& s) {
    if (s == "beta-prime") return ThresholdPriorType::BetaPrime;
    if (s == "normal") return ThresholdPriorType::Normal;
    Rf_error("Unknown threshold prior type: '%s'", s.c_str());
    return ThresholdPriorType::BetaPrime; // unreachable
}

// =============================================================================
// Log-density of the threshold prior
//
// For beta-prime: log p(x) = alpha * x - (alpha + beta) * log(1 + exp(x))
//   This is the log-density of the logit-Beta distribution
//   (i.e., sigma(x) ~ Beta(alpha, beta))
//
// For normal: log p(x) = log Normal(x; 0, scale)
// =============================================================================
inline double threshold_prior_logp(ThresholdPriorType type,
                                   double x,
                                   double alpha, double beta,
                                   double scale) {
    switch (type) {
        case ThresholdPriorType::BetaPrime:
            return x * alpha - std::log1p(std::exp(x)) * (alpha + beta);
        case ThresholdPriorType::Normal:
            return R::dnorm(x, 0.0, scale, true);
    }
    return 0.0; // unreachable
}

// =============================================================================
// Gradient d/dx log p(x) for the threshold prior
//
// For beta-prime: alpha - (alpha + beta) * sigmoid(x)
//   where sigmoid(x) = exp(x) / (1 + exp(x))
//
// For normal: -x / scale^2
// =============================================================================
inline double threshold_prior_grad(ThresholdPriorType type,
                                   double x,
                                   double alpha, double beta,
                                   double scale) {
    switch (type) {
        case ThresholdPriorType::BetaPrime: {
            double p = 1.0 / (1.0 + std::exp(-x));  // sigmoid
            return alpha - (alpha + beta) * p;
        }
        case ThresholdPriorType::Normal: {
            double s2 = scale * scale;
            return -x / s2;
        }
    }
    return 0.0; // unreachable
}

#endif // BGMS_INTERACTION_PRIOR_H
