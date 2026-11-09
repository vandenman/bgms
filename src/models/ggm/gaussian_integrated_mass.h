#pragma once
#include <cmath>
#include <stdexcept>

/** Log E[exp(-a X^2/2 + b X)] for X ~ Normal(mean, variance). */
inline double gaussian_integrated_log_mass(double a, double b, double mean,
                                           double variance) {
  const double av = a * variance;
  const double value =
      -0.5 * std::log1p(av) +
      (-0.5 * a * mean * mean + b * mean + 0.5 * b * b * variance) / (1.0 + av);
  if (!(a >= 0) || !(variance > 0) || !std::isfinite(av) ||
      !std::isfinite(value))
    throw std::runtime_error(
        "Gaussian integrated edge mass is not representable");
  return value;
}
