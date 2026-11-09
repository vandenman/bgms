library(bgms)
M = matrix(.1, 2, 2)
S = matrix(.2, 2, 2)
normalizer = list(
  family = "informed-normal", mean = M, sd = S, rate = 2, delta = .5,
  control = bgms:::resolve_informed_control()
)
T = matrix(c(8, 2, 2, 12), 2)
m = -.2
tau = .4
logprior = log(bgms:::informed_normal_edge_integral(m, tau, c(1, 1), .5)$value)
logpost = log(bgms:::informed_normal_edge_integral(m - tau^2 * T[1, 2], tau, 1 + diag(T) / 2, 5.5)$value)
expected = plogis(-m * T[1, 2] + tau^2 * T[1, 2]^2 / 2 + logpost - logprior)
results = bgms:::sample_ggm(
  list(
    n = 10L, suf_stat = T, pairwise_scale = NA_real_, interaction_prior_type = "informed-normal",
    interaction_mean = M, interaction_sd = S, scale_prior_type = "exponential", scale_shape = 1, scale_rate = 2
  ),
  matrix(.5, 2, 2), matrix(1L, 2, 2), 200000L, 2000L, 4L, TRUE, "gibbs", 42L, 4L, 0L,
  delta = .5, zratio_spec = normalizer
)
stopifnot(!any(vapply(results, function(x) x$error, logical(1))))
values = sapply(results, function(x) x$indicator_samples[2, ])
batches = unlist(lapply(1:4, function(k) colMeans(matrix(values[, k], 1000))))
estimate = mean(values)
half = qt(.995, length(batches) - 1) * sd(batches) / sqrt(length(batches))
print(c(expected = expected, estimate = estimate, half_width = half, error_plus_half = abs(estimate - expected) + half))
saveRDS(list(expected = expected, estimate = estimate, half_width = half), file.path(tempdir(), "bgms-informed-posterior.rds"))
stopifnot(abs(estimate - expected) + half <= 0.005)
