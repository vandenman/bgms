library(bgms)
probability = as.numeric(Sys.getenv("BGMS_ENUM_PROB", "0.3"))
iterations = as.integer(Sys.getenv("BGMS_ENUM_ITER", "400000"))
stopifnot(probability > 0, probability < 1, iterations >= 10000L, iterations %% 1000L == 0L)
p = 3L
M = matrix(c(0, -.015, .025, -.015, 0, -.01, .025, -.01, 0), p)
S = matrix(c(0, .02, .04, .02, 0, .06, .04, .06, 0), p)
graphs = lapply(0:7, function(code) {
  G = matrix(0L, p, p)
  G[upper.tri(G)] = as.integer(intToBits(code))[1:3]
  G + t(G)
})
edges = which(upper.tri(M), arr.ind = TRUE)
potentials = function(rate, delta) {
  value = half = numeric(8)
  for(code in 1:7) {
    e = max(which(as.integer(intToBits(code))[1:3] == 1L))
    parent = code - 2^(e - 1)
    ref = bgms:::informed_normal_reference(graphs[[parent + 1]], M, S, rate, delta,
      edges[e, 1], edges[e, 2],
      draws = 200000L, warmup = 2000L, chains = 4L, comparisons = 22L
    )
    stopifnot(ref$half_width <= .0005)
    value[code + 1] = value[parent + 1] - ref$log_ratio
    half[code + 1] = half[parent + 1] + ref$half_width
  }
  list(value = value, half = half)
}
prior = potentials(2, 1)
posterior = potentials(10, 6)
weights = exp(posterior$value - prior$value) * vapply(graphs, function(G) probability^(sum(G) / 2) * (1 - probability)^(3 - sum(G) / 2), numeric(1))
weights = weights / sum(weights)
reference_allowance = expm1(2 * max(prior$half + posterior$half)) / 2
spec = list(
  family = "informed-normal", mean = M, sd = S, rate = 2, delta = 1,
  control = bgms:::resolve_informed_control()
)
raw = bgms:::sample_ggm(
  list(
    n = 10L, suf_stat = diag(8, p), pairwise_scale = NA_real_,
    interaction_prior_type = "informed-normal", interaction_mean = M, interaction_sd = S,
    scale_prior_type = "exponential", scale_shape = 1, scale_rate = 2
  ),
  matrix(probability, p, p), matrix(1L, p, p), iterations, 2000L, 4L, TRUE, "gibbs", 42L, 4L, 0L,
  delta = 1, zratio_spec = spec
)
stopifnot(!any(vapply(raw, function(x) x$error, logical(1))))
indicators = lapply(raw, function(x) t(x$indicator_samples[c(2, 3, 5), , drop = FALSE]))
codes = lapply(indicators, function(x) as.vector(x %*% c(1, 2, 4)))
counts = sapply(codes, function(x) tabulate(x + 1, nbins = 8) / length(x))
estimate = rowMeans(counts)
batches = do.call(cbind, lapply(codes, function(x) {
  vapply(split(x, rep(seq_len(iterations / 1000), each = 1000)), function(z) tabulate(z + 1, nbins = 8) / 1000, numeric(8))
}))
half = qt(1 - .01 / (2 * 22), ncol(batches) - 1) * apply(batches, 1, sd) / sqrt(ncol(batches))
tv = sum(abs(estimate - weights)) / 2
tv_upper = tv + sum(half) / 2 + reference_allowance
expected_pip = as.vector(crossprod(weights, t(vapply(graphs, function(G) G[upper.tri(G)], numeric(3)))))
pip = colMeans(do.call(rbind, indicators))
pip_upper = abs(pip - expected_pip) + sum(half) + reference_allowance
print(c(tv = tv, tv_upper = tv_upper, reference_allowance = reference_allowance))
print(data.frame(expected = expected_pip, estimate = pip, error_upper = pip_upper))
stopifnot(tv_upper <= .005, all(pip_upper <= .01))
