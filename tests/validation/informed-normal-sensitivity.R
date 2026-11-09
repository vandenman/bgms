library(bgms)

settings = expand.grid(mean = c(-.5, 0, .5), sd = c(.1, .3, 1), n = c(0L, 4L))
probability = .3
results = vector("list", nrow(settings))
for(k in seq_len(nrow(settings))) {
  setting = settings[k, ]
  T = if(setting$n == 0L) matrix(0, 2, 2) else matrix(c(4, 2.4, 2.4, 4), 2)
  M = matrix(setting$mean, 2, 2)
  S = matrix(setting$sd, 2, 2)
  normalizer = list(
    family = "informed-normal", mean = M, sd = S, rate = 2,
    delta = .5, control = bgms:::resolve_informed_control()
  )
  m = -2 * setting$mean
  tau = 2 * setting$sd
  log_prior = log(bgms:::informed_normal_edge_integral(m, tau, c(1, 1), .5)$value)
  log_posterior = log(bgms:::informed_normal_edge_integral(
    m - tau^2 * T[1, 2], tau, 1 + diag(T) / 2, .5 + setting$n / 2
  )$value)
  expected = plogis(qlogis(probability) - m * T[1, 2] +
    tau^2 * T[1, 2]^2 / 2 + log_posterior - log_prior)
  chains = bgms:::sample_ggm(
    list(
      n = as.integer(setting$n), suf_stat = T, pairwise_scale = NA_real_,
      interaction_prior_type = "informed-normal", interaction_mean = M,
      interaction_sd = S, scale_prior_type = "exponential", scale_shape = 1, scale_rate = 2
    ),
    matrix(probability, 2, 2), matrix(1L, 2, 2), 50000L, 2000L, 4L,
    TRUE, "gibbs", 152L, 4L, 0L,
    delta = .5, zratio_spec = normalizer
  )
  stopifnot(!any(vapply(chains, function(x) x$error, logical(1))))
  indicators = sapply(chains, function(x) x$indicator_samples[2, ])
  theta = sapply(chains, function(x) -.5 * x$samples[2, ])
  batch = unlist(lapply(seq_len(ncol(indicators)), function(j) {
    colMeans(matrix(indicators[, j], 500))
  }))
  half = qt(1 - .01 / (2 * nrow(settings)), length(batch) - 1) *
    sd(batch) / sqrt(length(batch))
  active = theta[indicators == 1]
  results[[k]] = data.frame(setting,
    pip_reference = expected,
    pip_estimate = mean(indicators), pip_half_width = half,
    included_theta_mean = mean(active), included_theta_sd = sd(active),
    included_positive = mean(active > 0), overall_theta_mean = mean(theta)
  )
}
results = do.call(rbind, results)
print(results, digits = 5, row.names = FALSE)
stopifnot(all(abs(results$pip_estimate - results$pip_reference) +
  results$pip_half_width <= .01))
saveRDS(results, file.path(tempdir(), "bgms-informed-sensitivity.rds"))
