library(bgms)
prior = informed_normal_prior(matrix(.15, 2, 2), matrix(.2, 2, 2))
report = list()
for(prob in c(.1, .3, .5, .4)) {
  values = sapply(1:4, function(seed) {
    fit = sample_ggm_prior(2, 100000, 2000,
      interaction_prior = prior,
      precision_scale_prior = exponential_prior(rate = 2), spec = "hierarchical", delta = .5,
      edge_inclusion_prob = prob, edge_prior = if(prob == .4) beta_bernoulli_prior(2, 3) else NULL,
      seed = seed, update_method = "gibbs", verbose = FALSE
    )
    fit$edge_indicators[, 1]
  })
  batch = unlist(lapply(1:4, function(k) colMeans(matrix(values[, k], 1000))))
  half = qt(1 - .01 / (2 * 4), length(batch) - 1) * sd(batch) / sqrt(length(batch))
  row = c(
    target = prob, estimate = mean(values), half_width = half,
    error_plus_half = abs(mean(values) - prob) + half
  )
  print(row)
  report[[as.character(prob)]] = row
}
saveRDS(report, file.path(tempdir(), "bgms-informed-prior-law.rds"))
stopifnot(all(vapply(report, function(x) x["error_plus_half"] <= 0.005, logical(1))))
