library(bgms)
bank = expand.grid(sd = c(.1, 1e-8, 1e-9), probability = c(.1, .3, .5, .9))
results = lapply(seq_len(nrow(bank)), function(k) {
  x = bank[k, ]
  values = sapply(1:4, function(seed) {
    fit = sample_ggm_prior(2, 200000, 2000,
      interaction_prior = informed_normal_prior(matrix(.5, 2, 2), matrix(x$sd, 2, 2)),
      precision_scale_prior = exponential_prior(rate = 2), spec = "hierarchical", delta = 0,
      edge_inclusion_prob = x$probability, seed = seed, update_method = "gibbs", verbose = FALSE
    )
    fit$edge_indicators[, 1]
  })
  batches = unlist(lapply(1:4, function(k) colMeans(matrix(values[, k], 1000))))
  half = qt(1 - .01 / (2 * nrow(bank)), length(batches) - 1) * sd(batches) / sqrt(length(batches))
  result = cbind(x,
    estimate = mean(values), half_width = half,
    error_plus_half = abs(mean(values) - x$probability) + half
  )
  print(result)
  result
})
results = do.call(rbind, results)
stopifnot(all(results$error_plus_half <= .005))
