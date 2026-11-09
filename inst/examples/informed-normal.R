library(bgms)

set.seed(42)
x = matrix(rnorm(300), ncol = 3)
colnames(x) = c("A", "B", "C")
M = matrix(.1, 3, 3, dimnames = list(colnames(x), colnames(x)))
S = matrix(.15, 3, 3, dimnames = dimnames(M))
# Only off-diagonal entries specify interaction factors, on theta = -K / 2.
diag(M) = diag(S) = 0

fit_informed = function(data, M, S, iter = 1000, warmup = 500,
                        chains = 2, cores = 2, seed = 42) {
  bgm(data,
    variable_type = "continuous",
    interaction_prior = informed_normal_prior(M, S),
    precision_scale_prior = exponential_prior(rate = 2),
    precision_graph_prior = "hierarchical",
    delta = 0, update_method = "gibbs",
    iter = iter, warmup = warmup, chains = chains, cores = cores, seed = seed
  )
}

elapsed = system.time(fit <- fit_informed(x, M, S))
print(elapsed)
print(fit$posterior_mean_indicator)
print(fit$posterior_mean_pairwise)
print(fit$normalizer_diagnostics)

# For another fit, change both symmetric entries and call fit_informed(x, M2, S2).
M2 = M
S2 = S
M2["A", "B"] = M2["B", "A"] = -.2
S2["A", "B"] = S2["B", "A"] = .05
