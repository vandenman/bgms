test_that("Integrate Gaussian edge masses without cancelling spike ordinates", {
  bank = expand.grid(
    a = c(.2, 8 / 3), b = c(-1.1, 0, .7),
    mean = c(-1, 0, .6), sd = c(.3, 1e-7, 1e-9, 2^-36)
  )
  for(k in seq_len(nrow(bank))) {
    x = bank[k, ]
    expected = log(integrate(function(z) {
      value = x$mean + x$sd * z
      exp(-x$a * value^2 / 2 + x$b * value) * dnorm(z)
    }, -Inf, Inf, rel.tol = 1e-11)$value)
    actual = informed_normal_mass_cpp(x$a, x$b, x$mean, x$sd^2)
    expect_equal(actual, expected, tolerance = 1e-10)
    for(prob in c(.1, .3, .5, .9)) {
      expect_equal(qlogis(prob) + actual, qlogis(prob) + expected, tolerance = 1e-10)
    }
  }
  expect_equal(informed_normal_mass_cpp(8 / 3, 0, -1, 4e-18), -4 / 3, tolerance = 1e-12)
})

test_that("Never drop a Gaussian row direction when scales are ill-conditioned", {
  for(scales in list(c(.8, .1), c(1e16, .1))) {
    result = tryCatch(informed_normal_row_cpp(diag(2), c(.1, .1), scales, 1), error = identity)
    if(inherits(result, "error")) {
      expect_match(conditionMessage(result), "Informed normalizer failure")
      expect_gt(max(scales), 1e15)
    } else {
      denominator = 1 + 2 * scales^2
      expected = sum(-log1p(2 * scales^2) / 2 - .1^2 / denominator)
      expect_equal(result$log_h, expected, tolerance = 1e-10)
      expect_equal(as.numeric(result$mean), .1 / denominator, tolerance = 1e-10)
      expect_equal(diag(result$covariance), scales^2 / denominator, tolerance = 1e-10)
    }
  }
  M = matrix(-.05, 3, 3)
  S = matrix(.05, 3, 3)
  S[1, 3] = S[3, 1] = 5e15
  G = matrix(0L, 3, 3)
  G[1, 3] = G[3, 1] = G[2, 3] = G[3, 2] = 1L
  spec = list(
    family = "informed-normal", mean = M, sd = S, rate = 2, delta = 0,
    control = resolve_informed_control(list(
      particles_initial = 4096L, particles_max = 4096L,
      replicas = 32L, rejuvenation_sweeps = 0L, log_z_tol = .01
    ))
  )
  result = tryCatch(informed_normal_potentials_cpp(spec, list(G)), error = identity)
  if(inherits(result, "error")) {
    expect_match(conditionMessage(result), "Informed normalizer failure")
  } else {
    expect_lt(abs(result$log_z[1] - (-37.37585916)), .01)
  }
})

test_that("Preserve the noncentral reference penalty as slab SD vanishes", {
  for(s in c(.1, 2^-20, 2^-36)) {
    expected = log(informed_normal_edge_integral(-1, 2 * s, c(1, 1), 0)$value)
    reference = informed_normal_reference(matrix(0L, 2, 2), matrix(.5, 2, 2),
      matrix(s, 2, 2), 2, 0, 1, 2,
      draws = 10000L, warmup = 0L, chains = 4L
    )
    expect_gt(reference$log_ratio, 1)
    expect_gt(reference$mcse, 0)
    expect_lt(abs(reference$log_ratio + expected), 5 * reference$mcse)
    if(reference$status == "resolved") expect_lte(abs(reference$log_ratio + expected), .0005)
    expect_equal(informed_reference_log_ratio_R(matrix(1, 1, 1), integer(), 1, -1, 2 * s, 1),
      -log1p(8 * s * s) / 2 - 1 / (1 + 8 * s * s),
      tolerance = 1e-12
    )
  }
  expect_identical(summarize_informed_reference(matrix(NaN, 100, 2), .99, 1)$status, "inconclusive")
  expect_identical(summarize_informed_reference(matrix(0, 100, 2), .99, 1)$status, "inconclusive")
})

test_that("Align named informed arrays independently against their context", {
  labels = c("a", "b", "c")
  M = matrix(c(0, .02, -.03, .02, 0, .04, -.03, .04, 0), 3, dimnames = list(labels, labels))
  S = matrix(c(0, .1, .5, .1, 0, 1, .5, 1, 0), 3, dimnames = list(labels, labels))
  G = matrix(0L, 3, 3, dimnames = list(labels, labels))
  perm = c(3, 2, 1)
  baseline = informed_normal_reference(G, M, S, 2, 0, 1, 2, draws = 100L, warmup = 0L, chains = 2L)
  for(mean_named in c(FALSE, TRUE)) {
    for(sd_named in c(FALSE, TRUE)) {
      mean = if(mean_named) M[perm, perm] else unname(M)
      sd = if(sd_named) S[perm, perm] else unname(S)
      aligned = align_informed_matrices(informed_normal_prior(mean, sd), 3, labels)
      expect_equal(unname(aligned$mean), unname(M))
      expect_equal(unname(aligned$sd), unname(S))
      expect_identical(informed_normal_reference(G, mean, sd, 2, 0, 1, 2,
        draws = 100L, warmup = 0L, chains = 2L
      ), baseline)
    }
  }
  context = informed_graph_context(G[perm, perm])
  expect_equal(align_informed_matrices(informed_normal_prior(M, S), 3, context$labels)$sd, S[perm, perm])
  fit = sample_ggm_prior(3, 20, 0,
    interaction_prior = informed_normal_prior(unname(M), S),
    precision_scale_prior = exponential_prior(rate = 2), verbose = FALSE, update_method = "gibbs"
  )
  expect_equal(fit$informed_prior$interaction_sd, S)
})

test_that("Propagate actual C++ normalizer failures through the public fit", {
  set.seed(3)
  expect_error(
    bgm(matrix(rnorm(60), 20, 3),
      variable_type = "continuous",
      interaction_prior = informed_normal_prior(matrix(.1, 3, 3), matrix(.3, 3, 3)),
      precision_scale_prior = exponential_prior(rate = 2), delta = 0, update_method = "gibbs",
      iter = 5, warmup = 300, chains = 2, cores = 2, display_progress = FALSE, verbose = FALSE,
      normalizer_control = list(particles_initial = 2L, particles_max = 2L, replicas = 2L, log_z_tol = 1e-12)
    ),
    "Informed normalizer failure"
  )
})

test_that("Differentiate informed priors on every constrained four-vertex graph", {
  set.seed(91)
  M = matrix(0, 4, 4)
  S = matrix(0, 4, 4)
  M[upper.tri(M)] = c(-.4, .1, .3, -.2, .5, -.1)
  S[upper.tri(S)] = c(.2, .7, .4, .1, 1, .3)
  M = M + t(M)
  S = S + t(S)
  for(code in 0:63) {
    graph = matrix(0L, 4, 4)
    graph[upper.tri(graph)] = as.integer(intToBits(code))[1:6]
    graph = graph + t(graph)
    for(shape in c(1, 2, 4)) {
      theta = rnorm(4 + sum(graph) / 2, sd = .1)
      evaluate = function(x) informed_normal_gradient_cpp(x, diag(4), 5L, graph, M, S, shape, 2)
      result = evaluate(theta)
      numeric = vapply(seq_along(theta), function(k) {
        plus = minus = theta
        plus[k] = plus[k] + 1e-6
        minus[k] = minus[k] - 1e-6
        (evaluate(plus)$value - evaluate(minus)$value) / 2e-6
      }, numeric(1))
      expect_lt(max(abs(numeric - result$gradient) / pmax(1, abs(numeric))), 1e-5)
    }
  }
})

test_that("Do not keep a successful chain beside a real fatal C++ result", {
  run = function(p, control) {
    M = matrix(.1, p, p)
    S = matrix(.3, p, p)
    sample_ggm(
      list(
        n = 0L, suf_stat = matrix(0, p, p), pairwise_scale = NA_real_,
        interaction_prior_type = "informed-normal", interaction_mean = M, interaction_sd = S,
        scale_prior_type = "exponential", scale_shape = 1, scale_rate = 2
      ),
      matrix(.5, p, p), matrix(1L, p, p), 5L, 300L, 1L, TRUE, "gibbs", 42L, 2L, 0L,
      delta = 0, zratio_spec = list(
        family = "informed-normal", mean = M, sd = S, rate = 2, delta = 0,
        control = resolve_informed_control(control)
      )
    )[[1]]
  }
  success = run(2, NULL)
  failure = run(3, list(particles_initial = 2L, particles_max = 2L, replicas = 2L, log_z_tol = 1e-12))
  expect_false(success$error)
  expect_true(failure$fatal_normalizer)
  local_mocked_bindings(run_sampler_ggm = function(spec) list(success, failure))
  expect_error(
    run_sampler(structure(list(model_type = "ggm"), class = "bgm_spec")),
    "Informed normalizer failure"
  )
})

test_that("Keep the independent R reference stable at narrow scales", {
  reference = informed_normal_reference_R(matrix(0L, 2, 2), matrix(.5, 2, 2),
    matrix(2^-36, 2, 2), 2, 0, 1, 2,
    draws = 200L, warmup = 0L, chains = 2L
  )
  expect_gt(reference$log_ratio, 1)
  expect_gt(reference$mcse, 0)
  expect_identical(reference$status, "inconclusive")
  set.seed(23)
  for(k in 1:12) {
    X = matrix(rnorm(9), 3)
    C = crossprod(X) + diag(3)
    m = c(-.3, .4, -.2)
    s = c(.1, .6, .2)
    beta = .8
    Q = diag(1 / s[1:2]^2) + 2 * beta * C[1:2, 1:2]
    q = 2 * beta * C[3, 3] - 4 * beta^2 * drop(crossprod(C[1:2, 3], solve(Q, C[1:2, 3])))
    shift = 2 * beta * sum(C[1:2, 3] * solve(Q, m[1:2] / s[1:2]^2))
    expected = log(integrate(function(z) {
      x = m[3] + s[3] * z
      exp(-q * x^2 / 2 - shift * x) * dnorm(z)
    }, -Inf, Inf, rel.tol = 1e-11)$value)
    expect_equal(informed_reference_log_ratio_R(C, 1:2, 3, m, s, beta), expected, tolerance = 1e-10)
  }
})
