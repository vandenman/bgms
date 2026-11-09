test_that("Validate informed matrices and numerical controls", {
  M = matrix(0.2, 3, 3)
  S = matrix(0.5, 3, 3)
  expect_s3_class(informed_normal_prior(M, S), "bgms_parameter_prior")
  expect_error(informed_normal_prior(M, -S), "positive")
  M[1, 2] = 1
  expect_error(informed_normal_prior(M, S), "symmetric")
  expect_error(resolve_informed_control(list(replicas = 1)), "replicas")
  expect_error(resolve_informed_control(list(unknown = 1)), "supported")
})

test_that("Evaluate indexed interaction densities and gradients", {
  M = matrix(0.2, 2, 2)
  S = matrix(0.3, 2, 2)
  result = informed_normal_density_cpp(M, S, 0.7, 0L, 1L)
  expect_equal(result[1], dnorm(0.7, 0.2, 0.3, log = TRUE))
  expect_equal(result[2], -(0.7 - 0.2) / 0.3^2)
})

test_that("Freeze reproducible graph potentials and match independent quadrature", {
  M = matrix(0.1, 2, 2)
  S = matrix(0.15, 2, 2)
  spec = list(
    family = "informed-normal", mean = M, sd = S, rate = 2,
    delta = 0.5, control = resolve_informed_control()
  )
  empty = matrix(0L, 2, 2)
  edge = matrix(c(0L, 1L, 1L, 0L), 2)
  a = informed_normal_potentials_cpp(spec, list(empty, edge, edge))
  b = informed_normal_potentials_cpp(spec, list(edge))
  reference = informed_normal_edge_integral(-0.2, 0.3, c(1, 1), 0.5)
  expect_identical(as.numeric(a$log_z[1]), 0)
  expect_identical(a$log_z[2], a$log_z[3])
  expect_identical(a$log_z[2], b$log_z[1])
  expect_lt(abs(a$log_z[2] - log(reference$value)), 0.003)
  expect_equal(a$hits, 1)
  expect_lte(a$half_width, 0.001)
})

test_that("Sample informed fixed graphs and reject unsupported scope", {
  prior = informed_normal_prior(matrix(0.1, 2, 2), matrix(0.2, 2, 2))
  fit = sample_ggm_prior(2, 50, 20,
    interaction_prior = prior,
    precision_scale_prior = exponential_prior(rate = 2),
    update_method = "gibbs", verbose = FALSE
  )
  expect_true(all(is.finite(fit$K_diag)))
  expect_true(all(fit$K_diag[, 1] * fit$K_diag[, 2] > fit$K_offdiag[, 1]^2))
  expect_error(sample_ggm_prior(2, 10,
    interaction_prior = prior,
    precision_scale_prior = exponential_prior(rate = 2), spec = "joint"
  ), "hierarchical")
})

test_that("Stop the whole fit on a production normalizer failure", {
  local_mocked_bindings(run_sampler_ggm = function(spec) {
    list(
      list(error = FALSE, userInterrupt = FALSE),
      list(
        error = TRUE, fatal_normalizer = TRUE,
        error_msg = "Informed normalizer failure: budget exhausted"
      )
    )
  })
  expect_error(
    run_sampler(structure(list(model_type = "ggm"), class = "bgm_spec")),
    "budget exhausted"
  )
})

test_that("Align both prior axes to data names", {
  M = matrix(c(0, .1, .2, .1, 0, .3, .2, .3, 0), 3,
    dimnames = list(c("c", "a", "b"), c("c", "a", "b"))
  )
  S = matrix(.2, 3, 3, dimnames = dimnames(M))
  spec = list(model_type = "ggm", data = list(
    num_variables = 3L,
    data_columnnames = c("a", "b", "c")
  ), prior = list(
    scale_shape = 1,
    scale_eta = NA_real_, edge_selection = FALSE
  ))
  result = attach_informed_prior(spec, informed_normal_prior(M, S))
  expect_identical(result$prior$interaction_mean, M[c("a", "b", "c"), c("a", "b", "c")])
})

test_that("Preserve homogeneous Gaussian Gibbs draws", {
  input = list(
    n = 0L, suf_stat = matrix(0, 3, 3), pairwise_scale = 0.3,
    interaction_prior_type = "normal", scale_prior_type = "exponential",
    scale_shape = 1, scale_rate = 2
  )
  run = function(input) {
    sample_ggm(input, matrix(.5, 3, 3), matrix(1L, 3, 3),
      100L, 20L, 1L, FALSE, "gibbs", 7L, 1L, 0L,
      delta = 0.5
    )
  }
  centered = run(input)
  input$interaction_prior_type = "informed-normal"
  input$interaction_mean = matrix(0, 3, 3)
  input$interaction_sd = matrix(.3, 3, 3)
  informed = run(input)
  expect_false(informed[[1]]$error)
  expect_equal(informed[[1]]$samples, centered[[1]]$samples, tolerance = 1e-12)
})

test_that("Fail explicitly when connected-graph precision cannot be resolved", {
  spec = list(
    family = "informed-normal", mean = matrix(.1, 3, 3),
    sd = matrix(.3, 3, 3), rate = 2, delta = 0,
    control = resolve_informed_control(list(
      particles_initial = 2L,
      particles_max = 2L, replicas = 2L, log_z_tol = 1e-12
    ))
  )
  expect_error(
    informed_normal_potentials_cpp(spec, list(matrix(1L, 3, 3))),
    "precision budget exhausted"
  )
})

test_that("Match the compiled isolated-edge backend to positive quadrature", {
  bank = expand.grid(
    delta = c(0, .5, 2), m = c(0, .25, -.25, 1, -1),
    tau = c(.1, .5, 1)
  )
  graph = matrix(c(0L, 1L, 1L, 0L), 2)
  for(k in seq_len(nrow(bank))) {
    x = bank[k, ]
    spec = list(
      family = "informed-normal", mean = matrix(-x$m / 2, 2, 2),
      sd = matrix(x$tau / 2, 2, 2), rate = 2, delta = x$delta,
      control = resolve_informed_control()
    )
    actual = informed_normal_potentials_cpp(spec, list(graph))$log_z[1]
    expected = log(informed_normal_edge_integral(x$m, x$tau, c(1, 1), x$delta)$value)
    expect_equal(actual, expected, tolerance = 1e-8)
  }
})

test_that("Factor disconnected components with the original determinant tilt", {
  spec = list(
    family = "informed-normal", mean = matrix(.1, 4, 4),
    sd = matrix(.2, 4, 4), rate = 2, delta = log(4) / 2,
    control = resolve_informed_control()
  )
  graph = matrix(0L, 4, 4)
  graph[1, 2] = graph[2, 1] = graph[3, 4] = graph[4, 3] = 1L
  result = informed_normal_potentials_cpp(spec, list(graph))
  expected = 2 * log(informed_normal_edge_integral(-.2, .4, c(1, 1), spec$delta)$value)
  expect_equal(result$log_z[1], expected, tolerance = 1e-8)
})

test_that("Return informed public fit diagnostics through the S7 contract", {
  set.seed(5)
  x = matrix(rnorm(80), 40, 2)
  prior = informed_normal_prior(matrix(.1, 2, 2), matrix(.2, 2, 2))
  fit = bgm(x,
    variable_type = "continuous", interaction_prior = prior,
    precision_scale_prior = exponential_prior(rate = 2), update_method = "gibbs",
    iter = 100, warmup = 300, chains = 2, cores = 2,
    display_progress = FALSE, verbose = FALSE
  )
  expect_length(fit$normalizer_diagnostics, 2L)
  expect_identical(fit$normalizer_diagnostics[[1]]$method, "smc")
  expect_null(fit$zratio_diag)
  expect_equal(fit$.bgm_spec$prior$interaction_mean, prior$hyper.parameters$mean)
})

test_that("Reject unsupported informed prior roles and scale parameterizations", {
  prior = informed_normal_prior(matrix(.1, 2, 2), matrix(.2, 2, 2))
  expect_error(sample_ggm_prior(2, 10, interaction_prior = prior), "rate")
  expect_error(sample_ggm_prior(2, 10, interaction_prior = prior,
    precision_scale_prior = gamma_prior(shape = 2, rate = 1)), "exponential_prior")
  expect_error(sample_ggm_prior(2, 10, normalizer_control = list()), "informed")
  x = matrix(seq_len(40), 20, 2)
  expect_error(bgm(x, means_prior = prior), "only for GGM interactions")
  expect_error(bgm(x, threshold_prior = prior), "only for GGM interactions")
  expect_error(bgm(x, variable_type = "ordinal", interaction_prior = prior), "pure GGM")
  expect_error(bgm(x, variable_type = "continuous", interaction_prior = prior,
    precision_scale_prior = exponential_prior(rate = 1),
    edge_selection = FALSE, normalizer_control = list()), "graph selection")
})

test_that("Ignore unused prior diagonals without clipping off-diagonal means", {
  M = matrix(1e6, 2, 2)
  S = matrix(.01, 2, 2)
  diag(M) = NA_real_
  diag(S) = Inf
  prior = informed_normal_prior(M, S)
  expect_equal(diag(prior$hyper.parameters$mean), c(0, 0))
  expect_equal(diag(prior$hyper.parameters$sd), c(0, 0))
  expect_identical(prior$hyper.parameters$mean[1, 2], 1e6)
})

test_that("Fail before sampling when Gaussian factors overflow double precision", {
  expect_error(informed_normal_density_cpp(matrix(1e308, 2, 2),
    matrix(.1, 2, 2), 0, 0L, 1L), "cannot be represented")
  expect_error(informed_normal_density_cpp(matrix(0, 2, 2),
    matrix(1e-200, 2, 2), 0, 0L, 1L), "cannot be represented")
  expect_identical(resolve_informed_control(list()), resolve_informed_control())
})

test_that("Account for total precision across nontrivial disconnected components", {
  controls = resolve_informed_control(list(particles_initial = 16L,
    particles_max = 16L, replicas = 4L, log_z_tol = .1))
  spec = list(family = "informed-normal", mean = matrix(.001, 6, 6),
    sd = matrix(.01, 6, 6), rate = 2, delta = log(6) / 2, control = controls)
  graph = matrix(0L, 6, 6)
  graph[1:3, 1:3] = graph[4:6, 4:6] = 1L
  diag(graph) = 0L
  both = informed_normal_potentials_cpp(spec, list(graph))
  spec$mean = spec$mean[1:3, 1:3]
  spec$sd = spec$sd[1:3, 1:3]
  spec$control$log_z_tol = controls$log_z_tol / 2
  spec$control$nominal_confidence = 1 - (1 - controls$nominal_confidence) / 2
  one = informed_normal_potentials_cpp(spec, list(graph[1:3, 1:3]))
  expect_equal(both$log_z[1], 2 * one$log_z[1], tolerance = 1e-12)
  expect_equal(both$half_width, 2 * one$half_width, tolerance = 1e-12)
  expect_lte(both$half_width, controls$log_z_tol)
})
