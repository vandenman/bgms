sharing_spec = function() {
  list(
    family = "informed-normal", mean = matrix(.02, 3, 3),
    sd = matrix(.03, 3, 3), rate = 2, delta = 1,
    control = resolve_informed_control()
  )
}

test_that("Share a frozen normalizer exactly once between concurrent clones", {
  spec = sharing_spec()
  graph = matrix(1L, 3, 3)
  shared = informed_normal_shared_cpp(spec, graph, 4L)
  separate = informed_normal_potentials_cpp(spec, list(graph))
  expect_true(all(shared$errors == ""))
  expect_equal(sum(shared$evaluations), 1)
  expect_equal(sum(shared$hits), 3)
  expect_identical(as.numeric(shared$values), rep(as.numeric(separate$log_z), 4))
  spec$mean[, ] = -.2
  different = informed_normal_shared_cpp(spec, graph, 2L)
  expect_false(identical(different$values[1], shared$values[1]))
})

test_that("Release all cache waiters after numerical failure or interruption", {
  spec = sharing_spec()
  graph = matrix(1L, 3, 3)
  interrupted = informed_normal_shared_cpp(spec, graph, 4L, TRUE)
  expect_true(all(grepl("interrupted by test", interrupted$errors)))
  spec$control = resolve_informed_control(list(
    particles_initial = 2L,
    particles_max = 2L, replicas = 2L, log_z_tol = 1e-12
  ))
  failed = informed_normal_shared_cpp(spec, graph, 4L)
  expect_true(all(grepl("precision budget exhausted", failed$errors)))
})

test_that("Preserve chain draws when sharing across serial or parallel execution", {
  spec = sharing_spec()
  run = function(cores) {
    sample_ggm(
      list(
        n = 10L, suf_stat = diag(8, 3),
        pairwise_scale = NA_real_, interaction_prior_type = "informed-normal",
        interaction_mean = spec$mean, interaction_sd = spec$sd,
        scale_prior_type = "exponential", scale_shape = 1, scale_rate = 2
      ),
      matrix(.3, 3, 3), matrix(1L, 3, 3), 20L, 300L, 2L, TRUE,
      "gibbs", 42L, cores, 0L,
      delta = 1, zratio_spec = spec
    )
  }
  serial = run(1L)
  parallel = run(2L)
  for(chain in 1:2) {
    expect_false(serial[[chain]]$error)
    expect_false(parallel[[chain]]$error)
    expect_identical(serial[[chain]]$samples, parallel[[chain]]$samples)
    expect_identical(serial[[chain]]$indicator_samples, parallel[[chain]]$indicator_samples)
  }
  computed = function(raw) sum(vapply(raw, function(x) x$informed_normalizer$unique_graphs, numeric(1)))
  expect_equal(computed(serial), computed(parallel))
  expect_lte(computed(parallel), 7)
})

test_that("Factor heterogeneous stars with the original global delta", {
  for(p in c(3, 5, 10, 20)) {
    spec = sharing_spec()
    spec$mean = matrix(rep(c(-.1, .2, -.3), length.out = p * p), p)
    spec$mean = (spec$mean + t(spec$mean)) / 2
    spec$sd = matrix(.15, p, p)
    spec$sd[1, p] = spec$sd[p, 1] = .4
    graph = matrix(0L, p, p)
    graph[p, seq_len(p - 1)] = graph[seq_len(p - 1), p] = 1L
    expected = sum(vapply(seq_len(p - 1), function(i) {
      log(informed_normal_edge_integral(-2 * spec$mean[i, p], 2 * spec$sd[i, p], c(1, 1), 1)$value)
    }, numeric(1)))
    actual = informed_normal_potentials_cpp(spec, list(graph))
    expect_equal(as.numeric(actual$log_z), expected, tolerance = 1e-8)
    spec$mean = spec$mean[p:1, p:1]
    spec$sd = spec$sd[p:1, p:1]
    reversed = informed_normal_potentials_cpp(spec, list(graph[p:1, p:1]))
    expect_equal(as.numeric(reversed$log_z), expected, tolerance = 1e-8)
  }
})

test_that("Ignore the diagonal entries of both informed factor matrices", {
  M = matrix(.1, 3, 3)
  S = matrix(.15, 3, 3)
  expected = informed_normal_prior(M, S)
  diag(M) = c(NA, Inf, -100)
  diag(S) = c(-10, NaN, 0)
  expect_identical(informed_normal_prior(M, S), expected)
  expect_equal(diag(expected$hyper.parameters$mean), rep(0, 3))
  expect_equal(diag(expected$hyper.parameters$sd), rep(0, 3))
})

test_that("Match scalar row shortcuts to the analytic Gaussian distribution", {
  bank = expand.grid(a = c(.1, 1, 4), sd = c(1e-9, .3, 1e4), beta = c(.2, 1, 3))
  for(i in seq_len(nrow(bank))) {
    x = bank[i, ]
    denominator = 1 + 2 * x$beta * x$sd^2 / x$a
    row = informed_normal_row_cpp(matrix(x$a, 1, 1), .3, x$sd, x$beta)
    expect_equal(row$log_h, -log(denominator) / 2 - x$beta * .3^2 / (x$a + 2 * x$beta * x$sd^2), tolerance = 1e-10)
    expect_equal(as.numeric(row$mean), .3 / denominator, tolerance = 1e-10)
    expect_equal(as.numeric(row$covariance), x$sd^2 / denominator, tolerance = 1e-10)
  }
})
