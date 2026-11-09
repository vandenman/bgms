test_that("Check connected graph ratios against an independent full-row reference", {
  skip_if(Sys.getenv("BGMS_RUN_SLOW_TESTS") != "true", "Slow informed normalizer assessment")
  prior = informed_normal_prior(matrix(.1, 3, 3), matrix(.15, 3, 3))
  spec = list(
    family = "informed-normal", mean = prior$hyper.parameters$mean,
    sd = prior$hyper.parameters$sd, rate = 2, delta = 0,
    control = resolve_informed_control()
  )
  path = matrix(c(0L, 1L, 1L, 1L, 0L, 0L, 1L, 0L, 0L), 3)
  complete = matrix(1L, 3, 3)
  diag(complete) = 0L
  production = informed_normal_potentials_cpp(spec, list(path, complete))$log_z
  reference = informed_normal_reference(path, spec$mean, spec$sd, 2, 0, 2, 3,
    draws = 500000L, warmup = 5000L, chains = 4L
  )
  expect_lte(reference$half_width, 0.0005)
  expect_lte(abs(production[1] - production[2] - reference$log_ratio) +
    reference$half_width, 0.003)
})
