# ==============================================================================
# Tests for Prior Class Interface
# ==============================================================================
#
# Verifies that all prior constructors work correctly with bgm() and
# bgmCompare() across all model types (ordinal, continuous, mixed,
# blume-capel).
#
# Tests cover:
#   1. Prior constructor output structure
#   2. bgm() runs without error for each prior x model_type combination
#   3. bgmCompare() runs without error for each prior combination
#   4. Backward compatibility of deprecated scalar parameters
# ==============================================================================


# ==============================================================================
# 1. Prior Constructor Tests
# ==============================================================================

test_that("cauchy_prior creates valid prior object", {
  p = cauchy_prior(scale = 2.5)
  expect_s3_class(p, "bgms_interaction_prior")
  expect_equal(p$family, "cauchy")
  expect_equal(p$hyper.parameters$scale, 2.5)
})

test_that("normal_prior creates valid prior object", {
  p = normal_prior(scale = 0.5)
  expect_s3_class(p, "bgms_interaction_prior")
  expect_equal(p$family, "normal")
  expect_equal(p$hyper.parameters$scale, 0.5)
})

test_that("beta_prime_prior creates valid prior object", {
  p = beta_prime_prior(alpha = 1, beta = 1)
  expect_s3_class(p, "bgms_threshold_prior")
  expect_equal(p$family, "beta-prime")
  expect_equal(p$hyper.parameters$alpha, 1)
  expect_equal(p$hyper.parameters$beta, 1)
})

test_that("normal_threshold_prior creates valid prior object", {
  p = normal_threshold_prior(scale = 2)
  expect_s3_class(p, "bgms_threshold_prior")
  expect_equal(p$family, "normal")
  expect_equal(p$hyper.parameters$scale, 2)
})

test_that("bernoulli_prior creates valid prior object", {
  p = bernoulli_prior(0.3)
  expect_s3_class(p, "bgms_edge_prior")
  expect_equal(p$family, "Bernoulli")
  expect_equal(p$hyper.parameters$inclusion_probability, 0.3)
})

test_that("beta_bernoulli_prior creates valid prior object", {
  p = beta_bernoulli_prior(alpha = 2, beta = 5)
  expect_s3_class(p, "bgms_edge_prior")
  expect_equal(p$family, "Beta-Bernoulli")
  expect_equal(p$hyper.parameters$alpha, 2)
  expect_equal(p$hyper.parameters$beta, 5)
})

test_that("sbm_prior creates valid prior object", {
  p = sbm_prior()
  expect_s3_class(p, "bgms_edge_prior")
  expect_equal(p$family, "Stochastic-Block")
})


# ==============================================================================
# 2. bgm() with Prior Objects — Ordinal
# ==============================================================================

test_that("bgm ordinal works with cauchy_prior", {
  data("Wenchuan", package = "bgms")
  fit = bgm(Wenchuan[1:50, 1:4],
    interaction_prior = cauchy_prior(scale = 2.5),
    iter = 25, warmup = 50, chains = 1,
    display_progress = "none")
  expect_s3_class(fit, "bgms")
  expect_true(all(is.finite(do.call(rbind, fit$raw_samples$pairwise))))
})

test_that("bgm ordinal works with normal_prior", {
  data("Wenchuan", package = "bgms")
  fit = bgm(Wenchuan[1:50, 1:4],
    interaction_prior = normal_prior(scale = 1),
    iter = 25, warmup = 50, chains = 1,
    display_progress = "none")
  expect_s3_class(fit, "bgms")
  expect_true(all(is.finite(do.call(rbind, fit$raw_samples$pairwise))))
})

test_that("bgm ordinal works with normal_threshold_prior", {
  data("Wenchuan", package = "bgms")
  fit = bgm(Wenchuan[1:50, 1:4],
    threshold_prior = normal_threshold_prior(scale = 1),
    iter = 25, warmup = 50, chains = 1,
    display_progress = "none")
  expect_s3_class(fit, "bgms")
  expect_true(all(is.finite(do.call(rbind, fit$raw_samples$main))))
})

test_that("bgm ordinal works with both normal priors", {
  data("Wenchuan", package = "bgms")
  fit = bgm(Wenchuan[1:50, 1:4],
    interaction_prior = normal_prior(scale = 0.5),
    threshold_prior = normal_threshold_prior(scale = 0.5),
    iter = 25, warmup = 50, chains = 1,
    display_progress = "none")
  expect_s3_class(fit, "bgms")
})

test_that("bgm ordinal works with normal priors + edge selection", {
  data("Wenchuan", package = "bgms")
  fit = bgm(Wenchuan[1:50, 1:4],
    interaction_prior = normal_prior(scale = 0.5),
    threshold_prior = normal_threshold_prior(scale = 0.5),
    edge_prior = bernoulli_prior(0.5),
    iter = 25, warmup = 50, chains = 1,
    display_progress = "none")
  expect_s3_class(fit, "bgms")
})

test_that("bgm ordinal works with normal priors + beta_bernoulli edge prior", {
  data("Wenchuan", package = "bgms")
  fit = bgm(Wenchuan[1:50, 1:4],
    interaction_prior = normal_prior(scale = 0.5),
    threshold_prior = normal_threshold_prior(scale = 0.5),
    edge_prior = beta_bernoulli_prior(1, 1),
    iter = 25, warmup = 50, chains = 1,
    display_progress = "none")
  expect_s3_class(fit, "bgms")
})


# ==============================================================================
# 3. bgm() with Prior Objects — GGM (Continuous)
# ==============================================================================

test_that("bgm ggm works with normal_prior", {
  set.seed(42)
  Y = as.data.frame(matrix(rnorm(200), nrow = 50, ncol = 4))
  fit = bgm(Y,
    variable_type = "continuous",
    interaction_prior = normal_prior(scale = 1),
    iter = 25, warmup = 50, chains = 1,
    display_progress = "none")
  expect_s3_class(fit, "bgms")
  expect_true(all(is.finite(do.call(rbind, fit$raw_samples$pairwise))))
})

test_that("bgm ggm works with cauchy_prior", {
  set.seed(42)
  Y = as.data.frame(matrix(rnorm(200), nrow = 50, ncol = 4))
  fit = bgm(Y,
    variable_type = "continuous",
    interaction_prior = cauchy_prior(scale = 2.5),
    iter = 25, warmup = 50, chains = 1,
    display_progress = "none")
  expect_s3_class(fit, "bgms")
})


# ==============================================================================
# 4. bgm() with Prior Objects — Mixed MRF
# ==============================================================================

test_that("bgm mixed works with default priors", {
  set.seed(42)
  data("Wenchuan", package = "bgms")
  dat = data.frame(Wenchuan[1:100, 1:4], V5 = rnorm(100), V6 = rnorm(100))
  fit = bgm(dat,
    variable_type = c(rep("ordinal", 4), rep("continuous", 2)),
    iter = 25, warmup = 100, chains = 1,
    display_progress = "none")
  expect_s3_class(fit, "bgms")
  expect_true(all(is.finite(do.call(rbind, fit$raw_samples$pairwise))))
})

test_that("bgm mixed works with normal_prior", {
  set.seed(42)
  data("Wenchuan", package = "bgms")
  dat = data.frame(Wenchuan[1:100, 1:4], V5 = rnorm(100), V6 = rnorm(100))
  fit = bgm(dat,
    variable_type = c(rep("ordinal", 4), rep("continuous", 2)),
    interaction_prior = normal_prior(scale = 0.5),
    iter = 25, warmup = 100, chains = 1,
    display_progress = "none")
  expect_s3_class(fit, "bgms")
  expect_true(all(is.finite(do.call(rbind, fit$raw_samples$pairwise))))
})

test_that("bgm mixed works with normal_threshold_prior", {
  set.seed(42)
  data("Wenchuan", package = "bgms")
  dat = data.frame(Wenchuan[1:100, 1:4], V5 = rnorm(100), V6 = rnorm(100))
  fit = bgm(dat,
    variable_type = c(rep("ordinal", 4), rep("continuous", 2)),
    threshold_prior = normal_threshold_prior(scale = 1),
    iter = 25, warmup = 100, chains = 1,
    display_progress = "none")
  expect_s3_class(fit, "bgms")
  expect_true(all(is.finite(do.call(rbind, fit$raw_samples$main))))
})

test_that("bgm mixed works with both normal priors", {
  set.seed(42)
  data("Wenchuan", package = "bgms")
  dat = data.frame(Wenchuan[1:100, 1:4], V5 = rnorm(100), V6 = rnorm(100))
  fit = bgm(dat,
    variable_type = c(rep("ordinal", 4), rep("continuous", 2)),
    interaction_prior = normal_prior(scale = 0.5),
    threshold_prior = normal_threshold_prior(scale = 0.5),
    iter = 25, warmup = 100, chains = 1,
    display_progress = "none")
  expect_s3_class(fit, "bgms")
})


# ==============================================================================
# 5. bgm() with Prior Objects — Blume-Capel
# ==============================================================================

test_that("bgm blume-capel works with normal priors", {
  data("Wenchuan", package = "bgms")
  fit = bgm(Wenchuan[1:50, 1:4],
    variable_type = "blume-capel",
    baseline_category = 2,
    interaction_prior = normal_prior(scale = 0.5),
    threshold_prior = normal_threshold_prior(scale = 0.5),
    iter = 25, warmup = 50, chains = 1,
    display_progress = "none")
  expect_s3_class(fit, "bgms")
  main_names = fit$raw_samples$parameter_names$main
  expect_true(any(grepl("linear", main_names)))
  expect_true(any(grepl("quadratic", main_names)))
})


# ==============================================================================
# 6. bgmCompare() with Prior Objects
# ==============================================================================

test_that("bgmCompare works with normal priors", {
  data("Wenchuan", package = "bgms")
  x = Wenchuan[1:25, 1:4]
  y = Wenchuan[26:50, 1:4]
  fit = bgmCompare(x = x, y = y,
    interaction_prior = normal_prior(scale = 0.5),
    threshold_prior = normal_threshold_prior(scale = 0.5),
    difference_selection = FALSE,
    iter = 25, warmup = 100, chains = 1,
    update_method = "adaptive-metropolis",
    display_progress = "none")
  expect_s3_class(fit, "bgmCompare")
})

test_that("bgmCompare works with normal priors + difference selection", {
  data("Wenchuan", package = "bgms")
  x = Wenchuan[1:25, 1:4]
  y = Wenchuan[26:50, 1:4]
  fit = bgmCompare(x = x, y = y,
    interaction_prior = normal_prior(scale = 0.5),
    threshold_prior = normal_threshold_prior(scale = 0.5),
    difference_selection = TRUE,
    iter = 25, warmup = 100, chains = 1,
    update_method = "adaptive-metropolis",
    display_progress = "none")
  expect_s3_class(fit, "bgmCompare")
})


# ==============================================================================
# 7. Backward Compatibility — Deprecated Scalar Parameters
# ==============================================================================

test_that("deprecated pairwise_scale still works", {
  data("Wenchuan", package = "bgms")
  expect_warning(
    fit <- bgm(Wenchuan[1:50, 1:4],
      pairwise_scale = 2.5,
      iter = 25, warmup = 50, chains = 1,
      display_progress = "none"),
    "pairwise_scale"
  )
  expect_s3_class(fit, "bgms")
})

test_that("deprecated main_alpha/main_beta still works", {
  data("Wenchuan", package = "bgms")
  expect_warning(
    fit <- bgm(Wenchuan[1:50, 1:4],
      main_alpha = 1, main_beta = 1,
      iter = 25, warmup = 50, chains = 1,
      display_progress = "none"),
    "main_alpha"
  )
  expect_s3_class(fit, "bgms")
})


# ==============================================================================
# 8. Prior Object Stored in Fit
# ==============================================================================

test_that("prior info is stored in .bgm_spec", {
  data("Wenchuan", package = "bgms")
  fit = bgm(Wenchuan[1:50, 1:4],
    interaction_prior = normal_prior(scale = 0.5),
    threshold_prior = normal_threshold_prior(scale = 0.5),
    iter = 25, warmup = 50, chains = 1,
    display_progress = "none")
  spec = fit$.bgm_spec
  expect_equal(spec$prior$interaction_prior_type, "normal")
  expect_equal(spec$prior$pairwise_scale, 0.5)
  expect_equal(spec$prior$threshold_prior_type, "normal")
  expect_equal(spec$prior$threshold_scale, 0.5)
})
