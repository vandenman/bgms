# --------------------------------------------------------------------------- #
# Tests for the C++ MCMC diagnostics: .compute_ess_cpp and .compute_rhat_cpp.
#
# These replace coda::effectiveSize and coda::gelman.diag. The tests verify
# correctness against coda on well-behaved input and safe behavior on
# pathological input (constant chains, NaN, Inf, very short chains, etc.).
# --------------------------------------------------------------------------- #

# Helper: build a 3D array [niter x nchains x nparam] from a matrix or vector.
make_array = function(x, niter, nchains, nparam = 1L) {
  array(x, dim = c(niter, nchains, nparam))
}


# ---- Concordance with coda ------------------------------------------------- #

test_that("ESS matches coda::effectiveSize to machine precision", {
  skip_if_not_installed("coda")
  set.seed(42)
  niter = 500
  nchains = 2
  nparam = 5
  draws = array(rnorm(niter * nchains * nparam), dim = c(niter, nchains, nparam))

  ess_cpp = bgms:::.compute_ess_cpp(draws)

  ess_coda = numeric(nparam)
  for(j in seq_len(nparam)) {
    mcmc_list = coda::mcmc.list(
      lapply(seq_len(nchains), function(c) coda::mcmc(draws[, c, j]))
    )
    ess_coda[j] = coda::effectiveSize(mcmc_list)
  }
  expect_equal(ess_cpp, ess_coda, tolerance = 1e-10)
})

test_that("Rhat matches coda::gelman.diag to machine precision", {
  skip_if_not_installed("coda")
  set.seed(42)
  niter = 500
  nchains = 2
  nparam = 5
  draws = array(rnorm(niter * nchains * nparam), dim = c(niter, nchains, nparam))

  rhat_cpp = bgms:::.compute_rhat_cpp(draws)

  rhat_coda = numeric(nparam)
  for(j in seq_len(nparam)) {
    mcmc_list = coda::mcmc.list(
      lapply(seq_len(nchains), function(c) coda::mcmc(draws[, c, j]))
    )
    rhat_coda[j] = coda::gelman.diag(mcmc_list, autoburnin = FALSE)$psrf[1]
  }
  expect_equal(rhat_cpp, rhat_coda, tolerance = 1e-10)
})

test_that("ESS concordance holds for autocorrelated draws", {
  skip_if_not_installed("coda")
  set.seed(99)
  niter = 1000
  nchains = 3
  # AR(1) process with phi = 0.9
  draws = array(NA_real_, dim = c(niter, nchains, 1L))
  for(c in seq_len(nchains)) {
    x = numeric(niter)
    x[1] = rnorm(1)
    for(i in 2:niter) x[i] = 0.9 * x[i - 1] + rnorm(1)
    draws[, c, 1] = x
  }

  ess_cpp = bgms:::.compute_ess_cpp(draws)
  mcmc_list = coda::mcmc.list(
    lapply(seq_len(nchains), function(c) coda::mcmc(draws[, c, 1]))
  )
  ess_coda = unname(coda::effectiveSize(mcmc_list))
  expect_equal(ess_cpp, ess_coda, tolerance = 1e-10)
})


# ---- Single chain ---------------------------------------------------------- #

test_that("ESS works for single chain", {
  set.seed(1)
  draws = make_array(rnorm(200), niter = 200, nchains = 1)
  ess = bgms:::.compute_ess_cpp(draws)
  expect_length(ess, 1)
  expect_true(is.finite(ess))
  expect_true(ess > 0)
})

test_that("Rhat returns NA for single chain", {
  set.seed(1)
  draws = make_array(rnorm(200), niter = 200, nchains = 1)
  rhat = bgms:::.compute_rhat_cpp(draws)
  expect_length(rhat, 1)
  expect_true(is.na(rhat))
})


# ---- Constant chains ------------------------------------------------------- #

test_that("ESS is NA for constant chain", {
  draws = make_array(rep(5.0, 200), niter = 100, nchains = 2)
  ess = bgms:::.compute_ess_cpp(draws)
  expect_true(is.na(ess))
})

test_that("Rhat is NA for constant chain", {
  draws = make_array(rep(5.0, 200), niter = 100, nchains = 2)
  rhat = bgms:::.compute_rhat_cpp(draws)
  expect_true(is.na(rhat))
})


# ---- Very short chains ----------------------------------------------------- #

test_that("ESS returns NA for single iteration", {
  draws = make_array(c(1.0, 2.0), niter = 1, nchains = 2)
  ess = bgms:::.compute_ess_cpp(draws)
  expect_true(is.na(ess))
})

test_that("Rhat returns NA for single iteration", {
  draws = make_array(c(1.0, 2.0), niter = 1, nchains = 2)
  rhat = bgms:::.compute_rhat_cpp(draws)
  expect_true(is.na(rhat))
})

test_that("ESS returns finite value for niter = 2", {
  set.seed(7)
  draws = make_array(rnorm(4), niter = 2, nchains = 2)
  ess = bgms:::.compute_ess_cpp(draws)
  expect_length(ess, 1)
  # May be NA if AR order saturates, but must not crash or be NaN
  expect_true(is.na(ess) || is.finite(ess))
})

test_that("ESS is finite for short chain (niter = 10)", {
  set.seed(3)
  draws = make_array(rnorm(20), niter = 10, nchains = 2)
  ess = bgms:::.compute_ess_cpp(draws)
  expect_true(is.finite(ess))
  expect_true(ess > 0)
})


# ---- NaN and Inf input ----------------------------------------------------- #

test_that("ESS returns NA when draws contain NaN", {
  draws = make_array(c(1, 2, NaN, 4, 5, 6, 7, 8, 9, 10), niter = 5, nchains = 2)
  ess = bgms:::.compute_ess_cpp(draws)
  expect_true(is.na(ess))
})

test_that("Rhat returns NA when draws contain NaN", {
  draws = make_array(c(1, 2, NaN, 4, 5, 6, 7, 8, 9, 10), niter = 5, nchains = 2)
  rhat = bgms:::.compute_rhat_cpp(draws)
  expect_true(is.na(rhat))
})

test_that("ESS returns NA when draws contain Inf", {
  draws = make_array(c(1, 2, Inf, 4, 5, 6, 7, 8, 9, 10), niter = 5, nchains = 2)
  ess = bgms:::.compute_ess_cpp(draws)
  expect_true(is.na(ess))
})

test_that("Rhat returns NA when draws contain Inf", {
  draws = make_array(c(1, 2, Inf, 4, 5, 6, 7, 8, 9, 10), niter = 5, nchains = 2)
  rhat = bgms:::.compute_rhat_cpp(draws)
  expect_true(is.na(rhat))
})

test_that("ESS returns NA when draws contain -Inf", {
  draws = make_array(c(1, 2, -Inf, 4, 5, 6, 7, 8, 9, 10), niter = 5, nchains = 2)
  ess = bgms:::.compute_ess_cpp(draws)
  expect_true(is.na(ess))
})


# ---- Mixed pathological and good parameters -------------------------------- #

test_that("NaN in one parameter does not corrupt other parameters", {
  set.seed(5)
  good = rnorm(200)
  bad = c(rnorm(99), NaN, rnorm(100))  # NaN in chain 1
  draws = array(c(good, bad), dim = c(100, 2, 2))

  ess = bgms:::.compute_ess_cpp(draws)
  expect_length(ess, 2)
  expect_true(is.finite(ess[1]))  # good parameter
  expect_true(ess[1] > 0)
  expect_true(is.na(ess[2]))     # bad parameter

  rhat = bgms:::.compute_rhat_cpp(draws)
  expect_length(rhat, 2)
  expect_true(is.finite(rhat[1]))
  expect_true(is.na(rhat[2]))
})


# ---- All-zero draws -------------------------------------------------------- #

test_that("ESS is NA for all-zero draws", {
  draws = make_array(rep(0.0, 200), niter = 100, nchains = 2)
  ess = bgms:::.compute_ess_cpp(draws)
  expect_true(is.na(ess))
})


# ---- Binary (0/1) draws ---------------------------------------------------- #

test_that("ESS is finite and positive for binary draws with variation", {
  set.seed(12)
  draws = make_array(sample(0:1, 200, replace = TRUE), niter = 100, nchains = 2)
  ess = bgms:::.compute_ess_cpp(draws)
  expect_true(is.finite(ess))
  expect_true(ess > 0)
})

test_that("Rhat is finite for binary draws with variation", {
  set.seed(12)
  draws = make_array(sample(0:1, 200, replace = TRUE), niter = 100, nchains = 2)
  rhat = bgms:::.compute_rhat_cpp(draws)
  expect_true(is.finite(rhat))
  expect_true(rhat > 0)
})


# ---- Multiple parameters --------------------------------------------------- #

test_that("batch ESS handles many parameters correctly", {
  set.seed(77)
  nparam = 50
  draws = array(rnorm(500 * 2 * nparam), dim = c(500, 2, nparam))
  ess = bgms:::.compute_ess_cpp(draws)
  expect_length(ess, nparam)
  expect_true(all(is.finite(ess)))
  expect_true(all(ess > 0))
})

test_that("batch Rhat handles many parameters correctly", {
  set.seed(77)
  nparam = 50
  draws = array(rnorm(500 * 2 * nparam), dim = c(500, 2, nparam))
  rhat = bgms:::.compute_rhat_cpp(draws)
  expect_length(rhat, nparam)
  expect_true(all(is.finite(rhat)))
  # Well-mixed iid draws should have Rhat close to 1
  expect_true(all(rhat > 0.95 & rhat < 1.05))
})


# ---- Near-constant draws (tiny variance) ----------------------------------- #

test_that("ESS handles near-constant draws without crashing", {
  set.seed(9)
  draws = make_array(1e10 + rnorm(200, sd = 1e-10), niter = 100, nchains = 2)
  ess = bgms:::.compute_ess_cpp(draws)
  expect_length(ess, 1)
  # Result may be NA (if variance is below threshold) or finite
  expect_true(is.na(ess) || is.finite(ess))
})


# ---- Output shape ---------------------------------------------------------- #

test_that("output length matches nparam dimension", {
  draws = array(rnorm(300), dim = c(50, 2, 3))
  expect_length(bgms:::.compute_ess_cpp(draws), 3)
  expect_length(bgms:::.compute_rhat_cpp(draws), 3)
})

test_that("ESS values are non-negative when finite", {
  set.seed(42)
  draws = array(rnorm(2000), dim = c(200, 2, 5))
  ess = bgms:::.compute_ess_cpp(draws)
  finite_ess = ess[is.finite(ess)]
  expect_true(all(finite_ess >= 0))
})


# ---- Rhat properties ------------------------------------------------------- #

test_that("Rhat is close to 1 for well-mixed iid chains", {
  set.seed(42)
  draws = array(rnorm(2000), dim = c(500, 2, 2))
  rhat = bgms:::.compute_rhat_cpp(draws)
  expect_true(all(abs(rhat - 1) < 0.05))
})

test_that("Rhat detects non-convergence (shifted chains)", {
  set.seed(42)
  niter = 500
  # Chain 1: mean 0, Chain 2: mean 10
  chain1 = rnorm(niter, mean = 0)
  chain2 = rnorm(niter, mean = 10)
  draws = array(c(chain1, chain2), dim = c(niter, 2, 1))
  rhat = bgms:::.compute_rhat_cpp(draws)
  expect_true(rhat > 1.5)
})
