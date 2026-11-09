test_that("Match the stable row integral to an independent Gaussian integral", {
  C = matrix(c(1.7, 0.4, 0.4, 0.9), 2)
  m = c(0.4, -0.2)
  tau = c(0.6, 0.8)
  beta = 0.7
  row = informed_normal_row(solve(C), 1:2, m, tau, beta)
  Q = diag(1 / tau^2) + 2 * beta * C
  h = m / tau^2
  reference = -sum(log(tau)) - as.numeric(determinant(Q)$modulus) / 2 -
    sum(m^2 / tau^2) / 2 + sum(h * solve(Q, h)) / 2
  expect_equal(row$log_h, reference, tolerance = 1e-12)
  expect_equal(exp(row$log_h), 0.4995574777107555, tolerance = 1e-12)
  expect_lte(row$log_h, 0)
  expect_equal(informed_normal_row(
    diag(2), integer(), numeric(),
    numeric(), beta
  )$log_h, 0)
  noise = backsolve(row$chol, diag(2))
  expect_equal(tau * tcrossprod(noise) * rep(tau, each = 2),
    solve(Q),
    tolerance = 1e-12
  )
})

test_that("Check the isolated-edge quadrature bank against Bessel integration", {
  bank = expand.grid(
    delta = c(0, 0.5, 2), rate = 1:2,
    m = c(0, 0.25, -0.25, 1, -1), tau = c(0.1, 0.5, 1)
  )
  for(i in seq_len(nrow(bank))) {
    x = bank[i, ]
    beta = if(x$rate == 1) c(1, 1) else c(0.5, 2)
    a = informed_normal_edge_integral(x$m, x$tau, beta, x$delta)
    b = informed_normal_edge_integral(x$m, x$tau, rev(beta), x$delta)
    reference = integrate(function(z) {
      u = abs(z) * sqrt(prod(beta))
      value = exp(log(2) + (x$delta + 1) * log(u) +
        log(besselK(2 * u, x$delta + 1, expon.scaled = TRUE)) -
        2 * u - lgamma(x$delta + 1))
      value[u == 0] = 1
      value * dnorm(z, x$m, x$tau)
    }, -Inf, Inf, rel.tol = 1e-10, subdivisions = 1000L)
    expect_equal(a$value, b$value, tolerance = 1e-9)
    expect_equal(a$value, reference$value, tolerance = 1e-9)
    expect_lt(a$abs.error, 1e-8)
    expect_lt(reference$abs.error, 1e-8)
  }
})
