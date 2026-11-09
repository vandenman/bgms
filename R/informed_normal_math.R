# Internal precision-coordinate kernels: m = -2 * mean, tau = 2 * sd.
informed_normal_row = function(A, neighbors, m, tau, beta) {
  k = length(neighbors)
  if(k == 0L) {
    return(list(
      log_h = 0, U = matrix(0, nrow(A), 0),
      mean = numeric(), chol = matrix(0, 0, 0)
    ))
  }
  E = matrix(0, nrow(A), k)
  E[cbind(neighbors, seq_len(k))] = tau
  U = forwardsolve(t(chol(A)), E)
  B = diag(k) + 2 * beta * crossprod(U)
  L = chol(B)
  v = m / tau
  u = backsolve(L, forwardsolve(t(L), v))
  log_h = -sum(log(diag(L))) -
    (sum((u - v)^2) + 2 * beta * sum((U %*% u)^2)) / 2
  if(!is.finite(log_h)) stop("Nonfinite informed Gaussian row integral.")
  list(log_h = log_h, U = U, mean = as.vector(u), chol = L)
}

informed_normal_edge_integral = function(m, tau, beta, delta,
                                         rel.tol = 1e-10) {
  stopifnot(
    length(beta) == 2L, all(is.finite(beta)), all(beta > 0),
    length(m) == 1L, is.finite(m), length(tau) == 1L,
    is.finite(tau), tau > 0, length(delta) == 1L,
    is.finite(delta), delta >= 0
  )
  integrate(function(a) {
    exp(stats::dgamma(a, delta + 1, rate = beta[1], log = TRUE) -
      log1p(2 * beta[2] * tau^2 / a) / 2 -
      beta[2] * m^2 / (a + 2 * beta[2] * tau^2))
  }, 0, Inf, rel.tol = rel.tol, subdivisions = 1000L)
}
