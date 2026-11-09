# Independent Q-form full-row integration and prior Gibbs reference.
informed_normal_reference_R = function(graph, mean, sd, rate, delta, i, j,
                                       draws = 10000L, warmup = 2000L,
                                       chains = 4L, seed = 1L) {
  stopifnot(i != j, graph[i, j] == 0, chains >= 2, draws >= 100)
  old = if(exists(".Random.seed", .GlobalEnv)) get(".Random.seed", .GlobalEnv) else NULL
  on.exit(if(is.null(old)) {
    if(exists(".Random.seed", .GlobalEnv)) rm(".Random.seed", envir = .GlobalEnv)
  } else {
    assign(".Random.seed", old, .GlobalEnv)
  })
  set.seed(seed)
  p = nrow(graph)
  m = -2 * mean
  tau = 2 * sd
  beta = rep(rate / 2, length.out = p)
  log_h = function(A, neighbors, vertex, rest) {
    if(!length(neighbors)) {
      return(0)
    }
    C = solve(A)[neighbors, neighbors, drop = FALSE]
    s = tau[rest[neighbors], vertex]
    mu = m[rest[neighbors], vertex]
    Q = diag(1 / s^2, length(s)) + 2 * beta[vertex] * C
    h = mu / s^2
    -sum(log(s)) - as.numeric(determinant(Q)$modulus) / 2 -
      sum(mu^2 / s^2) / 2 + sum(h * solve(Q, h)) / 2
  }
  values = matrix(0, draws, chains)
  for(chain in seq_len(chains)) {
    K = diag((delta + 1) / beta * (0.5 + chain / chains))
    for(iter in seq_len(draws + warmup)) {
      for(vertex in seq_len(p)) {
        rest = setdiff(seq_len(p), vertex)
        N = which(graph[rest, vertex] != 0)
        A = K[rest, rest, drop = FALSE]
        b = numeric(p - 1L)
        if(length(N)) {
          C = solve(A)[N, N, drop = FALSE]
          s = tau[rest[N], vertex]
          Q = diag(1 / s^2, length(s)) + 2 * beta[vertex] * C
          b[N] = solve(Q, m[rest[N], vertex] / s^2) +
            backsolve(chol(Q), rnorm(length(N)))
        }
        K[rest, vertex] = K[vertex, rest] = b
        K[vertex, vertex] = rgamma(1, delta + 1, rate = beta[vertex]) +
          sum(b * solve(A, b))
      }
      if(iter > warmup) {
        rest = setdiff(seq_len(p), j)
        N = which(graph[rest, j] != 0)
        A = K[rest, rest, drop = FALSE]
        values[iter - warmup, chain] = exp(
          log_h(A, c(N, match(i, rest)), j, rest) - log_h(A, N, j, rest)
        )
      }
    }
  }
  batch_size = floor(sqrt(draws))
  batches = floor(draws / batch_size)
  batch_means = unlist(lapply(seq_len(chains), function(k) {
    colMeans(matrix(values[seq_len(batch_size * batches), k], batch_size))
  }))
  avg = mean(values)
  mcse = sd(batch_means) / sqrt(length(batch_means)) / avg
  list(
    log_ratio = -log(avg), mcse = mcse,
    half_width = stats::qt(0.995, length(batch_means) - 1L) * mcse,
    method = "independent full-row Gibbs, batch-means nominal 99% interval",
    chain_means = colMeans(values), draws = draws, chains = chains
  )
}

informed_normal_reference = function(graph, mean, sd, rate, delta, i, j,
                                     draws = 10000L, warmup = 2000L,
                                     chains = 4L, seed = 1L,
                                     confidence = 0.99, comparisons = 1L) {
  stopifnot(
    is.matrix(graph), nrow(graph) == ncol(graph),
    identical(graph, t(graph)), all(graph %in% 0:1),
    i != j, i >= 1, j >= 1, i <= nrow(graph), j <= nrow(graph),
    graph[i, j] == 0, chains >= 2, draws >= 100, warmup >= 0,
    is.finite(rate), rate > 0, is.finite(delta), delta >= 0,
    confidence > 0, confidence < 1, comparisons >= 1
  )
  prior = informed_normal_prior(mean, sd)$hyper.parameters
  stopifnot(nrow(prior$mean) == nrow(graph))
  values = informed_normal_reference_cpp(
    graph, prior$mean, prior$sd, rate,
    delta, as.integer(i - 1), as.integer(j - 1), as.integer(draws),
    as.integer(warmup), as.integer(chains), as.integer(seed)
  )
  batch_size = floor(sqrt(draws))
  batches = floor(draws / batch_size)
  batch_means = unlist(lapply(seq_len(chains), function(k) {
    colMeans(matrix(values[seq_len(batch_size * batches), k], batch_size))
  }))
  avg = mean(values)
  se = sd(batch_means) / sqrt(length(batch_means))
  radius = stats::qt(
    1 - (1 - confidence) / (2 * comparisons),
    length(batch_means) - 1L
  ) * se
  half = if(radius < avg) -log1p(-radius / avg) else Inf
  list(
    log_ratio = -log(avg), mcse = se / avg, half_width = half,
    status = if(is.finite(half) && half <= 0.0005) "resolved" else "inconclusive",
    method = "independent Q-form full-row Gibbs; nominal batch-means interval",
    confidence = confidence, comparisons = comparisons,
    chain_means = colMeans(values), draws = draws, chains = chains
  )
}
