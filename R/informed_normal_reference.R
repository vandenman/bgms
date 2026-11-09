informed_reference_inputs = function(graph, mean, sd, rate, delta, i, j, draws, warmup, chains, seed) {
  context = informed_graph_context(graph)
  graph = context$graph
  stopifnot(
    is.matrix(graph), nrow(graph) == ncol(graph),
    isSymmetric(unname(graph)), all(graph %in% 0:1)
  )
  for(n in c("i", "j", "draws", "warmup", "chains", "seed")) {
    validate_integer(get(n), n, min_value = switch(n,
      draws = 100L,
      chains = 2L,
      i = 1L,
      j = 1L,
      0L
    ))
  }
  stopifnot(
    i != j, i <= nrow(graph), j <= nrow(graph), graph[i, j] == 0,
    length(rate) == 1L, is.finite(rate), rate > 0,
    length(delta) == 1L, is.finite(delta), delta >= 0
  )
  prior = align_informed_matrices(informed_normal_prior(mean, sd), nrow(graph), context$labels)
  list(graph = graph, mean = prior$mean, sd = prior$sd)
}

informed_reference_inconclusive = function(reason) {
  list(
    log_ratio = NA_real_, mcse = Inf, half_width = Inf,
    status = "inconclusive", reason = reason
  )
}

summarize_informed_reference = function(values, confidence, comparisons) {
  stopifnot(length(confidence) == 1L, confidence > 0, confidence < 1)
  validate_integer(comparisons, "comparisons", min_value = 1L)
  if(any(!is.finite(values)) || any(values < 0) || mean(values) <= 0) {
    return(informed_reference_inconclusive("Nonfinite or zero reference mass"))
  }
  draws = nrow(values)
  chains = ncol(values)
  batch_size = floor(sqrt(draws))
  batches = floor(draws / batch_size)
  batch_means = unlist(lapply(seq_len(chains), function(k) {
    colMeans(matrix(values[seq_len(batch_size * batches), k], batch_size))
  }))
  avg = mean(values)
  se = sd(batch_means) / sqrt(length(batch_means))
  radius = stats::qt(1 - (1 - confidence) / (2 * comparisons), length(batch_means) - 1L) * se
  half = if(radius < avg) -log1p(-radius / avg) else Inf
  list(
    log_ratio = -log(avg), mcse = se / avg, half_width = half,
    status = if(is.finite(half) && half <= 0.0005) "resolved" else "inconclusive",
    method = "independent candidate-edge Schur ratio; nominal batch-means interval",
    confidence = confidence, comparisons = comparisons,
    chain_means = colMeans(values), draws = draws, chains = chains
  )
}

informed_reference_log_ratio_R = function(C, neighbors, candidate, mean, sd, beta) {
  q = 2 * beta * C[candidate, candidate]
  k = 0
  if(length(neighbors)) {
    s = sd[neighbors]
    mu = mean[neighbors]
    c = C[neighbors, candidate]
    Coo = C[neighbors, neighbors, drop = FALSE]
    S = diag(s, length(s))
    B = diag(length(s)) + 2 * beta * S %*% Coo %*% S
    cross = sqrt(2 * beta) * s * c
    joint = rbind(cbind(B, cross), c(cross, C[candidate, candidate]))
    if(any(!is.finite(joint)) || rcond(joint) < 1e-14) stop("Reference Schur complement is numerically unresolved")
    L = chol(joint)
    q = 2 * beta * tail(diag(L), 1)^2
    Q = diag(1 / s^2, length(s)) + 2 * beta * Coo
    a = solve(Q, mu / s^2)
    k = 2 * beta * sum(c * a)
  }
  m = mean[candidate]
  variance = sd[candidate]^2
  value = -0.5 * log1p(q * variance) +
    (variance * k^2 / 2 - m * k - q * m^2 / 2) / (1 + q * variance)
  if(!is.finite(value) || q <= 0) stop("Reference candidate mass is numerically unresolved")
  value
}

informed_normal_reference_R = function(graph, mean, sd, rate, delta, i, j,
                                       draws = 10000L, warmup = 2000L,
                                       chains = 4L, seed = 1L,
                                       confidence = 0.99, comparisons = 1L) {
  inputs = informed_reference_inputs(graph, mean, sd, rate, delta, i, j, draws, warmup, chains, seed)
  graph = inputs$graph
  old = if(exists(".Random.seed", .GlobalEnv)) get(".Random.seed", .GlobalEnv) else NULL
  on.exit(if(is.null(old)) {
    if(exists(".Random.seed", .GlobalEnv)) rm(".Random.seed", envir = .GlobalEnv)
  } else {
    assign(".Random.seed", old, .GlobalEnv)
  })
  set.seed(seed)
  p = nrow(graph)
  m = -2 * inputs$mean
  tau = 2 * inputs$sd
  beta = rate / 2
  values = tryCatch(
    {
      values = matrix(0, draws, chains)
      for(chain in seq_len(chains)) {
        K = diag(rep((delta + 1) / beta * (0.5 + chain / chains), p))
        for(iter in seq_len(draws + warmup)) {
          for(vertex in seq_len(p)) {
            rest = setdiff(seq_len(p), vertex)
            N = which(graph[rest, vertex] != 0)
            C = solve(K[rest, rest, drop = FALSE])
            b = numeric(p - 1L)
            if(length(N)) {
              s = tau[rest[N], vertex]
              Q = diag(1 / s^2, length(s)) + 2 * beta * C[N, N, drop = FALSE]
              b[N] = solve(Q, m[rest[N], vertex] / s^2) + backsolve(chol(Q), rnorm(length(N)))
            }
            K[rest, vertex] = K[vertex, rest] = b
            K[vertex, vertex] = rgamma(1, delta + 1, rate = beta) + sum(b * (C %*% b))
          }
          if(iter > warmup) {
            rest = setdiff(seq_len(p), j)
            N = which(graph[rest, j] != 0)
            C = solve(K[rest, rest, drop = FALSE])
            values[iter - warmup, chain] = exp(informed_reference_log_ratio_R(
              C, N, match(i, rest), m[rest, j], tau[rest, j], beta
            ))
          }
        }
      }
      values
    },
    error = identity
  )
  if(inherits(values, "error")) {
    return(informed_reference_inconclusive(conditionMessage(values)))
  }
  summarize_informed_reference(values, confidence, comparisons)
}

informed_normal_reference = function(graph, mean, sd, rate, delta, i, j,
                                     draws = 10000L, warmup = 2000L,
                                     chains = 4L, seed = 1L,
                                     confidence = 0.99, comparisons = 1L) {
  inputs = informed_reference_inputs(graph, mean, sd, rate, delta, i, j, draws, warmup, chains, seed)
  values = tryCatch(informed_normal_reference_cpp(
    inputs$graph, inputs$mean, inputs$sd, rate, delta,
    as.integer(i - 1), as.integer(j - 1), as.integer(draws),
    as.integer(warmup), as.integer(chains), as.integer(seed)
  ), error = identity)
  if(inherits(values, "error")) {
    return(informed_reference_inconclusive(conditionMessage(values)))
  }
  summarize_informed_reference(values, confidence, comparisons)
}
