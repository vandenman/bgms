library(bgms)
control = bgms:::resolve_informed_control()
report = list()
check_potentials = function(M, S, graphs, delta = 1) {
  spec = list(family = "informed-normal", mean = M, sd = S, rate = 2, delta = delta, control = control)
  bgms:::informed_normal_potentials_cpp(spec, graphs)$log_z
}
for(p in c(3, 5, 10, 20)) {
  M = S = matrix(0, p, p)
  M[upper.tri(M)] = rep(c(-.015, .025, -.01), length.out = p * (p - 1) / 2)
  S[upper.tri(S)] = rep(c(.02, .04, .06), length.out = p * (p - 1) / 2)
  M = M + t(M)
  S = S + t(S)
  G = matrix(0L, p, p)
  G[seq_len(p - 1), p] = G[p, seq_len(p - 1)] = 1L
  exact = sum(vapply(seq_len(p - 1), function(i) {
    log(bgms:::informed_normal_edge_integral(-2 * M[i, p], 2 * S[i, p], c(1, 1), 1)$value)
  }, numeric(1)))
  production = check_potentials(M, S, list(G))
  result = data.frame(p = p, exact = exact, production = production, error = abs(production - exact))
  print(result)
  report[[paste0("star", p)]] = result
  stopifnot(result$error <= .003)
}
for(p in c(3, 4)) {
  edges = which(upper.tri(matrix(0, p, p)), arr.ind = TRUE)
  ne = nrow(edges)
  M = S = matrix(0, p, p)
  M[upper.tri(M)] = rep(c(-.015, .025, -.01), length.out = ne)
  S[upper.tri(S)] = rep(c(.02, .04, .06), length.out = ne)
  M = M + t(M)
  S = S + t(S)
  graphs = lapply(0:(2^ne - 1), function(code) {
    G = matrix(0L, p, p)
    G[upper.tri(G)] = as.integer(intToBits(code))[seq_len(ne)]
    G + t(G)
  })
  production = as.numeric(check_potentials(M, S, graphs))
  comparisons = ne * 2^(ne - 1)
  checks = list()
  reference = numeric(length(graphs))
  uncertainty = numeric(length(graphs))
  for(code in 0:(2^ne - 2)) {
    if(code %% 8L == 0L) message("Checking p=", p, ", graph code ", code)
    for(e in seq_len(ne)) {
      if(bitwAnd(code, bitwShiftL(1L, e - 1L))) next
      next_code = code + 2^(e - 1)
      ref = bgms:::informed_normal_reference(graphs[[code + 1]], M, S, 2, 1,
        edges[e, 1], edges[e, 2],
        draws = 100000L, warmup = 2000L, chains = 4L,
        comparisons = comparisons, seed = 1700L + code * ne + e
      )
      if(ref$half_width > .0005) {
        ref = bgms:::informed_normal_reference(graphs[[code + 1]], M, S, 2, 1,
          edges[e, 1], edges[e, 2],
          draws = 500000L, warmup = 5000L, chains = 4L,
          comparisons = comparisons, seed = 1700L + code * ne + e
        )
      }
      discrepancy = abs(production[code + 1] - production[next_code + 1] - ref$log_ratio)
      checks[[length(checks) + 1]] = data.frame(
        code = code, edge = e, half_width = ref$half_width,
        discrepancy = discrepancy, upper = discrepancy + ref$half_width
      )
      if(e == max(which(as.integer(intToBits(next_code))[seq_len(ne)] == 1L))) {
        reference[next_code + 1] = reference[code + 1] - ref$log_ratio
        uncertainty[next_code + 1] = uncertainty[code + 1] + ref$half_width
      }
    }
  }
  checks = do.call(rbind, checks)
  weights = exp(reference - production)
  weights = weights / sum(weights)
  tv = sum(abs(weights - 1 / length(weights))) / 2
  allowance = expm1(2 * max(uncertainty)) / 2
  print(c(
    p = p, max_ratio_upper = max(checks$upper), max_reference_half = max(checks$half_width),
    tv = tv, tv_upper = tv + allowance
  ))
  report[[paste0("graphs", p)]] = list(checks = checks, tv = tv, allowance = allowance)
  saveRDS(report, file.path(tempdir(), "bgms-informed-heldout.rds"))
  stopifnot(all(checks$half_width <= .0005), all(checks$upper <= .003), tv + allowance <= .005)
}

for(p in c(5, 10, 20)) {
  M = S = matrix(0, p, p)
  M[upper.tri(M)] = rep(c(-.015, .025, -.01), length.out = p * (p - 1) / 2)
  S[upper.tri(S)] = rep(c(.02, .04, .06), length.out = p * (p - 1) / 2)
  M = M + t(M)
  S = S + t(S)
  path = matrix(0L, p, p)
  for(i in seq_len(p - 1)) path[i, i + 1] = path[i + 1, i] = 1L
  cycle = path
  cycle[1, p] = cycle[p, 1] = 1L
  production = as.numeric(check_potentials(M, S, list(path, cycle)))
  ref = bgms:::informed_normal_reference(path, M, S, 2, 1, 1, p,
    draws = 100000L, warmup = 2000L, chains = 4L, comparisons = 3L
  )
  upper = abs(production[1] - production[2] - ref$log_ratio) + ref$half_width
  print(c(cycle_size = p, reference_half = ref$half_width, ratio_upper = upper))
  stopifnot(ref$half_width <= .0005, upper <= .003)
}
