library(bgms)
for(p in c(3, 5)) {
  M = S = matrix(0, p, p)
  M[upper.tri(M)] = rep(c(-.4, .3, .5), length.out = p * (p - 1) / 2)
  S[upper.tri(S)] = rep(c(.08, .2, .35), length.out = p * (p - 1) / 2)
  M = M + t(M)
  S = S + t(S)
  G = matrix(0L, p, p)
  G[seq_len(p - 1), p] = G[p, seq_len(p - 1)] = 1L
  spec = list(
    family = "informed-normal", mean = M, sd = S, rate = 2, delta = 2,
    control = bgms:::resolve_informed_control()
  )
  exact = sum(vapply(seq_len(p - 1), function(i) {
    log(bgms:::informed_normal_edge_integral(-2 * M[i, p], 2 * S[i, p], c(1, 1), 2)$value)
  }, numeric(1)))
  for(permutation in list(seq_len(p), rev(seq_len(p)))) {
    spec$mean = M[permutation, permutation]
    spec$sd = S[permutation, permutation]
    result = bgms:::informed_normal_potentials_cpp(spec, list(G[permutation, permutation]))
    print(c(p = p, hub = which(permutation == p), exact = exact, production = result$log_z, error = abs(exact - result$log_z)))
    stopifnot(abs(exact - result$log_z) <= .003)
  }
  if(p == 3) {
    spec$mean = M
    spec$sd = S
    plus = G
    plus[1, 2] = plus[2, 1] = 1L
    ref = bgms:::informed_normal_reference(G, M, S, 2, 2, 1, 2, draws = 1000000L, warmup = 5000L, chains = 4L)
    potential = bgms:::informed_normal_potentials_cpp(spec, list(G, plus))$log_z
    upper = abs(potential[1] - potential[2] - ref$log_ratio) + ref$half_width
    print(c(cycle_ratio = potential[1] - potential[2], reference = ref$log_ratio, half = ref$half_width, upper = upper))
    stopifnot(ref$half_width <= .0005, upper <= .003)
  }
}
