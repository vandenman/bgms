library(bgms)
M = matrix(.1, 3, 3)
S = matrix(.15, 3, 3)
g = matrix(0L, 3, 3)
g[1, 2] = g[2, 1] = 1L
pathref = bgms:::informed_normal_reference(g, M, S, 2, 0, 1, 3, draws = 500000, warmup = 5000, chains = 4, comparisons = 2)
print(pathref)
g[1, 3] = g[3, 1] = 1L
triref = bgms:::informed_normal_reference(g, M, S, 2, 0, 2, 3, draws = 500000, warmup = 5000, chains = 4, comparisons = 2)
print(triref)
ell_edge = log(bgms:::informed_normal_edge_integral(-.2, .3, c(1, 1), 0)$value)
ell_path = ell_edge - pathref$log_ratio
ell_triangle = ell_path - triref$log_ratio
reference = c(0, ell_edge, ell_edge, ell_path, ell_edge, ell_path, ell_path, ell_triangle)
graphs = lapply(0:7, function(code) {
  graph = matrix(0L, 3, 3)
  graph[upper.tri(graph)] = as.integer(intToBits(code))[1:3]
  graph + t(graph)
})
spec = list(
  family = "informed-normal", mean = M, sd = S, rate = 2, delta = 0,
  control = bgms:::resolve_informed_control()
)
production = as.numeric(bgms:::informed_normal_potentials_cpp(spec, graphs)$log_z)
weights = exp(reference - production)
weights = weights / sum(weights)
tv = sum(abs(weights - 1 / 8)) / 2
allowance = expm1(2 * (pathref$half_width + triref$half_width)) / 2
print(c(tv = tv, reference_allowance = allowance, upper = tv + allowance))
saveRDS(list(path = pathref, triangle = triref, tv = tv, allowance = allowance), file.path(tempdir(), "bgms-informed-graph-law.rds"))
stopifnot(pathref$half_width <= 0.0005, triref$half_width <= 0.0005, tv + allowance <= 0.005)

ratio_checks = do.call(rbind, lapply(seq_along(graphs), function(k) {
  absent = which(upper.tri(graphs[[k]]) & graphs[[k]] == 0L, arr.ind = TRUE)
  if(!nrow(absent)) {
    return(NULL)
  }
  do.call(rbind, lapply(seq_len(nrow(absent)), function(e) {
    next_graph = graphs[[k]]
    ij = absent[e, ]
    next_graph[ij[1], ij[2]] = next_graph[ij[2], ij[1]] = 1L
    next_index = which(vapply(graphs, identical, logical(1), next_graph))
    edges = sum(graphs[[k]]) / 2
    half = switch(as.character(edges),
      "0" = 0,
      "1" = pathref$half_width,
      "2" = triref$half_width
    )
    discrepancy = abs((production[k] - production[next_index]) - (reference[k] - reference[next_index]))
    data.frame(
      graph = k, next_graph = next_index, discrepancy = discrepancy,
      reference_half_width = half, upper = discrepancy + half
    )
  }))
}))
print(ratio_checks)
stopifnot(
  nrow(ratio_checks) == 12L, all(ratio_checks$reference_half_width <= .0005),
  all(ratio_checks$upper <= .003)
)
