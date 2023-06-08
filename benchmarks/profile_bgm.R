rm(list = ls())
library(bgms)

# based on http://minimallysufficient.github.io/r/programming/c++/2018/02/16/profiling-rcpp-packages.html

bgms:::start_profiler("benchmarks/test_profile.out")
fit <- bgm(x = Wenchuan, iter = 100)
bgms:::stop_profiler()
profvis::profvis(prof_input = "benchmarks/test_profile.out")

bgms:::start_profiler("benchmarks/log_pseudolikelihood_ratio.out")
for (i in 1:1000) {
  bgms:::log_pseudolikelihood_ratio(
    interactions,
    thresholds,
    observations,
    no_categories,
    no_persons,
    node1,
    node2,
    proposed_state,
    current_state,
    rest_matrix
  )
}
bgms:::stop_profiler()

bgms:::start_profiler("benchmarks/log_pseudolikelihood_ratio_parallel.out")
for (i in 1:1000) {
  bgms:::log_pseudolikelihood_ratio_parallel(
    interactions,
    thresholds,
    observations,
    no_categories,
    no_persons,
    node1,
    node2,
    proposed_state,
    current_state,
    rest_matrix
  )
}
bgms:::stop_profiler()

