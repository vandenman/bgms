rm(list = ls())
library(ggplot2)
library(bench)
library(bgms)

simulate_interactions <- function(no_nodes, density, strength) {

  interactions <- matrix(0, nrow = no_nodes, ncol = no_nodes)

  all_edges <- which(lower.tri(interactions), arr.ind = TRUE)
  n_edges   <- nrow(all_edges)
  sampled_edges <- sample(n_edges, size = ceiling(density * n_edges))

  interactions[all_edges[sampled_edges, ]] <-
    interactions[all_edges[sampled_edges, 2:1]] <-
    strength
  return(interactions)
}

set.seed(123)
no_persons     <- 5e3
no_nodes       <- 20
max_categories <- 7
no_categories <- sample(1:max_categories, size = no_nodes, replace = TRUE)


# interactions = matrix(0, nrow = no_nodes, ncol = no_nodes)
# interactions[2, 1] = interactions[4, 1] = interactions[3, 2] =
#   interactions[5, 2] = interactions[5, 4] = .25
# interactions = interactions + t(interactions)

interactions <- simulate_interactions(no_nodes, .2, .25)

thresholds <- matrix(0, nrow = no_nodes, ncol = max(no_categories))
observations <- mrfSampler(
  no_states = no_persons,
  no_nodes = no_nodes,
  no_categories = no_categories,
  interactions = interactions,
  thresholds = thresholds
)

node1 <- 1L
node2 <- 2L
current_state <- 1.23
proposed_state <- 1.45
rest_matrix <- bgms:::create_rest_matrix(no_persons, no_nodes, observations, interactions)

# node1 <- 3L
# person <- 3L
# bgms:::compute_ratio_node_orig(
#   thresholds     = thresholds,
#   no_categories  = no_categories,
#   bound          = -3,
#   rest_score     = 0.0,
#   proposed_state = proposed_state,
#   current_state  = current_state,
#   obs_score      = observations[person, node1],
#   node           = node1
# )
#
# bgms:::compute_ratio_node_new(
#   thresholds     = thresholds,
#   no_categories  = no_categories,
#   bound          = -3,
#   rest_score     = 0.0,
#   proposed_state = proposed_state,
#   current_state  = current_state,
#   obs_score      = observations[person, node1],
#   node           = node1
# )
# timing <- bench::mark(
#   original = bgms:::compute_ratio_node_orig(
#     thresholds     = thresholds,
#     no_categories  = no_categories,
#     bound          = -3,
#     rest_score     = 0.0,
#     proposed_state = proposed_state,
#     current_state  = current_state,
#     obs_score      = observations[person, node1],
#     node           = node1
#   ),
#   new = bgms:::compute_ratio_node_new(
#     thresholds     = thresholds,
#     no_categories  = no_categories,
#     bound          = -3,
#     rest_score     = 0.0,
#     proposed_state = proposed_state,
#     current_state  = current_state,
#     obs_score      = observations[person, node1],
#     node           = node1
#   ),
#   min_time = 3,
#   max_iterations = 1e5
# )
# summary(timing, relative = TRUE)

result_ref <-
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

result_parallel <-
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

stopifnot(all.equal(result_ref, result_parallel))

RcppParallel::setThreadOptions(numThreads = "auto", stackSize = "auto")
timing <- bench::mark(
  serial = bgms:::log_pseudolikelihood_ratio(
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
  ),
  parallel = bgms:::log_pseudolikelihood_ratio_parallel(
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
  ), min_time = 3, max_iterations = 1e5
)
summary(timing, relative = TRUE)

set.seed(42)
timing <- bench::press(
  no_persons     = c(1e2, 1e3, 1e4),
  no_nodes       = c(5, 10, 20),
  max_categories = c(3, 5, 7),
  {

    no_categories <- sample(1:max_categories, size = no_nodes, replace = TRUE)

    interactions <- simulate_interactions(no_nodes = no_nodes, density = .2, strength = runif(1, .05, .25))

    thresholds <- matrix(0, nrow = no_nodes, ncol = max(no_categories))

    # ensure node1 < node2
    node1 <- sample(no_nodes - 1, 1)
    node2 <- sample(node1:no_nodes, 1)

    # cat("calling mrfSampler\n")
    observations <- mrfSampler(
      no_states     = no_persons,
      no_nodes      = no_nodes,
      no_categories = no_categories,
      interactions  = interactions,
      thresholds    = thresholds
    )

    current_state <- rnorm(1)
    proposed_state <- current_state + rnorm(1, 0, .1)

    # cat("calling bgms:::create_rest_matrix\n")
    rest_matrix <- bgms:::create_rest_matrix(no_persons, no_nodes, observations, interactions)

    bench::mark(
      serial = bgms:::log_pseudolikelihood_ratio(
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
      ),
      parallel = bgms:::log_pseudolikelihood_ratio_parallel(
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
    )
  }
)

bench:::autoplot.bench_mark()
autoplot(timing, type = "violin")
