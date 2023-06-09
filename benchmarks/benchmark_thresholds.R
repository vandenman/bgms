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

temp <- unlist(lapply(no_categories, seq_len))
no_nodes

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

category <- 1L
g_serial <- numeric(no_persons)
q_serial <- numeric(no_persons)
g_parallel <- numeric(no_persons)
q_parallel <- numeric(no_persons)
exp_current <- 0.1
bgms:::compute_c(no_persons, node1, category, exp_current, rest_matrix, no_categories, thresholds, g_serial, q_serial)
bgms:::compute_c_parallel(no_persons, node1, category, exp_current, rest_matrix, no_categories, thresholds, g_parallel, q_parallel)

all.equal(g_serial, g_parallel)
all.equal(q_serial, q_parallel)

RcppParallel::setThreadOptions(numThreads = "auto", stackSize = "auto")
RcppParallel::setThreadOptions(numThreads = 4, stackSize = "auto")
timing <- bench::mark(
  serial   = bgms:::compute_c(no_persons, node1, category, exp_current, rest_matrix, no_categories, thresholds, g_serial, q_serial),
  parallel = bgms:::compute_c_parallel(no_persons, node1, category, exp_current, rest_matrix, no_categories, thresholds, g_parallel, q_parallel),
  min_time = 3, max_iterations = 1e5
)
summary(timing)
summary(timing, relative = TRUE)
no_persons
