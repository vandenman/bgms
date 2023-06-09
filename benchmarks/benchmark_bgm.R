rm(list = ls())
library(bgms)
library(bench)
library(ggplot2)

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

set.seed(42)
timing <- bench::press(
  no_persons     = c(1e2, 5e2, 1e3),#, 1e4),
  no_nodes       = c(5, 10),#, 20),
  # max_categories = c(3, 5, 7),
  {

    no_categories <- sample(1:5, size = no_nodes, replace = TRUE)

    interactions <- simulate_interactions(no_nodes = no_nodes, density = .2, strength = runif(1, .05, .25))

    thresholds <- matrix(0, nrow = no_nodes, ncol = max(no_categories))

    # cat("calling mrfSampler\n")
    observations <- mrfSampler(
      no_states     = no_persons,
      no_nodes      = no_nodes,
      no_categories = no_categories,
      interactions  = interactions,
      thresholds    = thresholds
    )

    bench::mark(
      serial   = bgms:::bgm(observations, iter = 400, burnin = 100, display_progress = FALSE, parallel = FALSE),
      parallel = bgms:::bgm(observations, iter = 400, burnin = 100, display_progress = FALSE, parallel = TRUE),
      check = FALSE, min_iterations = 5
    )
  }
)

# pdf("~/bgm_timings.pdf")
autoplot(timing)
# dev.off()
