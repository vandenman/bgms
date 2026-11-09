#' Informed Gaussian interaction factors for a Gaussian graphical model
#'
#' @description Specify fixed Gaussian factors on interactions theta = -K/2.
#' Means and standard deviations describe the factors before the positive
#' definiteness restriction, not the resulting marginal distributions.
#' @param mean Numeric symmetric matrix of interaction means.
#' @param sd Numeric symmetric matrix of positive interaction standard deviations.
#' @return A bgms_parameter_prior object. Unused diagonals are stored as zero.
#' @details Matrices must have the same dimensions. Symmetry is checked with
#' relative tolerance 1e-12. Named axes must have unique, nonempty names and
#' matching sets; both axes are aligned to the fitted data columns. Unnamed
#' matrices follow data column order. Only pure GGMs with exponential precision
#' diagonals and an explicit raw rate are supported. Graph selection requires
#' the hierarchical specification. The SMC correction defines an approximate
#' posterior with a frozen normalizer for each graph.
#' @seealso \code{normal_prior}, \code{exponential_prior}, \code{bgm}
#' @examples
#' M = matrix(0.1, 3, 3)
#' S = matrix(0.5, 3, 3)
#' informed_normal_prior(M, S)
#' @export
informed_normal_prior = function(mean, sd) {
  validate = function(x, label) {
    if(!is.matrix(x) || !is.numeric(x) || nrow(x) != ncol(x) || nrow(x) < 2L) {
      stop(label, " must be a numeric square matrix of dimension at least two.")
    }
    dn = dimnames(x)
    if(!is.null(dn)) {
      valid = function(n) !is.null(n) && !anyNA(n) && all(nzchar(n)) && !anyDuplicated(n)
      if(!valid(dn[[1]]) || !valid(dn[[2]]) || !setequal(dn[[1]], dn[[2]])) {
        stop(label, " must have unique nonempty matching row and column names.")
      }
      x = x[, rownames(x), drop = FALSE]
    }
    diag(x) = 0
    if(any(!is.finite(x))) stop(label, " off-diagonal entries must be finite.")
    if(any(abs(x - t(x)) > 1e-12 * pmax(1, abs(x), abs(t(x))))) {
      stop(label, " must be symmetric (tolerance 1e-12).")
    }
    x / 2 + t(x) / 2
  }
  mean = validate(mean, "mean")
  sd = validate(sd, "sd")
  if(!identical(dim(mean), dim(sd))) stop("mean and sd dimensions must match.")
  if(any(sd[row(sd) != col(sd)] <= 0)) stop("Off-diagonal sd must be positive.")
  structure(
    list(
      family = "informed-normal",
      hyper.parameters = list(mean = mean, sd = sd)
    ),
    class = "bgms_parameter_prior"
  )
}

resolve_informed_control = function(control = NULL) {
  defaults = list(
    method = "smc", master_seed = 1L, log_z_tol = 0.001,
    nominal_confidence = 0.99, particles_initial = 1024L,
    particles_multiplier = 4L, particles_max = 1048576L,
    replicas = 32L, rejuvenation_sweeps = 1L,
    resampling = "systematic_every_prefix", vertex_order = "data_columns",
    failure = "error"
  )
  if(!is.null(control) && (!is.list(control) || (length(control) > 0L && is.null(names(control))) ||
    anyDuplicated(names(control)) || any(!names(control) %in% names(defaults)))) {
    stop("normalizer_control must be a named list of supported controls.")
  }
  defaults[names(control)] = control
  for(n in c(
    "master_seed", "particles_initial", "particles_multiplier",
    "particles_max", "replicas", "rejuvenation_sweeps"
  )) {
    v = defaults[[n]]
    minimum = if(n == "rejuvenation_sweeps") 0 else if(n %in% c("replicas", "particles_multiplier")) 2 else 1
    if(!is.numeric(v) || length(v) != 1L || !is.finite(v) ||
      v < minimum || v > .Machine$integer.max || v != floor(v)) {
      stop("Invalid normalizer control: ", n)
    }
    defaults[[n]] = as.integer(v)
  }
  for(n in c("log_z_tol", "nominal_confidence")) {
    v = defaults[[n]]
    if(!is.numeric(v) || length(v) != 1L || !is.finite(v) || v <= 0 || v >= 1) {
      stop("Invalid normalizer control: ", n)
    }
  }
  if(defaults$particles_max < defaults$particles_initial) stop("particles_max is below particles_initial.")
  for(n in c("method", "resampling", "vertex_order", "failure")) {
    expected = c(
      method = "smc", resampling = "systematic_every_prefix",
      vertex_order = "data_columns", failure = "error"
    )
    if(!identical(defaults[[n]], unname(expected[n]))) stop("Unsupported normalizer control: ", n)
  }
  defaults
}

attach_informed_prior = function(spec, prior, control = NULL) {
  informed = identical(prior$family, "informed-normal")
  if(!is.null(control) && (!informed || spec$model_type != "ggm" || !spec$prior$edge_selection)) {
    stop("normalizer_control requires informed GGM graph selection.")
  }
  if(!informed) {
    return(spec)
  }
  p = spec$prior
  if(spec$model_type != "ggm") stop("informed_normal_prior() requires a pure GGM.")
  if(p$scale_shape != 1 || !is.na(p$scale_eta)) {
    stop("Use exponential_prior(rate = ...) with informed_normal_prior().")
  }
  if(p$edge_selection && p$precision_graph_prior != "hierarchical") {
    stop("Informed graph selection requires precision_graph_prior = 'hierarchical'.")
  }
  hp = prior$hyper.parameters
  for(n in c("mean", "sd")) {
    x = hp[[n]]
    if(nrow(x) != spec$data$num_variables) stop("Informed prior dimensions must match the data.")
    if(!is.null(dimnames(x))) {
      labels = spec$data$data_columnnames
      if(anyDuplicated(labels) || !setequal(rownames(x), labels)) {
        stop("Informed prior names must match the fitted data columns exactly.")
      }
      x = x[labels, labels, drop = FALSE]
    }
    spec$prior[[paste0("interaction_", n)]] = x
  }
  spec$prior$normalizer_control = if(p$edge_selection) resolve_informed_control(control) else NULL
  spec
}

informed_normal_spec = function(p) {
  list(
    family = "informed-normal", mean = p$interaction_mean,
    sd = p$interaction_sd, rate = p$scale_rate, delta = p$delta,
    control = p$normalizer_control, gauge_sweeps = 0L
  )
}
