# Mixed MRF — Implementation Plan

**Date:** 2026-02-25 (updated 2026-03-04; review amendments applied 2026-03-04)
**Branch:** `ggm_mixed` (PR #78)
**Goal:** Build a monolithic `MixedMRFModel` in C++ that supports both
conditional and marginal pseudo-likelihood, with and without edge selection.

**Reference code:** `MaartenMarsman/mixedGM` repository (R + Rcpp prototype)
**Theory:** `dev/plans/mixedMRF/A_Mixed_Graphical_Model_for_Continuous_and_Ordinal_Variables (1).pdf`

---

## Prototype Status (mixedGM)

The `mixedGM` package (`/Users/maartenmarsman/Documents/GitHub/mixedGM`) provides
a complete R + Rcpp prototype with:

### Implemented & Tested Components

| Component | Location | Status |
|-----------|----------|--------|
| Conditional OMRF likelihood | `src/log_likelihoods.cpp` | ✅ C++ (Rcpp) |
| Marginal OMRF likelihood | `src/log_likelihoods.cpp` | ✅ C++ (Rcpp) |
| Conditional GGM likelihood | `src/log_likelihoods.cpp` | ✅ C++ (Rcpp) |
| Θ computation | `src/log_likelihoods.cpp` | ✅ C++ (Rcpp) |
| Gibbs data generator | `src/mixed_gibbs.cpp` | ✅ C++ (Rcpp) |
| Cholesky update/downdate | `src/cholupdate.cpp` | ✅ C++ (Rcpp) |
| MH parameter updates | `R/cond_*_mh_update_functions.R` | R only |
| Edge selection | `R/cond_*_mh_update_functions.R` | R only |
| Main sampler loop | `R/mixed_sampler.R` | R only |
| Stan exact-likelihood model | `inst/stan/mixed_mrf_exact.stan` | ✅ Stan |
| Unit tests (likelihood) | `tests/testthat/test-likelihood-correctness.R` | ✅ |
| Parameter recovery tests | `tests/testthat/test-parameter-recovery.R` | ✅ |
| Edge selection tests | `tests/testthat/test-edge-selection.R` | ✅ |

### Bug Fixes Already Applied

The following issues identified in the plan reviews have been fixed in mixedGM:

| Issue | Status | Location |
|-------|--------|----------|
| Missing factor of 2 in Θ | ✅ Fixed | `rcpp_compute_Theta()`: `Kxx + 2.0 * Kxy * Sigma_yy * Kxy.t()` |
| `dnorm` without `log=TRUE` | ✅ Fixed | All edge selection functions use `log = TRUE` |
| Cache invalidation | ✅ Fixed | `update_Kyy_cache()` called after every Kyy change |
| Marginal PL Kyy acceptance | ✅ Fixed | Includes all p marginal OMRF terms |

### Validation Infrastructure

- **Stan exact model** (`inst/stan/mixed_mrf_exact.stan`): Enumerates all ordinal
  configurations for p ≤ 5 as gold-standard posterior. Serves as non-circular
  validation against the pseudolikelihood.
- **Simulation study** (`dev/simulation_study_plan.md`): 54-cell design covering
  p ∈ {5,10,15}, q ∈ {5,10,15}, n ∈ {250,500,1000}.

---

## Table of contents

1. [Model overview](#1-model-overview)
2. [Two pseudo-likelihood approaches](#2-two-pseudo-likelihood-approaches)
3. [Parameter groups and update schedule](#3-parameter-groups-and-update-schedule)
4. [File layout](#4-file-layout)
5. [Implementation phases](#5-implementation-phases)
6. [Phase A — Skeleton and data structures](#phase-a--skeleton-and-data-structures)
7. [Phase B — Conditional pseudo-likelihood](#phase-b--conditional-pseudo-likelihood)
8. [Phase C — Marginal pseudo-likelihood](#phase-c--marginal-pseudo-likelihood)
9. [Phase D — Edge selection](#phase-d--edge-selection)
10. [Phase E — R interface and integration](#phase-e--r-interface-and-integration)
11. [Phase F — Warmup, adaptation, and diagnostics](#phase-f--warmup-adaptation-and-diagnostics)
12. [Phase G — Simulation and prediction](#phase-g--simulation-and-prediction)
13. [Testing strategy](#testing-strategy)
14. [Reuse inventory](#reuse-inventory)
15. [Risk register](#risk-register)

---

## 1. Model overview

The mixed MRF models the joint distribution of $p$ ordinal variables
$x$ ($x_s \in \{0, 1, \ldots, C_s\}$) and $q$ continuous variables $y$:

$$\log f(x, y) \propto \sum_s \mu_{x,s}(x_s) + x^\top K_{xx}\, x
  - \tfrac{1}{2}(y - \mu_y)^\top K_{yy}\,(y - \mu_y)
  + 2\, x^\top K_{xy}\, y$$

Parameters:

| Symbol | Storage | Dimension | Role |
|--------|---------|-----------|------|
| $\mu_x$ | `mux_` | $p \times \max(C_s)$ | Ordinal thresholds (main effects) |
| $K_{xx}$ | `Kxx_` | $p \times p$ symmetric, zero diag | Discrete pairwise interactions |
| $K_{yy}$ | `Kyy_` | $q \times q$ SPD | Continuous precision matrix |
| $K_{xy}$ | `Kxy_` | $p \times q$ | Cross-type interactions |
| $\mu_y$ | `muy_` | $q$-vector | Continuous means |
| $G$ | `edge_indicators_` | $(p+q) \times (p+q)$ | Edge inclusion indicators |

The factor 2 on $x^\top K_{xy}\, y$ reflects a symmetric parameterization:
the bilinear coupling between ordinal variable $x_s$ and continuous variable
$y_j$ contributes once as $(x_s, y_j)$ and once as $(y_j, x_s)$ in the
sufficient statistics sum over node pairs.  Absorbing both contributions into
a single $K_{xy}$ matrix introduces the factor 2.  This is a pure convention
(not a free parameter) and must be applied consistently in all likelihoods,
conditional means, and $\Theta$ computations.

### Conventions carried into C++

- $K_{yy}$ stores the **positive-definite precision** $\Sigma_{yy}^{-1}$. The
  LaTeX note writes $K_{yy} = -\tfrac{1}{2}\Sigma_{yy}^{-1}$; we absorb the
  $-\tfrac{1}{2}$ into the log-density and always work with SPD matrices.
- The joint density is written as $-\tfrac{1}{2}(y - \mu_y)^\top K_{yy}
  (y - \mu_y)$ so that $\mu_y$ is the literal continuous mean. This matches
  the R prototype and keeps the conditional mean expression short.
- With this convention, all marginal rest-scores that arise from integrating
  out $y$ must include $K_{yy}^{-1}$ explicitly. Whenever the LaTeX note shows
  $\Sigma_{yy}$, substitute $K_{yy}^{-1}$.

---

## 2. Two pseudo-likelihood approaches

Both approaches approximate the intractable joint $f(x, y)$ using
pseudo-likelihoods. They share the same GGM part but differ in how the
discrete pseudo-likelihood handles the coupling to $y$.

### 2.1 Conditional pseudo-likelihood

$$\text{PL}_{\text{cond}}(x, y) =
  \underbrace{f(y \mid x)}_{\text{conditional GGM}} \cdot
  \prod_{s=1}^{p} \underbrace{f(x_s \mid x_{-s}, y)}_{\text{conditional OMRF}}$$

**Conditional OMRF** — full conditional of $x_s$ given $x_{-s}$ **and** $y$:

$$r_s = x_{-s}^\top K_{xx,-s,s} + 2\,y^\top K_{yx,s}$$
$$\log f(x_s = c \mid x_{-s}, y) = \mu_{x,s,c} + c \cdot r_s
  - \log\!\Bigl(1 + \sum_{c'=1}^{C_s} \exp(\mu_{x,s,c'} + c' \cdot r_s)\Bigr)$$

The rest-score $r_s$ depends on $K_{xx}$ and $K_{xy}$ but **not** on $K_{yy}$.

**Conditional GGM** — $y \mid x$ is multivariate Gaussian:

$$y \mid x \sim N\bigl(\mu_y + 2\,x\,K_{xy}\,K_{yy}^{-1},\; K_{yy}^{-1}\bigr)$$
$$\log f(y \mid x) = \frac{n}{2}\log|K_{yy}|
  - \frac{1}{2}\sum_{v=1}^{n}(y_v - M_v)^\top K_{yy}\,(y_v - M_v)$$

where $M = \mathbf{1}\mu_y^\top + 2\,x\,K_{xy}\,K_{yy}^{-1}$ is the
$n \times q$ conditional mean matrix.

### 2.2 Marginal pseudo-likelihood

$$\text{PL}_{\text{marg}}(x, y) =
  \underbrace{f(y \mid x)}_{\text{conditional GGM}} \cdot
  \prod_{s=1}^{p} \underbrace{f(x_s \mid x_{-s})}_{\text{marginal OMRF}}$$

**Marginal OMRF** — full conditional of $x_s$ after integrating out $y$:

$$\Theta = K_{xx} + 2\,K_{xy}\,K_{yy}^{-1}\,K_{yx}$$
$$r_s = \bigl(x^\top \Theta_{\cdot,s}\bigr) - x_s\,\Theta_{ss} + 2\,(K_{xy})_{s\cdot}\,K_{yy}^{-1}\,\mu_y$$

The self-interaction $x_s\,\Theta_{ss}$ must be subtracted because the rest-score
conditions only on $x_{-s}$.  In practice:
- Compute $r^{\text{row}} = x^\top \Theta_{\cdot,s}$ (matrix-vector product over all observations)
- Subtract `x_dbl.col(s) * Theta_(s,s)` from the result
- Add the scalar bias `2.0 * arma::dot(Kxy_.row(s), Kyy_inv_ * muy_)` (same for every observation)

This matches the mixedGM implementation in `rcpp_log_pl_marginal_omrf` exactly.

Same categorical form as conditional PL, but the effective interaction
matrix $\Theta$ absorbs the continuous variables. This means:
- Changing $K_{yy}$ or $K_{xy}$ requires recomputing $\Theta$
- Changing $\mu_y$ changes all rest-scores through the
  $K_{yy}^{-1}\,\mu_y$ term

### 2.3 Which parameters affect which likelihoods

| Parameter | Conditional OMRF | Conditional GGM | Marginal OMRF |
|-----------|:---:|:---:|:---:|
| $\mu_x$ | ✓ | | ✓ |
| $K_{xx}$ | ✓ | | ✓ (via $\Theta$) |
| $K_{yy}$ | | ✓ | ✓ (via $\Theta$) |
| $K_{xy}$ | ✓ | ✓ | ✓ (via $\Theta$) |
| $\mu_y$ | | ✓ | ✓ |

---

## 3. Parameter groups and update schedule

Each MCMC iteration updates 5 parameter groups in sequence, matching
the R prototype. The table shows which log-likelihood components enter
the MH acceptance ratio for each group.

### 3.1 Conditional PL mode

| Step | Parameter | Components in acceptance ratio |
|------|-----------|-------------------------------|
| 1 | $\mu_{x,s,c}$ (one threshold) | `cond_omrf(s)` + prior |
| 2 | $\mu_{y,j}$ (one mean) | `cond_ggm()` + prior |
| 3 | $K_{xx,ij}$ (one pair) | `cond_omrf(i) + cond_omrf(j)` + prior |
| 4 | $K_{yy}$ (one element) | `cond_ggm()` + prior (Cholesky proposal) |
| 5 | $K_{xy,ij}$ (one element) | `cond_omrf(i) + cond_ggm()` + prior |

### 3.2 Marginal PL mode

| Step | Parameter | Components in acceptance ratio |
|------|-----------|-------------------------------|
| 1 | $\mu_{x,s,c}$ | `marg_omrf(s)` + prior |
| 2 | $\mu_{y,j}$ | `cond_ggm()` + $\sum_s$ `marg_omrf(s)` + prior |
| 3 | $K_{xx,ij}$ | `marg_omrf(i) + marg_omrf(j)` + prior |
| 4 | $K_{yy}$ | `cond_ggm()` + $\sum_s$ `marg_omrf(s)` + prior (Cholesky proposal) |
| 5 | $K_{xy,ij}$ | $\sum_s$ `marg_omrf(s)` + `cond_ggm()` + prior |

**Key difference:** In marginal mode, updating $\mu_y$ requires evaluating
ALL $p$ marginal OMRF terms because $\mu_y$ enters every rest-score. The
same is true for $K_{yy}$ and every $K_{xy,ij}$ because $\Theta$ changes
globally with those parameters. Marginal PL is therefore much more
expensive per iteration.

### 3.3 Edge selection (post-warmup)

Three independent edge-selection sweeps, same in both PL modes:

| Edge type | Indicator | RJ proposal | Acceptance components |
|-----------|-----------|-------------|----------------------|
| Discrete-discrete | $G_{xx}$ | Toggle + spike-and-slab | `omrf(i) + omrf(j)` |
| Continuous-continuous | $G_{yy}$ | Cholesky-based toggle | `cond_ggm()` |
| Cross | $G_{xy}$ | Toggle + spike-and-slab | `omrf(i) + cond_ggm()` |

---

## 4. File layout

```
src/
  models/
    mixed/
      mixed_mrf_model.h            # Class declaration (follows GGMModel pattern)
      mixed_mrf_model.cpp          # Constructor, clone, vectorization
      mixed_mrf_likelihoods.cpp    # Port from mixedGM/src/log_likelihoods.cpp
      mixed_mrf_metropolis.cpp     # Port MH updates from mixedGM R code
      mixed_mrf_edge_selection.cpp # Port edge selection from mixedGM R code
      mixed_mrf_cholesky.cpp       # Cholesky permute/R() (port from mixedGM R)
  sample_mixed.cpp                 # Rcpp interface (copy sample_ggm.cpp pattern)
  mrf_simulation.cpp               # Extend with mixed_gibbs_generate()
  mrf_prediction.cpp             # Add mixed MRF prediction
R/
  bgm.R                          # Extend bgm() to dispatch mixed data
  validate_data.R                # Add mixed data validation
  validate_model.R               # Add mixed model validation
tests/
  testthat/
    test-mixed-mrf-likelihood.R   # Likelihood correctness vs R prototype
    test-mixed-mrf-sampling.R     # Recovery tests
    test-mixed-mrf-edge-sel.R     # Edge selection tests
```

`configure` + `inst/generate_makevars_sources.R` already glob every `.cpp`
under `src/`, but Phase A must still run the script (or re-run `configure`)
so the new translation units show up in `sources.mk`/`Makevars`. Likewise,
add the new Rcpp export to `src/RcppExports.cpp`, `R/RcppExports.R`, and
`NAMESPACE` when `sample_mixed.cpp` lands.

---

## 5. Implementation phases

| Phase | What | Depends on | Deliverable |
|-------|------|------------|-------------|
| **A** | Skeleton: class, data structures, `BaseModel` overrides | — | Compiles, no sampling |
| **B** | Conditional PL: all 5 MH updates, no edge selection | A | Recovery test passes (cond PL, estimation only) |
| **C** | Marginal PL: $\Theta$ caching, marginal OMRF, $\mu_y$ full sweep | B | Recovery test passes (marg PL, estimation only) |
| **D** | Edge selection: 3 RJ sweeps | B | Structure recovery test passes |
| **E** | R interface: `bgm()` dispatch, output formatting | B | End-to-end `bgm(mixed_data)` works |
| **F** | Warmup schedule, adaptation, diagnostics | E | Full warmup pipeline |
| **G** | Simulation and prediction | E | `simulate.bgms` and `predict.bgms` for mixed |

---

## Phase A — Skeleton and data structures

### A.1 Create `mixed_mrf_model.h`

```cpp
class MixedMRFModel : public BaseModel {
public:
    // Construction
    MixedMRFModel(
        const arma::imat& x,           // n × p discrete (0-based categories)
        const arma::mat& y,             // n × q continuous
        const arma::ivec& num_categories, // p-vector
        bool edge_selection,
        const std::string& pseudolikelihood, // "conditional" or "marginal"
        int seed
    );

    MixedMRFModel(const MixedMRFModel& other);

    // BaseModel overrides (all 13 pure virtuals)
    void do_one_metropolis_step(int iteration = -1) override;
    void update_edge_indicators() override;
    size_t parameter_dimension() const override;
    arma::vec get_vectorized_parameters() const override;
    void set_vectorized_parameters(const arma::vec& params) override;
    arma::ivec get_vectorized_indicator_parameters() override;
    size_t full_parameter_dimension() const override;
    arma::vec get_full_vectorized_parameters() const override;
    void set_seed(int seed) override;
    std::unique_ptr<BaseModel> clone() const override;
    SafeRNG& get_rng() override;
    const arma::imat& get_edge_indicators() const override;
    arma::mat& get_inclusion_probability() override;
    int get_num_variables() const override;
    int get_num_pairwise() const override;
    void prepare_iteration() override;
    void set_edge_selection_active(bool active) override;
    void initialize_graph() override;
    void init_metropolis_adaptation(const WarmupSchedule& schedule) override;
    void tune_proposal_sd(int iteration, const WarmupSchedule& schedule) override;
    bool has_missing_data() const override;
    void impute_missing() override;

    // Capability queries
    bool has_edge_selection() const override;
    bool has_adaptive_metropolis() const override;

private:
    // --- Data ---
    arma::imat x_;                    // n × p discrete observations
    arma::mat y_;                     // n × q continuous observations
    int n_, p_, q_;
    arma::ivec num_categories_;       // p-vector
    int max_cats_;                    // max(num_categories)

    // --- Parameters ---
    arma::mat mux_;                   // p × max_cats thresholds
                                     //   mux_(s, c) is the threshold for category c+1 of variable s;
                                     //   category 0 is the reference level (threshold fixed at 0).
                                     //   In C++ loops always use index c-1 when accessing mux_ for category c.
    arma::vec muy_;                   // q-vector continuous means
    arma::mat Kxx_;                   // p × p discrete interactions (diagonal always zero;
                                     //   enforced by construction — not a free parameter)
    arma::mat Kyy_;                   // q × q SPD precision
    arma::mat Kxy_;                   // p × q cross interactions

    // --- Edge indicators ---
    // Single combined (p+q)×(p+q) matrix (Decision: Option A).
    //   Gxx block : rows [0,p),    cols [0,p)    — symmetric, zero diag
    //   Gyy block : rows [p,p+q),  cols [p,p+q) — symmetric, zero diag
    //   Gxy block : rows [0,p),    cols [p,p+q) — full p×q rectangle
    //              (lower-left mirror [p,p+q)×[0,p) unused; Gxy is not symmetric)
    //
    // Accessor helpers (prefer over raw index arithmetic throughout):
    //   int& gxx(int i, int j)  { return edge_indicators_(i, j); }
    //   int& gyy(int i, int j)  { return edge_indicators_(p_+i, p_+j); }
    //   int& gxy(int i, int j)  { return edge_indicators_(i, p_+j); }
    //
    // Serialization order for get_vectorized_indicator_parameters():
    //   1. upper-tri(Gxx) row-major — length p(p-1)/2
    //   2. upper-tri(Gyy) row-major — length q(q-1)/2
    //   3. full Gxy      row-major — length p*q
    //   Total length: p(p-1)/2 + q(q-1)/2 + p*q
    arma::imat edge_indicators_;      // (p+q) × (p+q)
    arma::mat inclusion_probability_; // (p+q) × (p+q)
    bool edge_selection_;
    bool edge_selection_active_;

    // --- Proposal SDs (Robbins-Monro) ---
    arma::mat prop_sd_Kxx_;           // p × p
    arma::mat prop_sd_Kyy_;           // q × q
    arma::mat prop_sd_Kxy_;           // p × q
    arma::mat prop_sd_mux_;           // p × max_cats
    arma::vec prop_sd_muy_;           // q-vector

    // --- Cached quantities ---
    arma::mat Kyy_inv_;              // q × q  inverse of Kyy (always maintained)
    arma::mat Kyy_chol_;             // q × q  upper Cholesky of Kyy
    double Kyy_log_det_;             // log|Kyy|
    arma::mat Theta_;                // p × p  Kxx + 2 * Kxy * Kyy_inv * Kyx
                                     //   (marginal PL only)
    arma::mat conditional_mean_;     // n × q  muy + 2 * x * Kxy * Kyy_inv

    // --- Configuration ---
    bool use_marginal_pl_;           // true = marginal, false = conditional
    SafeRNG rng_;

    // --- Edge update order ---
    arma::uvec edge_order_;          // shuffled pair indices
};
```

### A.2 Implement trivial overrides

Implement `parameter_dimension`, `get/set_vectorized_parameters`,
`get_full_vectorized_parameters`, `clone`, `set_seed`, `get_rng`,
`get_edge_indicators`, `get_inclusion_probability`, `get_num_variables`,
`get_num_pairwise`, `has_edge_selection`, `has_adaptive_metropolis`.

**Parameter vectorization order (free parameters only):**
1. `mux_`: For each variable $s = 0,\ldots,p-1$, copy columns $0 \ldots C_s-1$
   of row $s$ (i.e., the thresholds for categories $1 \ldots C_s$; category 0
   is the fixed reference level). Each variable contributes $C_s$ entries;
   the total count is $\sum_s C_s$. Variables with different $C_s$ produce
   runs of different length — **do not use a fixed stride**.
   ```
   idx = 0
   for s in 0..p-1:
       for c in 0..num_categories_[s]-1:
           out[idx++] = mux_(s, c)   // threshold for category c+1
   ```
2. `Kxx_`: strictly upper-triangular entries (symmetry supplies the lower
   half). Count = $p(p-1)/2$. Diagonal entries are always zero and excluded.
3. `muy_`: all $q$ means.
4. `Kyy_`: upper-triangle **including** the diagonal. Count = $q(q+1)/2$.
5. `Kxy_`: all $p\times q$ cross entries, row-major.

**`prepare_iteration()`** should be included in Phase A.2's list of trivial
overrides: it shuffles `edge_order_` (same pattern as `OMRFModel`) so that
RJ sweeps have no order bias.

`parameter_dimension()` returns the number of currently **active**
parameters under edge selection (mirrors `OMRFModel`).
`full_parameter_dimension()` always returns the total count above so that
sample buffers have a fixed width even while RJ toggles edges on/off.

`get_vectorized_indicator_parameters()` serializes in three contiguous
blocks (matching the layout documented in the class header comment):
1. Upper triangle of $G_{xx}$ (rows/cols $[0,p)$) — length $p(p-1)/2$
2. Upper triangle of $G_{yy}$ (rows/cols $[p,p+q)$) — length $q(q-1)/2$
3. Full $G_{xy}$ block (rows $[0,p)$, cols $[p,p+q)$) row-major — length $pq$

Total length: $p(p-1)/2 + q(q-1)/2 + pq$.
The diagonal of `edge_indicators_` is always zero and excluded throughout.

### A.3 Testing checkpoint

- Class compiles and links
- `parameter_dimension()` returns correct count
- Round-trip: `set_vectorized_parameters(get_vectorized_parameters())` is identity

---

## Phase B — Conditional pseudo-likelihood

### B.1 Implement log-likelihood functions

In `mixed_mrf_likelihoods.cpp`:

#### `log_conditional_omrf(int s)` — per-variable discrete PL

Computes $\log f(x_s \mid x_{-s}, y)$ summed over all $n$ observations.

```
rest_score = x_[, -s] * Kxx_[-s, s] + 2 * y_ * Kxy_[s, ]^T
             ^-- n×(p-1) * (p-1)×1     ^-- n×q * q×1
           = n-vector

For each observation v:
  For c = 1..C_s:
    eta[v,c] = mux_[s, c] + c * rest_score[v]
  log_Z[v] = log(1 + sum_c exp(eta[v,c]))   (log-sum-exp stabilized)
  ll += mux_[s, x_[v,s]] + x_[v,s] * rest_score[v] - log_Z[v]
        (only if x_[v,s] > 0 for the threshold part)
```

**Reuse opportunity:** The inner loop (compute log-partition for
one ordinal variable given a rest-score vector) is identical to
`OMRFModel::ordinal_log_pseudolikelihood_ratio()`. Extract or
duplicate the stabilized log-sum-exp computation.

#### `log_conditional_ggm()` — conditional GGM log-likelihood

Computes $\log f(y \mid x)$ using cached `Kyy_inv_`, `Kyy_log_det_`,
and `conditional_mean_`.

```
conditional_mean_ = 1*muy_^T + 2 * x_ * Kxy_ * Kyy_inv_
D = y_ - conditional_mean_
quad_sum = sum((D * Kyy_) .* D)       // trace of Kyy * D^T * D
ll = n/2 * (-q * log(2*pi) + Kyy_log_det_) - quad_sum / 2
```

**Reuse opportunity:** This is structurally identical to `GGMModel::log_density_impl`
but with a non-zero, observation-dependent conditional mean.

### B.2 Implement cache maintenance

#### `recompute_conditional_mean()`
Recompute `conditional_mean_ = 1*muy^T + 2 * x_ * Kxy_ * Kyy_inv_`.
Called after any change to `muy_`, `Kxy_`, or `Kyy_`.

#### `recompute_Kyy_decomposition()`
Recompute `Kyy_chol_`, `Kyy_inv_`, `Kyy_log_det_` from `Kyy_`.
Called after any change to `Kyy_`.

#### `recompute_Theta()` (marginal PL only)
Recompute `Theta_ = Kxx_ + 2 * Kxy_ * Kyy_inv_ * Kxy_^T`.
Called after any change to `Kxx_`, `Kxy_`, or `Kyy_`.

**Cache update order — proposed-state bookkeeping:**

The guiding rule is: **all caches must be consistent with the proposed
parameter value before evaluating the proposed likelihood**. On rejection,
temporaries are discarded cheaply (Armadillo move semantics).

1. **Kyy proposal:** Build proposed `Kyy_chol_prop`, `Kyy_inv_prop`,
   `Kyy_log_det_prop`, and `conditional_mean_prop` (and `Theta_prop` if
   marginal PL) into temporaries first. Evaluate proposed likelihood with
   these temporaries. On acceptance, swap into cached members in the order:
   `Kyy_` → `Kyy_chol_`/`Kyy_inv_`/`Kyy_log_det_` → `conditional_mean_`
   → `Theta_` (if needed). On rejection, discard temporaries.
2. **Kxx proposal (marginal PL mode):** In marginal PL mode, `Theta_`
   changes when `Kxx_` changes. Build `Theta_prop` from the proposed Kxx
   into a temporary before evaluating `log_marginal_omrf`. On acceptance,
   swap `Kxx_` and `Theta_`. On rejection, discard `Theta_prop`. In
   conditional PL mode, no cache depends on `Kxx_`, so no temporary is
   needed.
3. **Kxy proposal:** Build `conditional_mean_prop` (both modes) and
   `Theta_prop` (marginal mode) into temporaries before evaluating the
   proposed likelihood. On acceptance, swap `Kxy_`, `conditional_mean_`,
   and (if marginal) `Theta_`. On rejection, discard temporaries.
4. **muy proposal (marginal PL mode):** `conditional_mean_` depends on
   `muy_` directly. Build `conditional_mean_prop` (and re-evaluate all
   rest-score bias terms `2 * dot(Kxy_.row(s), Kyy_inv_ * muy_prop)`) before
   evaluating the proposed likelihood. Swap on acceptance.
5. **Every accepted RJ edge toggle (Phase D)** must trigger the same
   cache-refresh logic as an MH acceptance for that parameter type.

### B.3 Implement MH updates (conditional PL)

Each update function follows the pattern:
1. Save current parameter value
2. Propose new value from $N(\text{current}, \text{prop\_sd})$
3. Compute log acceptance ratio (log-likelihood change + log-prior change)
4. Accept/reject
5. Robbins-Monro update of proposal SD

#### `update_threshold(int s, int c)` — one threshold $\mu_{x,s,c}$
- Propose: $\mu'_{x,s,c} \sim N(\mu_{x,s,c}, \sigma_{\mu_x,s,c})$
- Acceptance: `log_conditional_omrf(s)` at proposed vs current + prior ratio
- Prior: Normal(0, $\sigma^2$)

#### `update_continuous_mean(int j)` — one mean $\mu_{y,j}$
- Propose: $\mu'_{y,j} \sim N(\mu_{y,j}, \sigma_{\mu_y,j})$
- Acceptance: `log_conditional_ggm()` at proposed vs current + prior ratio
- Must update `conditional_mean_` before evaluating proposed likelihood
- Prior: Normal(0, $\sigma^2$)

#### `update_Kxx(int i, int j)` — one discrete interaction
- Propose: $K'_{xx,ij} \sim N(K_{xx,ij}, \sigma_{K_{xx},ij})$
- Set both $K_{xx,ij}$ and $K_{xx,ji}$ (symmetric)
- Acceptance: `log_conditional_omrf(i) + log_conditional_omrf(j)` + prior
- Prior: Cauchy(0, scale)
- Only if $G_{xx,ij} = 1$

#### `update_Kyy_element(int i, int j)` — one precision element (Cholesky)
- Uses the same Cholesky-based proposal as `GGMModel`:
  1. Permute rows/cols so (i,j) maps to (q-1, q) position
  2. Compute Cholesky, extract constants
  3. Propose on the Cholesky scale: $\phi' \sim N(\phi, \sigma_{K_{yy},ij})$
  4. Rebuild $K'_{yy}$ maintaining positive definiteness
  5. Unpermute
- Acceptance: `log_conditional_ggm()` at proposed vs current + priors
- Priors: Cauchy(0, scale) on off-diagonal; Gamma(shape, rate) on diagonal
- Must update all Kyy-dependent caches after acceptance

Unlike `GGMModel`, no rank-2 determinant lemma shortcut is available: the
conditional mean depends on $K_{yy}^{-1}$, so each proposal evaluates the
full `log_conditional_ggm()` with freshly computed `Kyy_inv_` and
`conditional_mean_`.

**Jacobian for diagonal Kyy proposals.** The diagonal element is proposed
on the log scale to guarantee positivity:
```
// Diagonal element i:
theta_curr = log(L(i, i))              // L is the Cholesky factor
theta_prop = rnorm(theta_curr, prop_sd)
L_prop(i, i) = exp(theta_prop)
// Include log-Jacobian in acceptance:
ln_alpha += theta_prop - theta_curr
```
Omitting this Jacobian biases the diagonal distribution. Off-diagonal
elements are proposed linearly and need no Jacobian.

**Cholesky update strategy.** `GGMModel` uses rank-1 Cholesky
update/downdate (`cholupdate.h`) applied to a single off-diagonal entry.
The mixed MRF follows the `mixedGM` approach instead: **permute** the
target row/column pair to the last two positions, perform the 2×2 block
update, then unpermute. These are distinct algorithms. Place the permute
helpers in `mixed_mrf_cholesky.cpp`, ported from
`mixedGM/R/continuous_variable_helper.R`. Do **not** attempt to reuse
`GGMModel`'s rank-1 routines for this purpose.

#### `update_Kxy(int i, int j)` — one cross interaction
- Propose: $K'_{xy,ij} \sim N(K_{xy,ij}, \sigma_{K_{xy},ij})$
- Acceptance: `log_conditional_omrf(i) + log_conditional_ggm()` + prior
- Must update `conditional_mean_` after modifying `Kxy_`
- Prior: Cauchy(0, scale)
- Only if $G_{xy,ij} = 1$

### B.4 Implement `do_one_metropolis_step(int iteration)`

```cpp
void MixedMRFModel::do_one_metropolis_step(int iteration) {
    // Step 1: Update all thresholds
    for (int s = 0; s < p_; ++s)
        for (int c = 0; c < num_categories_[s]; ++c)
            update_threshold(s, c, iteration);

    // Step 2: Update all continuous means
    for (int j = 0; j < q_; ++j)
        update_continuous_mean(j, iteration);

    // Step 3: Update Kxx (upper triangle, edge-gated)
    for (int i = 0; i < p_ - 1; ++i)
        for (int j = i + 1; j < p_; ++j)
            if (!edge_selection_active_ || gxx(i, j) == 1)
                update_Kxx(i, j, iteration);

    // Step 4: Update Kyy (off-diag + diagonal, edge-gated)
    for (int i = 0; i < q_ - 1; ++i)
        for (int j = i + 1; j < q_; ++j)
            if (!edge_selection_active_ || gyy(i, j) == 1)
                update_Kyy_offdiag(i, j, iteration);
    for (int i = 0; i < q_; ++i)
        update_Kyy_diag(i, iteration);  // diagonal always active

    // Step 5: Update Kxy (edge-gated)
    for (int i = 0; i < p_; ++i)
        for (int j = 0; j < q_; ++j)
            if (!edge_selection_active_ || gxy(i, j) == 1)
                update_Kxy(i, j, iteration);
}
```

### B.5 Testing checkpoint — conditional PL estimation

**Test 1: Likelihood agreement**
- Generate data from `mixed_gibbs_generate()` in R
- Compute `log_conditional_omrf(s)` and `log_conditional_ggm()` in both
  R (prototype) and C++ at the same parameter values
- Assert agreement to machine precision

**Test 2: Recovery (estimation only, no edge selection)**
- Generate data from known parameters (p=3 ordinal, q=2 continuous)
- Run mixed sampler with conditional PL, no edge selection
- Check posterior means recover true parameters (correlation > 0.9)
- Use `dev/plans/mixedMRF/mixedGM/dev/conditional_vs_marginal_pl.R`
  as template

---

## Phase C — Marginal pseudo-likelihood

### C.1 Implement `log_marginal_omrf(int s)`

Same structure as `log_conditional_omrf(s)` but uses $\Theta$ instead
of $K_{xx}$ and adds the $\mu_y$ bias term.  The self-interaction must be
excluded explicitly:

```
Theta_ = Kxx_ + 2 * Kxy_ * Kyy_inv_ * Kxy_^T     (cached; see C.3)

// n-vector of rest-scores:
rest_score = x_dbl * Theta_.col(s)              // include all n obs, all p vars
           - x_dbl.col(s) * Theta_(s, s)        // subtract self-interaction
           + 2.0 * arma::dot(Kxy_.row(s),
                             Kyy_inv_ * muy_)   // scalar bias, same for all obs
```

This matches `rcpp_log_pl_marginal_omrf` in `mixedGM/src/log_likelihoods.cpp`
exactly.  The inner log-partition loop is then identical to `log_conditional_omrf`.

### C.2 Modify `update_continuous_mean(int j)` for marginal mode

In marginal PL mode, changing $\mu_y$ affects all $p$ marginal OMRF terms.
The acceptance ratio becomes:

```
sum_{s=1}^{p} [log_marginal_omrf(s, proposed) - log_marginal_omrf(s, current)]
+ [log_conditional_ggm(proposed) - log_conditional_ggm(current)]
+ log_prior_ratio
```

This is more expensive; patch the R prototype (factor-of-two fix +
`dnorm(log = TRUE)`) before generating fixtures so C++ and R target the
same expression.

### C.3 Cache invalidation for marginal PL

**Per-proposal proposed-Theta rule** (see Phase B.2): When proposing a
$K_{xx}$, $K_{xy}$, or $K_{yy}$ move in marginal PL mode, the proposed
marginal OMRF likelihood must be evaluated with a proposed $\Theta$
computed from the proposed parameter values.  This means each of those
update functions must build a local `Theta_prop` before calling
`log_marginal_omrf(s)` with the proposed value.  See Phase B.2 for the
exact temporary-variable protocol.

**Θ recompute granularity.** Recomputing $\Theta$ after every individual
element change is $O(p^2 q + pq^2)$ per move, which is expensive for
large $p, q$. The mitigation already noted in the risk register covers
the accepted-Theta path:  after **accepting** a $K_{xx}$/$K_{xy}$/$K_{yy}$
move, update the cached `Theta_` once from the new parameter state.  Do
not recompute `Theta_` more than once per accepted move.

For initial implementation, full recompute
is simpler and correct. Rank-1 shortcuts for single $K_{xy,ij}$ changes
are a future optimization.

### C.4 Configuration dispatch

`do_one_metropolis_step` dispatches between conditional and marginal
via `use_marginal_pl_`. The dispatch happens inside each update function,
not at the loop level — most updates have different acceptance ratios
in the two modes.

### C.5 Testing checkpoint — marginal PL estimation

**Test 3: Marginal likelihood agreement**
- Same data as Test 1, compute `log_marginal_omrf(s)` in R and C++
- Assert agreement

**Test 4: Recovery (marginal PL)**
- Same setup as Test 2 but `pseudolikelihood = "marginal"`
- Check posterior means recover true parameters

**Test 5: Conditional vs marginal agreement**
- Run both modes on the same data, compare posterior means
- They should be similar (not identical — different approximations)

---

## Phase D — Edge selection

### D.1 Discrete edge selection (`update_edge_indicator_Kxx`)

For each pair $(i, j)$ with $i < j$:
1. Propose $G'_{xx,ij} = 1 - G_{xx,ij}$
2. If **birth** ($G_{xx,ij}=0 \to 1$): propose $K'_{xx,ij} \sim N(K_{xx,ij}, \sigma)$;
   `k_curr = Kxx_(i,j)`, `k_prop = rnorm(k_curr, sigma)` (k_curr = 0 on a true birth).
3. If **death** ($G_{xx,ij}=1 \to 0$): set $K'_{xx,ij} = 0$.
4. Compute log acceptance ratio:
   - Likelihood: `omrf(i) + omrf(j)` at proposed vs current parameters
   - Slab prior: on birth add `log_slab_prior(k_prop)`, on death subtract `log_slab_prior(k_curr)`
   - **Hastings ratio** (proposal asymmetry):
     - On birth: subtract `dnorm(k_prop, k_curr, sigma, log=true)` (cost of generating the proposed value)
     - On death: add `dnorm(k_curr, k_prop, sigma, log=true)` = `dnorm(k_curr, 0, sigma, log=true)`
       (cost of the reverse birth, which would propose k_curr from a Normal centred on 0)
   - Inclusion prior: $\log(\pi) - \log(1 - \pi)$ ratio (or reverse on death)
5. After an **accepted** move, refresh `Theta_` (marginal mode).

All log-density calls must use `log = true` to keep all terms on the log scale.
Follows `cond_omrf_update_association_indicator_pair` in the R prototype.
Transplant the Hastings terms verbatim from that R code; do not re-derive.

### D.2 Continuous edge selection (`update_edge_indicator_Kyy`)

Cholesky-based birth/death as in `GGMModel`:
1. Permute, Cholesky, extract constants
2. If birth: propose $\phi' \sim N(0, \sigma)$, rebuild
3. If death: set off-diagonal to 0, rebuild
4. Acceptance: `cond_ggm()` + $\sum_s$ `marg_omrf(s)` (marginal mode only)
  + inclusion prior + slab/proposal density
5. On acceptance, run the same cache pipeline as an MH precision update
  (`Kyy` → decompositions → `conditional_mean_` → `Theta_`).

Follows `cond_ggm_update_precision_indicator_pair` in the R prototype.

**Reuse opportunity:** Directly reuse `GGMModel::update_edge_indicator_parameter_pair`
logic, adapted for the mixed model's conditional GGM likelihood.

### D.3 Cross edge selection (`update_edge_indicator_Kxy`)

Cross-edges $G_{xy,ij}$ share the same Bernoulli inclusion prior $\pi$ as
$G_{xx}$ and $G_{yy}$ (decided).  A single $\pi$ keeps the SBM prior
uniform across all edge types; if different sparsity assumptions are needed
for cross-type edges in the future, a separate hyperparameter can be added
then.

For each pair $(i, j)$ where $i \in \{1..p\}$, $j \in \{1..q\}$:
1. Propose $G'_{xy,ij} = 1 - G_{xy,ij}$
2. If birth: propose $K'_{xy,ij} \sim N(K_{xy,ij}, \sigma)$
3. If death: set $K'_{xy,ij} = 0$
4. Acceptance: `omrf(i) + cond_ggm()` in conditional mode;
  `cond_ggm() + \sum_s marg_omrf(s)` in marginal mode; spike-and-slab priors
  plus log-dense Hastings terms.
5. On acceptance, update `conditional_mean_` (both modes) and `Theta_`
  (marginal mode) before the next likelihood evaluation.

Follows `cond_omrf_update_cross_association_indicator_pair` in the R prototype.

### D.4 Implement `update_edge_indicators()`

```cpp
void MixedMRFModel::update_edge_indicators() {
    if (!edge_selection_active_) return;

    // Shuffle edge order
    // ...

    // Discrete-discrete edges
    for (int i = 0; i < p_ - 1; ++i)
        for (int j = i + 1; j < p_; ++j)
            update_edge_indicator_Kxx(i, j);

    // Continuous-continuous edges
    for (int i = 0; i < q_ - 1; ++i)
        for (int j = i + 1; j < q_; ++j)
            update_edge_indicator_Kyy(i, j);

    // Cross edges
    for (int i = 0; i < p_; ++i)
        for (int j = 0; j < q_; ++j)
            update_edge_indicator_Kxy(i, j);
}
```

### D.5 Testing checkpoint — edge selection

**Test 6: Structure recovery**
- Generate data from a sparse mixed graph (some edges zero)
- Run with edge selection, check posterior inclusion probabilities
  recover the true structure (true edges have high PIP, false edges low)

---

## Phase E — R interface and integration

### E.1 Create `sample_mixed.cpp`

Rcpp interface function `sample_mixed_mrf_cpp(...)`:
- Takes R data (integer matrix `x`, numeric matrix `y`)
- Creates `MixedMRFModel`
- Creates edge prior
- Calls `run_mcmc_sampler()`
- Returns results as `Rcpp::List`

Follow the pattern of `sample_ggm.cpp` and `sample_omrf.cpp`.

### E.2 Extend `bgm()` in R

The user interface uses **Option A** (decided): a single data frame plus a
`variable_type` argument:

```r
bgm(data, variable_type = c("o", "o", "c", ...))
```

- `data`: an $n \times (p+q)$ data frame or matrix with all variables.
- `variable_type`: character vector, length $p+q$; values `"o"` (ordinal)
  or `"c"` (continuous). Column order must match `variable_type`.

`bgm()` splits `data` into the integer matrix `x` (ordinal columns) and
numeric matrix `y` (continuous columns), then dispatches to
`sample_mixed_mrf_cpp()`. The split indices must be stored in the output
object so that `coef`, `predict`, and `simulate` methods can reconstruct
the original column order.

**Missing data** is an explicit non-goal for this PR.
`has_missing_data()` returns `false` and `impute_missing()` is an empty
override so the shared sampler pipeline compiles. Future imputation support
for mixed data should be a separate phase (Phase H).

### E.3 Extend `build_output.R`

The output structure needs to accommodate:
- Separate interaction matrices: `Kxx`, `Kyy`, `Kxy` (or a combined one)
- Threshold samples: `mux` array
- Mean samples: `muy` matrix
- Edge indicators decomposed by type

### E.4 Testing checkpoint — end-to-end

**Test 7: `bgm()` with mixed data**
- Call `bgm()` with `variable_type` containing both ordinal and continuous
- Check output structure matches expected format
- Verify S3 methods work (`print`, `summary`, `coef`, `predict`)

---

## Phase F — Warmup, adaptation, and diagnostics

### F.1 Warmup schedule

**Pre-condition check (before Phase F begins):** Verify that `WarmupSchedule`
and `ChainRunner` support a Metropolis-only model without NUTS mass-matrix
or step-size hooks.  If `WarmupSchedule::stage_2_windows()` triggers
NUTS-specific adaptation that `MixedMRFModel` cannot satisfy, the class
must either no-op those hooks or a simplified schedule must be added.
Check `GGMModel`'s implementation of `init_metropolis_adaptation` and
`tune_proposal_sd` as the reference pattern.

The mixed MRF is Metropolis-only (no NUTS/HMC), matching the GGM model.
Use the same warmup schedule:
- Stage 1: Initial fast adaptation (75 iterations)
- Stage 2: Doubling windows for covariance adaptation
- Stage 3a: Terminal fast adaptation
- Stage 3b: Proposal SD tuning with edge selection (if enabled)
- Stage 3c: Step-size re-adaptation with edge selection active

### F.2 Robbins-Monro adaptation

Per-parameter proposal SD adaptation, matching the R prototype:
```
sigma_new = sigma_old + (acceptance - target) * weight
weight = (1/iter)^0.6
target = 0.44
```

Clamp to [0.001, 2.0].

### F.3 Init metropolis adaptation

Override `init_metropolis_adaptation(const WarmupSchedule&)` to store
the schedule for use in `tune_proposal_sd()`.

Override `tune_proposal_sd(int iteration, const WarmupSchedule&)` — the
Robbins-Monro is already embedded in each update function.

---

## Phase G — Simulation and prediction

### G.1 Mixed MRF simulation

Extend `mrf_simulation.cpp` to support mixed data generation via
block Gibbs sampling, matching `mixed_gibbs_generate()`:

1. Sample $x_s \mid x_{-s}, y$: categorical from log-sum-exp stabilized
   probabilities
2. Sample $y \mid x$: multivariate Normal with conditional mean
   $\mu_y + 2\,x\,K_{xy}\,K_{yy}^{-1}$ and covariance $K_{yy}^{-1}$

### G.2 Mixed MRF prediction

Extend `mrf_prediction.cpp` for posterior predictive checks on mixed data.

---

## Testing strategy

### Existing test infrastructure (from mixedGM)

The mixedGM prototype already provides comprehensive tests that can be adapted
for the bgms C++ implementation:

| mixedGM Test File | What it covers | Adaptation for bgms |
|-------------------|----------------|---------------------|
| `test-likelihood-correctness.R` | C++ likelihoods vs hand-computed values | Use as reference for bgms unit tests |
| `test-parameter-recovery.R` | Posterior mean recovery (p=2,q=1 and larger) | Port test scenarios |
| `test-edge-selection.R` | Edge birth/death moves, PIP calibration | Port test scenarios |
| `test-edge-cases.R` | p=1, q=1, binary-only ordinal | Port edge cases |
| `test-cholesky-update.R` | Cholesky update/downdate correctness | Already covered by bgms GGM tests |
| `test-data-generator.R` | Gibbs generator sanity checks | Port |
| `test-mcmc-diagnostics.R` | ESS, R-hat convergence checks | Port |
| `test-pl-comparison.R` | Conditional vs marginal PL agreement | Port |

### Test fixture availability

The mixedGM prototype implements all C++ likelihood functions as Rcpp exports.
This means test fixtures can be generated by running mixedGM R code and
comparing against the bgms C++ port:

```r
# Generate fixture in mixedGM
library(mixedGM)
set.seed(42)   # ALWAYS use a documented seed
ll_cond_omrf = rcpp_log_pl_conditional_omrf(x, y, Kxx, Kxy, mux, num_categories, i)
# Compare against bgms MixedMRFModel::log_conditional_omrf(i)
```

All fixture generation must use `set.seed(<documented value>)` before any
RNG-dependent step.  The seed and the mixedGM package version must be
recorded in a comment at the top of each fixture file.  The mixedGM
tests use `set.seed(42)` consistently; use the same seed for bgms fixtures
unless a specific test requires otherwise.

No need to patch the prototype — all math bugs have already been fixed.

Cross-reference the existing mixedGM test files for each scenario:
- `tests/testthat/test-likelihood-correctness.R` (likelihood unit tests)
- `tests/testthat/test-parameter-recovery.R` (recovery scenarios)
- `tests/testthat/test-edge-selection.R` (structure recovery scenarios)
- `tests/testthat/test-edge-cases.R` (p=1, q=1, binary ordinal)

### Unit tests (per-function correctness)

| Test | What | How |
|------|------|-----|
| T1 | `log_conditional_omrf(s)` | Compare C++ vs R prototype at known parameters |
| T2 | `log_conditional_ggm()` | Compare C++ vs R prototype at known parameters |
| T3 | `log_marginal_omrf(s)` | Compare C++ vs R prototype at known parameters |
| T4 | `parameter_dimension()` | Check count for known p, q, num_categories |
| T5 | Vectorization round-trip | `set(get(params)) == params` |
| T6 | Cholesky permutation | Verify permute is involution, PD maintained |
| T7 | Analytic Gaussian check | Set $K_{xy}=0$ so `log_conditional_ggm()` reduces to a standard MVN and compare against closed form |
| T8 | Fixture replay | Load deterministic R fixtures and ensure C++ reproduces saved log-likelihood components bit-for-bit |
| T9 | Cache freshness | After each parameter tweak, recompute `conditional_mean_` and (if needed) `Theta_` from scratch and compare with cached copies |

### Integration tests (sampling correctness)

| Test | What | How |
|------|------|-----|
| T7 | Conditional PL recovery | Generate → estimate → check cor > 0.9 |
| T8 | Marginal PL recovery | Generate → estimate → check cor > 0.9 |
| T9 | Edge selection recovery | Sparse graph → check PIP > 0.5 for true, < 0.5 for false |
| T10 | Cond vs marginal agreement | Both modes → posterior means close |
| T11 | Reproducibility | Same seed → identical output |
| T12 | Multi-chain | 4 chains → R-hat < 1.1 |

### Edge-case & stress tests

| Test | What | How |
|------|------|-----|
| T13 | Kyy PD invariant | Run 1k iterations with aggressive proposals; Cholesky of $K_{yy}$ must succeed after every accepted move |
| T14 | Cache consistency under RJ | After edge births/deaths, recompute caches and assert equality (debug build) |
| T15 | Degenerate $p=1$ | Single ordinal variable, ensure sampler runs and recovers $K_{xy}, K_{yy}$ |
| T16 | Degenerate $q=1$ | Scalar precision, ensure Cholesky permutation code handles the trivial case |
| T17 | Binary-only ordinal | All $C_s=1$, verify conditional PL matches logistic rest-scores |
| T18 | Gibbs generator sanity | Compare empirical moments from `mixed_gibbs_generate()` against theoretical targets before using it for fixtures |

### Regression tests

| Test | What |
|------|------|
| R1 | Existing GGM tests still pass |
| R2 | Existing OMRF tests still pass |
| R3 | Existing bgmCompare tests still pass |

---

## Reuse inventory

### From mixedGM prototype (port to bgms)

| Component | mixedGM Location | Target in bgms | Action |
|-----------|-----------------|----------------|--------|
| Conditional OMRF likelihood | `src/log_likelihoods.cpp` | `mixed_mrf_likelihoods.cpp` | Port (remove Rcpp exports) |
| Marginal OMRF likelihood | `src/log_likelihoods.cpp` | `mixed_mrf_likelihoods.cpp` | Port |
| Conditional GGM likelihood | `src/log_likelihoods.cpp` | `mixed_mrf_likelihoods.cpp` | Port |
| Θ computation | `src/log_likelihoods.cpp` | `mixed_mrf_likelihoods.cpp` | Port |
| Log-sum-exp helper | `src/log_likelihoods.cpp` | `mixed_mrf_likelihoods.cpp` | Port |
| Gibbs data generator | `src/mixed_gibbs.cpp` | `mrf_simulation.cpp` | Extend |
| Cholesky helpers | `src/cholupdate.cpp` | Already in bgms | N/A |
| Cholesky permute/R() | `R/continuous_variable_helper.R` | `mixed_mrf_cholesky.cpp` | Port to C++ |
| MH Kxx updates | `R/cond_omrf_mh_update_functions.R` | `mixed_mrf_metropolis.cpp` | Port to C++ |
| MH Kyy updates | `R/cond_ggm_mh_update_functions.R` | `mixed_mrf_metropolis.cpp` | Port to C++ |
| MH Kxy updates | `R/cond_omrf_mh_update_functions.R` | `mixed_mrf_metropolis.cpp` | Port to C++ |
| Edge selection (Kxx) | `R/cond_omrf_mh_update_functions.R` | `mixed_mrf_edge_selection.cpp` | Port to C++ |
| Edge selection (Kyy) | `R/cond_ggm_mh_update_functions.R` | `mixed_mrf_edge_selection.cpp` | Port to C++ |
| Edge selection (Kxy) | `R/cond_omrf_mh_update_functions.R` | `mixed_mrf_edge_selection.cpp` | Port to C++ |
| Test fixtures | `tests/testthat/test-*.R` | `tests/testthat/test-mixed-*.R` | Adapt |

### From bgms existing infrastructure (direct reuse)

| Component | Source | Reuse type |
|-----------|--------|------------|
| Cholesky update/downdate | `src/models/ggm/cholupdate.h` | Direct include |
| Log-sum-exp stabilization | `OMRFModel::compute_logZ_*` | Adapt pattern |
| Robbins-Monro adaptation | `OMRFModel::robbins_monro_*` | Direct reuse |
| Edge prior (SBM) | `src/priors/sbm_edge_prior.h` | Direct reuse |
| WarmupSchedule | `src/mcmc/execution/warmup_schedule.h` | Direct reuse |
| ChainRunner | `src/mcmc/execution/chain_runner.h` | Direct reuse |
| SafeRNG | `src/rng/rng_utils.h` | Direct reuse |
| Progress manager | `src/utils/progress_manager.h` | Direct reuse |
| R output builder | `R/build_output.R` | Extend |
| Validation functions | `R/validate_data.R` | Extend |
| Rcpp interface pattern | `src/sample_ggm.cpp` | Copy pattern |

---

## Risk register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Kyy inversion per MH step is $O(q^3)$ | Slow for large $q$ | Cache `Kyy_inv_`; incremental rank-1 updates after Cholesky proposals |
| Marginal PL $\mu_y$ update evaluates all $p$ OMRF terms | Slow for large $p$ | Cache rest-scores; provide conditional PL as faster default |
| $\Theta$ recomputation after every $K_{yy}$/$K_{xy}$ change | $O(p^2 q + pq^2)$ | Defer $\Theta$ recompute to once per accepted move, not per-element; proposed-state uses a temporary (Phase B.2, C.3) |
| Edge selection order effects | Bias | Shuffle edge order each iteration (already done in OMRF) |
| **Sticky edge indicators** | Poor PIP mixing | Shuffle cross-edge and xx-edge update order each iteration; monitor PIP stability and autocorrelation across chains |
| Numerical instability in log-sum-exp | NaN/Inf | Use stabilized version from OMRF (subtract max) |
| Factor 2 convention mismatch | Wrong posteriors | Document consistently; unit-test against R prototype |
| PD violation during Kyy proposals | Crash | Cholesky-based proposals guarantee PD by construction |
| Large parameter space mixing | Poor ESS | Per-parameter Robbins-Monro; future: block updates or HMC |

---

## Updated implementation workflow

Given that mixedGM provides tested C++ likelihood implementations and
validated R MH updates, the recommended workflow is:

### Phase 1: Port likelihoods (1-2 days)

1. Create `src/models/mixed/` directory
2. Copy `mixedGM/src/log_likelihoods.cpp` → `mixed_mrf_likelihoods.cpp`
3. Remove Rcpp exports, convert to internal functions
4. Adapt to bgms coding conventions (SafeRNG, etc.)
5. **Validation**: Call mixedGM Rcpp functions from R and compare to bgms

### Phase 2: Skeleton class (1-2 days)

1. Create `mixed_mrf_model.h` following GGMModel pattern
2. Implement all BaseModel pure virtuals (stubs where needed)
3. Implement `parameter_dimension()`, `get/set_vectorized_parameters()`
4. Implement `clone()`, `set_seed()`, lifecycle hooks
5. **Validation**: Class compiles, vectorization round-trip passes

### Phase 3: MH updates (2-3 days)

1. Port `cond_omrf_update_*` functions from R to C++
2. Port `cond_ggm_update_*` functions from R to C++
3. Port `cond_update_cross_associations` from R to C++
4. Implement `do_one_metropolis_step()` calling all update functions
5. **Validation**: Compare posterior samples vs mixedGM on same data

### Phase 4: Edge selection (2-3 days)

1. Port `*_update_*_indicator_pair` functions from R to C++
2. Implement `update_edge_indicators()` calling all three edge types
3. Implement `initialize_graph()`, `prepare_iteration()`
4. **Validation**: Run edge selection tests from mixedGM

### Phase 5: R interface (1-2 days)

1. Create `sample_mixed.cpp` following `sample_ggm.cpp` pattern
2. Extend `bgm()` to detect and dispatch mixed data
3. Extend `build_output.R` for mixed model output structure
4. **Validation**: End-to-end `bgm()` call works

### Phase 6: Gibbs generator (1 day)

1. Port `mixed_gibbs_generate_cpp()` to `mrf_simulation.cpp`
2. Extend `simulate.bgms()` for mixed models
3. **Validation**: Generated data matches mixedGM output

### Total estimated effort: 8-13 days

The C++ likelihood implementations (Phase B-C in the original plan) are
largely complete via mixedGM, reducing the remaining work to porting MH
updates and wiring into the bgms infrastructure.
