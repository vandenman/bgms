// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// sample_omrf_gibbs
IntegerMatrix sample_omrf_gibbs(int no_states, int no_variables, IntegerVector no_categories, NumericMatrix interactions, NumericMatrix thresholds, int iter);
RcppExport SEXP _bgms_sample_omrf_gibbs(SEXP no_statesSEXP, SEXP no_variablesSEXP, SEXP no_categoriesSEXP, SEXP interactionsSEXP, SEXP thresholdsSEXP, SEXP iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type no_states(no_statesSEXP);
    Rcpp::traits::input_parameter< int >::type no_variables(no_variablesSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type no_categories(no_categoriesSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type interactions(interactionsSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type thresholds(thresholdsSEXP);
    Rcpp::traits::input_parameter< int >::type iter(iterSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_omrf_gibbs(no_states, no_variables, no_categories, interactions, thresholds, iter));
    return rcpp_result_gen;
END_RCPP
}
// sample_bcomrf_gibbs
IntegerMatrix sample_bcomrf_gibbs(int no_states, int no_variables, IntegerVector no_categories, NumericMatrix interactions, NumericMatrix thresholds, StringVector variable_type, IntegerVector reference_category, int iter);
RcppExport SEXP _bgms_sample_bcomrf_gibbs(SEXP no_statesSEXP, SEXP no_variablesSEXP, SEXP no_categoriesSEXP, SEXP interactionsSEXP, SEXP thresholdsSEXP, SEXP variable_typeSEXP, SEXP reference_categorySEXP, SEXP iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type no_states(no_statesSEXP);
    Rcpp::traits::input_parameter< int >::type no_variables(no_variablesSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type no_categories(no_categoriesSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type interactions(interactionsSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type thresholds(thresholdsSEXP);
    Rcpp::traits::input_parameter< StringVector >::type variable_type(variable_typeSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type reference_category(reference_categorySEXP);
    Rcpp::traits::input_parameter< int >::type iter(iterSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_bcomrf_gibbs(no_states, no_variables, no_categories, interactions, thresholds, variable_type, reference_category, iter));
    return rcpp_result_gen;
END_RCPP
}
// gibbs_sampler
List gibbs_sampler(IntegerMatrix observations, IntegerMatrix gamma, NumericMatrix interactions, NumericMatrix thresholds, IntegerVector no_categories, double interaction_scale, NumericMatrix proposal_sd, NumericMatrix proposal_sd_blumecapel, String edge_prior, NumericMatrix theta, double beta_bernoulli_alpha, double beta_bernoulli_beta, IntegerMatrix Index, int iter, int burnin, IntegerMatrix n_cat_obs, IntegerMatrix sufficient_blume_capel, double threshold_alpha, double threshold_beta, bool na_impute, IntegerMatrix missing_index, LogicalVector variable_bool, IntegerVector reference_category, bool save, bool display_progress, bool edge_selection);
RcppExport SEXP _bgms_gibbs_sampler(SEXP observationsSEXP, SEXP gammaSEXP, SEXP interactionsSEXP, SEXP thresholdsSEXP, SEXP no_categoriesSEXP, SEXP interaction_scaleSEXP, SEXP proposal_sdSEXP, SEXP proposal_sd_blumecapelSEXP, SEXP edge_priorSEXP, SEXP thetaSEXP, SEXP beta_bernoulli_alphaSEXP, SEXP beta_bernoulli_betaSEXP, SEXP IndexSEXP, SEXP iterSEXP, SEXP burninSEXP, SEXP n_cat_obsSEXP, SEXP sufficient_blume_capelSEXP, SEXP threshold_alphaSEXP, SEXP threshold_betaSEXP, SEXP na_imputeSEXP, SEXP missing_indexSEXP, SEXP variable_boolSEXP, SEXP reference_categorySEXP, SEXP saveSEXP, SEXP display_progressSEXP, SEXP edge_selectionSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< IntegerMatrix >::type observations(observationsSEXP);
    Rcpp::traits::input_parameter< IntegerMatrix >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type interactions(interactionsSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type thresholds(thresholdsSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type no_categories(no_categoriesSEXP);
    Rcpp::traits::input_parameter< double >::type interaction_scale(interaction_scaleSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type proposal_sd(proposal_sdSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type proposal_sd_blumecapel(proposal_sd_blumecapelSEXP);
    Rcpp::traits::input_parameter< String >::type edge_prior(edge_priorSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type beta_bernoulli_alpha(beta_bernoulli_alphaSEXP);
    Rcpp::traits::input_parameter< double >::type beta_bernoulli_beta(beta_bernoulli_betaSEXP);
    Rcpp::traits::input_parameter< IntegerMatrix >::type Index(IndexSEXP);
    Rcpp::traits::input_parameter< int >::type iter(iterSEXP);
    Rcpp::traits::input_parameter< int >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< IntegerMatrix >::type n_cat_obs(n_cat_obsSEXP);
    Rcpp::traits::input_parameter< IntegerMatrix >::type sufficient_blume_capel(sufficient_blume_capelSEXP);
    Rcpp::traits::input_parameter< double >::type threshold_alpha(threshold_alphaSEXP);
    Rcpp::traits::input_parameter< double >::type threshold_beta(threshold_betaSEXP);
    Rcpp::traits::input_parameter< bool >::type na_impute(na_imputeSEXP);
    Rcpp::traits::input_parameter< IntegerMatrix >::type missing_index(missing_indexSEXP);
    Rcpp::traits::input_parameter< LogicalVector >::type variable_bool(variable_boolSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type reference_category(reference_categorySEXP);
    Rcpp::traits::input_parameter< bool >::type save(saveSEXP);
    Rcpp::traits::input_parameter< bool >::type display_progress(display_progressSEXP);
    Rcpp::traits::input_parameter< bool >::type edge_selection(edge_selectionSEXP);
    rcpp_result_gen = Rcpp::wrap(gibbs_sampler(observations, gamma, interactions, thresholds, no_categories, interaction_scale, proposal_sd, proposal_sd_blumecapel, edge_prior, theta, beta_bernoulli_alpha, beta_bernoulli_beta, Index, iter, burnin, n_cat_obs, sufficient_blume_capel, threshold_alpha, threshold_beta, na_impute, missing_index, variable_bool, reference_category, save, display_progress, edge_selection));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_bgms_sample_omrf_gibbs", (DL_FUNC) &_bgms_sample_omrf_gibbs, 6},
    {"_bgms_sample_bcomrf_gibbs", (DL_FUNC) &_bgms_sample_bcomrf_gibbs, 8},
    {"_bgms_gibbs_sampler", (DL_FUNC) &_bgms_gibbs_sampler, 26},
    {NULL, NULL, 0}
};

RcppExport void R_init_bgms(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
