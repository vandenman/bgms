#include "models/ggm/zratio_engine.h"
#include "priors/parameter_prior.h"
#include <RcppArmadillo.h>

// [[Rcpp::export]]
Rcpp::List informed_normal_potentials_cpp(Rcpp::List spec, Rcpp::List graphs) {
    auto engine = zratio_engine_from_spec(spec);
    arma::vec potentials(graphs.size());
    for (int k = 0; k < graphs.size(); ++k)
        potentials(k) = engine->informed.potential(Rcpp::as<arma::imat>(graphs[k]));
    return Rcpp::List::create(Rcpp::Named("log_z") = potentials,
                              Rcpp::Named("hits") = engine->informed.hits,
                              Rcpp::Named("evaluations") = engine->informed.evaluations,
                              Rcpp::Named("half_width") = engine->informed.largest_half_width);
}

// [[Rcpp::export]]
Rcpp::NumericVector informed_normal_density_cpp(arma::mat mean, arma::mat sd, double x, int i,
                                                int j) {
    InformedNormalPrior prior(mean, sd);
    return Rcpp::NumericVector::create(prior.logp(x, i, j), prior.grad(x, i, j));
}

// [[Rcpp::export]]
arma::mat informed_normal_reference_cpp(const arma::imat& graph,
                                        const arma::mat& mean, const arma::mat& sd,
                                        double rate, double delta, int i, int j,
                                        int draws, int warmup, int chains, int seed) {
    const size_t p=graph.n_rows;
    arma::mat values(draws,chains), m=-2*mean, tau=2*sd;
    const double beta=rate/2;
    auto logh=[&](const arma::mat& C, const arma::uvec& neighbors,
                   const arma::uvec& rest) {
        if(neighbors.is_empty()) return 0.0;
        arma::vec prec(neighbors.n_elem), mu(neighbors.n_elem);
        for(size_t k=0;k<neighbors.n_elem;++k) {
            prec(k)=1/std::pow(tau(rest(neighbors(k)),j),2);
            mu(k)=m(rest(neighbors(k)),j);
        }
        arma::mat Q=2*beta*C.submat(neighbors,neighbors)+arma::diagmat(prec);
        arma::vec h=prec%mu;
        arma::mat L=arma::chol(Q);
        arma::vec solved=arma::solve(Q,h,arma::solve_opts::likely_sympd);
        return .5*arma::sum(arma::log(prec))-arma::sum(arma::log(L.diag()))
            -.5*arma::dot(mu%prec,mu)+.5*arma::dot(h,solved);
    };
    for(int chain=0;chain<chains;++chain) {
        SafeRNG rng(seed+chain);
        arma::mat K=arma::eye(p,p)*((delta+1)/beta*(.5+double(chain+1)/chains));
        for(int iter=0;iter<draws+warmup;++iter) {
            for(size_t vertex=0;vertex<p;++vertex) {
                arma::uvec rest(p-1); size_t at=0;
                for(size_t k=0;k<p;++k) if(k!=vertex) rest(at++)=k;
                arma::ivec column=graph.col(vertex);
                arma::uvec N=arma::find(column.elem(rest));
                arma::mat C=arma::inv_sympd(K.submat(rest,rest));
                arma::vec b(p-1,arma::fill::zeros);
                if(!N.is_empty()) {
                    arma::vec prec(N.n_elem), mu(N.n_elem);
                    for(size_t k=0;k<N.n_elem;++k) {
                        prec(k)=1/std::pow(tau(rest(N(k)),vertex),2);
                        mu(k)=m(rest(N(k)),vertex);
                    }
                    arma::mat Q=2*beta*C.submat(N,N)+arma::diagmat(prec);
                    arma::mat L=arma::chol(Q);
                    b.elem(N)=arma::solve(Q,prec%mu,arma::solve_opts::likely_sympd)
                        +arma::solve(arma::trimatu(L),arma_rnorm_vec(rng,N.n_elem));
                }
                for(size_t k=0;k<rest.n_elem;++k) K(rest(k),vertex)=K(vertex,rest(k))=b(k);
                K(vertex,vertex)=rgamma(rng,delta+1,beta)+arma::as_scalar(b.t()*C*b);
            }
            if(iter>=warmup) {
                arma::uvec rest(p-1); size_t at=0,candidate=0;
                for(size_t k=0;k<p;++k) if(k!=size_t(j)) {if(k==size_t(i)) candidate=at;rest(at++)=k;}
                arma::ivec column=graph.col(j); arma::uvec N=arma::find(column.elem(rest));
                arma::uvec plus(N.n_elem+1); if(!N.is_empty()) plus.head(N.n_elem)=N;
                plus(N.n_elem)=candidate;
                arma::mat C=arma::inv_sympd(K.submat(rest,rest));
                values(iter-warmup,chain)=std::exp(logh(C,plus,rest)-logh(C,N,rest));
            }
        }
    }
    return values;
}
