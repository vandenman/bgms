#include <Rcpp.h>

class ParameterPrior
{

public:

    virtual double log_pdf(double x, int node1, int node2) const { Rf_error("you must override logpdf."); };
    virtual double rand(int node1, int node2)              const { Rf_error("you must override rand.");   };

};

class CauchyPrior: public ParameterPrior
{

public:

  // constructor
  CauchyPrior(double location, double scale): _location{location}, _scale{scale}
  {}

  double log_pdf(double x, int, int) const override { return R::dcauchy(x, _location, _scale, true); };
  double rand(int, int)              const override { return R::rcauchy(   _location, _scale);       };

private:

  const double _location,
               _scale;

};

class UnitInformationPrior: public ParameterPrior
{

public:

  // constructor
  UnitInformationPrior(Rcpp::NumericMatrix proposal_sd): _proposal_sd{proposal_sd}
  {}

  double log_pdf(double x, int node1, int node2) const override { return R::dnorm(x, 0.0, _proposal_sd(node1, node2), true); };
  double rand(int node1, int node2)              const override { return R::rnorm(   0.0, _proposal_sd(node1, node2));       };

private:

  const Rcpp::NumericMatrix _proposal_sd;

};