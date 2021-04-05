#ifndef ADADELTA_H
#define ADADELTA_H

#include <iostream>
#include "xtensor/xarray.hpp"

class Adadelta{
public:
    double eps;
    double rho;
    xt::xarray<double> w_updt;
    xt::xarray<double> E_w_updt;
    xt::xarray<double> E_grad;
    bool isFirst;
    std::string optimizerType = "Adadelta";

    Adadelta(double r = 0.95, double eps1 = 0.000001);

    xt::xarray<double> update(xt::xarray<double> w, xt::xarray<double> grad_wrt_w);
};

#endif