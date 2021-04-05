#ifndef STOCHASTIC_H
#define STOCHASTIC_H

#include <iostream>
#include "xtensor/xarray.hpp"


class StochasticGradientDescent{
public:
    double learning_rate;
    double momentum;
    xt::xarray<double> w_updt;
    bool isFirst;

    std::string optimizerType = "SGD";

    StochasticGradientDescent(double lr = 0.01, double m = 0);
    xt::xarray<double> update(xt::xarray<double> w, xt::xarray<double> grad_wrt_w);

};

#endif
