#ifndef RMSPROP_H
#define RMSPROP_H

#include <iostream>
#include "xtensor/xarray.hpp"

class RMSprop{
public:
    double learning_rate;
    double rho;
    double eps = 0.00000001;
    bool isFirst;
    xt::xarray<double> Eg;

    RMSprop(double lr = 0.01, double r = 0.9);

    xt::xarray<double> update(xt::xarray<double> w, xt::xarray<double> grad_wrt_w);
};

#endif
