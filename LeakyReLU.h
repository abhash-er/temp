#ifndef LEAKYRELU_H
#define LEAKYRELU_H

#include <iostream>
#include "xtensor/xarray.hpp"


class LeakyReLU{
public:
    double alpha;
    
    LeakyReLU(double a = 0.2);

    xt::xarray<double> leakyReLU(xt::xarray<double> x);
    xt::xarray<double> gradient(xt::xarray<double> x);
};

#endif