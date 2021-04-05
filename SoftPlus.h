#ifndef SOFTPLUS_H
#define SOFTPLUS_H 

#include <iostream>
#include "xtensor/xarray.hpp"


class SoftPlus{
    public:
        xt::xarray<double> softplus(xt::xarray<double> x);
        xt::xarray<double> gradient(xt::xarray<double> x);
};

#endif