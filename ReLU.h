#ifndef RELU_H
#define RELU_H 

#include <iostream>
#include "xtensor/xarray.hpp"


class ReLU{
    public:
        xt::xarray<double> relu(xt::xarray<double> x);
        xt::xarray<double> gradient(xt::xarray<double> x);
};

#endif