#ifndef SOFTMAX_H
#define SOFTMAX_H 

#include <iostream>
#include "xtensor/xarray.hpp"


class SoftMax{
    public:
        xt::xarray<double> softmax(xt::xarray<double> x);
        xt::xarray<double> gradient(xt::xarray<double> x);
};

#endif