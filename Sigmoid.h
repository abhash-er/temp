#ifndef SIGMOID_H
#define SIGMOID_H 

#include <iostream>
#include "xtensor/xarray.hpp"


class Sigmoid{
    public:
        xt::xarray<double> sigmoid(xt::xarray<double> arr);
        xt::xarray<double> gradient(xt::xarray<double> arr);
};

#endif
