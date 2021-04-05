#ifndef TANH_H
#define TANH_H 

#include <iostream>
#include "xtensor/xarray.hpp"


class Tanh{
    public:
        xt::xarray<double> tanh(xt::xarray<double> arr);
        xt::xarray<double> gradient(xt::xarray<double> arr);
};

#endif
