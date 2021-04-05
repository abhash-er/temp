#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H

#include <iostream>
#include "xtensor/xarray.hpp"


class CrossEntropy{
public:

    xt::xarray<double> loss(xt::xarray<double> y,xt::xarray<double> p);
    xt::xarray<double> acc(xt::xarray<double> y,xt::xarray<double> p);
    xt::xarray<double> gradient(xt::xarray<double> y,xt::xarray<double> p);
};

#endif