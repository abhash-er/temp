#ifndef SQUARELOSS_H
#define SQUARELOSS_H

#include <iostream>
#include "xtensor/xarray.hpp"


class SquareLoss{
public:

    xt::xarray<double> loss(xt::xarray<double> y,xt::xarray<double> y_pred);
    xt::xarray<double> acc(xt::xarray<double> y,xt::xarray<double> y_pred){
        return 0;
    }
    xt::xarray<double> gradient(xt::xarray<double> y,xt::xarray<double> y_pred);
};

#endif