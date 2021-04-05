#ifndef ADAM_H
#define ADAM_H

#include <iostream>
#include "xtensor/xarray.hpp"

class Adam{
public:
    double learning_rate;
    double eps = 0.00000001;
    xt::xarray<double> m;
    xt::xarray<double> v;
    xt::xarray<double> w_updt;
    bool isFirst;
    double b1;
    double b2;
    std::string optimizerType = "Adam";

    Adam(double lr= 0.001, double b11 = 0.9, double b22 = 0.999);

    xt::xarray<double> update(xt::xarray<double> w, xt::xarray<double> grad_wrt_w);
    
};

#endif