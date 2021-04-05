#include "Softmax.h"

xt::xarray<double> SoftMax::softmax(xt::xarray<double> x){
    int ax = -1;
    for(auto& e1 : x.shape()){ax++;}

    auto e_x = xt::exp(x - xt::sum(x,{ax}, xt::keep_dims | xt::evaluation_strategy::immediate));

    ax = -1;
    for(auto& e1 : e_x.shape()){ax++;}

    return e_x / xt::sum(e_x,{ax}, xt::keep_dims | xt::evaluation_strategy::immediate);
}

xt::xarray<double> SoftMax::gradient(xt::xarray<double> x){
    auto p = SoftMax::softmax(x);
    return p*(1-p);
}