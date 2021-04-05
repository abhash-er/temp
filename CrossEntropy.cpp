#include "CrossEntropy.h"

xt::xarray<double> CrossEntropy::loss(xt::xarray<double> y,xt::xarray<double> p){
    p = xt::clip(p, 1e-15, 1 - 1e-15);
    return -y*xt::log(p) - (1-y)*xt::log(1-p);
}

xt::xarray<double> CrossEntropy::acc(xt::xarray<double> y,xt::xarray<double> p){
    int y_true_len;
    for(auto i : y.shape()){
        y_true_len = i;
        break;
    } 

    return xt::sum(xt::equal(y,p), 0) / y_true_len;
}

xt::xarray<double> CrossEntropy::gradient(xt::xarray<double> y,xt::xarray<double> p){
    p = xt::clip(p,1e-15,1 - 1e-15);
    return - (y/p) + (1-y) / (1-p);
}