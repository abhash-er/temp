#include "SoftPlus.h"

xt::xarray<double> SoftPlus::softplus(xt::xarray<double> x){
    return xt::log(1 + xt::exp(x));
}

xt::xarray<double> SoftPlus::gradient(xt::xarray<double> x){
    return 1 / (1 + xt::exp(-x));
}
