#include "Tanh.h"

xt::xarray<double> Tanh::tanh(xt::xarray<double> x){
    return 2 / (1+xt::exp(-2*x)) - 1; 
}

xt::xarray<double> Tanh::gradient(xt::xarray<double> x){
    auto a = Tanh::tanh(x);
    return 1 - xt::pow(a,2);

}