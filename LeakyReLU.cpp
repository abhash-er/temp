#include "LeakyReLU.h"

LeakyReLU::LeakyReLU(double a){
    alpha = a;
}

xt::xarray<double> LeakyReLU::leakyReLU(xt::xarray<double> x){
    return xt::where(x >=0, x, alpha*x);
}
xt::xarray<double> LeakyReLU::gradient(xt::xarray<double> x){
    return xt::where(x >=0, 1, alpha);
}