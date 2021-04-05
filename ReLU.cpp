#include "ReLU.h"

xt::xarray<double> ReLU::relu(xt::xarray<double> x){
    return xt::where(x >= 0,x,0);
}
xt::xarray<double> ReLU::gradient(xt::xarray<double> x){
    return xt::where(x>=0,1,0);
}