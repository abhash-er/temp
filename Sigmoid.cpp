#include "Sigmoid.h"


xt::xarray<double> Sigmoid::sigmoid(xt::xarray<double> arr){
    return 1.0/(1.0+xt::exp(-arr));
}

xt::xarray<double> Sigmoid::gradient(xt::xarray<double> arr){
    xt::xarray<double> a = Sigmoid::sigmoid(arr);
    return a*(1.0-a);

}
