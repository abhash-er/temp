#include "SquareLoss.h"

xt::xarray<double> SquareLoss::loss(xt::xarray<double> y,xt::xarray<double> y_pred){
    return 0.5 * xt::pow((y-y_pred),2);
}

xt::xarray<double> SquareLoss::gradient(xt::xarray<double> y,xt::xarray<double> y_pred){
    return -(y-y_pred);
}