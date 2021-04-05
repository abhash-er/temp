#include "RMSprop.h"

RMSprop::RMSprop(double lr, double r){
    learning_rate = lr;
    rho = r;
    isFirst = true;
}

xt::xarray<double> RMSprop::update(xt::xarray<double> w, xt::xarray<double> grad_wrt_w){
    if(isFirst){
        Eg = xt::zeros<double>(grad_wrt_w.shape());
        isFirst = false;
    }
    Eg = rho*Eg + (1-rho)*xt::pow(grad_wrt_w,2);

    return w- learning_rate * grad_wrt_w / xt::sqrt(Eg + eps); 
}
