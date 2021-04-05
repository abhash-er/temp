#include "Adadelta.h"

Adadelta::Adadelta(double r, double eps1){
    rho = r;
    eps = eps1;
    isFirst = true;
}

xt::xarray<double> Adadelta::update(xt::xarray<double> w, xt::xarray<double> grad_wrt_w){
    if(isFirst){
        w_updt = xt::zeros<double>(w.shape());
        E_w_updt = xt::zeros<double>(w.shape());
        E_grad = xt::zeros<double>(grad_wrt_w.shape());
        isFirst = false;
    }

    E_grad = rho*E_grad + (1-rho)*xt::pow(grad_wrt_w,2);

    auto RMS_delta_w = xt::sqrt(E_w_updt + eps);
    auto RMS_grad = xt::sqrt(E_grad + eps);

    auto adaptive_lr = RMS_delta_w / RMS_grad ; 

    w_updt = adaptive_lr * grad_wrt_w;
    E_w_updt = rho*E_w_updt + (1-rho)*xt::pow(w_updt,2);

    return w - w_updt;
}