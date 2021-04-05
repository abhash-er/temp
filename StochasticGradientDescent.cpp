#include "StochasticGradientDescent.h"

StochasticGradientDescent::StochasticGradientDescent(double lr, double m){
    learning_rate = lr;
    momentum = m;
    isFirst = true;
}

xt::xarray<double> StochasticGradientDescent::update(xt::xarray<double> w, xt::xarray<double> grad_wrt_w){
    if(isFirst){
        w_updt  =  xt::zeros<double>(w.shape());
        isFirst = false;
    }
    
    w_updt = momentum*w_updt + (1.0-momentum)*grad_wrt_w;
    return w-(learning_rate*w_updt);
}

