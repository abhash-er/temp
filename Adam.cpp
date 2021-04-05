#include "Adam.h"

Adam::Adam(double lr, double b11, double b22){
    learning_rate = lr;
    b1 = b11;
    b2 = b22;
    isFirst = true;
}

xt::xarray<double> Adam::update(xt::xarray<double> w, xt::xarray<double> grad_wrt_w){
    if(isFirst){ 
        m = xt::zeros<double>(grad_wrt_w.shape());
        v = xt::zeros<double>(grad_wrt_w.shape()); 
        isFirst = false;
    }
    m = b1*m + (1-b1)*grad_wrt_w;
    v = b2*v + (1-b2)*xt::pow(grad_wrt_w,2);

    xt::xarray<double> m_hat = m / (1-b1);
    xt::xarray<double> v_hat = v / (1-b2);

    w_updt = learning_rate * m_hat / ((xt::sqrt(v_hat)) + eps); 

    return w - w_updt; 
}