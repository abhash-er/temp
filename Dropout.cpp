#include "Dropout.h"
#include "xtensor/xrandom.hpp"

Dropout::Dropout(){
    p = 0.2;
}

Dropout::Dropout(double p_var = 0.2){
    p = p_var;
}

xt::xarray<double> Dropout::forward_pass(xt::xarray<double> X, bool training = true){
    double c = 1-p;
    std::vector<int> x_size;
    for(auto i: X.shape()){
        x_size.push_back(i);
    }
    if(training){
        mask = xt::random::rand<double>(x_size) > p;
        return X*mask;
    }
    else{
        return X*c;
    }
}

xt::xarray<double> Dropout::backward_pass(xt::xarray<double> accum_grad){
    return accum_grad*mask;
}

std::vector<int> Dropout::output_shape(){
    return input_shape;
}
