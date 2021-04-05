#include "Reshape.h"

Reshape::Reshape(){
    input_shape = {128,128};
    isTrainable = true;
    shape = {128,128};
}

Reshape::Reshape(std::vector<int> shape, std::vector<int> input_shape){
    input_shape = input_shape;
    isTrainable = true;
    shape = shape;
}



xt::xarray<double> Reshape::forward_pass(xt::xarray<double> X, bool training = true){
    auto temp_shape = X.shape();
    prev_shape = {};
    for(auto i:temp_shape){
        prev_shape.push_back(i);
    }

    std::vector<int> new_shape = {};
    new_shape.push_back(prev_shape[0]);
    for(auto i :shape){
        new_shape.push_back(i);
    }

    return X.reshape(new_shape);
}

xt::xarray<double> Reshape::backward_pass(xt::xarray<double> accum_grad){
    return accum_grad.reshape(prev_shape);
}

std::vector<int> Reshape::output_shape(){
    return shape;
}