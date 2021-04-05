#include "Dense.h"
#include <cmath> 
#include "xtensor/xrandom.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor-blas/xlinalg.hpp"


Dense::Dense(){
    input_shape = {128,128};
    n_units = 64;
    isTrainable = true;
}

Dense::Dense(int units, std::vector<int> shape){
    input_shape = shape;
    n_units = units;
    isTrainable = true;
}

void Dense::intialize(std::string optimizer_name, StochasticGradientDescent opt_sgd, Adam opt_adam, RMSprop opt_rms_prop, Adadelta opt_ada){
    double limit = 1 / sqrt(input_shape[0]);
    W = xt::random::rand<double>({input_shape[0],n_units},-limit,limit);
    wo = xt::zeros<double>({1,n_units});

    opt_name = optimizer_name;
    
    if(optimizer_name == "SGD"){
        W_opt_sgd = StochasticGradientDescent(opt_sgd);
        wo_opt_sgd = StochasticGradientDescent(opt_sgd);
    }

    else if(optimizer_name == "Adam"){
        W_opt_adam =  Adam(opt_adam);
        wo_opt_adam = Adam(opt_adam);
    }

    else if(optimizer_name == "RMSprop"){
        W_opt_rms = RMSprop(opt_rms_prop);
        wo_opt_rms = RMSprop(opt_rms_prop);

    }
    else if(optimizer_name == "Adadelta"){
        W_opt_ada = Adadelta(opt_ada);
        wo_opt_ada = Adadelta(opt_ada);
    } 

}

xt::xarray<double> Dense::parameters(){
    std::vector<int> shape_w;
    std::vector<int> shape_w0;
    for(auto i: W.shape()){
        shape_w.push_back(i);
    }
    for(auto i: wo.shape()){
        shape_w0.push_back(i);
    }
    std::vector<std::size_t> shape1 = {shape_w.size(),1};
    std::vector<std::size_t> shape2 = {shape_w0.size(),1};

    auto W_shape= xt::adapt(shape_w, shape1);
    auto w0_shape = xt::adapt(shape_w0, shape2);

    return xt::prod(W_shape) + xt::prod(w0_shape); 
}

xt::xarray<double> Dense::forward_pass(xt::xarray<double> X, bool training){
    layer_input = X;
    return xt::linalg::dot(X, W) + wo;
}


xt::xarray<double> Dense::backward_pass(xt::xarray<double> accum_grad){
    auto W_temp = W;

    if(isTrainable){
        auto grad_w = xt::linalg::dot(xt::transpose(layer_input),accum_grad);
        auto grad_w0 = xt::sum(accum_grad,{0}, xt::keep_dims | xt::evaluation_strategy::immediate);
 
        if(opt_name == "SGD"){
            W = W_opt_sgd.update(W,grad_w);
            wo = wo_opt_sgd.update(wo,grad_w0);
        }

        else if(opt_name == "Adam"){
            W = W_opt_adam.update(W,grad_w);
            wo = wo_opt_adam.update(wo,grad_w0);
        }

        else if(opt_name == "RMSprop"){
            W = W_opt_rms.update(W,grad_w);
            wo = wo_opt_rms.update(wo,grad_w0);
        }

        else if(opt_name == "Adadelta"){
            W = W_opt_ada.update(W,grad_w);
            wo = wo_opt_ada.update(wo,grad_w0);
        }
    }

    accum_grad = xt::linalg::dot(accum_grad,xt::transpose(W));
    return accum_grad;
}

std::vector<int> Dense::output_shape(){
    return {n_units,};
}