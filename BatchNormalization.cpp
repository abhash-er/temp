#include "BatchNormalization.h"
#include <xtensor/xadapt.hpp>

BatchNormalization::BatchNormalization(){
    momentum = 0.99;
    eps = 0.01;
    running_mean = 0;
    running_var = 0;
    isFirst = true;
    isTrainable = true;
}

BatchNormalization::BatchNormalization(double mom =0.99){
    momentum = mom;
    eps = 0.01;
    running_mean = 0;
    running_var = 0;
    isFirst = true;
    isTrainable = true;
}

void BatchNormalization::initialize(std::string optimizer_name, StochasticGradientDescent opt_sgd, Adam opt_adam, RMSprop opt_rms_prop, Adadelta opt_ada){
    gamma = xt::ones<double>(input_shape);
    beta = xt::zeros<double>(input_shape);

    opt_name = optimizer_name;
    
    if(optimizer_name == "SGD"){
        gamma_opt_sgd = StochasticGradientDescent(opt_sgd);
        beta_opt_sgd = StochasticGradientDescent(opt_sgd);
    }

    else if(optimizer_name == "Adam"){
        gamma_opt_adam =  Adam(opt_adam);
        beta_opt_adam = Adam(opt_adam);
    }

    else if(optimizer_name == "RMSprop"){
        gamma_opt_rmsprop = RMSprop(opt_rms_prop);
        beta_opt_rmsprop = RMSprop(opt_rms_prop);

    }
    else if(optimizer_name == "Adadelta"){
        gamma_opt_adadelta = Adadelta(opt_ada);
        beta_opt_adadelta = Adadelta(opt_ada);
    }
}

int BatchNormalization::parameters(){
    return xt::prod(gamma.shape()) + xt::prod(beta.shape());
}

xt::xarray<double> BatchNormalization::forward_pass(xt::xarray<double> X, bool training = true){
    if(isFirst){
        running_mean = xt::mean(X);
        running_var = xt::variance(X);
        isFirst = false;
    }

    xt::xarray<double> mean, var;
    if(training && isTrainable){
        mean = xt::mean(X,{0});
        var = xt::variance(X,{0});
        running_mean = momentum*running_mean + (1-momentum)*mean;
        running_var = momentum*running_var + (1-momentum)*var;
    }
    else{
        mean = running_mean;
        var = running_var;
    }

    X_centered = X - mean;
    stddev_inv = 1 / (xt::sqrt(var + eps));

    return gamma * (X_centered * stddev_inv) + beta; 



}

xt::xarray<double> BatchNormalization::backward_pass(xt::xarray<double> accum_grad){
    auto gamma_temp = gamma;
    
    if(isTrainable){
        xt::xarray<double> X_norm = X_centered*stddev_inv;
        auto grad_gamma = xt::sum(accum_grad*X_norm,{0});
        auto grad_beta = xt::sum(accum_grad,{0});

        if(opt_name == "SGD"){
            gamma = gamma_opt_sgd.update(gamma, grad_gamma);
            beta = beta_opt_sgd.update(beta, grad_beta);
        }
        else if(opt_name == "Adam"){
            gamma = gamma_opt_adam.update(gamma, grad_gamma);
            beta = beta_opt_adam.update(beta, grad_beta);
        }
        else if(opt_name == "RMSprop"){
            gamma = gamma_opt_rmsprop.update(gamma, grad_gamma);
            beta = beta_opt_rmsprop.update(beta, grad_beta);
        }
        else if(opt_name == "Adadelta"){
            gamma = gamma_opt_adadelta.update(gamma, grad_gamma);
            beta = beta_opt_adadelta.update(beta, grad_beta);
        }
    }

    std::vector<double> batch_size_temp;
    for(auto i:accum_grad.shape()){
        batch_size_temp.push_back(i);
    }

    std::vector<std::size_t> meta_size = {batch_size_temp.size(),1};
    auto batch_size = xt::adapt(batch_size_temp,meta_size);

    accum_grad = (1/batch_size) * gamma_temp * stddev_inv * (batch_size * accum_grad - xt::sum(accum_grad,{0}) - X_centered * xt::pow(stddev_inv,2) * xt::sum(accum_grad * X_centered,{0}));

    return accum_grad;
}

std::vector<int> BatchNormalization::output_shape(){
    return input_shape;
}

void BatchNormalization::set_input_shape(std::vector<int> shape){
    input_shape = shape;
}