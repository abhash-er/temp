#ifndef BATCHNORM_H
#define BATCHNORM_H

#include <iostream>
#include "xtensor/xarray.hpp"
#include "StochasticGradientDescent.h"
#include "Adam.h"
#include "RMSprop.h"
#include "Adadelta.h"



class BatchNormalization{

    //Parameter -> takes in the name of the function

public:
    bool isTrainable;
    double momentum;
    double eps;
    xt::xarray<double> running_mean;
    xt::xarray<double> running_var;

    std::vector<int> input_shape;
    xt::xarray<double> gamma;
    xt::xarray<double> beta;

    xt::xarray<double> X_centered;
    xt::xarray<double> stddev_inv;

    std::string opt_name;

    StochasticGradientDescent gamma_opt_sgd;
    Adam gamma_opt_adam;
    RMSprop gamma_opt_rmsprop;
    Adadelta gamma_opt_adadelta;

    StochasticGradientDescent beta_opt_sgd;
    Adam beta_opt_adam;
    RMSprop beta_opt_rmsprop;
    Adadelta beta_opt_adadelta;

    bool isFirst;
     

    BatchNormalization();
    BatchNormalization(double mom);
    void initialize(std::string optimizer_name,StochasticGradientDescent opt_sgd = StochasticGradientDescent(), Adam opt_adam = Adam(), RMSprop opt_rms_prop = RMSprop(), Adadelta opt_ada = Adadelta());
    int parameters();  
    void set_input_shape(std::vector<int> shape);
    xt::xarray<double> forward_pass(xt::xarray<double> X, bool training = true);
    xt::xarray<double> backward_pass(xt::xarray<double> accum_grad); 
    std::vector<int> output_shape();
    

};

#endif 