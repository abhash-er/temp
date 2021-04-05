#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <iostream>
#include "xtensor/xarray.hpp"




class Activation{

    //Parameter -> takes in the name of the function

public:
    bool isTrainable;
    std::string layer_name;
    std::string activation_name;
    int parameters;
    std::vector<int> input_shape;
    xt::xarray<double> layer_input;

    Activation();
    Activation(std::string func_name);
    void set_input_shape(std::vector<int> shape);
    xt::xarray<double> forward_pass(xt::xarray<double> X, bool training = true);
    xt::xarray<double> backward_pass(xt::xarray<double> accum_grad); 
    std::vector<int> output_shape();
    

};

#endif 