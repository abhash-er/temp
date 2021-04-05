#ifndef DROPOUT_H
#define DROPOUT_H

#include <iostream>
#include "xtensor/xarray.hpp"




class Dropout{

    //Parameter -> takes in the name of the function

public:
    double p;
    std::vector<int> input_shape;
    xt::xarray<double> mask;
    bool isTrainable;

    Dropout();
    Dropout(double p_var);
    xt::xarray<double> forward_pass(xt::xarray<double> X, bool training);
    xt::xarray<double> backward_pass(xt::xarray<double> accum_grad);
    std::vector<int> output_shape();
    void set_input_shape(std::vector<int> shape){
        input_shape = shape;
    }

};

#endif 