#ifndef RESHAPE_H
#define RESHAPE_H

#include <iostream>
#include "xtensor/xarray.hpp"

class Reshape{

    //Parameter -> takes in the name of the function

public:
    bool isTrainable;
    std::vector<int> shape;
    std::vector<int> input_shape;
    std::vector<int> prev_shape;

    Reshape();
    Reshape(std::vector<int> shape, std::vector<int> input_shape);
    xt::xarray<double> forward_pass(xt::xarray<double> X, bool training);
    xt::xarray<double> backward_pass(xt::xarray<double> accum_grad);
    std::vector<int> output_shape();
    void set_input_shape(std::vector<int> shape){
        input_shape = shape;
    }

};

#endif 