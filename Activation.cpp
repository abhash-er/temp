#include "Activation.h"

#include "ReLU.h"
#include "Sigmoid.h"
#include "Softmax.h"
#include "SoftPlus.h"
#include "Tanh.h"
#include "LeakyReLU.h"


Activation::Activation(){
    activation_name = "relu";
    isTrainable = true;
}

Activation::Activation(std::string func_name){
    activation_name = func_name;
    isTrainable = true;
}

void Activation::set_input_shape(std::vector<int> shape){
    input_shape = shape;
}

xt::xarray<double> Activation::forward_pass(xt::xarray<double> X, bool training = true){
    layer_input = X;

    if(activation_name == "relu"){
        ReLU reluFunc = ReLU(); 
        return reluFunc.relu(X);
    }
    else if(activation_name == "sigmoid"){
        Sigmoid sigmoidFunc = Sigmoid();   
        return sigmoidFunc.sigmoid(X);
    }
    else if(activation_name == "softmax"){
        SoftMax softmaxFunc = SoftMax();
        return softmaxFunc.softmax(X);
    }
    else if(activation_name == "leaky_relu"){
        LeakyReLU leakyreluFunc = LeakyReLU(); 
        return leakyreluFunc.leakyReLU(X);
    }
    else if(activation_name == "tanh"){
        Tanh tanhFunc = Tanh();
        return tanhFunc.tanh(X);
    }
    else if(activation_name == "softplus"){
        SoftPlus softplusFunc = SoftPlus();
        return softplusFunc.softplus(X);
    }

    
}

xt::xarray<double> Activation::backward_pass(xt::xarray<double> accum_grad){

    if(activation_name == "relu"){
        ReLU reluFunc = ReLU(); 
        return accum_grad*reluFunc.gradient(layer_input);
    }
    else if(activation_name == "sigmoid"){
        Sigmoid sigmoidFunc = Sigmoid();   
        return sigmoidFunc.gradient(layer_input);
    }
    else if(activation_name == "softmax"){
        SoftMax softmaxFunc = SoftMax();
        return softmaxFunc.gradient(layer_input);
    }
    else if(activation_name == "leaky_relu"){
        LeakyReLU leakyreluFunc = LeakyReLU(); 
        return leakyreluFunc.gradient(layer_input);
    }
    else if(activation_name == "tanh"){
        Tanh tanhFunc = Tanh();
        return tanhFunc.gradient(layer_input);
    }
    else if(activation_name == "softplus"){
        SoftPlus softplusFunc = SoftPlus();
        return softplusFunc.gradient(layer_input);
    }
}

std::vector<int> Activation::output_shape(){
    return input_shape;
}


