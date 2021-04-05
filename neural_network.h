#ifndef NEURALNET_H
#define NEURALNET_H

#include <iostream>
#include "xtensor/xarray.hpp"
#include <unordered_map>
#include "StochasticGradientDescent.h"
#include "Adam.h"
#include "RMSprop.h"
#include "Adadelta.h"

#include "Sigmoid.h"
#include "Softmax.h"
#include "Tanh.h"
#include "SoftPlus.h"


#include "Dense.h"
#include "Activation.h"
#include "BatchNormalization.h"
#include "Dropout.h"
#include "Reshape.h"

#include "CrossEntropy.h"
#include "SquareLoss.h"

class layer_container: public Dense, public Activation, public BatchNormalization, public Dropout, public Reshape{
 public:   
    std::string layer_name;
    Dense dense;
    Activation act;
    BatchNormalization bn;
    Dropout dropout;
    Reshape reshape;


    layer_container(){
        layer_name = "Dense";
        dense = Dense();
    }

    layer_container(Dense a){
        layer_name = "Dense";
        dense = a;
    }

    layer_container(Activation a){
        layer_name = "Activation";
        act = a;
    }

    layer_container(BatchNormalization a){
        layer_name = "BatchNormalization";
        bn = a;
    }

    layer_container(Dropout a){
        layer_name = "Dropout";
        dropout = a;
    }

    layer_container(Reshape a){
        layer_name = "Reshape";
        reshape = a;
    }

    
};

class loss_container:public CrossEntropy, public SquareLoss{
public:
    CrossEntropy cross;
    SquareLoss square;
    std::string loss_name;

    loss_container(){
        loss_name = "CrossEntropy";
        cross = CrossEntropy();
    }

    loss_container(CrossEntropy loss_cross){
        loss_name = "CrossEntropy";
        cross = loss_cross;
    }

    loss_container(SquareLoss loss_sqaure){
        loss_name = "SqaureLoss";
        square = loss_sqaure;
    }

    xt::xarray<double> _loss(xt::xarray<double> y, xt::xarray<double> y_pred){
        if(loss_name == "CrossEntropy"){
            return cross.loss(y,y_pred);
        }
        else{
            return square.loss(y,y_pred);
        }
    }

    xt::xarray<double> _acc(xt::xarray<double> y, xt::xarray<double> y_pred){
        if(loss_name == "CrossEntropy"){
            return cross.acc(y,y_pred);
        }
        else{
            return square.acc(y,y_pred);
        }
    } 

    xt::xarray<double> _gradient(xt::xarray<double> y, xt::xarray<double> y_pred){
        if(loss_name == "CrossEntropy"){
            return cross.gradient(y,y_pred);
        }
        else{
            return square.gradient(y,y_pred);
        }
    } 

};

class optimizer_container:public StochasticGradientDescent, public Adam, public RMSprop, public Adadelta{
public:
    StochasticGradientDescent sgd;
    Adam adam;
    Adadelta ada;
    RMSprop rms;
    std::string opt_name;
    

    optimizer_container(){
        opt_name = "Adam";
        adam = Adam(); 
    }
    optimizer_container(StochasticGradientDescent x){
        opt_name = "SGD";
        sgd = x;
    }
    optimizer_container(Adam y){
        opt_name = "Adam";
        adam = y;
    }

    optimizer_container(Adadelta z){
        opt_name = "Adadelta";
        ada = z;
    }
    optimizer_container(RMSprop w){
        opt_name = "RMSProp";
        rms = w;
    }

    xt::xarray<double> update(xt::xarray<double> w, xt::xarray<double> grad_wrt_w){
        if(opt_name == "Adam"){
            return adam.update(w,grad_wrt_w);
        }
        else if(opt_name == "SGD"){
            return sgd.update(w, grad_wrt_w);
        }
        else if(opt_name == "Adadelta"){
            return ada.update(w, grad_wrt_w);
        }
        else if(opt_name == "RMSProp"){
            return rms.update(w, grad_wrt_w);
        }
    }
};


class NeuralNetwork{

public:
    optimizer_container optimizer;

    std::vector<layer_container> layers;
    loss_container loss_function;
    std::vector<int> prev_out_shape;

    std::unordered_map<std::string, xt::xarray<double>> val_set;
    std::unordered_map<std::string, std::vector<double>> errors;
    bool isValidationPresent;
    
    NeuralNetwork(optimizer_container optimizer_var = optimizer_container(Adam()),loss_container loss = loss_container(SquareLoss()),xt::xarray<double> X = {0}, xt::xarray<double> y = {0}, bool isValidation = false);
    void set_trainable(bool trainable);
    void add(layer_container layer);
    
    std::vector<double> test_on_batch(xt::xarray<double> X, xt::xarray<double> y);
    std::vector<double> train_on_batch(xt::xarray<double> X, xt::xarray<double> y);
    std::vector<std::vector<double>> fit(xt::xarray<double> X, xt::xarray<double> y, int n_epochs, int batch_size);
    xt::xarray<double> _forward_pass(xt::xarray<double> X, bool isTrainable);
    void _backward_pass(xt::xarray<double> loss_grad);
    void summary(std::string name);//not implemented
    xt::xarray<double> predict(xt::xarray<double> X);
};

#endif 