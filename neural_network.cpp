#include "neural_network.h"

#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xadapt.hpp"

std::vector<std::vector<xt::xarray<double>>> batch_iterator(xt::xarray<double> X, xt::xarray<double> y, int batch_size = 64){
    int n_samples;
    for(auto i : X.shape()){
        n_samples = i;
        break;
    }

    std::vector<std::vector<xt::xarray<double>>> batch_container;

    for(auto i: xt::arange(0,n_samples,batch_size)){
        int begin = i;
        int end = std::min(i+batch_size, n_samples);
        auto slice_X = xt::view(X, xt::range(begin,end), xt::all());
        auto slice_y = xt::view(y, xt::range(begin,end), xt::all());
        batch_container.push_back({slice_X,slice_y});
    }

    return batch_container;     
}



NeuralNetwork::NeuralNetwork(optimizer_container optimizer_var,loss_container loss,xt::xarray<double> X, xt::xarray<double> y, bool isValidation){
    optimizer = optimizer_var;
    loss_function = loss;
    isValidationPresent = isValidation;

    if(isValidation){
        val_set["X"] = X; 
        val_set["y"] = y; 
    }

    errors["training"] = {};
    errors["validation"] = {};
}

void NeuralNetwork::set_trainable(bool trainable){
    for(auto layer:layers){
        if(layer.layer_name == "Dense"){
            layer.dense.isTrainable = trainable;
        }
        else if(layer.layer_name == "Activation"){
            layer.act.isTrainable = trainable;
        }
        else if(layer.layer_name == "BatchNormalization"){
            layer.bn.isTrainable = trainable;
        }
        else if(layer.layer_name == "Dropout"){
            layer.dropout.isTrainable = trainable;
        }
        else if(layer.layer_name == "Reshape"){
            layer.reshape.isTrainable = trainable;
        }
        
    }
}




void NeuralNetwork::add(layer_container layer){

    if(layer.layer_name == "Dense"){
        layer.dense.set_input_shape(prev_out_shape);
        prev_out_shape = layer.dense.output_shape();
    }
    else if(layer.layer_name == "Activation"){
        layer.act.set_input_shape(prev_out_shape);
        prev_out_shape = layer.act.output_shape();
    }
    else if(layer.layer_name == "BatchNormalization"){
        layer.bn.set_input_shape(prev_out_shape);
        prev_out_shape = layer.bn.output_shape();
    }
    else if(layer.layer_name == "Dropout"){
        layer.dropout.set_input_shape(prev_out_shape);
        prev_out_shape = layer.dropout.output_shape();
    }
    else if(layer.layer_name == "Reshape"){
        layer.reshape.set_input_shape(prev_out_shape);
        prev_out_shape = layer.reshape.output_shape();
    }

    //initialize 
    if(layer.layer_name == "Dense"){
        layer.dense.intialize(optimizer.opt_name,optimizer.sgd,optimizer.adam,optimizer.rms,optimizer.ada);
    }
    else if(layer.layer_name == "BatchNormalization"){
        layer.bn.initialize(optimizer.opt_name,optimizer.sgd,optimizer.adam,optimizer.rms,optimizer.ada);
    }

    layers.push_back(layer);
    
}


std::vector<double> NeuralNetwork::test_on_batch(xt::xarray<double> X, xt::xarray<double> y){
    auto y_pred = _forward_pass(X,false);
    auto loss = xt::mean(loss_function._loss(y,y_pred))[0];
    auto acc = loss_function._acc(y,y_pred);

    return {loss,acc[0]};
}

std::vector<double> NeuralNetwork::train_on_batch(xt::xarray<double> X, xt::xarray<double> y){
    auto y_pred = _forward_pass(X,true);
    auto loss = xt::mean(loss_function._loss(y,y_pred))[0];
    auto acc = loss_function._acc(y,y_pred);
    auto loss_grad = loss_function._gradient(y,y_pred);
    _backward_pass(loss_grad);

    return {loss,acc[0]}; 
}

std::vector<std::vector<double>> NeuralNetwork::fit(xt::xarray<double> X, xt::xarray<double> y, int n_epochs, int batch_size){
    for(int i=0;i<n_epochs;i++){
        std::vector<double> batch_error = {};
        for(auto return_space : batch_iterator(X,y,batch_size)){
            auto X_batch = return_space[0];
            auto y_batch = return_space[1];
            
            auto return_train_batch = train_on_batch(X_batch, y_batch);
            auto loss = return_train_batch[0];

            batch_error.push_back(loss);
        }
        std::vector<std::size_t> batch_error_shape = { batch_error.size(), 1 };
        auto batch_error_tensor = xt::adapt(batch_error,batch_error_shape);
        errors["training"].push_back(xt::mean(batch_error_tensor)[0]); 

        if(isValidationPresent){
            auto return_test_batch = test_on_batch(val_set["X"],val_set["y"]);
            auto val_loss = return_test_batch[0];
            errors["validation"].push_back(val_loss);
        }

    }

    return {errors["training"],errors["validation"]};
}

xt::xarray<double> NeuralNetwork::_forward_pass(xt::xarray<double> X, bool isTrainable = true){
    auto layer_output = X;
    for(auto layer:layers){
        if(layer.layer_name == "Dense"){
            layer_output = layer.dense.forward_pass(layer_output,isTrainable);
        }
        else if(layer.layer_name == "Activation"){
            layer_output = layer.act.forward_pass(layer_output,isTrainable);
        }
        else if(layer.layer_name == "BatchNormalization"){
            layer_output = layer.bn.forward_pass(layer_output,isTrainable);
        }
        else if(layer.layer_name == "Dropout"){
            layer_output = layer.dropout.forward_pass(layer_output,isTrainable);
        }
        else if(layer.layer_name == "Reshape"){
            layer_output = layer.reshape.forward_pass(layer_output,isTrainable);
        }
        
    }

    return layer_output;
}

void NeuralNetwork::_backward_pass(xt::xarray<double> loss_grad){
    xt::xarray<double> loss_grad; 
    for(auto layer = layers.end();layer-- != layers.begin();){        
        if(layer->layer_name == "Dense"){
            loss_grad = layer->dense.backward_pass(loss_grad);
        }
        else if(layer->layer_name == "Activation"){
            loss_grad = layer->act.backward_pass(loss_grad);
        }
        else if(layer->layer_name == "BatchNormalization"){
            loss_grad = layer->bn.backward_pass(loss_grad);
        }
        else if(layer->layer_name == "Dropout"){
            loss_grad = layer->dropout.backward_pass(loss_grad);
        }
        else if(layer->layer_name == "Reshape"){
            loss_grad = layer->reshape.backward_pass(loss_grad);
        }
    }
}

void NeuralNetwork::summary(std::string name){
    std::cout << "Not Implemented";   
}

xt::xarray<double> NeuralNetwork::predict(xt::xarray<double> X){
    return _forward_pass(X,false);
}