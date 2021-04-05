#include <iostream>
#include "xtensor/xarray.hpp"
#include "neural_network.h"
#include "xtensor/xrandom.hpp"
#include "xtensor/xindex_view.hpp"

class Autoencoder{
public:
    int rows;
    int cols;
    int dim;
    int latent_dim;

    NeuralNetwork encoder;
    NeuralNetwork decoder;
    NeuralNetwork autoencoder;

    Autoencoder(){
        rows = 28;
        cols = 28;
        dim = rows * cols;
        latent_dim = 128;
        auto optimizer = optimizer_container(Adam(0.0002,0.5));
        auto loss_function = loss_container(SquareLoss());

        encoder = build_encoder(optimizer, loss_function);
        decoder = build_decoder(optimizer, loss_function);
        autoencoder = NeuralNetwork(optimizer, loss_function, xt::xarray<double> {0}, xt::xarray<double> {0}, false);

        
        for(auto i : encoder.layers){
            autoencoder.layers.push_back(i);
        }

        for(auto i : decoder.layers){
            autoencoder.layers.push_back(i);
        }
    }

    NeuralNetwork build_encoder(optimizer_container optimizer, loss_container loss){
        auto encoder_temp = NeuralNetwork(optimizer,loss);
        encoder_temp.add(layer_container(Dense(512, {dim, })));
        encoder_temp.add(layer_container(Activation("leaky_relu")));
        encoder_temp.add(layer_container(BatchNormalization(0.8)));
        encoder_temp.add(layer_container(Dense(256,encoder_temp.prev_out_shape)));
        encoder_temp.add(layer_container(Activation("leaky_relu")));
        encoder_temp.add(layer_container(BatchNormalization(0.8)));
        encoder_temp.add(layer_container(Dense(latent_dim,encoder_temp.prev_out_shape)));

        return encoder_temp;   
    }

    NeuralNetwork build_decoder(optimizer_container optimizer, loss_container loss){
        auto decoder_temp = NeuralNetwork(optimizer,loss);
        decoder_temp.add(layer_container(Dense(256,{latent_dim,})));
        decoder_temp.add(layer_container(Activation("leaky_relu")));
        decoder_temp.add(layer_container(BatchNormalization(0.8)));
        decoder_temp.add(layer_container(Dense(512,decoder_temp.prev_out_shape)));
        decoder_temp.add(layer_container(Activation("leaky_relu")));
        decoder_temp.add(layer_container(BatchNormalization(0.8)));
        decoder_temp.add(layer_container(Dense(dim,decoder_temp.prev_out_shape)));
        decoder_temp.add(layer_container(Activation("tanh")));
    
        return decoder_temp;
    }

    void train(xt::xarray<double> X,xt::xarray<double> y,int n_epochs, int batch_size = 128){
        //rescale in [-1,1]
        X = (xt::cast<double>(X) - 127.5)/127.5;

        for(int epoch =0;epoch < n_epochs;epoch++){
            int x_shape;
            for(auto i:X.shape()){
                x_shape = i;
                break;
            }
            auto idx =  xt::random::randint({batch_size,},0,x_shape);
            auto imgs = xt::index_view(X, idx);
            auto return_train = autoencoder.train_on_batch(imgs,imgs);
            auto loss = return_train[0];

            std::cout << "Loss at epoch" << epoch << ": " << loss;
        }
    }

};

int main(){
    Autoencoder autoenc = Autoencoder();
    

    return 0;
}