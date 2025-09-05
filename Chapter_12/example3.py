#  Example 12.3 VAE - Example_3_MNIST Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import objectives
from keras.datasets import mnist

import h5py
from pathlib import Path
from PIL import Image

from keras import backend as KBE
KBE.set_image_data_format('channels_last')

save_files = False

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.dirname(current_dir)) # path to parent dir
from DLBasics_Utilities import File_Helper
file_helper = File_Helper(save_files)

random_seed = 42
np.random.seed(random_seed)

make_20_small_only = True


def get_big_VAE_models(latent_dim):
    batch_size = 100
    epsilon_std = 1.0
    
    # These routines are part of the model, so we can't use numpy functions.
    # Instead, we use Keras backend functions that know how to talk to layers
    # and handle the data coming in and going out
    
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = KBE.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + KBE.exp(z_log_var / 2) * epsilon
    
    def vae_loss(input_layer, output_layer):
        image_loss = original_dim * objectives.binary_crossentropy(input_layer, decoder_output)
        kl_loss = - 0.5 * KBE.sum(1 + z_log_var - KBE.square(z_mean) - KBE.exp(z_log_var), axis=-1)
        return image_loss + kl_loss

    # build the encoder stage
    input_layer = Input(batch_shape=(batch_size, original_dim))
    encoder_hidden_1 = Dense(1000, activation='relu')(input_layer)
    encoder_hidden_2 = Dense(500, activation='relu')(encoder_hidden_1)
    encoder_hidden_3 = Dense(250, activation='relu')(encoder_hidden_2)
    encoder_hidden_4 = Dense(latent_dim, activation='relu')(encoder_hidden_3)
    
    # the fancy split and sampling stages
    z_mean = Dense(latent_dim)(encoder_hidden_4)
    z_log_var = Dense(latent_dim)(encoder_hidden_4)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # by saving the decoder layers we can use them again to make the generator
    decoder_hidden_1 = Dense(250, activation='relu')
    decoder_hidden_2 = Dense(500, activation='relu')
    decoder_hidden_3 = Dense(1000, activation='relu')
    output_layer = Dense(original_dim, activation='sigmoid')
    
    # build the decoder stage
    decoder_stack_1 = decoder_hidden_1(z)
    decoder_stack_2 = decoder_hidden_2(decoder_stack_1)
    decoder_stack_3 = decoder_hidden_3(decoder_stack_2)
    decoder_output = output_layer(decoder_stack_3)

    # build and compile the start-to-finish VAE model
    VAE = Model(input_layer, decoder_output)
    VAE.compile(optimizer='adam', loss=vae_loss)

    # save models for the mean and var encoders, and the full VAE encoder stage
    mean_encoder = Model(input_layer, z_mean)
    var_encoder = Model(input_layer, z_log_var)
    encoder = Model(input_layer, z)

    # re-use the decoder layers to build a standalone generator
    generator_input = Input(shape=(latent_dim,))
    generator_stack_1 = decoder_hidden_1(generator_input)
    generator_stack_2 = decoder_hidden_2(generator_stack_1)
    generator_stack_3 = decoder_hidden_3(generator_stack_2)
    generator_output = output_layer(generator_stack_3)
    generator = Model(generator_input, generator_output)
    
    weights_filename = 'NB4-VAE_big_weights_only_latent_dim_'+str(latent_dim)
        
    return (latent_dim, weights_filename, VAE, encoder, generator)


# constants for all models
original_dim = 784
batch_size = 100

def get_VAE_models(latent_dim):
   
    (latent_dim, weights_filename, VAE, encoder, generator) = get_big_VAE_models(latent_dim)
    np.random.seed(42)
    if not file_helper.load_model_weights(VAE, weights_filename):
        print("No weights file - training the model")
        np.random.seed(random_seed)
        number_of_epochs = 25
        history = VAE.fit(X_train, X_train,
            shuffle=True,
            epochs=number_of_epochs,
            batch_size=batch_size,
            validation_data=(X_test, X_test))
        file_helper.save_model_weights(VAE, weights_filename)  
        
    return (latent_dim, VAE, encoder, generator)

# Read MNIST data. We won't be using the y_train or y_test data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
pixels_per_image = np.prod(X_train.shape[1:])

# Cast values into the current floating-point type
X_train = KBE.cast_to_floatx(X_train)
X_test = KBE.cast_to_floatx(X_test)

# Normalize the range from [0,255] to [0,1]
X_train /= 255.
X_test /= 255.

# Reshape the data into a grid with one row per sample, each row 784 (28*28) pixels
X_train = X_train.reshape((len(X_train), pixels_per_image))
X_test = X_test.reshape((len(X_test), pixels_per_image))

print("X_train.shape = ",X_train.shape, " X_test.shape = ",X_test.shape)

(latent_dim, VAE, encoder, generator) = get_VAE_models(30)

print("latent_dim=",latent_dim)
print("VAE:")
VAE.summary()
print("encoder:")
encoder.summary()
print("generator:")
generator.summary()

predictions = VAE.predict(X_test, batch_size=batch_size)
print(predictions[0])

