"""
 @file   keras_model.py
 @brief  Script for keras model definition

"""

########################################################################
# import python-library
########################################################################
# from import
import keras.models
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Reshape, Flatten
from keras.layers import Conv2D, Cropping2D, Conv2DTranspose, Dense
from keras.utils.vis_utils import plot_model
from keras.backend import int_shape

########################################################################

def get_data_shape(layer):
    return tuple(int_shape(layer)[1:])

########################################################################
# keras model
########################################################################
def get_model(inputDim, latentDim):
    """
    define the keras model
    the model based on the simple convolutional auto encoder
    """
    input_img = Input(shape=(inputDim[0], inputDim[1], 1))  # adapt this if using 'channels_first' image data format

    # encoder
    x = Conv2D(32, (5, 5),strides=(1,2), padding='same')(input_img)   #32x128 -> 32x64
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (5, 5),strides=(1,2), padding='same')(x)           #32x32
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (5, 5),strides=(2,2), padding='same')(x)          #16x16
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3),strides=(2,2), padding='same')(x)          #8x8
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3),strides=(2,2), padding='same')(x)          #4x4
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    volumeSize = int_shape(x)
    # at this point the representation size is latentDim i.e. latentDim-dimensional
    x = Conv2D(latentDim, (4,4), strides=(1,1), padding='valid')(x)
    encoded = Flatten()(x)


    # decoder
    x = Dense(volumeSize[1] * volumeSize[2] * volumeSize[3])(encoded)
    x = Reshape((volumeSize[1], volumeSize[2], 512))(x)                #4x4

    x = Conv2DTranspose(256, (3, 3),strides=(2,2), padding='same')(x)  #8x8
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(128, (3, 3),strides=(2,2), padding='same')(x)  #16x16
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(64, (5, 5),strides=(2,2), padding='same')(x)   #32x32
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(32, (5, 5),strides=(1,2), padding='same')(x)   #32x64
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    decoded = Conv2DTranspose(1, (5, 5),strides=(1,2), padding='same')(x)

    return Model(inputs=input_img, outputs=decoded)
#########################################################################


def load_model(file_path):
    return keras.models.load_model(file_path)

def plot(model):
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
