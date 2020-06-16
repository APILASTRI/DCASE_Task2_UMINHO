"""
 @file   00_train.py
 @brief  Script for training

"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import sys
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
import math
import numpy as np
import keras
import random

import librosa
import librosa.core
import librosa.feature

from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed

#import librosa.core
# from import
from tqdm import tqdm
# original lib
import common as com
import keras_model as keras_model

########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################

# set seed
########################################################################
set_random_seed(1234)
########################################################################


########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save png file path.

        return : None
        """
        self.plt.savefig(name)


########################################################################

def file_list_generator(target_dir,
                        dir_name="train",
                        ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    dir_name : str (default="train")
        directory name containing training data
    ext : str (default="wav")
        file extension of audio files

    return :
        train_files : list [ str ]
            file list of wav files for training
    """
    com.logger.info("target_dir : {}".format(target_dir))

    # generate training list
    if dir_name==None:
        training_list_path = os.path.abspath("{dir}/*.{ext}".format(dir=target_dir, ext=ext))
    else: 
        training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
        com.logger.exception("no_wav_file!!")

    com.logger.info("train_file num : {num}".format(num=len(files)))
    return files


########################################################################~
# Data Loader
########################################################################
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(32,128), shuffle=True, step=8):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle

        self.data = np.load(self.list_IDs[0] , mmap_mode='r')
        
        self.step = step
        self.indexes_start = np.arange(self.data.shape[1]-self.dim[0]+self.step, step=self.step)
        self.max = len(self.indexes_start)
        self.indexes = np.arange(self.data.shape[0])
        
        self.indexes = np.repeat(self.indexes, self.max )
        self.indexes_start = np.repeat(self.indexes_start, self.data.shape[0])
    
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data.shape[0] * self.max  / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        indexes_start = self.indexes_start[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X = self.__data_generation(indexes, indexes_start).reshape((self.batch_size, *self.dim, 1))

        return X, X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            np.random.shuffle(self.indexes_start)


    def __data_generation(self, indexes, index_start):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, (id_file, id_start) in enumerate(zip(indexes, index_start)):

            x = self.data[id_file,]
            length, mels = x.shape

            start = id_start

            start = min(start, length - self.dim[0])
            
            # crop part of sample
            crop = x[start:start+self.dim[0], :]

            X[i,] = crop
        return X
            

########################################################################


########################################################################
# main 00_train.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode, target = com.command_line_chk()
    if mode is None:
        sys.exit(-1)
    
    # make output directory
    os.makedirs(param["model_directory"], exist_ok=True)

    # initialize the visualizer
    visualizer = visualizer()

    # load base_directory list
    dirs = com.select_dirs(param=param, mode=mode, target=target)


    # loop of the base directory (machine types)
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))

        # set path
        machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory"],
                                                                    machine_type=machine_type)
        best_model_filepath = "{model}/bestmodel_{machine_type}_".format(model=param["model_directory"],
                                                                    machine_type=machine_type)
        history_img = "{model}/history__{machine_type}.png".format(model=param["model_directory"],
                                                                    machine_type=machine_type)
        features_file_path = "{features}/{machine_type}".format(features=param["features_directory"],
                                                                    machine_type=machine_type)
        features_dir_path = os.path.abspath(features_file_path)

        
        if os.path.exists(model_file_path):
            com.logger.info("model exists")
            continue


        # get features
        # get npy files list (features files)
        list_files_npy_train = file_list_generator(features_dir_path, dir_name="train", ext="npy")
        list_files_npy_val = file_list_generator(features_dir_path, dir_name="val", ext="npy")
        
        if len(list_files_npy_train)==0 or len(list_files_npy_val)==0:
            com.logger.exception("no_npy_files!!")
            sys.exit(-1)  


        shape0_feat = param["autoencoder"]["shape0"]
        shape1_feat = param["feature"]["n_mels"]

        # load data 
        gen_train = DataGenerator(list_files_npy_train, batch_size=param["fit"]["batch_size"], dim=(shape0_feat,shape1_feat), step=param["step"])
        gen_val = DataGenerator(list_files_npy_val,  batch_size=param["fit"]["batch_size"], dim=(shape0_feat,shape1_feat), shuffle=False, step=param["step"])
        

        # train model
        print("============== MODEL TRAINING ==============")
        
        # checkpoint
        model_checkpoint = ModelCheckpoint(best_model_filepath+"{epoch:02d}.hdf5", monitor='val_loss', verbose=1, save_best_only=True)
        early = EarlyStopping(monitor='val_loss', mode='min', patience=10, min_delta=0.0001)

        # create model
        model = keras_model.get_model((shape0_feat, shape1_feat), param["autoencoder"]["latentDim"])
        model.summary()
    
        
        #train model
        model.compile(**param["fit"]["compile"])
        history = model.fit_generator(gen_train, 
                            validation_data=gen_val,
                            epochs=param["fit"]["epochs"], 
                            verbose=param["fit"]["verbose"],
                            callbacks=[model_checkpoint, early])
        
        visualizer.loss_plot(history.history["loss"], history.history["val_loss"])
        visualizer.save_figure(history_img)
        model.save(model_file_path)
        com.logger.info("save_model -> {}".format(model_file_path))
        print("============== END TRAINING ==============")

    
        
