"""
 @file   00_train.py
 @brief  Script for extract and save features

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
#import keras

import librosa
import librosa.core
import librosa.feature

from sklearn.externals.joblib import load, dump
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#import librosa.core
# from import
from tqdm import tqdm
# original lib
import common as com

########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################



########################################################################
# feature extractor
########################################################################

def list_to_vector_array(file_list, feat_path, feature_range, scaler, 
                            msg="calc...",
                            n_mels=64,
                            frames=5,
                            n_fft=1024,
                            hop_length=512,
                            power=2.0):
    """
    convert the file_list to features and save features.
    file_to_vector_array() is iterated, and the output vector array is saved.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.
    """
    ###### uncomment to compute scaler
    #scaler = preprocessing.StandardScaler()

    
    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):

        vector_array = com.file_to_vector_array(file_list[idx], feature_range, scaler, 
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power)
        ###### uncomment to compute scaler
        #scaler.partial_fit(X=vector_array)
        
        if idx == 0:
            X = np.empty((len(file_list), vector_array.shape[0], vector_array.shape[1]))
        X[idx,] = vector_array

    #save features 
    numpy.save(feat_path+"\\data.npy", X)
        
    ###### uncomment to compute scaler
    '''
    #save scaler
    scaler_file_path = "scalers_std_add/{machine_type}".format(machine_type=machine_type)
    # make scaler directory
    os.makedirs(scaler_file_path, exist_ok=True)
    scaler_file_path = os.path.abspath(scaler_file_path)
    dump(scaler, scaler_file_path+"/scaler_{machine_type}.bin".format(machine_type=machine_type), compress=True)
    print("dump scaler")'''
           


def file_list_generator(target_dir,
                        dir_name=None,
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




########################################################################
# main features.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode, target = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    # load base_directory list
    dirs = com.select_dirs(param=param, mode=mode, target=target)


    # loop of the base directory (machine types)
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))

    

        # set path
        machine_type = os.path.split(target_dir)[1]
        features_dir_train = "{features}/{machine_type}/{tip}".format(features=param["features_directory"],
                                                                    machine_type=machine_type, tip="train")
        features_dir_train = os.path.abspath(features_dir_train)
        features_dir_val = "{features}/{machine_type}/{tip}".format(features=param["features_directory"],
                                                                    machine_type=machine_type, tip="val")
        features_dir_val = os.path.abspath(features_dir_val)
        

        # create directory 
        if not os.path.isdir(features_dir_train):
            os.makedirs(features_dir_train) 
        else:
            # delete existing features
            list_files_npy = file_list_generator(features_dir_train, ext="npy")
            for file in list_files_npy:
                os.remove(file)

        if not os.path.isdir(features_dir_val):
            os.makedirs(features_dir_val) 
        else:
            # delete existing features
            list_files_npy = file_list_generator(features_dir_val, ext="npy")
            for file in list_files_npy:
                os.remove(file)
        
        # load scaler
        scaler_file_path = "{scalers}/{machine_type}".format(scalers=param["scalers_directory"], machine_type=machine_type)
        scaler_file_path = os.path.abspath(scaler_file_path)
        scaler = load(scaler_file_path+"/scaler_{machine_type}.bin".format(machine_type=machine_type))

        
        # generate features
        print("============== FEATURES_GENERATOR ==============")

        # get wav files list
        list_files_wav = file_list_generator(target_dir, dir_name="train")

        train_filenames, val_filenames= train_test_split( 
                        list_files_wav, test_size=param["fit"]["validation_split"], random_state=1)

        # extract and save features
        list_to_vector_array(train_filenames, features_dir_train, param["train_data"][machine_type], scaler,
                                        msg="generate train dataset",
                                        n_mels=param["feature"]["n_mels"],
                                        frames=param["feature"]["frames"],
                                        n_fft=param["feature"]["n_fft"],
                                        hop_length=param["feature"]["hop_length"],
                                        power=param["feature"]["power"])
        
        list_to_vector_array(val_filenames, features_dir_val, param["train_data"][machine_type], scaler,
                                        msg="generate val dataset",
                                        n_mels=param["feature"]["n_mels"],
                                        frames=param["feature"]["frames"],
                                        n_fft=param["feature"]["n_fft"],
                                        hop_length=param["feature"]["hop_length"],
                                        power=param["feature"]["power"])

