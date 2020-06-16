########################################################################
# import python-library
########################################################################
# default

import csv
import glob
import multiprocessing
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import sys
from functools import partial
from venv import logger
import re
import itertools

import librosa
import librosa.core
import librosa.feature

# additional
import numpy
import pandas as pd
from sklearn.metrics import roc_auc_score
import keras.models
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from tqdm import tqdm
import tensorflow as tf

tf.random.set_seed(1234)


def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


########################################################################

def file_load_stream(wav_name, mono=False):
    try:
        sr = librosa.get_samplerate(wav_name)
        frameSize = sr
        hoplength = frameSize // 2
        stream = librosa.stream(wav_name,
                                block_length=1,
                                frame_length=frameSize,
                                hop_length=hoplength,
                                mono=mono,
                                fill_value=0)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))
    return stream


########################################################################
# feature extractor
########################################################################
tamanhoVec = []


def file_to_vector_array(file_name,
                         n_mels=128,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames
    global tamanhoVec
    # 02 generate melspectrogram using librosa
    y, sr = file_load(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1
    tamanhoVec.append(vector_array_size)
    # 05 skip too short clips
    if vector_array_size < 1:
        return numpy.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = numpy.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

    return vector_array


contador = 0


def file_to_vector_array_stream(file_name,
                                n_mels=128,
                                frames=5,
                                n_fft=1024,
                                hop_length=512,
                                power=2):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions

    dims = n_mels * frames
    global tamanhoVec
    global contador
    # 02 generate melspectrogram using librosa
    stream = file_load_stream(file_name)
    sr = librosa.get_samplerate(file_name)
    lista = []
    for n, y in enumerate(stream):
        mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                         sr=sr,
                                                         n_fft=n_fft,
                                                         hop_length=hop_length,
                                                         n_mels=n_mels,
                                                         power=power)

        audio = librosa.effects.pitch_shift(y, sr, 3)

        mel_spectogram_audioPitched = librosa.feature.melspectrogram(y=audio,
                                                                     sr=sr,
                                                                     n_fft=n_fft,
                                                                     hop_length=hop_length,
                                                                     n_mels=n_mels,
                                                                     power=power)
        # 03 convert melspectrogram

        log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)
        log_mel_spectrogramPitched = 20.0 / power * numpy.log10(mel_spectogram_audioPitched + sys.float_info.epsilon)
        # 04 calculate total vector size
        vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1  # aqui será len(
        # log_mel_spectrogram[0, :])
        vector_array_size2 = len(log_mel_spectrogramPitched[0, :]) - frames + 1
        # vezes 2 visto que teremos a versão "normal" e warped!

        tamanhoVec.append(vector_array_size)
        # 05 skip too short clips
        if vector_array_size < 1:
            return numpy.empty((0, dims))
        if vector_array_size2 < 1:
            return numpy.empty((0, dims))

        # 06 generate feature vectors by concatenating multiframes
        vector_array = numpy.zeros((vector_array_size, dims))
        vector_array2 = numpy.zeros((vector_array_size2, dims))
        for t in range(frames):
            vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

        for t in range(frames):
            vector_array2[:, n_mels * t: n_mels * (t + 1)] = mel_spectogram_audioPitched[:, t: t + vector_array_size2].T

        lista.append(vector_array)
        lista.append(vector_array2)
        contador += 1

    lista2 = numpy.asarray(lista)
    lista2 = lista2.reshape(lista2.shape[0] * lista2.shape[1], lista2.shape[2])
    return lista2


def file_to_vector_array_stream_test_data(file_name,
                                          n_mels=128,
                                          frames=5,
                                          n_fft=1024,
                                          hop_length=512,
                                          power=1):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions

    dims = n_mels * frames
    global tamanhoVec
    global contador
    # 02 generate melspectrogram using librosa
    stream = file_load_stream(file_name)
    sr = librosa.get_samplerate(file_name)
    lista = []
    for n, y in enumerate(stream):
        mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                         sr=sr,
                                                         n_fft=n_fft,
                                                         hop_length=hop_length,
                                                         n_mels=n_mels,
                                                         power=power)

        # 03 convert melspectrogram

        log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

        # 04 calculate total vector size
        vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1  # aqui será len(
        # log_mel_spectrogram[0, :])

        # vezes 2 visto que teremos a versão "normal" e warped!

        tamanhoVec.append(vector_array_size)
        # 05 skip too short clips
        if vector_array_size < 1:
            return numpy.empty((0, dims))

        # 06 generate feature vectors by concatenating multiframes
        vector_array = numpy.zeros((vector_array_size, dims))

        for t in range(frames):
            vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

        lista.append(vector_array)

        contador += 1

    lista2 = numpy.asarray(lista)
    lista2 = lista2.reshape(lista2.shape[0] * lista2.shape[1], lista2.shape[2])
    return lista2


def numpy_dataset(train_files):
    from tqdm import tqdm
    par_file = partial(file_to_vector_array)
    tq_files = tqdm(train_files)
    np_data = numpy.concatenate(list(map(par_file, tq_files)))
    return np_data


def numpy_dataset_stream(train_files):
    par_file = partial(file_to_vector_array_stream)
    # tq_files = tqdm(train_files)
    from pqdm.processes import pqdm
    import multiprocessing
    d = pqdm(train_files, par_file, n_jobs=multiprocessing.cpu_count())
    list_audio = []
    for i in d:
        list_audio.append(i)
    newlist = numpy.asarray(list_audio)
    newlist = newlist.reshape(newlist.shape[0] * newlist.shape[1], newlist.shape[2])
    return newlist


def numpy_dataset_stream_noaugment(train_files):
    par_file = partial(file_to_vector_array_stream_test_data)

    from pqdm.processes import pqdm
    import multiprocessing
    d = pqdm(train_files, par_file, n_jobs=multiprocessing.cpu_count())
    list_audio = []
    for i in d:
        list_audio.append(i)
    newlist = numpy.asarray(list_audio)
    newlist = newlist.reshape(newlist.shape[0] * newlist.shape[1], newlist.shape[2])
    return newlist


def get_model(inputDim, name_for_model=""):
    """
    define the keras model
    the model based on the simple dense auto encoder
    (128*128*128*128*8*128*128*128*128)
    """
    # x (512*512*512*512*8*512*512*512*512) activation Relu, optimizer adam base x spectogram = 2' --> best para o
    # slider!
    # ver o caso para o toycar! com elu e o que acontecerá?
    inputLayer = Input(shape=(inputDim,))

    h = Dense(512)(inputLayer)  # estava 128
    h = BatchNormalization()(h)
    h = keras.layers.ReLU()(h)

    h = Dense(512)(h)
    h = BatchNormalization()(h)
    h = keras.layers.ReLU()(h)

    h = Dense(512)(h)
    h = BatchNormalization()(h)
    h = keras.layers.ReLU()(h)

    h = Dense(512)(h)
    h = BatchNormalization()(h)
    h = keras.layers.ReLU()(h)

    h = Dense(8)(h)
    h = BatchNormalization()(h)
    h = keras.layers.ReLU()(h)

    h = Dense(512)(h)
    h = BatchNormalization()(h)
    h = keras.layers.ReLU()(h)

    h = Dense(512)(h)
    h = BatchNormalization()(h)
    h = keras.layers.ReLU()(h)

    h = Dense(512)(h)
    h = BatchNormalization()(h)
    h = keras.layers.ReLU()(h)

    h = Dense(512)(h)
    h = BatchNormalization()(h)
    h = keras.layers.ReLU()(h)

    h = Dense(inputDim)(h)

    vae2 = Model(inputLayer, h, name='AE' + name_for_model)

    vae2.compile(optimizer='adam', loss="mse")
    return vae2


def get_machine_id_list_for_test(target_dir,
                                 dir_name="test",
                                 ext="wav"):
    """
    target_dir : str
        base directory path of "dev_data" or "eval_data"
    test_dir_name : str (default="test")
        directory containing test data
    ext : str (default="wav)
        file extension of audio files
    return :
        machine_id_list : list [ str ]
            list of machine IDs extracted from the names of test files
    """
    # create test files
    dir_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(dir_path))
    # extract id
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list


def eval_model(target_dir, model, machine_type):
    ### load parameters
    # initialize lines in csv for AUC and pAUC
    csv_lines = [[machine_type], ["id", "AUC", "pAUC"]]
    # results by type
    performance = []

    machine_id_list = get_machine_id_list_for_test(target_dir)

    for id_str in machine_id_list:
        # load test file
        test_files, y_true = test_file_list_generator(target_dir, id_str)

        # setup anomaly score file path
        anomaly_score_csv = ("{result}/anomaly_score_{machine_type}_{id_str}.csv"
                             .format(result="models/results/",
                                     machine_type=machine_type,
                                     id_str=id_str))

        anomaly_score_list = []
        print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
        y_pred = [0. for k in test_files]
        for file_idx, file_path in tqdm(enumerate(test_files),
                                        total=len(test_files)):
            try:
                data = file_to_vector_array(file_path)
                data = data.reshape(data.shape[0], data.shape[1], 1)
                from sklearn.metrics.pairwise import cosine_similarity
                errors = numpy.mean(numpy.square(data - model.predict(data)),
                                    axis=1)
                y_pred[file_idx] = numpy.mean(errors)
                anomaly_score_list.append([os.path.basename(file_path),
                                           y_pred[file_idx]])
            except:
                print("file broken!!: {}".format(file_path))

        # save anomaly score
        save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
        print("anomaly score result ->  {}".format(anomaly_score_csv))

        # append AUC and pAUC to lists
        auc = roc_auc_score(y_true, y_pred)
        p_auc = roc_auc_score(y_true, y_pred, max_fpr=0.1)
        csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
        performance.append([auc, p_auc])
        print("AUC : {}".format(auc))
        print("pAUC : {}".format(p_auc))

        print("\n============ END OF TEST FOR A MACHINE ID ============")

    return csv_lines


def eval_model_stream(target_dir, model, machine_type):
    ### load parameters
    # initialize lines in csv for AUC and pAUC
    csv_lines = [[machine_type], ["id", "AUC", "pAUC"]]
    # results by type
    performance = []

    machine_id_list = get_machine_id_list_for_test(target_dir)

    for id_str in machine_id_list:
        # load test file
        test_files, y_true = test_file_list_generator(target_dir, id_str)

        # setup anomaly score file path
        anomaly_score_csv = ("{result}/anomaly_score_{machine_type}_{id_str}.csv"
                             .format(result="models/results/",
                                     machine_type=machine_type,
                                     id_str=id_str))

        anomaly_score_list = []
        print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
        y_pred = [0. for k in test_files]
        for file_idx, file_path in tqdm(enumerate(test_files),
                                        total=len(test_files)):
            data = file_to_vector_array_stream_test_data(file_path)
            errors = numpy.mean(numpy.square(data - model.predict(data)),
                                axis=1)
            y_pred[file_idx] = numpy.mean(errors)
            anomaly_score_list.append([os.path.basename(file_path),
                                       y_pred[file_idx]])

        # save anomaly score
        save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
        print("anomaly score result ->  {}".format(anomaly_score_csv))

        # append AUC and pAUC to lists
        auc = roc_auc_score(y_true, y_pred)
        p_auc = roc_auc_score(y_true, y_pred, max_fpr=0.1)
        csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
        performance.append([auc, p_auc])
        print("AUC : {}".format(auc))
        print("pAUC : {}".format(p_auc))

        print("\n============ END OF TEST FOR A MACHINE ID ============")

    return csv_lines


def eval_model_stream_evaluation(target_dir, model, machine_type):
    ### load parameters
    # initialize lines in csv for AUC and pAUC
    csv_lines = [[machine_type], ["id", "AUC", "pAUC"]]
    # results by type
    performance = []

    machine_id_list = get_machine_id_list_for_test(target_dir)

    for id_str in machine_id_list:
        # load test file
        test_files = test_file_list_generator_eval(target_dir, id_str)

        # setup anomaly score file path
        anomaly_score_csv = ("{result}/anomaly_score_{machine_type}_{id_str}.csv"
                             .format(result="models/results/",
                                     machine_type=machine_type,
                                     id_str=id_str))

        anomaly_score_list = []
        print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
        y_pred = [0. for k in test_files]
        for file_idx, file_path in tqdm(enumerate(test_files),
                                        total=len(test_files)):
            data = file_to_vector_array_stream_test_data(file_path)
            errors = numpy.mean(numpy.square(data - model.predict(data)),
                                axis=1)
            y_pred[file_idx] = numpy.mean(errors)
            anomaly_score_list.append([os.path.basename(file_path),
                                       y_pred[file_idx]])

        # save anomaly score
        save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
        print("anomaly score result ->  {}".format(anomaly_score_csv))

        print("\n============ END OF TEST FOR A MACHINE ID ============")

    return None


def test_file_list_generator(target_dir, id_name, dir_name="test",
                             prefix_normal="normal", prefix_anomaly="anomaly",
                             ext="wav"):
    normal_files = sorted(
        glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}"
                  .format(dir=target_dir, dir_name=dir_name,
                          prefix_normal=prefix_normal,
                          id_name=id_name, ext=ext)))

    normal_labels = numpy.zeros(len(normal_files))
    anomaly_files = sorted(
        glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}"
                  .format(dir=target_dir, dir_name=dir_name,
                          prefix_anomaly=prefix_anomaly,
                          id_name=id_name, ext=ext)))

    anomaly_labels = numpy.ones(len(anomaly_files))
    files = numpy.concatenate((normal_files, anomaly_files), axis=0)
    labels = numpy.concatenate((normal_labels, anomaly_labels), axis=0)

    if len(files) == 0:
        print("no_wav_file!!")

    return files, labels


def test_file_list_generator_eval(target_dir, id_name, dir_name="test",
                                  ext="wav"):
    files = sorted(
        glob.glob("{dir}/{dir_name}/{id_name}*.{ext}"
                  .format(dir=target_dir, dir_name=dir_name,
                          id_name=id_name, ext=ext)))

    if len(files) == 0:
        print("no_wav_file!!")

    return files


def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n', delimiter=';')
        writer.writerows(save_data)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    path = os.getcwd()

    folders = []
    foldersForChallenge = ["ToyCar", "ToyConveyor", "valve", "slider", "pump", "fan"]  # machines

    # iterate machines
    for f in foldersForChallenge:
        callbacks_listv1 = [

        files_Read = []
        print("Machine Processing", f)

        # load model
        AEBest = keras.models.load_model("modelBest" + "_" + f + "Additional.H5")

        # eval model
        eval_model_stream_evaluation(path + "/" + f, AEBest, f)
