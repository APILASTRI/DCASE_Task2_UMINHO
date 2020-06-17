# DCASE_Task2_UMINHO

Anomalous sound detection (ASD) is the task of identifying whether the sound emitted from a target machine is normal or anomalous. The automatic detection of mechanical failures can bring numerous benefits for industry 4.0 and for factory automation using artificial intelligence[1]. 
The challenge (dcase 2020 task 2) is to detect unknown anomalies sounds under the condition that only normal sound samples have were provided as training data. For this task, a baseline system implementation was provided for comparison purposes [1].  
About the machine learning model to detect anomalies, two methods involving autoencoder were used: Deep Autoencoder and Convolutional Autoencoder. About the audio features,the log energies derived directly from the filter bank energies (called MFECs) were used.

## :bookmark_tabs: Table of contents
- [Libraries](#libraries)
- [Dense Autoencoder](#dense-autoencoder)
- [Convolutional Autoencoder](#convolutional-autoencoder)
- [Results](#results)
- [References](#references)


## Libraries
- Keras                   
- matplotlib                    
- numpy                         
- PyYAML                        
- scikit-learn                 
- librosa                       
- tensorflow                    
- tqdm                        

#### Usage

##### 1. Clone repository
Clone this repository from Github. 

##### 2. Download and Unzip datasets
- Development dataset
  - Download `dev_data_<Machine_Type>.zip` from https://zenodo.org/record/3678171.
- "Additional training dataset", i.e. the evaluation dataset for training
  - After launch, download `eval_data_train_<Machine_Type>.zip` from https://zenodo.org/record/3727685 (not available until April. 1).
- "Evaluation dataset", i.e. the evaluation for test
  - After launch, download `eval_data_test_<Machine_Type>.zip` from https://zenodo.org/record/3841772 (not available until June. 1).

## Dense Autoencoder

A simple Dense AE was proposed. The overall system architecture is shown in the following figure.

<img src="https://user-images.githubusercontent.com/62994395/84875585-0eddb480-b07e-11ea-88dc-214a0d93adf6.png" width="480" height="220" />

## Convolutional Autoencoder

A simple Convolutional AE was proposed. The overall system architecture is shown in the following figure. 

<img src="https://user-images.githubusercontent.com/62994395/84875963-94616480-b07e-11ea-9ac1-fe62baa35201.png" width="520" height="250" />


#### Description
The Convolutional AE system consists of three main scripts:
- `features.py`
  - This script generates and saves the features for each Machine Type by using the directory **dev_data/<Machine_Type>/train/** or **eval_data/<Machine_Type>/train/**.
- `00_train.py`
  - This script trains models for each Machine Type by using the features extracted with the previous script.
- `01_test.py`
  - This script makes csv files for each Machine ID including the anomaly scores for each wav file in the directory **dev_data/<Machine_Type>/test/** or **eval_data/<Machine_Type>/test/**.
  - The csv files will be stored in the directory **result/**.
  - If the mode is "development", it also makes the csv files including the AUC and pAUC for each Machine ID. 

#### Usage

##### 1. Directory structure
Make the directory structure as follows:
- ./dcase2020
    - /ConvAE
        - /00_train.py
        - /01_test.py
        - /common.py
        - /keras_model.py
        - /config.yaml
    - /dev_data
        - /ToyCar
            - /train (Only normal data for all Machine IDs are included.)
                - /normal_id_01_00000000.wav
                - ...
                - /normal_id_04_00000999.wav
            - /test (Normal and anomaly data for all Machine IDs are included.)
                - /normal_id_01_00000000.wav
                - ...
                - /anomaly_id_04_00000264.wav
        - /ToyConveyor (The other Machine Types have the same directory structure as ToyCar.)
        - /fan
        - /pump
        - /slider
        - /valve
    - /eval_data
        - /ToyCar
            - /train ( "additional training dataset". Only normal data for all Machine IDs are included.)
                - /normal_id_05_00000000.wav
                - ...
                - /normal_id_07_00000999.wav
            - /test ("evaluation dataset". Normal and anomaly data for all Machine IDs are included, but there is no label about normal or anomaly.)
                - /id_05_00000000.wav
                - ...
                - /id_07_00000514.wav
        - /ToyConveyor (The other machine types have the same directory structure as ToyCar.)
        - /fan
        - /pump
        - /slider
        - /valve


#### 2. Change parameters
You can change the parameters for feature extraction and model definition by editing `config.yaml`.

#### 3. Run training script 
Run the training script `00_train.py`. 
Use the option `-d` for the development dataset or `-e` for the evaluation dataset.
Use the option `--target` to select only one machine type (optional)
```
$ python 00_train.py -d --target "ToyCar"
```

`00_train.py` trains the models and saves the trained models in the directory **model/**.

#### 4. Run test script (for development dataset)
Run the test script `01_test.py`.
Use the option  `-d` for the development dataset or `-e` for the evaluation dataset.
Use the option `--target` to select only one machine type (optional).
```
$ python 01_test.py -d  --target "ToyCar"
```
The options for `01_test.py` are the same as those for `00_train.py`.
`01_test.py` calculates the anomaly scores for each wav file in the directory **dev_data/<Machine_Type>/test/**.
The csv files for each Machine ID including the anomaly scores will be stored in the directory **result/**.
If the mode is "development", the script also makes the csv files including the AUCs and pAUCs for each Machine ID. 





## :chart_with_upwards_trend: Results
The table below shows the performance results of DCASE 2020 Task 2 for the development dataset in which the best (mean) results are in bold. Best mean for each machine type: 
- **ToyCar**: Dense AE with 80.79% AUC and 71.17% pAUC
- **ToyConveyor**: Dense AE with 76.43% AUC and 63.79% pAUC
- **fan**: Dense AR with 72.03% AUC and 53.25% pAUC
- **pump**: Dense AE with 73.06% AUC and Conv AE with 60.96% pAUC
- **slider**: Conv AE with 91.77% AUC and 76.20% pAUC
- **valve**: Conv AE with 78.83% AUC and 53.10% pAUC

![result-crop](https://user-images.githubusercontent.com/23443227/84788627-3c2b5380-afe7-11ea-8a7f-a69a950ce9fa.png)

## :page_with_curl: References

[1] Koizumi, Y., Kawaguchi, Y., Imoto, K., Nakamura, T., Nikaido, Y., Tanabe, R., ... & Harada, N. (2020). Description and Discussion on DCASE2020 Challenge Task2: Unsupervised Anomalous Sound Detection for Machine Condition Monitoring. arXiv preprint arXiv:2006.05822.
