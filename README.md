# DCASE_Task2_UMINHO

Anomalous sound detection (ASD) is the task of identifying whether the sound emitted from a target machine is normal or anomalous. The automatic detection of mechanical failures can bring numerous benefits for industry 4.0 and for factory automation using artificial intelligence (AI)[1]. 
The challenge (dcase 2020 task 2) is to detect unknown anomalies sounds under the condition that only normal sound samples have were provided as training data. For this task, a baseline system implementation was provided for comparison purposes [1].  
About the machine learning model to detect anomalies, two methods involving autoencoder were used: Deep Autoencoder and Convulacional Autoencoder. About the features of the audios, It was used the log energies derived directly from the filter bank energies (called MFECs)....

## :bookmark_tabs: Table of contents
- [Libraries](#libraries)
    - [Convolutional Autoencoder](#convolutional-autoencoder-lib)
    - [Dense Autoencoder](#dense-autoencoder-lib) 
- [Dense Autoencoder](#dense-autoencoder)
- [Convolutional Autoencoder](#convolutional-autoencoder)
- [Results](#results)
- [References](#references)


## Libraries
### Convolutional Autoencoder
Libraries:
- Lib1
- Lib2
### Dense Autoencoder
Libraries:
- Lib1
- Lib2

## Dense Autoencoder

## Convolutional Autoencoder


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
